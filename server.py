#!/usr/bin/env python3
"""Durable, scoped memory retrieval with optional synchronous semantic indexing."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import socket
import sqlite3
import threading
import time
import unicodedata
from urllib.parse import urlsplit


DEFAULT_DB = Path(__file__).resolve().parent / ".local" / "memory.sqlite3"
MAX_BODY_BYTES = 8 * 1024 * 1024
REQUEST_HEADER_TIMEOUT_SECONDS = 30
AUTHENTICATED_BODY_TIMEOUT_SECONDS = 1800
MAX_QUERY_TERMS = 64
MAX_RETRIEVAL_CANDIDATES = 512
MAX_EVIDENCE_CHARACTERS = 500_000
MAX_WINDOW_CHARACTERS = 16_000
STOP_WORDS = frozenset("a an and are as at be been by can could did do does for from had has have how i if in is it its me my of on or our s should that the their them then there these they this to us was were what when where which who why will with would you your".split())


class ValidationError(ValueError):
    pass


class PayloadCapacityError(ValidationError):
    """Valid schema, but input exceeds the declared semantic payload capacity."""


class ConflictError(Exception):
    pass


class UnavailableError(Exception):
    """A retryable operation failed without a successful acknowledgement."""
    pass


def _object(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValidationError("Expected a JSON object")
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing:
        raise ValidationError("Missing fields: " + ", ".join(sorted(missing)))
    if extra:
        raise ValidationError("Unknown fields: " + ", ".join(sorted(extra)))


def _nonempty_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(name + " must be a nonempty string")


def validate_add(payload):
    _object(payload, ("request_id", "messages", "user_id", "session_id"))
    for name in ("request_id", "user_id", "session_id"):
        _nonempty_string(payload[name], name)
    messages = payload["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a nonempty array")
    for message in messages:
        _object(message, ("role", "content"), ("timestamp",))
        _nonempty_string(message["role"], "role")
        if message["role"] not in ("user", "assistant"):
            raise ValidationError("role must be user or assistant")
        _nonempty_string(message["content"], "content")
        if "timestamp" in message:
            timestamp = message["timestamp"]
            if type(timestamp) is not int or not -(2 ** 63) <= timestamp < 2 ** 63:
                raise ValidationError("timestamp must be a signed 64-bit Unix millisecond integer")


def validate_search(payload):
    _object(payload, ("query", "user_id", "top_k"), ("options",))
    if not isinstance(payload["query"], str):
        raise ValidationError("query must be a string")
    _nonempty_string(payload["user_id"], "user_id")
    if type(payload["top_k"]) is not int or not 1 <= payload["top_k"] <= 100:
        raise ValidationError("top_k must be an integer from 1 through 100")
    if "options" in payload:
        options = payload["options"]
        if not isinstance(options, list) or any(not isinstance(v, str) for v in options):
            raise ValidationError("options must be an array of strings")


def _cjk(character):
    number = ord(character)
    return (0x3400 <= number <= 0x4DBF or 0x4E00 <= number <= 0x9FFF
            or 0xF900 <= number <= 0xFAFF or 0x3040 <= number <= 0x30FF
            or 0x20000 <= number <= 0x3134F)


def lexical_tokens(text):
    """Unicode word tokens; Chinese/Japanese runs use characters and bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result = []
    run = []
    previous_kind = None

    def flush():
        if not run:
            return
        value = "".join(run)
        if previous_kind == "cjk":
            result.extend(value)
            result.extend(value[i:i + 2] for i in range(len(value) - 1))
        else:
            result.append(value)
        run.clear()

    for character in normalized:
        kind = "cjk" if _cjk(character) else (
            "word" if unicodedata.category(character)[0] in ("L", "N") else None)
        if kind != previous_kind or kind is None:
            flush()
        if kind is not None:
            run.append(character)
        previous_kind = kind
    flush()
    return result


def _canonical_json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class MemoryStore:
    """Per-operation snapshots and exact, binary user isolation in every index.

    Corpus/term statistics are persisted per user. SQLite ranks postings without
    copying every hit or its original content into Python. There is still work
    proportional to matching postings; the long-history load test reports it.
    """

    def __init__(self, db_path, semantic_backend=None, context_radius=1, semantic_weight=1.0,
                 context_grouping="reserved"):
        if (isinstance(semantic_weight, bool) or not isinstance(semantic_weight, (int, float))
                or not math.isfinite(semantic_weight) or not 0 <= semantic_weight <= 4):
            raise ValueError("semantic_weight must be finite and from 0 through 4")
        if type(context_grouping) is not str or context_grouping not in ("reserved", "emitted"):
            raise ValueError("context_grouping must be reserved or emitted")
        self.semantic_weight = float(semantic_weight)
        self.context_grouping = context_grouping
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.semantic = semantic_backend
        self.context_radius = context_radius
        self._write_lock = threading.Lock()
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    user_id TEXT NOT NULL COLLATE BINARY,
                    request_id TEXT NOT NULL COLLATE BINARY,
                    session_id TEXT NOT NULL COLLATE BINARY,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (user_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL COLLATE BINARY,
                    session_id TEXT NOT NULL COLLATE BINARY,
                    request_id TEXT NOT NULL COLLATE BINARY,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_timestamp_ms INTEGER,
                    ingested_at_ms INTEGER NOT NULL,
                    doc_length INTEGER NOT NULL,
                    UNIQUE (user_id, request_id, message_index),
                    FOREIGN KEY (user_id, request_id)
                        REFERENCES requests (user_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS messages_user ON messages (user_id);
                CREATE INDEX IF NOT EXISTS messages_session_sequence
                    ON messages (user_id, session_id, sequence);
                CREATE INDEX IF NOT EXISTS messages_session_time
                    ON messages (user_id, session_id, source_timestamp_ms, sequence);
                CREATE TABLE IF NOT EXISTS terms (
                    term TEXT NOT NULL,
                    message_id TEXT NOT NULL REFERENCES messages (id),
                    frequency INTEGER NOT NULL,
                    PRIMARY KEY (term, message_id)
                );
                CREATE TABLE IF NOT EXISTS user_statistics (
                    user_id TEXT PRIMARY KEY COLLATE BINARY,
                    message_count INTEGER NOT NULL,
                    total_length INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS postings (
                    user_id TEXT NOT NULL COLLATE BINARY,
                    term TEXT NOT NULL COLLATE BINARY,
                    message_id TEXT NOT NULL REFERENCES messages (id),
                    frequency INTEGER NOT NULL,
                    doc_length INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    PRIMARY KEY (user_id, term, message_id)
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS term_statistics (
                    user_id TEXT NOT NULL COLLATE BINARY,
                    term TEXT NOT NULL COLLATE BINARY,
                    document_count INTEGER NOT NULL,
                    PRIMARY KEY (user_id, term)
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS vectors (
                    user_id TEXT NOT NULL COLLATE BINARY,
                    message_id TEXT NOT NULL REFERENCES messages (id),
                    segment_index INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL COLLATE BINARY,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (user_id, fingerprint, message_id, segment_index)
                ) WITHOUT ROWID;
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY
                );
            """)
            # Upgrade pre-scoped databases atomically. New inserts use postings;
            # the legacy terms table is kept only as a migration source.
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute("SELECT 1 FROM schema_migrations WHERE name=?",
                                      ("scoped-postings-v1",)).fetchone():
                connection.execute("""
                    INSERT OR IGNORE INTO postings
                    SELECT m.user_id, t.term, t.message_id, t.frequency,
                           m.doc_length, m.sequence
                    FROM terms t JOIN messages m ON m.id=t.message_id
                """)
                connection.execute("""
                    INSERT OR REPLACE INTO user_statistics
                    SELECT user_id, COUNT(*), SUM(doc_length) FROM messages GROUP BY user_id
                """)
                connection.execute("""
                    INSERT OR REPLACE INTO term_statistics
                    SELECT user_id, term, COUNT(*) FROM postings GROUP BY user_id, term
                """)
                connection.execute("INSERT INTO schema_migrations VALUES (?)",
                                   ("scoped-postings-v1",))
            if self.semantic is not None:
                # A switched model or incomplete old database must not quietly
                # present itself as a completely indexed semantic submission.
                missing = connection.execute("""
                    SELECT 1 FROM messages m WHERE NOT EXISTS (
                        SELECT 1 FROM vectors v WHERE v.user_id=m.user_id
                        AND v.message_id=m.id AND v.fingerprint=?) LIMIT 1
                """, (self.semantic.fingerprint,)).fetchone()
                if missing:
                    raise ValueError("Existing messages lack the configured semantic index; "
                                     "use a fresh database or explicitly rebuild before startup")
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-4096")
        return connection

    @staticmethod
    def _acknowledgement(payload):
        return {"success": True, "request_id": payload["request_id"],
                "user_id": payload["user_id"], "session_id": payload["session_id"]}

    @staticmethod
    def _existing(connection, scope, serialized):
        existing = connection.execute(
            "SELECT payload_json FROM requests WHERE user_id=? AND request_id=?", scope).fetchone()
        if existing is not None and existing["payload_json"] != serialized:
            raise ConflictError("The same user_id/request_id has different content")
        return existing is not None

    def add(self, payload):
        validate_add(payload)
        serialized = _canonical_json(payload)
        scope = (payload["user_id"], payload["request_id"])
        connection = self._connect()
        try:
            # A normal retry does not call a model, rebuild postings, or wait on
            # the writer. Its previous committed response is immediately valid.
            if self._existing(connection, scope, serialized):
                return self._acknowledgement(payload)
        finally:
            connection.close()
        prepared = []
        for index, message in enumerate(payload["messages"]):
            identity = json.dumps(list(scope) + [index], ensure_ascii=False,
                                  separators=(",", ":")).encode("utf-8")
            prepared.append(("msg_" + hashlib.sha256(identity).hexdigest(),
                             Counter(lexical_tokens(message["content"]))))
        document_vectors = None
        if self.semantic is not None:
            from semantic import EmbeddingCapacityError, EmbeddingError, encode_vector, normalize_vector
            try:
                document_vectors = self.semantic.embed_documents(
                    [message["content"] for message in payload["messages"]])
                if len(document_vectors) != len(prepared) or any(not row for row in document_vectors):
                    raise EmbeddingError("Embedding backend returned incomplete document vectors")
                document_vectors = [[encode_vector(normalize_vector(vector, self.semantic.dimension))
                                     for vector in row] for row in document_vectors]
            except EmbeddingCapacityError as error:
                raise PayloadCapacityError(
                    "Input exceeds the declared semantic capacity; Add was not committed") from error
            except EmbeddingError as error:
                raise UnavailableError("Semantic indexing failed; Add was not committed") from error
        # Preprocessing and model calls above cannot hold SQLite's single writer.
        # Bound local wait separately from the cross-process SQLite busy timeout.
        if not self._write_lock.acquire(timeout=120):
            raise UnavailableError("Storage writer queue is busy; retry the same request")
        connection = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            if not self._existing(connection, scope, serialized):
                connection.execute("INSERT INTO requests VALUES (?, ?, ?, ?)",
                                   scope + (payload["session_id"], serialized))
                now = time.time_ns() // 1_000_000
                total_length = 0
                term_counts = Counter()
                for index, (message, (message_id, frequencies)) in enumerate(
                        zip(payload["messages"], prepared)):
                    doc_length = sum(frequencies.values())
                    total_length += doc_length
                    cursor = connection.execute("""
                        INSERT INTO messages
                            (id, user_id, session_id, request_id, message_index, role, content,
                             source_timestamp_ms, ingested_at_ms, doc_length)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (message_id, payload["user_id"], payload["session_id"], payload["request_id"],
                          index, message["role"], message["content"], message.get("timestamp"),
                          now, doc_length))
                    connection.executemany("INSERT INTO postings VALUES (?, ?, ?, ?, ?, ?)",
                                           ((payload["user_id"], term, message_id, count,
                                             doc_length, cursor.lastrowid)
                                            for term, count in frequencies.items()))
                    term_counts.update(frequencies.keys())
                    if document_vectors is not None:
                        for segment_index, vector in enumerate(document_vectors[index]):
                            connection.execute("INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?)",
                                               (payload["user_id"], message_id, segment_index,
                                                self.semantic.fingerprint, self.semantic.dimension, vector))
                connection.execute("""
                    INSERT INTO user_statistics VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        message_count=message_count+excluded.message_count,
                        total_length=total_length+excluded.total_length
                """, (payload["user_id"], len(prepared), total_length))
                connection.executemany("""
                    INSERT INTO term_statistics VALUES (?, ?, ?)
                    ON CONFLICT(user_id, term) DO UPDATE SET
                        document_count=document_count+excluded.document_count
                """, ((payload["user_id"], term, count) for term, count in term_counts.items()))
            connection.commit()
        except sqlite3.OperationalError as error:
            if connection is not None:
                connection.rollback()
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise UnavailableError("Storage is busy; retry the same request") from error
            raise
        except Exception:
            if connection is not None:
                connection.rollback()
            raise
        finally:
            if connection is not None:
                connection.close()
            self._write_lock.release()
        return self._acknowledgement(payload)

    def _lexical_candidates(self, connection, payload, limit):
        statistics = connection.execute("SELECT * FROM user_statistics WHERE user_id=?",
                                        (payload["user_id"],)).fetchone()
        if statistics is None:
            return []
        weights = {term: 1.0 for term in lexical_tokens(payload["query"])}
        for option in payload.get("options", []):
            for term in lexical_tokens(option):
                weights.setdefault(term, 0.35)
        if not weights:
            return []
        if any(term not in STOP_WORDS for term in weights):
            weights = {term: weight for term, weight in weights.items() if term not in STOP_WORDS}
        # Read only small term statistics, never source texts or all postings.
        term_rows = []
        terms = list(weights)
        for start in range(0, len(terms), 400):
            batch = terms[start:start + 400]
            term_rows.extend(connection.execute(
                "SELECT term, document_count FROM term_statistics WHERE user_id=? AND term IN ("
                + ",".join("?" for _ in batch) + ")", (payload["user_id"], *batch)))
        count = statistics["message_count"]
        ranked_terms = [(row["term"], weights[row["term"]], math.log(
            1 + (count - row["document_count"] + 0.5) / (row["document_count"] + 0.5)))
            for row in term_rows]
        # Keep original question terms before optional answer choices, and prefer
        # informative terms for oversized inputs. This is query-independent of
        # dataset IDs and disclosed as a retrieval bound rather than truncation.
        ranked_terms.sort(key=lambda row: (-row[1], -row[2], row[0]))
        ranked_terms = ranked_terms[:MAX_QUERY_TERMS]
        if not ranked_terms:
            return []
        parameters = [value for term in ranked_terms for value in term]
        values = ",".join("(?,?,?)" for _ in ranked_terms)
        average_length = max(statistics["total_length"] / count, 1.0)
        rows = connection.execute("""
            WITH query_terms(term, weight, inverse_frequency) AS (VALUES """ + values + """)
            SELECT p.message_id, SUM(q.weight*q.inverse_frequency*p.frequency*2.2 /
                (p.frequency+1.2*(0.25+0.75*p.doc_length/?))) AS score,
                MIN(p.sequence) AS sequence
            FROM query_terms q JOIN postings p ON p.term=q.term AND p.user_id=?
            GROUP BY p.message_id ORDER BY score DESC, sequence ASC LIMIT ?
        """, (*parameters, average_length, payload["user_id"], limit))
        return [(row["message_id"], row["score"]) for row in rows]

    def _fuse(self, rankings):
        scores = Counter()
        lexical_ranks = {}
        for index, (weight, ranking) in enumerate(zip((1.0, self.semantic_weight), rankings)):
            for rank, (message_id, _) in enumerate(ranking, 1):
                scores[message_id] += weight / (60 + rank)
                if index == 0:
                    lexical_ranks.setdefault(message_id, rank)
        return sorted(scores.items(), key=lambda row: (
            -row[1], lexical_ranks.get(row[0], math.inf), row[0]))

    def search(self, payload):
        validate_search(payload)
        query_parts = [payload["query"]] + payload.get("options", [])
        if not any(part.strip() for part in query_parts):
            return {"data": []}
        query_vector = None
        if self.semantic is not None and self.semantic_weight > 0:
            from semantic import EmbeddingCapacityError, EmbeddingError
            try:
                query_vector = self.semantic.embed_query("\n".join(query_parts))
            except EmbeddingCapacityError as error:
                raise PayloadCapacityError(
                    "Query exceeds the declared semantic capacity; no successful acknowledgement") from error
            except EmbeddingError as error:
                raise UnavailableError("Semantic query embedding failed") from error
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            candidate_limit = min(MAX_RETRIEVAL_CANDIDATES, max(128, payload["top_k"] * 4))
            lexical = self._lexical_candidates(connection, payload, candidate_limit)
            ranking = lexical
            if query_vector is not None:
                from semantic import EmbeddingError, rank_vectors
                rows = connection.execute("""
                    SELECT message_id, vector FROM vectors WHERE user_id=? AND fingerprint=?
                """, (payload["user_id"], self.semantic.fingerprint))
                try:
                    semantic = rank_vectors(query_vector,
                                            ((row["message_id"], row["vector"]) for row in rows),
                                            candidate_limit)
                except EmbeddingError as error:
                    raise UnavailableError("Semantic index retrieval failed") from error
                ranking = self._fuse((lexical, semantic))
            ids = [message_id for message_id, _ in ranking[:payload["top_k"]]]
            if not ids:
                return {"data": []}
            rows = connection.execute(
                "SELECT * FROM messages WHERE user_id=? AND id IN ("
                + ",".join("?" for _ in ids) + ")", (payload["user_id"], *ids))
            # Even backend-produced IDs are re-scoped before any raw text read.
            documents = {row["id"]: row for row in rows}
            reserved_anchors = set(documents) if self.context_grouping == "reserved" else set()
            emitted_sources = set()
            output = []
            used_characters = 0
            for message_id, score in ranking[:payload["top_k"]]:
                document = documents.get(message_id)
                if document is None or (self.context_grouping == "emitted" and message_id in emitted_sources):
                    continue
                content = self._evidence(document)
                pending_sources = {message_id}
                neighbors = self._neighbors(connection, document)
                for neighbor in neighbors:
                    if (neighbor["id"] in reserved_anchors or neighbor["id"] in emitted_sources
                            or neighbor["id"] in pending_sources):
                        continue
                    extra = "\n\n[adjacent context]\n" + self._evidence(neighbor)
                    if len(content) + len(extra) <= MAX_WINDOW_CHARACTERS:
                        content += extra
                        pending_sources.add(neighbor["id"])
                # Never truncate source text. Stop at a whole evidence boundary;
                # preserve one oversized first item so its evidence is not lost.
                if output and used_characters + len(content) > MAX_EVIDENCE_CHARACTERS:
                    break
                output.append({"id": message_id, "content": content, "score": float(score)})
                used_characters += len(content)
                # The emitted research mode can include a later anchor as
                # context; skip it later only after its full text was returned.
                # The default reserved mode keeps all selected anchors separate.
                emitted_sources.update(pending_sources)
            return {"data": output}
        finally:
            connection.close()

    def _neighbors(self, connection, document):
        if not self.context_radius:
            return []
        base = (document["user_id"], document["session_id"])
        rows = []
        use_source_time = document["source_timestamp_ms"] is not None
        # Missing dates in the received neighborhood make event-time adjacency
        # unknown. Inspect only bounded metadata before choosing one ordering;
        # do not replace an undated reply with distant, dated source text.
        if use_source_time:
            for operator, order in (("<", "DESC"), (">", "ASC")):
                timestamps = connection.execute("""
                    SELECT source_timestamp_ms FROM messages
                    WHERE user_id=? AND session_id=? AND sequence """
                    + operator + " ? ORDER BY sequence " + order + " LIMIT ?",
                    (*base, document["sequence"], self.context_radius))
                if any(row["source_timestamp_ms"] is None for row in timestamps):
                    use_source_time = False
                    break
        if use_source_time:
            for operator, order in (("<", "DESC"), (">", "ASC")):
                rows.extend(connection.execute("""
                    SELECT * FROM messages WHERE user_id=? AND session_id=?
                    AND source_timestamp_ms IS NOT NULL
                    AND (source_timestamp_ms, sequence) """ + operator + """ (?, ?)
                    ORDER BY source_timestamp_ms """ + order + ", sequence " + order + " LIMIT ?",
                    (*base, document["source_timestamp_ms"], document["sequence"], self.context_radius)))
        else:
            for operator, order in (("<", "DESC"), (">", "ASC")):
                rows.extend(connection.execute("""
                    SELECT * FROM messages WHERE user_id=? AND session_id=? AND sequence """
                    + operator + " ? ORDER BY sequence " + order + " LIMIT ?",
                    (*base, document["sequence"], self.context_radius)))
        if use_source_time:
            return sorted(rows, key=lambda row: (row["source_timestamp_ms"], row["sequence"]))
        return sorted(rows, key=lambda row: row["sequence"])

    @staticmethod
    def _evidence(document):
        source = {"role": document["role"], "session_id": document["session_id"],
                  "request_id": document["request_id"], "message_index": document["message_index"],
                  "message_id": document["id"], "received_sequence": document["sequence"]}
        if document["source_timestamp_ms"] is not None:
            source["timestamp_ms"] = document["source_timestamp_ms"]
            try:
                instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
                    milliseconds=document["source_timestamp_ms"])
                source["timestamp_utc"] = instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            except OverflowError:
                # Signed-64-bit protocol values can exceed datetime's year range;
                # retain their exact source integer without fabricating a date.
                pass
        prefix = "[source " + json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "]\n"
        return prefix + document["content"]

    def healthy(self):
        connection = self._connect()
        try:
            connection.execute("SELECT 1 FROM requests LIMIT 1").fetchone()
        finally:
            connection.close()


def make_server(address, store, api_token=None, allow_insecure=False):
    if not api_token and not allow_insecure:
        raise ValueError("MEMORY_API_TOKEN is required; use --allow-insecure only for local experiments")

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 closes each connection, including failures with unread request bodies.
        protocol_version = "HTTP/1.0"

        def setup(self):
            self.request.settimeout(REQUEST_HEADER_TIMEOUT_SECONDS)
            super().setup()

        def log_message(self, format_string, *args):
            logging.info("%s %s", self.command, urlsplit(self.path).path)

        def _send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if status == 401:
                self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(body)

        def _authenticated(self):
            if not api_token and allow_insecure:
                return True
            scheme, _, supplied = self.headers.get("Authorization", "").partition(" ")
            return scheme.casefold() == "bearer" and hmac.compare_digest(
                supplied.encode("utf-8"), api_token.encode("utf-8"))

        def do_GET(self):
            if urlsplit(self.path).path != "/health":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                store.healthy()
                self._send_json(200, {"status": "ok"})
            except Exception:
                logging.exception("Health check failed")
                self._send_json(503, {"error": "Storage unavailable"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/add", "/search"):
                self._send_json(404, {"error": "Not found"})
                return
            if not self._authenticated():
                self._send_json(401, {"error": "Valid Bearer token required"})
                return
            length = self.headers.get("Content-Length")
            if length is None:
                self._send_json(411, {"error": "Content-Length required"})
                return
            try:
                length = int(length)
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length"})
                return
            if length < 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "Request body exceeds the size limit"})
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                self._send_json(415, {"error": "Content-Type must be application/json"})
                return
            self.connection.settimeout(AUTHENTICATED_BODY_TIMEOUT_SECONDS)
            try:
                try:
                    raw_body = self.rfile.read(length)
                except socket.timeout:
                    self._send_json(408, {"error": "Request body timed out; no successful acknowledgement"})
                    return
                finally:
                    self.connection.settimeout(REQUEST_HEADER_TIMEOUT_SECONDS)
                if len(raw_body) != length:
                    raise ValidationError("Incomplete request body")
                payload = json.loads(raw_body.decode("utf-8"))
                result = store.add(payload) if path == "/add" else store.search(payload)
                self._send_json(200, result)
            except PayloadCapacityError as error:
                self._send_json(422, {"error": str(error)})
            except (ValidationError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
            except ConflictError as error:
                self._send_json(409, {"error": str(error)})
            except UnavailableError as error:
                self._send_json(503, {"error": str(error)})
            except Exception:
                logging.exception("Memory operation failed")
                self._send_json(500, {"error": "Operation failed; no successful acknowledgement"})

    class MemoryHTTPServer(ThreadingHTTPServer):
        # The stdlib default backlog of five creates artificial connect delays
        # even in a sixteen-client load test.
        request_queue_size = 128

    server = MemoryHTTPServer(address, Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--context-radius", type=int, default=1,
                        help="Same-session neighbors on each side, 0 through 2; never invent source dates")
    parser.add_argument("--context-grouping", choices=("reserved", "emitted"),
                        default=os.environ.get("MEMORY_CONTEXT_GROUPING", "reserved"),
                        help="reserved keeps candidate anchors separate; emitted enables experimental source grouping")
    parser.add_argument("--semantic-weight", type=float, default=1.0,
                        help="Global semantic RRF weight, finite 0 through 4; lexical weight is 1. "
                             "Zero skips semantic Search, while configured Add indexing is preserved")
    parser.add_argument("--allow-insecure", action="store_true",
                        help="Disable auth ONLY for a local experiment; never use for a formal submission")
    args = parser.parse_args()
    if not 0 <= args.context_radius <= 2:
        parser.error("--context-radius must be from 0 through 2")
    if args.context_grouping not in ("reserved", "emitted"):
        parser.error("--context-grouping or MEMORY_CONTEXT_GROUPING must be reserved or emitted")
    if not math.isfinite(args.semantic_weight) or not 0 <= args.semantic_weight <= 4:
        parser.error("--semantic-weight must be finite and from 0 through 4")
    token = os.environ.get("MEMORY_API_TOKEN")
    if not token and not args.allow_insecure:
        parser.error("Set MEMORY_API_TOKEN before startup; formal use must enable authentication")
    if args.allow_insecure and args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("--allow-insecure is limited to loopback addresses")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not token:
        logging.warning("Local experiment without authentication; unsuitable for formal use")
    from semantic import EmbeddingConfigError, build_backend
    try:
        backend = build_backend()
        store = MemoryStore(args.db, semantic_backend=backend, context_radius=args.context_radius,
                            semantic_weight=args.semantic_weight, context_grouping=args.context_grouping)
    except (EmbeddingConfigError, ValueError) as error:
        parser.error(str(error))
    server = make_server((args.host, args.port), store, token, args.allow_insecure)
    logging.info("Memory service listening on %s:%s; semantic indexing %s", args.host,
                 server.server_port, "enabled" if backend is not None else "disabled")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
