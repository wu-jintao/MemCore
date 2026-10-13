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

import coding


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
                 context_grouping="reserved", profile="textual", coding_weight=1.4,
                 coding_dense_rerank_weight=0.0, coding_result_limit=8,
                 coding_character_limit=120000):
        if (isinstance(semantic_weight, bool) or not isinstance(semantic_weight, (int, float))
                or not math.isfinite(semantic_weight) or not 0 <= semantic_weight <= 4):
            raise ValueError("semantic_weight must be finite and from 0 through 4")
        if type(context_grouping) is not str or context_grouping not in ("reserved", "emitted"):
            raise ValueError("context_grouping must be reserved or emitted")
        if profile not in ("textual", "coding"):
            raise ValueError("profile must be textual or coding")
        if (isinstance(coding_weight, bool) or not isinstance(coding_weight, (int, float))
                or not math.isfinite(coding_weight) or not 0 <= coding_weight <= 4):
            raise ValueError("coding_weight must be finite and from 0 through 4")
        if (isinstance(coding_dense_rerank_weight, bool)
                or not isinstance(coding_dense_rerank_weight, (int, float))
                or not math.isfinite(coding_dense_rerank_weight)
                or not 0 <= coding_dense_rerank_weight <= 4):
            raise ValueError("coding_dense_rerank_weight must be finite and from 0 through 4")
        if coding_dense_rerank_weight > 0 and profile != "coding":
            raise ValueError("coding_dense_rerank_weight requires the coding profile")
        if coding_dense_rerank_weight > 0 and (semantic_backend is None
                or not callable(getattr(semantic_backend, "embed_query_segments", None))):
            raise ValueError("Coding dense reranking requires query segment embeddings")
        if type(coding_result_limit) is not int or not 1 <= coding_result_limit <= 100:
            raise ValueError("coding_result_limit must be from 1 through 100")
        if type(coding_character_limit) is not int or not 1000 <= coding_character_limit <= 500000:
            raise ValueError("coding_character_limit must be from 1000 through 500000")
        self.semantic_weight = float(semantic_weight)
        self.context_grouping = context_grouping
        self.profile = profile
        self.coding_weight = float(coding_weight)
        self.coding_dense_rerank_weight = float(coding_dense_rerank_weight)
        self.coding_result_limit = coding_result_limit
        self.coding_character_limit = coding_character_limit
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
            if self.profile == "coding":
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS code_documents (
                        message_sequence INTEGER PRIMARY KEY REFERENCES messages(sequence),
                        quality REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS code_features (
                        feature TEXT NOT NULL COLLATE BINARY,
                        message_sequence INTEGER NOT NULL REFERENCES messages(sequence),
                        weight INTEGER NOT NULL,
                        PRIMARY KEY (feature, message_sequence)
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS coding_postings (
                        term TEXT NOT NULL COLLATE BINARY,
                        message_sequence INTEGER NOT NULL REFERENCES messages(sequence),
                        frequency INTEGER NOT NULL,
                        PRIMARY KEY (term, message_sequence)
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS coding_embedding_cache (
                        fingerprint TEXT NOT NULL COLLATE BINARY,
                        content_sha256 TEXT NOT NULL COLLATE BINARY,
                        segment_index INTEGER NOT NULL,
                        dimension INTEGER NOT NULL,
                        vector BLOB NOT NULL,
                        PRIMARY KEY (fingerprint, content_sha256, segment_index)
                    ) WITHOUT ROWID;
                """)
                code_columns = [row[1] for row in connection.execute(
                    "PRAGMA table_info(code_features)")]
                posting_columns = [row[1] for row in connection.execute(
                    "PRAGMA table_info(coding_postings)")]
                cache_columns = [row[1] for row in connection.execute(
                    "PRAGMA table_info(coding_embedding_cache)")]
                if code_columns != ["feature", "message_sequence", "weight"] \
                        or posting_columns != ["term", "message_sequence", "frequency"] \
                        or cache_columns != ["fingerprint", "content_sha256", "segment_index", "dimension", "vector"]:
                    raise ValueError("Existing Coding database uses an incompatible index schema; use a fresh database")
                coding_marker = connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE name=?", ("coding-postings-v1",)).fetchone()
                cache_marker = connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE name=?", ("coding-embedding-cache-v1",)).fetchone()
                existing_coding_messages = connection.execute("SELECT 1 FROM messages LIMIT 1").fetchone()
                if existing_coding_messages and (not coding_marker or not cache_marker):
                    raise ValueError("Existing Coding database lacks a current Coding index marker; use a fresh database")
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
            if self.profile == "coding" and not connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE name=?", ("coding-postings-v1",)).fetchone():
                connection.execute("INSERT INTO schema_migrations VALUES (?)", ("coding-postings-v1",))
            if self.profile == "coding" and not connection.execute(
                    "SELECT 1 FROM schema_migrations WHERE name=?", ("coding-embedding-cache-v1",)).fetchone():
                connection.execute("INSERT INTO schema_migrations VALUES (?)", ("coding-embedding-cache-v1",))
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

    def _request_record(self, payload):
        serialized = _canonical_json(payload)
        if self.profile == "coding":
            return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return serialized

    @staticmethod
    def _existing(connection, scope, serialized):
        existing = connection.execute(
            "SELECT payload_json FROM requests WHERE user_id=? AND request_id=?", scope).fetchone()
        if existing is not None and existing["payload_json"] != serialized:
            raise ConflictError("The same user_id/request_id has different content")
        return existing is not None

    def add(self, payload):
        validate_add(payload)
        serialized = self._request_record(payload)
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
            code_features = coding.extract_features(message["content"]) if self.profile == "coding" else None
            lexical_text = (coding.semantic_document(message["content"])
                            if self.profile == "coding" else message["content"])
            prepared.append(("msg_" + hashlib.sha256(identity).hexdigest(),
                             Counter(lexical_tokens(lexical_text)), code_features))
        document_vectors = None
        cache_hashes = None
        cache_misses = {}
        if self.semantic is not None:
            from semantic import EmbeddingCapacityError, EmbeddingError, encode_vector, normalize_vector
            try:
                embedding_texts = [message["content"] for message in payload["messages"]]
                if self.profile == "coding":
                    embedding_texts = [coding.semantic_document(text) for text in embedding_texts]
                    cache_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest()
                                    for text in embedding_texts]
                    cached = {}
                    connection = self._connect()
                    try:
                        for start in range(0, len(set(cache_hashes)), 400):
                            batch = sorted(set(cache_hashes))[start:start + 400]
                            rows = connection.execute(
                                "SELECT content_sha256,segment_index,dimension,vector "
                                "FROM coding_embedding_cache WHERE fingerprint=? AND content_sha256 IN ("
                                + ",".join("?" for _ in batch) + ") ORDER BY content_sha256,segment_index",
                                (self.semantic.fingerprint, *batch))
                            for row in rows:
                                if row["dimension"] != self.semantic.dimension:
                                    raise EmbeddingError("Cached Coding embedding dimension mismatch")
                                cached.setdefault(row["content_sha256"], []).append(
                                    (row["segment_index"], row["vector"]))
                    finally:
                        connection.close()
                    normalized_cache = {}
                    for key, values in cached.items():
                        values.sort()
                        if [index for index, _ in values] != list(range(len(values))):
                            raise EmbeddingError("Cached Coding embedding segments are incomplete")
                        normalized_cache[key] = [blob for _, blob in values]
                    missing_order = []
                    for key, text in zip(cache_hashes, embedding_texts):
                        if key not in normalized_cache and key not in cache_misses:
                            cache_misses[key] = text
                            missing_order.append(key)
                    if missing_order:
                        embedded = self.semantic.embed_documents([cache_misses[key] for key in missing_order])
                        if len(embedded) != len(missing_order) or any(not row for row in embedded):
                            raise EmbeddingError("Embedding backend returned incomplete document vectors")
                        for key, vectors in zip(missing_order, embedded):
                            normalized_cache[key] = [encode_vector(normalize_vector(vector, self.semantic.dimension))
                                                     for vector in vectors]
                    document_vectors = [normalized_cache[key] for key in cache_hashes]
                else:
                    embedded = self.semantic.embed_documents(embedding_texts)
                    if len(embedded) != len(prepared) or any(not row for row in embedded):
                        raise EmbeddingError("Embedding backend returned incomplete document vectors")
                    document_vectors = [[encode_vector(normalize_vector(vector, self.semantic.dimension))
                                         for vector in row] for row in embedded]
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
                if cache_hashes is not None:
                    unique_rows = {}
                    for key, vectors in zip(cache_hashes, document_vectors):
                        unique_rows.setdefault(key, vectors)
                    connection.executemany(
                        "INSERT OR IGNORE INTO coding_embedding_cache VALUES (?, ?, ?, ?, ?)",
                        ((self.semantic.fingerprint, key, segment_index,
                          self.semantic.dimension, blob)
                         for key, vectors in unique_rows.items()
                         for segment_index, blob in enumerate(vectors)))
                total_length = 0
                term_counts = Counter()
                for index, (message, (message_id, frequencies, code_features)) in enumerate(
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
                    if self.profile == "coding":
                        connection.executemany("INSERT INTO coding_postings VALUES (?, ?, ?)",
                                               ((term, cursor.lastrowid, count)
                                                for term, count in frequencies.items()))
                    else:
                        connection.executemany("INSERT INTO postings VALUES (?, ?, ?, ?, ?, ?)",
                                               ((payload["user_id"], term, message_id, count,
                                                 doc_length, cursor.lastrowid)
                                                for term, count in frequencies.items()))
                    term_counts.update(frequencies.keys())
                    if code_features is not None:
                        connection.execute("INSERT INTO code_documents VALUES (?, ?)",
                                           (cursor.lastrowid, coding.quality_prior(code_features)))
                        connection.executemany("INSERT INTO code_features VALUES (?, ?, ?)",
                                               ((feature, cursor.lastrowid, weight)
                                                for feature, weight in code_features.weighted.items()))
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
        if self.profile == "coding":
            rows = connection.execute("""
                WITH query_terms(term, weight, inverse_frequency) AS (VALUES """ + values + """)
                SELECT m.id AS message_id,
                    SUM(q.weight*q.inverse_frequency*p.frequency*2.2 /
                    (p.frequency+1.2*(0.25+0.75*m.doc_length/?))) AS score,
                    MIN(m.sequence) AS sequence
                FROM query_terms q JOIN coding_postings p ON p.term=q.term
                JOIN messages m ON m.sequence=p.message_sequence AND m.user_id=?
                GROUP BY m.id ORDER BY score DESC, sequence ASC LIMIT ?
            """, (*parameters, average_length, payload["user_id"], limit))
        else:
            rows = connection.execute("""
                WITH query_terms(term, weight, inverse_frequency) AS (VALUES """ + values + """)
                SELECT p.message_id, SUM(q.weight*q.inverse_frequency*p.frequency*2.2 /
                    (p.frequency+1.2*(0.25+0.75*p.doc_length/?))) AS score,
                    MIN(p.sequence) AS sequence
                FROM query_terms q JOIN postings p ON p.term=q.term AND p.user_id=?
                GROUP BY p.message_id ORDER BY score DESC, sequence ASC LIMIT ?
            """, (*parameters, average_length, payload["user_id"], limit))
        return [(row["message_id"], row["score"]) for row in rows]

    def _coding_candidates(self, connection, payload, limit):
        if self.profile != "coding" or self.coding_weight <= 0:
            return []
        query = "\n".join([payload["query"], *payload.get("options", [])])
        profile = coding.query_profile(query)
        features = profile.weighted
        if not features:
            return []
        selected = sorted(features.items(), key=lambda item: (-item[1], item[0]))[:96]
        names = [name for name, _ in selected]
        counts = dict(connection.execute(
            "SELECT f.feature,COUNT(*) FROM code_features f "
            "JOIN messages m ON m.sequence=f.message_sequence "
            "WHERE m.user_id=? AND f.feature IN ("
            + ",".join("?" for _ in names) + ") GROUP BY f.feature",
            (payload["user_id"], *names)))
        statistics = connection.execute("SELECT message_count FROM user_statistics WHERE user_id=?",
                                        (payload["user_id"],)).fetchone()
        if statistics is None:
            return []
        document_count = statistics["message_count"]
        ranked = [(name, weight, math.log(1 + (document_count - counts.get(name, 0) + .5)
                                              / (counts.get(name, 0) + .5)))
                  for name, weight in selected if counts.get(name, 0)]
        if not ranked:
            return []
        values = ",".join("(?,?,?)" for _ in ranked)
        parameters = [value for row in ranked for value in row]
        rows = connection.execute("""
            WITH query_features(feature,weight,inverse_frequency) AS (VALUES """ + values + """)
            SELECT m.id AS message_id,
                SUM(q.weight*q.inverse_frequency*f.weight)+d.quality AS score,
                MIN(m.sequence) AS sequence
            FROM query_features q JOIN code_features f ON f.feature=q.feature
            JOIN messages m ON m.sequence=f.message_sequence AND m.user_id=?
            JOIN code_documents d ON d.message_sequence=m.sequence
            GROUP BY m.id ORDER BY score DESC, sequence ASC LIMIT ?
        """, (*parameters, payload["user_id"], limit))
        confidence = coding.query_confidence(profile)
        return ([(row["message_id"], row["score"]) for row in rows], confidence)

    @staticmethod
    def _fuse(weighted_rankings):
        scores = Counter()
        lexical_ranks = {}
        for index, (weight, ranking) in enumerate(weighted_rankings):
            for rank, (message_id, _) in enumerate(ranking, 1):
                scores[message_id] += weight / (60 + rank)
                if index == 0:
                    lexical_ranks.setdefault(message_id, rank)
        return sorted(scores.items(), key=lambda row: (
            -row[1], lexical_ranks.get(row[0], math.inf), row[0]))

    def _dense_post_rerank(self, connection, payload, output, query_segments):
        """Reorder Coding's already packed evidence set using segment-pair maxima."""
        if self.coding_dense_rerank_weight <= 0 or len(output) < 2:
            return output
        from semantic import EmbeddingError, decode_vector, normalize_vector
        try:
            queries = tuple(normalize_vector(vector, self.semantic.dimension)
                            for vector in query_segments)
            if not queries:
                raise EmbeddingError("Query segment embeddings are empty")
            identities = [item["id"] for item in output]
            rows = connection.execute(
                "SELECT message_id, vector FROM vectors WHERE user_id=? AND fingerprint=? "
                "AND message_id IN (" + ",".join("?" for _ in identities) + ") "
                "ORDER BY message_id, segment_index",
                (payload["user_id"], self.semantic.fingerprint, *identities),
            )
            documents = {identity: [] for identity in identities}
            for row in rows:
                documents[row["message_id"]].append(
                    decode_vector(row["vector"], self.semantic.dimension))
            if any(not documents[identity] for identity in identities):
                raise EmbeddingError("Selected Coding evidence lacks segment embeddings")
            dense_scores = {
                identity: max(math.fsum(a * b for a, b in zip(query, document))
                              for query in queries for document in documents[identity])
                for identity in identities
            }
        except EmbeddingError as error:
            raise UnavailableError("Coding dense reranking failed") from error
        dense_order = sorted(identities, key=lambda identity: (-dense_scores[identity], identity))
        dense_ranks = {identity: rank for rank, identity in enumerate(dense_order, 1)}
        original_ranks = {identity: rank for rank, identity in enumerate(identities, 1)}
        scores = {
            identity: (1.0 / (60 + original_ranks[identity])
                       + self.coding_dense_rerank_weight / (60 + dense_ranks[identity]))
            for identity in identities
        }
        ranked = sorted(output, key=lambda item: (
            -scores[item["id"]], original_ranks[item["id"]], item["id"]))
        return [{**item, "score": float(scores[item["id"]])} for item in ranked]

    def search(self, payload):
        validate_search(payload)
        query_parts = [payload["query"]] + payload.get("options", [])
        if not any(part.strip() for part in query_parts):
            return {"data": []}
        query_vector = None
        query_segments = None
        needs_query_embedding = (self.semantic_weight > 0
                                 or self.coding_dense_rerank_weight > 0)
        if self.semantic is not None and needs_query_embedding:
            from semantic import EmbeddingCapacityError, EmbeddingError
            try:
                embedding_query = "\n".join(query_parts)
                if self.profile == "coding":
                    embedding_query = coding.semantic_query(embedding_query)
                if self.coding_dense_rerank_weight > 0:
                    query_vector, query_segments = self.semantic.embed_query_segments(embedding_query)
                else:
                    query_vector = self.semantic.embed_query(embedding_query)
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
            weighted_rankings = [(1.0, lexical)]
            if query_vector is not None and self.semantic_weight > 0:
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
                weighted_rankings.append((self.semantic_weight, semantic))
            coding_result = self._coding_candidates(connection, payload, candidate_limit)
            if coding_result:
                coding_ranking, coding_confidence = coding_result
                weighted_rankings.append((self.coding_weight * coding_confidence, coding_ranking))
            ranking = lexical if len(weighted_rankings) == 1 else self._fuse(weighted_rankings)
            result_limit = min(payload["top_k"], self.coding_result_limit) if self.profile == "coding" else payload["top_k"]
            ids = [message_id for message_id, _ in ranking[:result_limit]]
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
            for message_id, score in ranking[:result_limit]:
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
                evidence_limit = self.coding_character_limit if self.profile == "coding" else MAX_EVIDENCE_CHARACTERS
                if output and used_characters + len(content) > evidence_limit:
                    break
                output.append({"id": message_id, "content": content, "score": float(score)})
                used_characters += len(content)
                # The emitted research mode can include a later anchor as
                # context; skip it later only after its full text was returned.
                # The default reserved mode keeps all selected anchors separate.
                emitted_sources.update(pending_sources)
            if self.profile == "coding" and self.coding_dense_rerank_weight > 0:
                output = self._dense_post_rerank(
                    connection, payload, output, query_segments)
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


def make_server(address, store, api_token=None, allow_insecure=False,
                max_body_bytes=MAX_BODY_BYTES):
    if not api_token and not allow_insecure:
        raise ValueError("MEMORY_API_TOKEN is required; use --allow-insecure only for local experiments")
    if type(max_body_bytes) is not int or not 1024 <= max_body_bytes <= 128 * 1024 * 1024:
        raise ValueError("max_body_bytes must be from 1 KiB through 128 MiB")

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
            if length < 0 or length > max_body_bytes:
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
    parser.add_argument("--profile", choices=("textual", "coding"),
                        default=os.environ.get("MEMORY_PROFILE", "textual"),
                        help="textual preserves the published v1.1 behavior; coding adds deterministic engineering feature retrieval")
    parser.add_argument("--coding-weight", type=float,
                        default=float(os.environ.get("MEMORY_CODING_WEIGHT", "1.4")))
    parser.add_argument("--coding-dense-rerank-weight", type=float,
                        default=float(os.environ.get("MEMORY_CODING_DENSE_RERANK_WEIGHT", "0.0")))
    parser.add_argument("--coding-result-limit", type=int,
                        default=int(os.environ.get("MEMORY_CODING_RESULT_LIMIT", "8")))
    parser.add_argument("--coding-character-limit", type=int,
                        default=int(os.environ.get("MEMORY_CODING_CHARACTER_LIMIT", "120000")))
    parser.add_argument("--max-body-bytes", type=int, default=MAX_BODY_BYTES,
                        help="Maximum authenticated JSON body bytes; Textual default remains 8 MiB")
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
                            semantic_weight=args.semantic_weight, context_grouping=args.context_grouping,
                            profile=args.profile, coding_weight=args.coding_weight,
                            coding_dense_rerank_weight=args.coding_dense_rerank_weight,
                            coding_result_limit=args.coding_result_limit,
                            coding_character_limit=args.coding_character_limit)
    except (EmbeddingConfigError, ValueError) as error:
        parser.error(str(error))
    server = make_server((args.host, args.port), store, token, args.allow_insecure,
                         max_body_bytes=args.max_body_bytes)
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
