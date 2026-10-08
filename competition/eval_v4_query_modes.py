#!/usr/bin/env python3
"""Query-only native v4 B/C replay of a completed Lo6 PUBLIC dev baseline.

Dry plan unless --execute. No Add, new document index, Answer/Judge, official
evaluation, or heldout/validation data. Eight existing document segments are
the sole document API equivalence probe; gold is opened only after both cached
query replays finish, for local paired postprocessing.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import secrets
import sqlite3
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from semantic import EmbeddingConfig, EmbeddingError, HttpEmbeddingBackend, decode_vector, normalize_vector, _utf8_segments
from server import MemoryStore
from eval_v4_budget import (BudgetLedger, BudgetStopped, NoRedirect, atomic_json, https_transport,
                            read_model_key, MODEL, DIMENSION, BATCH_SIZE, SEGMENT_BYTES, PRIVATE_ROOT)
from eval_prepare import SOURCE_MAPPING_VERSION, file_hash, json_line, normalized
from eval_retrieval import (HTTPClient, KS, aggregate, evidence_hits, percentiles, read_jsonl,
                            requests_for_history, validate_search)
from eval_compare import compare

INSTRUCT = "Given a question, retrieve relevant conversation passages that provide evidence for answering it."
NATIVE_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
EXPECTED_QUESTIONS = 547
EXPECTED_HISTORIES = 6


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_native_endpoint(compatible, native):
    left, right = urlsplit(compatible), urlsplit(native)
    for parsed in (left, right):
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or "agentmemoryleaderboard.ai" in parsed.hostname.lower()):
            raise ValueError("Use credential-free HTTPS endpoints in the same Aliyun workspace")
    if (left.netloc != right.netloc or left.path != "/compatible-mode/v1/embeddings"
            or right.path != NATIVE_PATH or not (left.hostname.endswith(".maas.aliyuncs.com")
                or left.hostname in ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"))):
        raise ValueError("Native and compatible routes must share the original Aliyun workspace origin")
    return native


class NativeV4Client:
    def __init__(self, compatible_endpoint, native_endpoint, key, ledger,
                 transport=https_transport, http_timeout=30):
        self.endpoint = validate_native_endpoint(compatible_endpoint, native_endpoint)
        self.key, self.ledger, self.transport = key, ledger, transport
        if not isinstance(http_timeout, (int, float)) or not math.isfinite(http_timeout) or not 0 < http_timeout <= 120:
            raise ValueError("Native request timeout must be positive and at most 120 seconds")
        self.http_timeout, self.closed = http_timeout, False

    def close(self):
        self.closed, self.key = True, ""

    def encode(self, texts, text_type, instruct=""):
        if self.closed or self.ledger.stopped:
            raise BudgetStopped("Native model calls are closed")
        if (not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= BATCH_SIZE
                or any(not isinstance(text, str) or not text.strip() for text in texts)
                or text_type not in ("document", "query")
                or instruct not in ("", INSTRUCT) or (instruct and text_type != "query")):
            self.ledger.stop("invalid_native_input")
            raise BudgetStopped("Invalid fixed native experiment input")
        parameters = {"dimension": DIMENSION, "output_type": "dense", "text_type": text_type}
        if instruct:
            parameters["instruct"] = instruct
        payload = {"model": MODEL, "input": {"texts": list(texts)}, "parameters": parameters}
        body = json.dumps(payload, ensure_ascii=False).encode()
        # BudgetLedger adds 32 per input. C reserves another 32 plus the full
        # UTF-8 instruction bytes per input; these synthetic reservation strings
        # are never sent or saved, and do not alter the original model inputs.
        reserved_texts = [text + instruct + " " * 32 for text in texts] if instruct else texts
        index = self.ledger.begin(reserved_texts, hashlib.sha256(body).hexdigest())
        with self.ledger.lock:
            self.ledger.state["calls"][index].update(text_type=text_type,
                instruct_sha256=digest(instruct), original_input_utf8_bytes=sum(len(t.encode()) for t in texts),
                instruction_extra_reservation_per_input=len(instruct.encode()) + 32 if instruct else 0)
            self.ledger._save()
        request = Request(self.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.key})
        limit = min(16 * 1024 * 1024, len(texts) * DIMENSION * 40 + 65536)
        outcome = queue.Queue(maxsize=1)

        def send_once():
            try:
                with self.ledger.lock:
                    if self.ledger.stopped or self.ledger.remaining_seconds() <= 0:
                        raise BudgetStopped("Deadline reached before native transport")
                    self.ledger.state["calls"][index]["transport_started"] = True
                    self.ledger._save()
                if self.ledger.stopped or self.ledger.remaining_seconds() <= 0:
                    raise BudgetStopped("Deadline reached before native transport")
                result = self.transport(request, min(self.http_timeout, self.ledger.remaining_seconds()), limit)
                outcome.put((True, result))
            except BaseException as error:
                outcome.put((False, error))

        usage, http_status = None, None
        reason = "malformed_native_response"
        try:
            remaining = min(self.http_timeout, self.ledger.remaining_seconds())
            if remaining <= 0:
                reason = "runtime_limit"
                raise ValueError()
            threading.Thread(target=send_once, daemon=True).start()
            try:
                good, result = outcome.get(timeout=remaining)
            except queue.Empty:
                reason = "runtime_limit" if self.ledger.remaining_seconds() <= 0 else "transport_timeout"
                raise ValueError() from None
            if not good:
                reason = "runtime_limit" if isinstance(result, BudgetStopped) else "transport_failure"
                if isinstance(result, HTTPError):
                    http_status, reason = result.code, "http_" + str(result.code)
                    result.close()
                raise ValueError()
            http_status, raw = result
            if http_status != 200:
                reason = "http_" + str(http_status)
                raise ValueError()
            if len(raw) > limit:
                reason = "response_size_limit"
                raise ValueError()
            response = json.loads(raw)
            total = response.get("usage", {}).get("total_tokens") if isinstance(response, dict) else None
            if type(total) is not int or total <= 0:
                reason = "missing_or_invalid_native_usage"
                raise ValueError()
            usage = {"input_tokens": total, "total_tokens": total}
            if response.get("status_code", 200) != 200 or response.get("model", MODEL) != MODEL:
                reason = "native_response_contract_mismatch"
                raise ValueError()
            values = response.get("output", {}).get("embeddings")
            if not isinstance(values, list) or len(values) != len(texts):
                reason = "invalid_embedding_count"
                raise ValueError()
            ordered = [None] * len(texts)
            for item in values:
                number = item.get("text_index") if isinstance(item, dict) else None
                if type(number) is not int or not 0 <= number < len(texts) or ordered[number] is not None:
                    reason = "invalid_embedding_indexes"
                    raise ValueError()
                ordered[number] = normalize_vector(item.get("embedding"), DIMENSION)
            if self.ledger.finish(index, "ok", usage, http_status) != "ok":
                raise BudgetStopped("Native usage or deadline exceeded its reservation")
            return ordered
        except BudgetStopped:
            raise
        except Exception:
            self.ledger.finish(index, reason, usage, http_status)
            raise BudgetStopped("Native request failed; all further model calls are locked") from None


class CachedQueryBackend:
    def __init__(self, source_fingerprint, query_vectors, variant_hash):
        self.fingerprint = source_fingerprint  # Selects existing corpus rows, not a query model claim.
        self.dimension, self.variant_hash = DIMENSION, variant_hash
        self.query_vectors = query_vectors

    def embed_query(self, text):
        if text not in self.query_vectors:
            raise EmbeddingError("Unplanned query is absent from the immutable query cache")
        return self.query_vectors[text]

    def embed_documents(self, texts):
        raise EmbeddingError("Document embedding and Add are forbidden in query-only replay")


def readonly_database(path):
    path = Path(path).resolve()
    # Closed WAL databases may retain a WAL header after their sidecars vanish.
    # mode=ro would then try to create sidecars. immutable is safe only when no
    # WAL is present; a live WAL always goes through SQLite's ordinary snapshot.
    options = "?mode=ro" if Path(str(path) + "-wal").exists() else "?mode=ro&immutable=1"
    connection = sqlite3.connect(path.as_uri() + options, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def database_identity(path):
    connection = readonly_database(path)
    try:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        result = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            rows = []
            for row in connection.execute("SELECT * FROM " + quoted):
                rows.append(json.dumps([{"blob_sha256": hashlib.sha256(item).hexdigest()} if isinstance(item, bytes)
                                        else item for item in row], ensure_ascii=False, separators=(",", ":")))
            result[table] = {"rows": len(rows), "sha256": digest(sorted(rows))}
        return result
    finally:
        connection.close()


def backup_database(source, destination):
    left = readonly_database(source)
    right = sqlite3.connect(destination)
    try:
        left.backup(right)
        right.commit()
    finally:
        right.close()
        left.close()
    os.chmod(destination, 0o600)


def inspect_source(args, expected_questions=EXPECTED_QUESTIONS, expected_histories=EXPECTED_HISTORIES):
    folder, data_root = Path(args.source_run).resolve(), Path(args.data_root).resolve()
    summary = json.loads((folder / "summary.json").read_text())
    config = summary["configuration"]
    parent_plan = json.loads((folder.parent / "plan.json").read_text())
    compatible = parent_plan["endpoint"]
    validate_native_endpoint(compatible, args.endpoint)
    if (summary.get("status") != "public_proxy_not_official_aml" or summary.get("dataset") != "locomo_refined"
            or summary.get("split") != "dev" or summary.get("question_count") != expected_questions
            or summary.get("history_count") != expected_histories or summary.get("ingestion_failed_histories") != 0
            or summary.get("server_source_changed_during_run") or config.get("embedding_provider") != "http"
            or config.get("model") != MODEL or config.get("dimension") != DIMENSION
            or config.get("source_mapping_version") != SOURCE_MAPPING_VERSION
            or config.get("top_k") != 100 or config.get("character_budget") != 0
            or config.get("context_radius") != 1 or config.get("context_grouping") != "reserved"
            or config.get("semantic_weight") != 1 or parent_plan.get("model") != MODEL
            or parent_plan.get("dimension") != DIMENSION or parent_plan.get("segment_bytes") != SEGMENT_BYTES):
        raise ValueError("A complete fixed Lo6 v4 dev baseline with exactly the pinned scope is required")
    core_hashes = {name: file_hash(ROOT / name) for name in ("server.py", "semantic.py")}
    if (config.get("core_source_sha256_at_start") != core_hashes
            or config.get("core_source_sha256_at_end") != core_hashes):
        raise ValueError("Replay core must match the baseline's unchanged source hashes")
    records = read_jsonl(folder / "retrieval.jsonl")
    ids = [row.get("query_id") for row in records]
    if len(records) != expected_questions or len(set(ids)) != len(ids) or any(row.get("status") != "ok" for row in records):
        raise ValueError("Baseline retrieval must include every successful unique original question")
    prepared = data_root / "prepared" / "locomo_refined" / "dev"
    specification = config["prepared_data"]
    # Gold is not opened, parsed, or hashed at this stage. compare verifies it later.
    for name in ("histories", "queries"):
        if file_hash(prepared / (name + ".jsonl")) != specification["files"][name]["sha256"]:
            raise ValueError("Prepared source/query checksum mismatch")
    queries = {row["query_id"]: row for row in read_jsonl(prepared / "queries.jsonl")}
    selected = []
    for row in records:
        query = queries[row["query_id"]]
        if row["sample_id"] != query["sample_id"] or not isinstance(query.get("query"), str) or not query["query"].strip() or query.get("options"):
            raise ValueError("Replay requires the exact original free-answer query namespace")
        selected.append({key: query[key] for key in ("query_id", "sample_id", "query")})
    samples = {query["sample_id"] for query in selected}
    histories = {row["sample_id"]: row for row in read_jsonl(prepared / "histories.jsonl") if row["sample_id"] in samples}
    if len(histories) != expected_histories:
        raise ValueError("Baseline history denominator differs")
    ingestion_run_id = summary.get("ingestion_run_id", summary["run_id"])
    expected_requests, expected_messages, users = {}, {}, {}
    for sample, history in histories.items():
        user, payloads, _, _ = requests_for_history(history, ingestion_run_id, SOURCE_MAPPING_VERSION)
        users[sample] = user
        for payload in payloads:
            scope = (user, payload["request_id"])
            if scope in expected_requests:
                raise ValueError("Ambiguous source request namespace")
            expected_requests[scope] = payload
            for index, message in enumerate(payload["messages"]):
                mid = "msg_" + hashlib.sha256(json.dumps([user, payload["request_id"], index], ensure_ascii=False,
                                                       separators=(",", ":")).encode()).hexdigest()
                expected_messages[mid] = (user, payload["session_id"], payload["request_id"], index,
                                         message["role"], message["content"], message.get("timestamp"))
    database = folder / "memory.sqlite3"
    config_for_fingerprint = EmbeddingConfig(provider="http", model=MODEL, dimension=DIMENSION,
        batch_size=BATCH_SIZE, concurrency=1, http_retries=0, endpoint=compatible, api_key="unused-no-model-call",
        allow_http=True, http_segment_bytes=SEGMENT_BYTES)
    fingerprint = HttpEmbeddingBackend(config_for_fingerprint).fingerprint
    connection = readonly_database(database)
    try:
        actual_requests = {(row[0], row[1]): json.loads(row[2]) for row in connection.execute(
            "SELECT user_id,request_id,payload_json FROM requests")}
        actual_messages = {row[0]: tuple(row[1:]) for row in connection.execute(
            "SELECT id,user_id,session_id,request_id,message_index,role,content,source_timestamp_ms FROM messages")}
        if actual_requests != expected_requests or actual_messages != expected_messages:
            raise ValueError("Stored source messages or ingestion namespaces differ from baseline provenance")
        expected_segments = {mid: _utf8_segments(message[5], SEGMENT_BYTES, 512) for mid, message in expected_messages.items()}
        counts = Counter()
        probes = []
        for row in connection.execute("SELECT user_id,message_id,segment_index,fingerprint,dimension,vector FROM vectors ORDER BY user_id,message_id,segment_index"):
            user, mid, index, fp, dimension, vector = row
            if (mid not in expected_messages or user != expected_messages[mid][0] or fp != fingerprint
                    or dimension != DIMENSION or len(vector) != DIMENSION * 4
                    or not 0 <= index < len(expected_segments[mid])):
                raise ValueError("Stored vector namespace, dimension or document segmentation differs")
            counts[mid] += 1
            if len(probes) < 8:
                probes.append({"text": expected_segments[mid][index], "vector": decode_vector(vector, DIMENSION),
                               "segment_sha256": hashlib.sha256(expected_segments[mid][index].encode()).hexdigest()})
        if len(probes) != 8 or any(counts[mid] != len(parts) for mid, parts in expected_segments.items()):
            raise ValueError("Every original document segment must be present; eight gate segments are required")
    finally:
        connection.close()
    segments = {query["query"]: _utf8_segments(query["query"], SEGMENT_BYTES, 8) for query in selected}
    unique_segments = list(dict.fromkeys(part for parts in segments.values() for part in parts))
    query_bytes = sum(len(part.encode()) + 32 for part in unique_segments)
    gate_reservation = sum(len(probe["text"].encode()) + 32 for probe in probes)
    plan = {"questions": len(selected), "histories": len(histories), "document_probe_segments": 8,
            "unique_query_segments_per_variant": len(unique_segments), "add_requests": 0,
            "planned_model_requests": 1 + 2 * math.ceil(len(unique_segments) / BATCH_SIZE),
            "reservation_upper_estimate": gate_reservation + 2 * query_bytes
                + len(unique_segments) * (len(INSTRUCT.encode()) + 32),
            "instruction": INSTRUCT, "instruction_sha256": digest(INSTRUCT),
            "source_corpus_fingerprint": fingerprint, "source_run_id": summary["run_id"],
            "ingestion_run_id": ingestion_run_id, "source_mapping_version": SOURCE_MAPPING_VERSION,
            "source_summary_sha256": file_hash(folder / "summary.json"),
            "source_retrieval_sha256": file_hash(folder / "retrieval.jsonl"),
            "source_db_sha256": file_hash(database), "source_database_identity": database_identity(database),
            "selection_sha256": digest(ids), "prepared_data": specification, "core_source_sha256": core_hashes,
            "query_mode_runner_sha256": file_hash(Path(__file__)),
            "research_helpers_sha256": {name: file_hash(HERE / name) for name in (
                "eval_v4_budget.py", "eval_prepare.py", "eval_retrieval.py", "eval_compare.py")},
            "compatible_endpoint": compatible, "native_endpoint": args.endpoint,
            "query_input": "original_question_only_no_options_dates_categories_or_gold",
            "document_gate": {"minimum_cosine": 0.999999, "maximum_absolute_difference": 1e-5},
            "full_capacity_validated": False}
    return {"plan": plan, "summary": summary, "source_run": folder, "database": database,
            "queries": selected, "histories": histories, "users": users, "probes": probes,
            "query_segments": segments, "unique_segments": unique_segments, "compatible_endpoint": compatible}


def document_equivalence(client, probes):
    vectors = client.encode([probe["text"] for probe in probes], "document")
    results = []
    for probe, vector in zip(probes, vectors):
        reference = normalize_vector(probe["vector"], DIMENSION)
        cosine = math.fsum(a * b for a, b in zip(reference, vector))
        difference = max(abs(a - b) for a, b in zip(reference, vector))
        results.append({"segment_sha256": probe["segment_sha256"], "cosine": cosine,
                        "max_absolute_difference": difference,
                        "passed": cosine >= 0.999999 and difference <= 1e-5})
    return {"passed": all(row["passed"] for row in results), "segments": results}


def prefetch_queries(client, source, instruct):
    cache = {}
    texts = source["unique_segments"]
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        cache.update(zip(batch, client.encode(batch, "query", instruct)))
    result = {}
    for query, parts in source["query_segments"].items():
        vectors = [cache[part] for part in parts]
        result[query] = normalize_vector(tuple(math.fsum(vector[index] for vector in vectors) / len(vectors)
                                               for index in range(DIMENSION)), DIMENSION)
    return result


@contextmanager
def search_only_store(database, backend):
    store = MemoryStore(database, semantic_backend=backend, context_radius=1,
                        semantic_weight=1, context_grouping="reserved")
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/search":
                self.send(405, {"error": "Search-only replay"})
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send(401, {"error": "Unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError()
                payload = json.loads(self.rfile.read(length))
                self.send(200, store.search(payload))
            except Exception:
                self.send(500, {"error": "Cached Search failed"})

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    service.daemon_threads = True
    worker = threading.Thread(target=service.serve_forever, daemon=True)
    worker.start()
    client = HTTPClient("http://127.0.0.1:" + str(service.server_port), token, 120)
    client.opener = build_opener(ProxyHandler({}), NoRedirect())
    try:
        yield client
    finally:
        service.shutdown()
        service.server_close()
        worker.join(timeout=5)


def replay(source, folder, variant, vectors, ledger):
    folder.mkdir(mode=0o700)
    database = folder / "memory.sqlite3"
    backup_database(source["snapshot"], database)
    expected_identity = source["plan"]["source_database_identity"]
    variant_spec = {"model": MODEL, "dimension": DIMENSION, "protocol": "dashscope-native",
                    "endpoint": source["plan"]["native_endpoint"], "text_type": "query",
                    "instruct": INSTRUCT if variant == "C" else "", "segment_bytes": SEGMENT_BYTES,
                    "long_query": "normalized-segment-mean-v1", "normalization": "l2"}
    variant_hash = digest(variant_spec)
    backend = CachedQueryBackend(source["plan"]["source_corpus_fingerprint"], vectors, variant_hash)
    records, started = [], time.monotonic()
    with search_only_store(database, backend) as client:
        with (folder / "retrieval.jsonl").open("w", encoding="utf-8") as output:
            for query in source["queries"]:
                row = {"query_id": query["query_id"], "sample_id": query["sample_id"]}
                if ledger.stopped or ledger.remaining_seconds() <= 0:
                    ledger.stop("runtime_limit" if not ledger.stopped else ledger.state["stop_reason"])
                    row.update(status="search_error", error="NotExecutedAfterTerminalStop")
                else:
                    try:
                        client.timeout = min(120, ledger.remaining_seconds())
                        status, response, seconds, size = client.request("/search", {
                            "query": query["query"], "user_id": source["users"][query["sample_id"]], "top_k": 100})
                        if status != 200:
                            raise ValueError()
                        row.update(status="ok", retrieved=validate_search(response, 100), search_seconds=seconds,
                                   response_bytes=size)
                    except Exception:
                        ledger.stop("cached_search_failure")
                        row.update(status="search_error", error="CachedSearchFailure")
                records.append(row)
                output.write(json_line(row))
                output.flush()
    if database_identity(database) != expected_identity:
        ledger.stop("copied_database_changed")
        raise ValueError("Search-only replay changed the copied corpus or index")
    configuration = {**source["summary"]["configuration"], "add_concurrency": 0,
                     "embedding_provider": "cached_native_query_only", "source_corpus_fingerprint": backend.fingerprint,
                     "query_variant_sha256": variant_hash, "query_variant": variant_spec,
                     "document_embeddings_reused": True, "gold_available_to_candidate": False}
    summary = {"status": "public_proxy_not_official_aml", "dataset": "locomo_refined", "split": "dev",
               "run_id": folder.name, "ingestion_run_id": source["plan"]["ingestion_run_id"],
               "model_label": "v4-native-query-" + variant, "question_count": len(records),
               "history_count": source["plan"]["histories"], "add_request_count": 0,
               "ingestion_failed_histories": 0, "add_latency": percentiles([]),
               "search_latency": percentiles([row["search_seconds"] for row in records if row["status"] == "ok"]),
               "elapsed_seconds": time.monotonic() - started, "configuration": configuration,
               "server_source_changed_during_run": False, "ingestion_provenance": source["plan"],
               "query_cache_sha256": digest({hashlib.sha256(text.encode()).hexdigest(): digest(vector)
                                             for text, vector in vectors.items()}),
               "evaluation_kind": "query_only_public_dev_no_add_or_document_reindex",
               "full_capacity_validated": False}
    atomic_json(folder / "summary.json", summary)
    return records


def postprocess_compare(source, arms, data_root, output):
    """All provider calls and candidates have closed before gold is read here."""
    prepared = Path(data_root) / "prepared" / "locomo_refined" / "dev"
    if file_hash(prepared / "gold.jsonl") != source["plan"]["prepared_data"]["files"]["gold"]["sha256"]:
        raise ValueError("Gold checksum changed before local postprocessing")
    labels = {row["query_id"]: row for row in read_jsonl(prepared / "gold.jsonl")}
    maps, frequencies = {}, {}
    for sample, history in source["histories"].items():
        _, _, sources, ids = requests_for_history(history, source["plan"]["ingestion_run_id"], SOURCE_MAPPING_VERSION)
        maps[sample] = (sources, ids)
        texts = {normalized(unit["text"]) for query in source["queries"] if query["sample_id"] == sample
                 for unit in labels[query["query_id"]]["gold_units"]}
        frequencies[sample] = Counter({text: sum(text in normalized(message["content"])
            for session in history["sessions"] for message in session["messages"]) for text in texts})
    for folder in arms:
        records = read_jsonl(folder / "retrieval.jsonl")
        for row in records:
            label = labels[row["query_id"]]
            if label["sample_id"] != row["sample_id"] or row["status"] != "ok":
                raise ValueError("Incomplete or mismatched query-only result cannot be scored")
            sources, ids = maps[row["sample_id"]]
            row.update(original_query_id=label["original_query_id"], category=label["category"])
            row["metrics"] = {str(k): {**evidence_hits(row["retrieved"][:k], label, sources, ids,
                frequencies[row["sample_id"]]), "returned_items": len(row["retrieved"][:k]),
                "returned_characters": sum(len(item["content"]) for item in row["retrieved"][:k])} for k in KS}
        with (folder / "retrieval.jsonl").open("w", encoding="utf-8") as stream:
            stream.writelines(json_line(row) for row in records)
        summary = json.loads((folder / "summary.json").read_text())
        summary["metrics_by_k"] = {str(k): aggregate(records, k, len(records)) for k in KS}
        atomic_json(folder / "summary.json", summary)
    result = compare([source["source_run"], *arms], Path(data_root), KS, [0])
    atomic_json(output / "comparison.json", result)
    return result


def run_experiment(args, source, output, transport=https_transport, key_reader=read_model_key):
    output = Path(output)
    ledger = BudgetLedger(output / "ledger.json", args.max_input_tokens, args.max_requests, args.max_runtime_seconds)
    report = {"status": "running_query_only_public_dev", "plan": source["plan"], "runs": [],
              "gold_read_before_retrieval_completed": False, "full_capacity_validated": False}
    report["runs"] = [{"variant": variant, "planned": len(source["queries"]), "completed": 0,
                       "successful": 0, "failed": 0, "unfinished": len(source["queries"])}
                      for variant in ("B", "C")]
    timer = threading.Timer(args.max_runtime_seconds, ledger.stop, args=("runtime_limit",))
    timer.daemon = True
    timer.start()
    native = None
    try:
        source["snapshot"] = output / "source-snapshot.sqlite3"
        backup_database(source["database"], source["snapshot"])
        if database_identity(source["snapshot"]) != source["plan"]["source_database_identity"]:
            raise ValueError("Baseline snapshot differs from audited source")
        key = key_reader(args.model_env_file, source["compatible_endpoint"], args.api_key_name)
        native = NativeV4Client(source["compatible_endpoint"], args.endpoint, key, ledger, transport, args.http_timeout)
        report["document_gate"] = document_equivalence(native, source["probes"])
        atomic_json(output / "document-gate.json", report["document_gate"])
        if not report["document_gate"]["passed"]:
            ledger.stop("document_equivalence_failure")
            raise BudgetStopped("Native document vectors failed the fixed equivalence gate")
        cache_b = prefetch_queries(native, source, "")
        cache_c = prefetch_queries(native, source, INSTRUCT)
        native.close()  # Model calls close before cached Search and any gold access.
        ledger.state["model_call_phase_closed"] = True
        ledger._save()
        arms = []
        for arm_index, (variant, vectors) in enumerate((("B", cache_b), ("C", cache_c))):
            folder = output / ("native-query-" + variant)
            records = replay(source, folder, variant, vectors, ledger)
            successful = sum(row["status"] == "ok" for row in records)
            unfinished = sum(row.get("error") == "NotExecutedAfterTerminalStop" for row in records)
            report["runs"][arm_index].update(directory=str(folder), successful=successful,
                completed=len(records) - unfinished, failed=len(records) - unfinished - successful,
                unfinished=unfinished)
            arms.append(folder)
            if ledger.stopped or any(row["status"] != "ok" for row in records):
                raise BudgetStopped("Cached query replay failed")
        if (file_hash(source["database"]) != source["plan"]["source_db_sha256"]
                or database_identity(source["database"]) != source["plan"]["source_database_identity"]
                or file_hash(source["source_run"] / "summary.json") != source["plan"]["source_summary_sha256"]
                or file_hash(source["source_run"] / "retrieval.jsonl") != source["plan"]["source_retrieval_sha256"]
                or {name: file_hash(ROOT / name) for name in ("server.py", "semantic.py")} != source["plan"]["core_source_sha256"]):
            raise ValueError("Baseline database or core source changed during replay")
        if ledger.stopped or ledger.remaining_seconds() <= 0:
            raise BudgetStopped("Experiment deadline reached before postprocessing")
        postprocess_compare(source, arms, args.data_root, output)
        if ledger.stopped or ledger.remaining_seconds() <= 0:
            raise BudgetStopped("Experiment deadline reached during postprocessing")
        report["status"] = "completed_query_only_public_dev"
    except Exception:
        ledger.stop("query_mode_runner_failure")
        report["status"] = "stopped_query_only_public_dev"
        report["error"] = "Terminal experiment failure; all further model calls are blocked"
    finally:
        if native is not None:
            native.close()
        timer.cancel()
        with ledger.lock:
            ledger._save()
            report["budget"] = {key: value for key, value in ledger.state.items() if key != "calls"}
        atomic_json(output / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--model-env-file", type=Path, required=True)
    result.add_argument("--api-key-name", choices=("MEMORY_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY"), default="MEMORY_EMBEDDING_API_KEY")
    result.add_argument("--endpoint", required=True, help="Same-workspace native endpoint; private env keeps its compatible URL")
    result.add_argument("--max-input-tokens", type=int, required=True)
    result.add_argument("--max-requests", type=int, required=True)
    result.add_argument("--max-runtime-seconds", type=float, default=900)
    result.add_argument("--http-timeout", type=float, default=30)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--execute", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.max_input_tokens <= 0 or args.max_requests <= 0 or not math.isfinite(args.max_runtime_seconds)
            or args.max_runtime_seconds <= 0 or not math.isfinite(args.http_timeout) or not 0 < args.http_timeout <= 120):
        raise ValueError("Positive finite token/request/runtime limits and a bounded HTTP timeout are required")
    output = args.output_dir.resolve()
    if not output.is_relative_to(PRIVATE_ROOT.resolve()) or output == PRIVATE_ROOT.resolve():
        raise ValueError("Output must be a new private starter/.local directory")
    source = inspect_source(args)
    mask = os.umask(0o077)
    try:
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        plan = {"status": "query_modes_dry_plan_no_model_calls", **source["plan"], "credential_file_read": False,
                "max_input_tokens": args.max_input_tokens, "max_requests": args.max_requests,
                "max_runtime_seconds": args.max_runtime_seconds,
                "budget_fits_conservative_plan": source["plan"]["reservation_upper_estimate"] <= args.max_input_tokens
                    and source["plan"]["planned_model_requests"] <= args.max_requests}
        atomic_json(output / "plan.json", plan)
        report = run_experiment(args, source, output) if args.execute else plan
        print(json.dumps({"status": report["status"], "output_directory": str(output),
                          "planned_questions": source["plan"]["questions"],
                          "planned_model_requests": source["plan"]["planned_model_requests"],
                          "reservation_upper_estimate": source["plan"]["reservation_upper_estimate"],
                          "model_requests": report.get("budget", {}).get("attempted_requests", 0)}, ensure_ascii=False))
        return 0 if report["status"] in ("query_modes_dry_plan_no_model_calls", "completed_query_only_public_dev") else 2
    finally:
        os.umask(mask)


if __name__ == "__main__":
    raise SystemExit(main())
