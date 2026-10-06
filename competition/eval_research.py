#!/usr/bin/env python3
"""Audited SEARCH-ONLY public-dev research using a copied completed index.

Default invocation only inspects; --execute-search explicitly enables fresh HTTP
Search calls. No Add route, paid provider, validation/heldout, or old-response
replay exists. By default a candidate may differ only in the whole-item char
cap. Exact reviewed _fuse/search AST hashes permit only those named methods;
every other server AST node and semantic.py must remain unchanged. These are
separate recorded policies, not a general permission to alter the service.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.parse

from eval_audit import audit
from eval_prepare import HERE, LEGACY_SOURCE_MAPPING_VERSION, SOURCE_MAPPING_VERSION, file_hash, json_line, normalized
from eval_retrieval import (GRADER_VERSION, HTTPClient, KS, aggregate, free_local_port,
                            local_embedding_environment, percentiles, requests_for_history, validate_search)

CORE_TABLES = {"requests": "user_id,request_id", "messages": "sequence",
               "vectors": "user_id,fingerprint,message_id,segment_index"}
COUNT_TABLES = ("requests", "messages", "vectors", "postings", "user_statistics", "term_statistics")


def immutable_database(path):
    wal = path.with_name(path.name + "-wal")
    if not path.is_file() or (wal.exists() and wal.stat().st_size):
        raise ValueError("Completed source DB is missing or has live WAL data")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def database_identity(path):
    """Hash source payloads, ordered messages/IDs and exact vector bytes; no text output."""
    digest = hashlib.sha256()
    with closing(immutable_database(path)) as connection:
        counts = {table: connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                  for table in COUNT_TABLES}
        for table, order in CORE_TABLES.items():
            digest.update(table.encode() + b"\0")
            for row in connection.execute("SELECT * FROM " + table + " ORDER BY " + order):
                for value in row:
                    if value is None:
                        tag, data = b"N", b""
                    elif isinstance(value, bytes):
                        tag, data = b"B", value
                    elif isinstance(value, str):
                        tag, data = b"S", value.encode("utf-8")
                    elif isinstance(value, int):
                        tag, data = b"I", str(value).encode()
                    else:
                        raise ValueError("Unexpected stored corpus value")
                    digest.update(tag + str(len(data)).encode() + b":" + data)
                digest.update(b"\n")
        groups = connection.execute(
            "SELECT fingerprint,dimension,length(vector),COUNT(*) FROM vectors "
            "GROUP BY fingerprint,dimension,length(vector)").fetchall()
    return {"core_sha256": digest.hexdigest(), "table_counts": counts, "vector_groups": groups}


def safe_environment(provider, model_dir):
    # Keep basic OS/runtime paths; do not pass paid-model credentials, proxies,
    # PYTHONPATH, or inherited embedding knobs to the candidate process.
    essential = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR")
    parent = {key: os.environ[key] for key in essential if key in os.environ}
    env = local_embedding_environment(parent, provider, model_dir)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def selected_jsonl(path, field, wanted):
    selected = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row[field]
            if key in wanted:
                if key in selected:
                    raise ValueError("Duplicate selected prepared identifier")
                selected[key] = row
                if set(selected) == wanted:
                    break
    if set(selected) != wanted:
        raise ValueError("Missing selected prepared data")
    return selected


def inspect_source(source_run, data_root, model_dir=None):
    # Gate split/completion BEFORE opening retrieval records, prepared data or DB.
    summary_path = source_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    config = summary["configuration"]
    if summary.get("split") != "dev" or summary.get("status") != "public_proxy_not_official_aml":
        raise ValueError("Only completed PUBLIC dev runs may be reused")
    if not config.get("launch_local") or summary.get("server_source_changed_during_run"):
        raise ValueError("Source must be a stable, locally launched complete ingestion run")
    if summary.get("ingestion_failed_histories") != 0 or summary.get("ingestion_run_id"):
        raise ValueError("Source must have complete original ingestion, not a partial/Search-only run")
    if config.get("top_k") != 100 or config.get("character_budget") != 0 or config.get("search_concurrency") != 1:
        raise ValueError("Source must have untrimmed top100 and serial Search for a matched rerun")
    provider = config.get("embedding_provider")
    if provider not in ("disabled", "local"):
        raise ValueError("Only verified disabled/local CPU providers are allowed; no paid HTTP provider")
    expected_backend = "lexical" if provider == "disabled" else "hybrid_local_e5"
    if config.get("forced_local_backend") != expected_backend:
        raise ValueError("Declared source backend is inconsistent")
    if provider == "local" and (model_dir is None or not model_dir.is_dir()):
        raise ValueError("Local source requires an explicit already downloaded pinned model directory")
    if provider == "disabled" and model_dir is not None:
        raise ValueError("Do not provide a model for a disabled source backend")
    source_url = urllib.parse.urlsplit(config["base_url"])
    if source_url.hostname not in ("127.0.0.1", "localhost", "::1") or not source_url.port:
        raise ValueError("Only a completed loopback source service is supported")
    family = socket.AF_INET6 if source_url.hostname == "::1" else socket.AF_INET
    with socket.socket(family) as probe:
        probe.settimeout(.5)
        if probe.connect_ex((source_url.hostname, source_url.port)) == 0:
            raise ValueError("Source service port is still open; close its service before copying")
    source_hashes = config["source_code_sha256"]
    for name, expected in (("server.py", source_hashes["server.py_at_start"]),
                           ("semantic.py", source_hashes["semantic.py"]),
                           ("eval_retrieval.py", source_hashes["eval_retrieval.py"]),
                           ("eval_prepare.py", source_hashes["eval_prepare.py"])):
        if not expected or file_hash(source_run / "source_snapshots" / name) != expected:
            raise ValueError("Source snapshots do not match recorded code hashes")
    if source_hashes["server.py_at_start"] != source_hashes["server.py_at_end"]:
        raise ValueError("Source server was modified")
    folder = data_root / "prepared" / summary["dataset"] / "dev"
    specification = config["prepared_data"]
    for name in ("histories", "queries", "gold"):
        if file_hash(folder / (name + ".jsonl")) != specification["files"][name]["sha256"]:
            raise ValueError("Prepared dev data differs from original ingestion")
    records = []
    with (source_run / "retrieval.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("status") != "ok":
                raise ValueError("Source has failed questions; preserve it as a failure audit")
            # Original response content is never replayed or used as new Search output.
            records.append({"query_id": row["query_id"], "sample_id": row["sample_id"]})
    query_ids = {row["query_id"] for row in records}
    if len(records) != summary["question_count"] or len(query_ids) != len(records):
        raise ValueError("Source records are incomplete or contain duplicate question IDs")
    queries = selected_jsonl(folder / "queries.jsonl", "query_id", query_ids)
    labels = selected_jsonl(folder / "gold.jsonl", "query_id", query_ids)
    sample_ids = {row["sample_id"] for row in records}
    if len(sample_ids) != summary["history_count"]:
        raise ValueError("Source result history scope differs from original completed ingestion")
    for row in records:
        if queries[row["query_id"]]["sample_id"] != row["sample_id"] or labels[row["query_id"]]["sample_id"] != row["sample_id"]:
            raise ValueError("Source query/gold/history scope differs")
    histories = selected_jsonl(folder / "histories.jsonl", "sample_id", sample_ids)
    mapping_version = config.get("source_mapping_version", LEGACY_SOURCE_MAPPING_VERSION)
    expected_requests, expected_messages, source_maps, frequencies = {}, {}, {}, {}
    for sample, history in histories.items():
        session_ids = [session["session_id"] for session in history["sessions"]]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Legacy ambiguous duplicate sessions cannot be reused")
        user, payloads, sources, ids = requests_for_history(history, summary["run_id"], mapping_version)
        source_maps[sample] = (user, sources, ids)
        texts = {normalized(unit["text"]) for gold in labels.values() if gold["sample_id"] == sample
                 for unit in gold["gold_units"]}
        frequencies[sample] = Counter({text: sum(text in normalized(message["content"])
                                               for session in history["sessions"] for message in session["messages"])
                                       for text in texts})
        for payload in payloads:
            key = (user, payload["request_id"])
            if key in expected_requests:
                raise ValueError("Source request namespace is not unique")
            expected_requests[key] = payload
            for index, message in enumerate(payload["messages"]):
                identity = json.dumps([user, payload["request_id"], index], ensure_ascii=False, separators=(",", ":")).encode()
                mid = "msg_" + hashlib.sha256(identity).hexdigest()
                expected_messages[mid] = (user, payload["session_id"], payload["request_id"], index,
                                          message["role"], message["content"], message.get("timestamp"))
        for gold in (label for label in labels.values() if label["sample_id"] == sample):
            positions = {source["position"]: source["content"] for source in sources.values()}
            for unit in gold["gold_units"]:
                if positions.get((unit["session_id"], unit["message_position"])) != unit.get("source_message_content"):
                    raise ValueError("Gold location does not identify its complete original message")
    database = data_root / "runs" / summary["run_id"] / "memory.sqlite3"
    physical_before = file_hash(database)
    with closing(immutable_database(database)) as connection:
        actual_requests = {}
        for user, rid, payload in connection.execute("SELECT user_id,request_id,payload_json FROM requests"):
            actual_requests[(user, rid)] = json.loads(payload)
        if actual_requests != expected_requests or len(actual_requests) != summary["add_request_count"]:
            raise ValueError("Stored ingestion payloads differ from prepared original source scope")
        actual_messages = {row[0]: tuple(row[1:]) for row in connection.execute(
            "SELECT id,user_id,session_id,request_id,message_index,role,content,source_timestamp_ms FROM messages")}
        if actual_messages != expected_messages:
            raise ValueError("Stored IDs, message content/timestamps or source positions differ")
        if provider == "local":
            orphaned = connection.execute(
                "SELECT COUNT(*) FROM vectors v LEFT JOIN messages m ON m.id=v.message_id "
                "WHERE m.id IS NULL OR v.user_id!=m.user_id").fetchone()[0]
            coverage = connection.execute("SELECT COUNT(DISTINCT message_id) FROM vectors").fetchone()[0]
            if orphaned or coverage != len(expected_messages):
                raise ValueError("Semantic index does not cover every original source message")
    identity = database_identity(database)
    semantic_audit = audit(model_dir, [source_run], data_root) if provider == "local" else None
    if provider == "disabled" and identity["table_counts"]["vectors"]:
        raise ValueError("Disabled backend source unexpectedly contains semantic vectors")
    if file_hash(database) != physical_before:
        raise ValueError("Source database changed during inspection")
    metadata = {"source_run_id": summary["run_id"], "dataset": summary["dataset"], "split": "dev",
                "source_summary_sha256": file_hash(summary_path),
                "source_retrieval_sha256": file_hash(source_run / "retrieval.jsonl"),
                "source_db_sha256": physical_before, "source_db_bytes": database.stat().st_size,
                "source_database_path": str(database.resolve()), "source_run_directory": str(source_run.resolve()),
                "source_mapping_version": mapping_version, "embedding_provider": provider,
                "actual_backend": expected_backend, "actual_database_identity": identity,
                "semantic_audit": semantic_audit,
                "matched_queries": len(records), "matched_histories": len(histories),
                "prepared_data": specification, "source_code_sha256": source_hashes}
    return {"metadata": metadata, "summary": summary, "database": database, "source_run": source_run,
            "ordered_queries": [queries[row["query_id"]] for row in records], "labels": labels,
            "source_maps": source_maps, "frequencies": frequencies}


def _method_ast_sha256(server_script, name):
    tree = ast.parse(server_script.read_text())
    stores = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MemoryStore"]
    if len(stores) != 1:
        raise ValueError("Expected exactly one MemoryStore class")
    methods = [node for node in stores[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == name]
    if len(methods) != 1 or not isinstance(methods[0], ast.FunctionDef):
        raise ValueError("Expected exactly one synchronous MemoryStore." + name + " method")
    return hashlib.sha256(ast.dump(methods[0], include_attributes=False).encode()).hexdigest()


def fuse_ast_sha256(server_script):
    """Return the exact MemoryStore._fuse AST hash for explicit review approval."""
    return _method_ast_sha256(server_script, "_fuse")


def search_ast_sha256(server_script):
    """Return the exact reviewed evidence-assembly/search implementation hash."""
    return _method_ast_sha256(server_script, "search")


def validate_candidate(candidate, source, approved_fuse_ast_sha256=None, approved_search_ast_sha256=None):
    if candidate.name != "server.py":
        raise ValueError("Candidate snapshot must contain server.py and the unchanged semantic.py")
    for name, approved in (("_fuse", approved_fuse_ast_sha256), ("search", approved_search_ast_sha256)):
        if approved is not None and (len(approved) != 64 or any(character not in "0123456789abcdef" for character in approved)):
            raise ValueError("Approved " + name + " AST hash must be an exact lowercase SHA256")
    reference = source["source_run"] / "source_snapshots" / "server.py"
    original_fuse_hash = fuse_ast_sha256(reference)
    candidate_fuse_hash = fuse_ast_sha256(candidate)
    if approved_fuse_ast_sha256 is not None and candidate_fuse_hash != approved_fuse_ast_sha256:
        raise ValueError("Candidate _fuse AST does not match the explicitly approved hash")
    original_search_hash = search_ast_sha256(reference)
    candidate_search_hash = search_ast_sha256(candidate)
    if approved_search_ast_sha256 is not None and candidate_search_hash != approved_search_ast_sha256:
        raise ValueError("Candidate search AST does not match the explicitly approved hash")
    methods = []
    if approved_fuse_ast_sha256 is not None:
        methods.append("_fuse")
    if approved_search_ast_sha256 is not None:
        methods.append("search")
    def parsed(path):
        tree = ast.parse(path.read_text())
        cap = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MAX_EVIDENCE_CHARACTERS"
                                                    for target in node.targets):
                cap = ast.literal_eval(node.value)
                node.value = ast.Constant(value=0)
        if type(cap) is not int or cap <= 0:
            raise ValueError("Whole-item character cap must be one literal positive integer")
        if methods:
            # Only this explicitly reviewed method may differ. Replace it at its
            # existing class-body position, so moving it/adding methods or changing
            # Add, Search scope, Handler, imports or constants is still rejected.
            store = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MemoryStore")
            for name in methods:
                index = next(index for index, node in enumerate(store.body)
                             if isinstance(node, ast.FunctionDef) and node.name == name)
                store.body[index] = ast.parse("def " + name + "():\n    pass\n").body[0]
        return ast.dump(tree, include_attributes=False), cap
    before, original_cap = parsed(reference)
    after, candidate_cap = parsed(candidate)
    if before != after:
        if methods:
            raise ValueError("Candidate may change only MAX_EVIDENCE_CHARACTERS and the approved " + ", ".join(methods) + "; other server logic differs")
        raise ValueError("Candidate may change only MAX_EVIDENCE_CHARACTERS; other server logic differs")
    expected_semantic = source["metadata"]["source_code_sha256"]["semantic.py"]
    if file_hash(candidate.parent / "semantic.py") != expected_semantic:
        raise ValueError("Candidate semantic code differs from original ingestion")
    policy = "cap_only_v1"
    if approved_fuse_ast_sha256 is not None:
        policy = "cap_plus_explicitly_approved_fuse_v1"
    if approved_search_ast_sha256 is not None:
        policy = ("cap_plus_explicitly_reviewed_search_and_fuse_v1" if approved_fuse_ast_sha256 is not None
                  else "cap_plus_explicitly_reviewed_search_v1")
    return {"server_sha256": file_hash(candidate), "semantic_sha256": expected_semantic,
            "original_service_whole_item_character_cap": original_cap,
            "service_whole_item_character_cap": candidate_cap,
            "only_server_ast_change": "MAX_EVIDENCE_CHARACTERS" + (" and exactly reviewed MemoryStore." + ", MemoryStore.".join(methods) if methods else ""),
            "server_ast_change_policy": policy,
            "original_fuse_ast_sha256": original_fuse_hash,
            "candidate_fuse_ast_sha256": candidate_fuse_hash,
            "approved_fuse_ast_sha256": approved_fuse_ast_sha256,
            "original_search_ast_sha256": original_search_hash,
            "candidate_search_ast_sha256": candidate_search_hash,
            "approved_search_ast_sha256": approved_search_ast_sha256}


def backup_source(source, target):
    if target.exists() or target.resolve() == source["database"].resolve():
        raise ValueError("Copied database must use a new independent path")
    if file_hash(source["database"]) != source["metadata"]["source_db_sha256"]:
        raise ValueError("Original DB changed before backup")
    target.parent.mkdir(parents=True)
    with closing(immutable_database(source["database"])) as original, closing(sqlite3.connect(target)) as copied:
        original.backup(copied)
    copied_identity = database_identity(target)
    if copied_identity != source["metadata"]["actual_database_identity"]:
        raise ValueError("SQLite backup does not preserve corpus IDs/content/index")
    if file_hash(source["database"]) != source["metadata"]["source_db_sha256"]:
        raise ValueError("Original DB changed during backup")
    return {"copied_database_path": str(target.resolve()), "sqlite_backup_sha256": file_hash(target),
            "copied_database_identity": copied_identity, "source_database_unchanged": True}


def search_records(source, client):
    """Always perform fresh HTTP Search, retaining ORIGINAL ingestion scope."""
    from eval_retrieval import evidence_hits
    for query in source["ordered_queries"]:
        gold = source["labels"][query["query_id"]]
        user, sources, ids = source["source_maps"][query["sample_id"]]
        row = {"query_id": query["query_id"], "original_query_id": gold["original_query_id"],
               "sample_id": query["sample_id"], "category": gold["category"]}
        try:
            status, response, elapsed, size = client.request("/search", {
                "query": query["query"], "user_id": user, "top_k": 100})
            if status != 200:
                raise ValueError("Search was not HTTP 200")
            data = validate_search(response, 100)
            metrics = {str(k): {**evidence_hits(data[:k], gold, sources, ids, source["frequencies"][query["sample_id"]]),
                                "returned_items": len(data[:k]),
                                "returned_characters": sum(len(item["content"]) for item in data[:k])}
                       for k in KS}
            yield {**row, "status": "ok", "search_seconds": elapsed, "response_bytes": size,
                   "metrics": metrics, "retrieved": data}
        except Exception as error:
            yield {**row, "status": "search_error", "error": type(error).__name__ + ": " + str(error)}


def execute_search(source, candidate, model_dir, results_root, db_root, model_label, timeout, startup_timeout,
                   approved_fuse_ast_sha256=None, approved_search_ast_sha256=None):
    candidate_info = validate_candidate(candidate, source, approved_fuse_ast_sha256, approved_search_ast_sha256)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-searchonly-" + secrets.token_hex(4)
    result_dir = results_root / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    database = db_root / run_id / "memory.sqlite3"
    process, log = None, None
    try:
        lineage = backup_source(source, database)
        snapshots = result_dir / "source_snapshots"
        snapshots.mkdir()
        import shutil
        component_paths = (Path(__file__), HERE / "eval_prepare.py", HERE / "eval_retrieval.py", HERE / "eval_audit.py")
        component_hashes = {path.name: file_hash(path) for path in component_paths}
        for path in (*component_paths, candidate, candidate.parent / "semantic.py"):
            shutil.copy2(path, snapshots / path.name)
        runtime_server = snapshots / "server.py"
        config = source["summary"]["configuration"]
        provider = source["metadata"]["embedding_provider"]
        env = safe_environment(provider, model_dir)
        token = secrets.token_urlsafe(32)
        env["MEMORY_API_TOKEN"] = token
        declared = {key: value for key, value in env.items()
                    if key.startswith("MEMORY_EMBEDDING_") or key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        port = free_local_port()
        url = "http://127.0.0.1:" + str(port)
        log = (result_dir / "service.log").open("w", encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(runtime_server), "--host", "127.0.0.1", "--port", str(port),
                                    "--db", str(database), "--context-radius", str(config["context_radius"]),
                                    "--semantic-weight", str(config["semantic_weight"])],
                                   env=env, stdout=log, stderr=log)
        client = HTTPClient(url, token, timeout)
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Candidate exited before becoming healthy")
            try:
                if 200 <= client.request("/health")[0] < 300:
                    break
            except Exception:
                pass
            time.sleep(.1)
        else:
            raise RuntimeError("Candidate did not become healthy")
        start = time.perf_counter()
        records = []
        with (result_dir / "retrieval.jsonl").open("w", encoding="utf-8") as out:
            for row in search_records(source, client):
                records.append(row)
                out.write(json_line(row))
                out.flush()
        elapsed = time.perf_counter() - start
        process.terminate()
        process.wait(timeout=10)
        process = None
        if database_identity(database) != source["metadata"]["actual_database_identity"]:
            raise ValueError("Candidate altered copied corpus/index; results are not valid matched research")
        if file_hash(source["database"]) != source["metadata"]["source_db_sha256"]:
            raise ValueError("Original source DB changed; audit before treating the comparison as valid")
        if (file_hash(candidate) != candidate_info["server_sha256"]
                or file_hash(candidate.parent / "semantic.py") != candidate_info["semantic_sha256"]
                or file_hash(runtime_server) != candidate_info["server_sha256"]
                or file_hash(snapshots / "semantic.py") != candidate_info["semantic_sha256"]
                or any(file_hash(path) != component_hashes[path.name] for path in component_paths)):
            raise ValueError("Candidate server source changed during Search")
        summary = {"status": "public_proxy_not_official_aml", "evaluation_kind": "search_only_copied_completed_dev_index",
                   "run_id": run_id, "ingestion_run_id": source["summary"]["run_id"], "model_label": model_label,
                   "dataset": source["summary"]["dataset"], "split": "dev", "grader_version": GRADER_VERSION,
                   "source_mapping_version": source["metadata"]["source_mapping_version"],
                   "question_count": len(records), "history_count": source["metadata"]["matched_histories"],
                   "add_request_count": 0, "ingestion_failed_histories": 0,
                   "add_latency": percentiles([]),
                   "search_latency": percentiles([row["search_seconds"] for row in records if row["status"] == "ok"]),
                   "elapsed_seconds": elapsed, "metrics_by_k": {str(k): aggregate(records, k, len(records)) for k in KS},
                   "ingestion_provenance": source["metadata"], "database_backup": lineage,
                   "configuration": {"base_url": url, "launch_local": True, "forced_local_backend": source["metadata"]["actual_backend"],
                                     "embedding_provider": provider, "declared_local_embedding_environment": declared,
                                     "context_radius": config["context_radius"], "semantic_weight": config["semantic_weight"],
                                     "prepared_data": config["prepared_data"], "top_k": 100, "character_budget": 0,
                                     "service_whole_item_character_cap": candidate_info["service_whole_item_character_cap"],
                                     "server_ast_change_policy": candidate_info["server_ast_change_policy"],
                                     "approved_fuse_ast_sha256": candidate_info["approved_fuse_ast_sha256"],
                                     "approved_search_ast_sha256": candidate_info["approved_search_ast_sha256"],
                                     "search_concurrency": 1, "add_concurrency": 0, "python_version": sys.version.split()[0],
                                     "source_mapping_version": source["metadata"]["source_mapping_version"],
                                     "source_code_sha256": {**component_hashes,
                                                           "server.py_at_start": candidate_info["server_sha256"],
                                                           "server.py_at_end": file_hash(candidate),
                                                           "semantic.py": candidate_info["semantic_sha256"]}},
                   "candidate": candidate_info, "server_source_changed_during_run": False,
                   "runtime_server_snapshot_path": str(runtime_server),
                   "limitations": ["Fresh Search-only requests; original Add/index build is reused, not repeated or timed.",
                                   "Search-only is not a complete Add/Search performance run or proof of production/Full capacity.",
                                   "Original ingestion_run_id MUST rebuild provenance; the new run_id is only the Search experiment identity.",
                                   "Candidate follows the recorded cap-only or exact reviewed _fuse/search AST policy; every other server AST node is unchanged.",
                                   "Use offline token-prefix analysis for the 117760 input constraint; character limits do not prove token budget compliance.",
                                   "No paid provider, Answer/Judge, validation/heldout, official Smoke/Full, or original-response replay."]}
        (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"result_directory": str(result_dir), "run_id": run_id, "ingestion_run_id": summary["ingestion_run_id"],
                "question_count": len(records), "failed_questions": summary["metrics_by_k"]["100"]["failed_questions"]}
    except BaseException as error:
        (result_dir / "error.json").write_text(json.dumps({"run_id": run_id,
            "status": "incomplete_or_invalid_search_only_public_dev_research",
            "error": type(error).__name__ + ": " + str(error)}, indent=2) + "\n")
        raise
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Original completed ingestion's prepared data AND runs DB root")
    parser.add_argument("--embedding-model-dir", type=Path, help="Pinned existing E5 directory, required only for a local source")
    parser.add_argument("--candidate-server-script", type=Path, help="Independent snapshot with identical semantic.py; default cap-only, or exact reviewed _fuse/search methods plus cap")
    parser.add_argument("--approved-fuse-ast-sha256", help="Explicit reviewed exact MemoryStore._fuse AST hash; allows only that Search method plus cap change")
    parser.add_argument("--approved-search-ast-sha256", help="Exact reviewed MemoryStore.search AST hash; allows only that method plus cap and optional exact _fuse change")
    parser.add_argument("--execute-search", action="store_true", help="Explicitly copy DB and issue fresh loopback Search; default only inspects")
    parser.add_argument("--model-label", default="search-only-whole-character-cap-research")
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument("--db-root", type=Path, default=HERE.parent / ".local" / "search-research-indexes")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("Timeouts must be positive")
    if args.execute_search and args.candidate_server_script is None:
        parser.error("--execute-search requires an independent candidate snapshot")
    if args.approved_fuse_ast_sha256 is not None and args.candidate_server_script is None:
        parser.error("--approved-fuse-ast-sha256 requires a candidate snapshot")
    if args.approved_search_ast_sha256 is not None and args.candidate_server_script is None:
        parser.error("--approved-search-ast-sha256 requires a candidate snapshot")
    model_dir = args.embedding_model_dir.resolve() if args.embedding_model_dir else None
    source = inspect_source(args.source_run.resolve(), args.data_root.resolve(), model_dir)
    if not args.execute_search:
        report = {"status": "read_only_inspection_no_database_copy_no_model_load_no_search",
                  "source": source["metadata"]}
        if args.candidate_server_script:
            report["candidate"] = validate_candidate(args.candidate_server_script.resolve(), source, args.approved_fuse_ast_sha256, args.approved_search_ast_sha256)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(json.dumps(execute_search(source, args.candidate_server_script.resolve(), model_dir,
                                   args.results_root.resolve(), args.db_root.resolve(), args.model_label,
                                   args.timeout, args.startup_timeout, args.approved_fuse_ast_sha256, args.approved_search_ast_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
