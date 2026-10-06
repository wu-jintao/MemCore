#!/usr/bin/env python3
"""Evaluate PUBLIC evidence recall against any conformant local Add/Search service.

Never calls an Answer/Judge or a paid provider directly. The target's backend is
the caller's responsibility; --launch-local is explicitly forced to lexical mode.
Questions and gold never enter Add. Validation is for selection; heldout is final.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from eval_prepare import DEFAULT_ROOT, HERE, MANIFEST, SOURCE_MAPPING_VERSION, LEGACY_SOURCE_MAPPING_VERSION, file_hash, json_line, normalized

SOURCE_LINE = re.compile(r"^\[source (\{[^\n]*\})\][ \t]*$", re.MULTILINE)
ADJACENT_WRAPPER = re.compile(r"\n\n\[adjacent context\]\n(?=\[source \{[^\n]*\}\][ \t]*(?:\n|$))")
GRADER_VERSION = "public-full-turn-body-only-v2"
KS = (5, 10, 20, 50, 100)


def local_embedding_environment(parent, provider, model_dir=None):
    """Explicit local experiment settings; inherited embedding knobs/keys cannot leak in."""
    env = {key: value for key, value in parent.items() if not key.startswith("MEMORY_EMBEDDING_")}
    env.update({"MEMORY_EMBEDDING_PROVIDER": provider, "MEMORY_EMBEDDING_ALLOW_HTTP": "0"})
    if provider == "local":
        env.update({"MEMORY_EMBEDDING_MODEL_DIR": str(model_dir.resolve()),
                    "MEMORY_EMBEDDING_MODEL": "intfloat/multilingual-e5-small",
                    "MEMORY_EMBEDDING_DIMENSION": "384", "MEMORY_EMBEDDING_THREADS": "2",
                    "MEMORY_EMBEDDING_BATCH_SIZE": "8", "MEMORY_EMBEDDING_CONCURRENCY": "1",
                    "MEMORY_EMBEDDING_SEGMENT_TOKENS": "384", "MEMORY_EMBEDDING_OVERLAP_TOKENS": "64",
                    "MEMORY_EMBEDDING_MAX_DOCUMENT_SEGMENTS": "512", "MEMORY_EMBEDDING_MAX_TOTAL_SEGMENTS": "4096",
                    "MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS": "8", "MEMORY_EMBEDDING_QUEUE_TIMEOUT": "120",
                    "MEMORY_EMBEDDING_OPERATION_TIMEOUT": "600", "MEMORY_EMBEDDING_QUERY_PREFIX": "query: ",
                    "MEMORY_EMBEDDING_DOCUMENT_PREFIX": "passage: ",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    return env


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentiles(values):
    if not values:
        return {"count": 0, "p50_seconds": None, "p95_seconds": None, "max_seconds": None}
    ordered = sorted(values)
    return {"count": len(values), "p50_seconds": statistics.median(values),
            "p95_seconds": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "max_seconds": ordered[-1]}


def chunks(session, max_messages=20, max_words=2000):
    """Local reproducible approximation: whitespace words, NOT official Adapter count."""
    result, current, count, start = [], [], 0, 0
    for position, message in enumerate(session["messages"]):
        words = len(message["content"].split())
        if current and (len(current) >= max_messages or count + words > max_words):
            result.append((start, current))
            current, count, start = [], 0, position
        current.append(message)
        count += words
    if current:
        result.append((start, current))
    return result


def requests_for_history(history, run_id, source_mapping_version=SOURCE_MAPPING_VERSION):
    # No source dataset names or native question/evidence IDs are exposed to the API.
    if source_mapping_version not in (SOURCE_MAPPING_VERSION, LEGACY_SOURCE_MAPPING_VERSION):
        raise ValueError("Unsupported public source mapping version")
    session_ids = [session["session_id"] for session in history["sessions"]]
    if source_mapping_version == SOURCE_MAPPING_VERSION and len(set(session_ids)) != len(session_ids):
        raise ValueError("Prepared history repeats session IDs; migrate dev to session-occurrence-v2 before running")
    user_id = "proxy/" + run_id + "/" + history["sample_id"]
    requests, sources, id_lookup = [], {}, {}
    for session in history["sessions"]:
        for chunk_index, (start, messages) in enumerate(chunks(session)):
            rid = "r_" + hashlib.sha256((run_id + "\0" + session["session_id"] + "\0" + str(chunk_index)).encode()).hexdigest()
            requests.append({"request_id": rid, "user_id": user_id, "session_id": session["session_id"], "messages": messages})
            for i, message in enumerate(messages):
                position = (session["session_id"], start + i)
                identity = json.dumps([user_id, rid, i], ensure_ascii=False, separators=(",", ":")).encode()
                message_id = "msg_" + hashlib.sha256(identity).hexdigest()
                sources[(session["session_id"], rid, i)] = {"position": position, "content": message["content"], "message_id": message_id}
                id_lookup[message_id] = sources[(session["session_id"], rid, i)]
    return user_id, requests, sources, id_lookup


class HTTPClient:
    def __init__(self, base_url, token, timeout):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        # This explicit opener does not send loopback traffic through HTTP_PROXY.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, payload=None):
        headers = {}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            headers["Authorization"] = "Bearer " + self.token
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers)
        begin = time.perf_counter()
        with self.opener.open(req, timeout=self.timeout) as response:
            data = response.read()
            status = response.status
        return status, json.loads(data), time.perf_counter() - begin, len(data)


def validate_search(response, top_k):
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise ValueError("Search must return a data array")
    data = response["data"]
    if len(data) > top_k:
        raise ValueError("Search exceeds requested top_k")
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ValueError("Search candidate has an invalid id")
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            raise ValueError("Search candidate has invalid text content")
    return data


def source_original(source, sources):
    if not isinstance(source, dict) or type(source.get("message_index")) is not int:
        return None
    if not isinstance(source.get("session_id"), str) or not isinstance(source.get("request_id"), str):
        return None
    original = sources.get((source.get("session_id"), source.get("request_id"), source["message_index"]))
    if original and source.get("message_id") is not None and source["message_id"] != original["message_id"]:
        return None
    return original


def source_segments(content, sources, anchor=None, original_texts=None):
    """Separate generated wrapper metadata from actual message bodies.

    Only the outer first-line source header and the documented adjacent-wrapper
    separator are structural. Once a mapped original starts a body, consume its
    exact length before inspecting more boundaries: source-shaped quotations in
    that original remain evidence. Plain, unwrapped original messages are kept.
    Never join two bodies, which could create a turn absent from either source.
    """
    original_texts = original_texts if original_texts is not None else {s["content"] for s in sources.values()}
    if content in original_texts or (anchor and content == anchor["content"]):
        yield None, content
        return
    match = SOURCE_LINE.match(content)
    if not match:
        boundary = ADJACENT_WRAPPER.search(content)
        if not boundary:
            yield None, content
            return
        if boundary.start():
            yield None, content[:boundary.start()]
        match = SOURCE_LINE.match(content, boundary.end())
    while match:
        try:
            source = json.loads(match[1])
        except json.JSONDecodeError:
            source = None
        start = match.end() + (1 if content[match.end():match.end() + 1] == "\n" else 0)
        original = source_original(source, sources)
        # Exact original length protects legitimate source/adjacency quotations.
        hint = original or anchor
        if not hint or not content.startswith(hint["content"], start):
            candidates = (text for text in original_texts if content.startswith(text, start))
            protected = max(candidates, key=len, default=None)
            hint = {"content": protected} if protected is not None else None
        if hint and content.startswith(hint["content"], start):
            end = start + len(hint["content"])
            yield source, content[start:end]
            boundary = ADJACENT_WRAPPER.match(content, end)
            if not boundary:
                if content[end:].strip():
                    yield None, content[end:]
                return
        else:
            boundary = ADJACENT_WRAPPER.search(content, start)
            end = boundary.start() if boundary else len(content)
            yield source, content[start:end]
            if not boundary:
                return
        match = SOURCE_LINE.match(content, boundary.end())


def evidence_hits(data, gold, sources, id_lookup, text_frequency):
    verified_positions = set()
    bodies = []
    original_texts = {source["content"] for source in sources.values()}
    for item in data:
        content = item["content"]
        # The known baseline ID is useful only when its entire original content is present.
        anchor = id_lookup.get(item["id"])
        segments = list(source_segments(content, sources, anchor, original_texts))
        item_bodies = [normalized(segment) for _, segment in segments]
        bodies.extend(item_bodies)
        if anchor and any(normalized(anchor["content"]) in body for body in item_bodies):
            verified_positions.add(anchor["position"])
        for source, segment in segments:
            original = source_original(source, sources)
            if not original:
                continue
            if normalized(original["content"]) in normalized(segment):
                verified_positions.add(original["position"])
    hits, strict_hits = set(), set()
    ambiguous_units = 0
    for unit in gold["gold_units"]:
        position = (unit["session_id"], unit["message_position"])
        if position in verified_positions:
            hits.add(unit["unit_id"])
            strict_hits.add(unit["unit_id"])
            continue
        text = normalized(unit["text"])
        if text_frequency[text] == 1:
            if any(text in body for body in bodies):
                hits.add(unit["unit_id"])
        else:
            # Repeated generic utterances cannot establish a location without provenance.
            ambiguous_units += 1
    gold_sessions = set(gold["gold_session_ids"])
    session_hits = {sid for sid, _ in verified_positions} & gold_sessions
    return {"gold_turns": len(gold["gold_units"]), "hit_turns": len(hits),
            "source_verified_hit_turns": len(strict_hits), "ambiguous_without_source": ambiguous_units,
            "gold_sessions": len(gold_sessions), "hit_sessions": len(session_hits),
            "recall_any": bool(hits), "recall_all": len(hits) == len(gold["gold_units"]),
            "turn_recall": len(hits) / len(gold["gold_units"]),
            "source_verified_turn_recall": len(strict_hits) / len(gold["gold_units"]),
            "session_recall": len(session_hits) / len(gold_sessions) if gold_sessions else None,
            "session_recall_all": session_hits == gold_sessions if gold_sessions else None}


def prefix_budget(data, k, character_budget):
    result, used = [], 0
    for item in data[:k]:
        size = len(item["content"])
        if character_budget and used + size > character_budget:
            break
        result.append(item)
        used += size
    return result, used


def aggregate(records, k, denominator):
    successful = [record["metrics"][str(k)] for record in records if record.get("status") == "ok"]
    def mean(key):
        # Failures remain in the requested-question denominator, not silently dropped.
        return sum(m[key] for m in successful) / denominator if denominator else None
    return {"requested_questions": denominator, "successful_questions": len(successful),
            "failed_questions": denominator - len(successful),
            "full_turn_recall_any": mean("recall_any"), "full_turn_recall_all": mean("recall_all"),
            "full_turn_recall_macro": mean("turn_recall"),
            "source_verified_turn_recall_macro": mean("source_verified_turn_recall"),
            "session_recall_macro": mean("session_recall"), "session_recall_all": mean("session_recall_all"),
            "gold_turn_annotations": sum(m["gold_turns"] for m in successful),
            "hit_turn_annotations": sum(m["hit_turns"] for m in successful),
            "ambiguous_turn_annotations_without_source": sum(m["ambiguous_without_source"] for m in successful),
            "mean_returned_characters": mean("returned_characters"), "mean_returned_items": mean("returned_items")}


def run_evaluation(args, client, run_id, result_dir):
    folder = args.data_root / "prepared" / args.dataset / args.split
    prep_summary = json.loads((args.data_root / "prepared" / "summary.json").read_text())
    specification = prep_summary["datasets"][args.dataset]["splits"][args.split]
    for name in ("histories", "queries", "gold"):
        if file_hash(folder / (name + ".jsonl")) != specification["files"][name]["sha256"]:
            raise ValueError("Prepared public data checksum mismatch: " + name)
    queries = read_jsonl(folder / "queries.jsonl")
    labels = {label["query_id"]: label for label in read_jsonl(folder / "gold.jsonl")}
    histories = read_jsonl(folder / "histories.jsonl")
    if args.max_histories:
        histories = histories[:args.max_histories]
    allowed = {h["sample_id"] for h in histories}
    queries = sorted((q for q in queries if q["sample_id"] in allowed), key=lambda q: (q["sample_id"], q["query_id"]))
    if args.max_queries:
        queries = queries[:args.max_queries]
    requested_ids = {q["sample_id"] for q in queries}
    histories = [h for h in histories if h["sample_id"] in requested_ids]
    if not queries:
        raise ValueError("No eligible public questions selected")
    # Validate every selected source namespace before the first Add/model call.
    for history in histories:
        session_ids = [session["session_id"] for session in history["sessions"]]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Selected preparation has duplicate session IDs; migrate to session-occurrence-v2")
    if args.split == "heldout":
        if not args.final_heldout:
            raise ValueError("Heldout requires explicit --final-heldout; do not use for selection")
        if args.max_histories or args.max_queries:
            raise ValueError("Final heldout must evaluate the complete prepared split")
        lock = folder / "heldout-evaluated.lock"
        # Mark the attempt before service interaction; a failed attempt must be audited.
        with lock.open("x", encoding="utf-8") as f:
            f.write(json_line({"run_id": run_id, "result_directory": str(result_dir), "model_label": args.model_label}))
    add_times, add_bytes, records = [], 0, []
    source_indexes = {}
    history_failures = {}
    for number, history in enumerate(histories, 1):
        user_id, payloads, sources, ids = requests_for_history(history, run_id)
        # text frequency is scoped to one entire history, not a shared benchmark corpus.
        frequencies = Counter()
        for label in labels.values():
            if label["sample_id"] == history["sample_id"]:
                for unit in label["gold_units"]:
                    # Count history occurrences, rather than annotation repetition across questions.
                    text = normalized(unit["text"])
                    if text not in frequencies:
                        frequencies[text] = sum(text in normalized(m["content"]) for s in history["sessions"] for m in s["messages"])
        source_indexes[history["sample_id"]] = (user_id, sources, ids, frequencies)
        try:
            for payload in payloads:
                status, response, elapsed, size = client.request("/add", payload)
                add_times.append(elapsed)
                add_bytes += size
                expected = {key: payload[key] for key in ("request_id", "user_id", "session_id")}
                if status != 200 or response.get("success") is not True or any(response.get(key) != value for key, value in expected.items()):
                    raise ValueError("Add did not return exact synchronous success")
        except Exception as exc:
            history_failures[history["sample_id"]] = type(exc).__name__ + ": " + str(exc)
        print(json.dumps({"stage": "ingest", "history": number, "total": len(histories), "add_requests": len(payloads), "status": "error" if history["sample_id"] in history_failures else "ok"}), flush=True)

    def search_one(query):
        gold = labels[query["query_id"]]
        record = {"query_id": query["query_id"], "original_query_id": gold["original_query_id"],
                  "sample_id": query["sample_id"], "category": gold["category"]}
        if query["sample_id"] in history_failures:
            return {**record, "status": "ingestion_error", "error": history_failures[query["sample_id"]]}
        user_id, sources, ids, frequencies = source_indexes[query["sample_id"]]
        payload = {"query": query["query"], "user_id": user_id, "top_k": args.top_k}
        # Query dates, category, labels and answers remain outside Search as well.
        try:
            status, response, elapsed, size = client.request("/search", payload)
            if status != 200:
                raise ValueError("Search response is not HTTP 200")
            data = validate_search(response, args.top_k)
            metrics = {}
            for k in (x for x in KS if x <= args.top_k):
                prefix, count = prefix_budget(data, k, args.character_budget)
                metrics[str(k)] = {**evidence_hits(prefix, gold, sources, ids, frequencies),
                                   "returned_items": len(prefix), "returned_characters": count}
            # Raw retrieved evidence is diagnostic; gold answers are never copied into results.
            return {**record, "status": "ok", "search_seconds": elapsed, "response_bytes": size,
                    "metrics": metrics, "retrieved": data}
        except Exception as exc:
            return {**record, "status": "search_error", "error": type(exc).__name__ + ": " + str(exc)}

    with (result_dir / "retrieval.jsonl").open("w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.search_concurrency) as executor:
            for record in executor.map(search_one, queries):
                records.append(record)
                out.write(json_line(record))
                out.flush()
    by_category = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    ks = [k for k in KS if k <= args.top_k]
    summary = {"status": "public_proxy_not_official_aml", "grader_version": GRADER_VERSION,
               "source_mapping_version": SOURCE_MAPPING_VERSION,
               "run_id": run_id, "model_label": args.model_label,
               "dataset": args.dataset, "split": args.split, "scope": "retrieval_only_no_answer_judge_or_paid_model_request_by_harness",
               "question_count": len(queries), "history_count": len(histories), "add_request_count": len(add_times),
               "ingestion_failed_histories": len(history_failures),
               "add_latency": percentiles(add_times), "search_latency": percentiles([r["search_seconds"] for r in records if r["status"] == "ok"]),
               "search_response_bytes": sum(r.get("response_bytes", 0) for r in records),
               "metrics_by_k": {str(k): aggregate(records, k, len(queries)) for k in ks},
               "categories": {name: {str(k): aggregate(rows, k, len(rows)) for k in ks} for name, rows in sorted(by_category.items())},
               "limitations": ["These are public source evidence-recall metrics, not AML Overall, Answer correctness or official Smoke/Full.",
                               "Only the selected development histories/questions are evaluated; they are not a representative capacity proof.",
                               "Full-turn recall requires complete unmodified annotated turn text, not merely a matching session or gold answer string.",
                               "Source provenance verifies original content; unique whole text is a fallback for services without source fields. Repeated text without provenance is not counted.",
                               "Character budget is characters, not tokens; no official answer-context budget is reproduced.",
                               "Question-date injection is not used; original session dates are explicit history text with a documented UTC encoding convention.",
                               "LongMemEval filler history sources may be reused across splits; evidence-source groups remain together."]}
    return summary, specification


def free_local_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("longmemeval_s", "locomo_refined"))
    parser.add_argument("--split", default="dev", choices=("dev", "validation", "heldout"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--allow-remote", action="store_true", help="Explicitly allow a non-loopback test service; never the AML evaluator")
    parser.add_argument("--token-env", default="MEMORY_API_TOKEN")
    parser.add_argument("--launch-local", action="store_true", help="Launch local server.py with a fresh database and random token")
    parser.add_argument("--embedding-provider", default="disabled", choices=("disabled", "local"), help="For local launch only; no paid HTTP provider option")
    parser.add_argument("--embedding-model-dir", type=Path, help="Already downloaded local e5 model; all loading is forced offline")
    parser.add_argument("--context-radius", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--semantic-weight", type=float, default=1.0, help="Global semantic RRF weight, lexical weight fixed at 1")
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--server-script", type=Path, default=HERE.parent / "server.py")
    parser.add_argument("--model-label", required=True, help="Honest method/version description; recorded, not inferred")
    parser.add_argument("--max-histories", type=int, default=0, help="0 means the complete selected split")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100, choices=KS)
    parser.add_argument("--character-budget", type=int, default=0, help="0 means no character-budget truncation")
    parser.add_argument("--search-concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--final-heldout", action="store_true")
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if min(args.max_histories, args.max_queries, args.character_budget) < 0 or args.search_concurrency < 1 or args.timeout <= 0:
        parser.error("Limits must be nonnegative; concurrency and timeout positive")
    if not math.isfinite(args.semantic_weight) or not 0 <= args.semantic_weight <= 4:
        parser.error("Semantic weight must be finite and between 0 and 4")
    if args.embedding_provider == "local" and (not args.launch_local or not args.embedding_model_dir or not args.embedding_model_dir.is_dir()):
        parser.error("Local embedding requires --launch-local and an existing --embedding-model-dir")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    result_dir = args.results_root / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    process = None
    log = None
    database = None
    embedding_env = None
    server_hash = file_hash(args.server_script) if args.launch_local else None
    snapshots = result_dir / "source_snapshots"
    snapshots.mkdir()
    for path in (Path(__file__), HERE / "eval_prepare.py", MANIFEST):
        shutil.copy2(path, snapshots / path.name)
    if args.launch_local:
        for path in (args.server_script, args.server_script.parent / "semantic.py"):
            if path.exists():
                shutil.copy2(path, snapshots / path.name)
    try:
        if args.launch_local:
            port = free_local_port()
            args.base_url = "http://127.0.0.1:" + str(port)
            token = secrets.token_urlsafe(32)
            env = local_embedding_environment(os.environ, args.embedding_provider, args.embedding_model_dir)
            env["MEMORY_API_TOKEN"] = token
            embedding_env = {key: value for key, value in env.items() if key.startswith("MEMORY_EMBEDDING_") or key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
            database = args.data_root / "runs" / run_id / "memory.sqlite3"
            database.parent.mkdir(parents=True, exist_ok=True)
            log = (result_dir / "service.log").open("w", encoding="utf-8")
            process = subprocess.Popen([sys.executable, str(args.server_script), "--host", "127.0.0.1", "--port", str(port), "--db", str(database),
                                        "--context-radius", str(args.context_radius), "--semantic-weight", str(args.semantic_weight)],
                                       env=env, stdout=log, stderr=log)
        else:
            token = os.environ.get(args.token_env, "")
            if not token:
                raise ValueError("Set the token environment variable; do not put a key on the command line")
        host = urllib.parse.urlsplit(args.base_url).hostname
        if host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote:
            raise ValueError("Non-loopback evaluation requires explicit --allow-remote")
        if "agentmemoryleaderboard.ai" in (host or ""):
            raise ValueError("This harness never calls AML formal/Smoke endpoints")
        client = HTTPClient(args.base_url, token, args.timeout)
        ready = False
        startup_deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < startup_deadline:
            if process and process.poll() is not None:
                raise RuntimeError("Local service exited before becoming healthy")
            try:
                status, _, _, _ = client.request("/health")
                if 200 <= status < 300:
                    ready = True
                    break
            except Exception:
                pass
            if not process:
                break
            time.sleep(.1)
        if not ready:
            raise RuntimeError("Test service is not healthy")
        started = time.perf_counter()
        summary, data_specification = run_evaluation(args, client, run_id, result_dir)
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["configuration"] = {"base_url": args.base_url, "launch_local": args.launch_local,
                                    "forced_local_backend": "hybrid_local_e5" if args.launch_local and args.embedding_provider == "local" else "lexical" if args.launch_local else None,
                                    "embedding_provider": args.embedding_provider if args.launch_local else "external_service_not_inferred",
                                    "declared_local_embedding_environment": embedding_env,
                                    "embedding_model_dir": str(args.embedding_model_dir.resolve()) if args.embedding_model_dir else None,
                                    "embedding_model_manifest_sha256": file_hash(args.embedding_model_dir / "download-manifest.json") if args.embedding_model_dir and (args.embedding_model_dir / "download-manifest.json").exists() else None,
                                    "context_radius": args.context_radius if args.launch_local else None,
                                    "semantic_weight": args.semantic_weight if args.launch_local else None,
                                    "top_k": args.top_k, "character_budget": args.character_budget, "search_concurrency": args.search_concurrency,
                                    "add_concurrency": 1, "max_histories": args.max_histories, "max_queries": args.max_queries,
                                    "source_mapping_version": SOURCE_MAPPING_VERSION,
                                    "chunking": "20 messages / 2000 whitespace-counted words; public proxy approximation",
                                    "timeout_seconds": args.timeout, "python_version": sys.version.split()[0],
                                    "source_code_sha256": {"eval_prepare.py": file_hash(HERE / "eval_prepare.py"), "eval_retrieval.py": file_hash(Path(__file__)),
                                                           "server.py_at_start": server_hash, "server.py_at_end": file_hash(args.server_script) if args.launch_local else None,
                                                           "semantic.py": file_hash(args.server_script.parent / "semantic.py") if args.launch_local and (args.server_script.parent / "semantic.py").exists() else None},
                                    "public_manifest_sha256": file_hash(MANIFEST), "prepared_data": data_specification}
        summary["server_source_changed_during_run"] = bool(args.launch_local and server_hash != file_hash(args.server_script))
        if database:
            summary["local_storage_bytes"] = {p.name: p.stat().st_size for p in database.parent.glob("memory.sqlite3*")}
        (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"result_directory": str(result_dir), "summary": summary}, ensure_ascii=False, indent=2))
    except Exception as exc:
        (result_dir / "error.json").write_text(json.dumps({"run_id": run_id, "error": type(exc).__name__ + ": " + str(exc), "status": "incomplete_public_proxy"}, indent=2) + "\n")
        raise
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log:
            log.close()


if __name__ == "__main__":
    main()
