#!/usr/bin/env python3
"""Synthetic long-history HTTP load/recovery test, never an official AML score.

Runs a separate loopback server process. The generated history deliberately has
high-frequency words, unique facts, neighboring dialogue, and twenty-message
Add requests. Client requests are measured end to end, including JSON and TCP.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from urllib import error, request


ROOT = Path(__file__).resolve().parent
TOKEN = "local-synthetic-load-test-only"
USER = "synthetic-load/run-A/long-history-user"


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def metrics(results, elapsed):
    latencies = [row["seconds"] for row in results]
    return {"requests": len(results), "failed_requests": sum(not row["ok"] for row in results),
            "elapsed_seconds": round(elapsed, 4),
            "requests_per_second": round(len(results) / max(elapsed, 1e-9), 3),
            "p50_seconds": round(statistics.median(latencies), 4) if latencies else None,
            "p95_seconds": round(percentile(latencies, .95), 4) if latencies else None,
            "p99_seconds": round(percentile(latencies, .99), 4) if latencies else None,
            "max_response_bytes": max((row["bytes"] for row in results), default=0),
            "errors": [row.get("error") for row in results if not row["ok"]][:5]}


def call(base, path, payload=None):
    started = time.monotonic()
    try:
        req = request.Request(base + path,
                              data=json.dumps(payload).encode() if payload is not None else None,
                              headers={"Content-Type": "application/json",
                                       "Authorization": "Bearer " + TOKEN})
        with request.urlopen(req, timeout=120) as response:
            raw = response.read()
            value = json.loads(raw)
            return value, {"ok": response.status == 200, "seconds": time.monotonic() - started,
                           "bytes": len(raw)}
    except Exception as failure:
        return None, {"ok": False, "seconds": time.monotonic() - started, "bytes": 0,
                      "error": type(failure).__name__ + ": " + str(failure)}


def payload(start, count=20):
    messages = []
    for index in range(start, start + count):
        family = index % 31
        content = ("Every day I record our common neighborhood activity. "
                   "We discuss the weather, travel, meals, hobbies, and future plans. "
                   "My enduring interest is topicfamily%d. " % family)
        # Repeated common prose is an intentional large-posting-list workload.
        content += ("I kept the original detail because our plans can change over time. " * 5)
        content += "The unique archive marker is needlefact%08d." % index
        messages.append({"role": "user" if index % 2 == 0 else "assistant",
                         "content": content, "timestamp": 1700000000000 + index * 1000})
    return {"user_id": USER, "session_id": "session-" + str(start // 200),
            "request_id": "chunk-" + str(start), "messages": messages}


def start_server(db, log, args, metadata_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = dict(os.environ, MEMORY_API_TOKEN=TOKEN)
    if args.backend == "lexical":
        environment["MEMORY_EMBEDDING_PROVIDER"] = "disabled"
    # Record the object actually built by the normal CLI in the child process.
    # This wrapper never embeds inputs, downloads weights, or prints credentials.
    bootstrap = """
import json
from pathlib import Path
import sys
import semantic
import server
metadata_path = Path(sys.argv.pop(1))
original_build_backend = semantic.build_backend
def recording_build_backend(config=None):
    config = semantic.EmbeddingConfig.from_env() if config is None else config
    backend = original_build_backend(config)
    metadata = {
        'provider': config.provider,
        'model': config.model if backend is not None else None,
        'dimension': backend.dimension if backend is not None else None,
        'fingerprint': backend.fingerprint if backend is not None else None,
        'batch_size': config.batch_size if backend is not None else None,
        'embedding_concurrency': config.concurrency if backend is not None else None,
        'inference_threads': config.threads if backend is not None else None,
        'segment_tokens': config.segment_tokens if backend is not None else None,
        'overlap_tokens': config.overlap_tokens if backend is not None else None,
        'max_document_segments': config.max_document_segments if backend is not None else None,
        'max_query_segments': config.max_query_segments if backend is not None else None,
        'max_total_segments': config.max_total_segments if backend is not None else None,
        'queue_timeout_seconds': config.acquire_timeout if backend is not None else None,
        'operation_timeout_seconds': config.operation_timeout if backend is not None else None,
    }
    metadata_path.write_text(json.dumps(metadata))
    return backend
semantic.build_backend = recording_build_backend
server.main()
"""
    process = subprocess.Popen([sys.executable, "-c", bootstrap, str(metadata_path),
                                "--port", str(port), "--db", str(db)], cwd=ROOT,
                               env=environment, stdout=log, stderr=log)
    base = "http://127.0.0.1:" + str(port)
    # Loading already-downloaded local weights may take more than five seconds.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Synthetic server failed to start; inspect its local log")
        response, result = call(base, "/health")
        if result["ok"]:
            return process, base, json.loads(metadata_path.read_text())
        time.sleep(.05)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError("Synthetic server did not become healthy")


def stop_server(process):
    process.terminate()
    process.wait(timeout=10)


def run(args):
    report = {"kind": "synthetic-long-history-http-load", "official_evaluation": False,
              "host": {"system": platform.system(), "machine": platform.machine(),
                       "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                       "label": args.host_label, "label_is_user_supplied": True},
              "configuration": {"initial_messages": args.messages, "messages_per_add": 20,
                                "client_concurrency": args.concurrency, "search_rounds": args.rounds,
                                "requested_backend_mode": args.backend, "top_k": 100},
              "limits": ["Synthetic data, one main user, no official questions or answer score",
                         "Loopback transport; no 3 Mbps public-network simulation",
                         "Host label is supplied by the operator, not proof of cloud identity",
                         "Matching-posting SQL work remains proportional to query frequency",
                         "Semantic ranking scans this user's vectors; at most two scoring operations run concurrently",
                         "Configured mode permits offline local embeddings or disabled provider only",
                         "No automatic model download or paid HTTP embedding requests"]}
    temporary_root = ROOT / ".local"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="load-", dir=temporary_root) as directory:
        db = Path(directory) / "memory.sqlite3"
        metadata_path = Path(directory) / "backend.json"
        with (Path(directory) / "server.log").open("w") as log:
            process, base, backend_metadata = start_server(db, log, args, metadata_path)
            report["configuration"]["actual_backend"] = backend_metadata
            try:
                def ingest(start):
                    value, result = call(base, "/add", payload(start, min(20, args.messages - start)))
                    if result["ok"]:
                        found, visible = call(base, "/search", {
                            "user_id": USER, "query": "needlefact%08d" % start, "top_k": 1})
                        result["immediately_visible"] = bool(visible["ok"] and found["data"]
                            and "needlefact%08d" % start in found["data"][0]["content"])
                    return result
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    added = list(pool.map(ingest, range(0, args.messages, 20)))
                report["add"] = metrics(added, time.monotonic() - started)
                report["add"]["immediate_visibility_failures"] = sum(
                    not row.get("immediately_visible", False) for row in added)

                def searching(index):
                    query = ("common neighborhood activity" if index % 3 == 0 else
                             "topicfamily%d future plans" % (index % 31) if index % 3 == 1 else
                             "needlefact%08d" % ((index * 997) % args.messages))
                    value, result = call(base, "/search", {
                        "user_id": USER, "query": query, "top_k": 100})
                    if result["ok"] and not value.get("data"):
                        result.update(ok=False, error="Expected evidence but received none")
                    return result
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    found = list(pool.map(searching, range(args.rounds)))
                report["search"] = metrics(found, time.monotonic() - started)

                def mixed(index):
                    if index % 4 == 0:
                        return call(base, "/add", payload(args.messages + index * 20))[1]
                    return searching(index)
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    mixed_results = list(pool.map(mixed, range(args.rounds)))
                report["mixed_add_search"] = metrics(mixed_results, time.monotonic() - started)
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    retries = list(pool.map(lambda _: call(base, "/add", payload(0))[1],
                                            range(args.concurrency * 2)))
                report["duplicate_retry"] = metrics(retries, time.monotonic() - started)
                isolated, result = call(base, "/search", {
                    "user_id": USER.replace("run-A", "run-B"), "query": "common", "top_k": 100})
                report["complete_user_scope_isolated"] = bool(result["ok"] and isolated == {"data": []})
            finally:
                stop_server(process)
            report["server_child_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                                                        * (1 if platform.system() == "Darwin" else 1024))
            # Include a second process to validate the same on-disk database.
            process, base, restarted_metadata = start_server(db, log, args, metadata_path)
            try:
                report["restart_backend_unchanged"] = restarted_metadata == backend_metadata
                restarted, result = call(base, "/search", {
                    "user_id": USER, "query": "needlefact00000000", "top_k": 1})
                report["restart_evidence_preserved"] = bool(result["ok"] and restarted["data"]
                    and "needlefact00000000" in restarted["data"][0]["content"])
            finally:
                stop_server(process)
            with sqlite3.connect(db) as connection:
                report["stored_messages"] = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                report["stored_postings"] = connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            report["database_bytes"] = sum(path.stat().st_size for path in Path(directory).glob("memory.sqlite3*"))
    report["passed"] = bool(all(report[stage]["failed_requests"] == 0
                                for stage in ("add", "search", "mixed_add_search", "duplicate_retry"))
                            and report["add"]["immediate_visibility_failures"] == 0
                            and report["complete_user_scope_isolated"]
                            and report["restart_backend_unchanged"]
                            and report["restart_evidence_preserved"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=96)
    parser.add_argument("--backend", choices=("lexical", "configured"), default="lexical",
                        help="lexical overrides provider to disabled; configured preserves offline embedding settings")
    parser.add_argument("--host-label", default="unspecified execution host",
                        help="Operator-supplied environment label; actual system/CPU details are also recorded")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.messages < 20 or args.messages % 20 or not 1 <= args.concurrency <= 64 or args.rounds < 1:
        parser.error("messages must be a positive multiple of 20; concurrency 1..64; rounds positive")
    if args.backend == "configured":
        from semantic import EmbeddingConfig, EmbeddingConfigError
        try:
            configuration = EmbeddingConfig.from_env()
        except EmbeddingConfigError as failure:
            parser.error(str(failure))
        if configuration.provider == "http":
            parser.error("This offline load test refuses paid/external HTTP embeddings; use local or disabled")
    report = run(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
