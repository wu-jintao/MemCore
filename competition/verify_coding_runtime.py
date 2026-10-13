#!/usr/bin/env python3
"""Bounded loopback runtime verification for the MemCore Coding candidate.

This launches the real stdlib HTTP server in a child process with a deterministic
2048-dimensional synthetic backend. It checks 16-way Add/Search concurrency,
concurrent idempotent retries, strict user isolation, deterministic ranking,
SIGKILL restart recovery, WAL-backed persistence, and database integrity. It
never contacts a model, AML, or a non-loopback endpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from urllib import error, request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import coding  # noqa: E402
import server  # noqa: E402

TOKEN = "synthetic-coding-runtime-token"
DIMENSION = 2048
ADD_WORKERS = 16
SEARCH_WORKERS = 16
UNIQUE_ADDS = 64
IDEMPOTENT_RETRIES = 32
SEARCH_CALLS = 64


class DeterministicCodingBackend:
    dimension = DIMENSION
    fingerprint = "synthetic-coding-runtime-v1"

    @staticmethod
    def _vector(text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        indexes = [int.from_bytes(digest[offset:offset + 2], "big") % DIMENSION
                   for offset in range(0, 16, 2)]
        vector = [0.0] * DIMENSION
        for rank, index in enumerate(indexes, 1):
            vector[index] += 1.0 / rank
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)

    def embed_documents(self, texts):
        return [[self._vector(text)] for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def call(base, path, payload=None, timeout=60):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": "Bearer " + TOKEN}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = request.Request(base + path, data=body, headers=headers,
                          method="GET" if body is None else "POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as response:
        return response.code, json.loads(response.read().decode("utf-8"))


def wait_ready(base, process, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Coding runtime exited before health became ready")
        try:
            status, body = call(base, "/health", timeout=2)
            if status == 200 and body == {"status": "ok"}:
                return
        except Exception:
            pass
        time.sleep(.1)
    raise RuntimeError("Coding runtime did not become healthy")


def task_payload(index, user="coding-runtime/user"):
    module = index % 8
    return {
        "request_id": f"runtime-{index}",
        "user_id": user,
        "session_id": f"history-{index}",
        "messages": [{
            "role": "assistant",
            "timestamp": 1700000000000 + index,
            "content": (
                f"Bug fix for src/module_{module}/worker_{index}.py Worker{index}.run TaskError{index}.\n"
                f"Root cause: cache_key_{index} was stale after retry handling.\n"
                f"Patch updated Worker{index}.run and tests/test_worker_{index}.py::test_retry_{index}.\n"
                f"Verified with pytest tests/test_worker_{index}.py -q; all regression tests passed."
            ),
        }],
    }


def serve(args):
    backend = DeterministicCodingBackend()
    store = server.MemoryStore(
        args.database, semantic_backend=backend, context_radius=0,
        semantic_weight=3.0, profile="coding", coding_weight=2.0,
        coding_result_limit=8, coding_character_limit=120000)
    http = server.make_server(("127.0.0.1", args.port), store, TOKEN)
    http.serve_forever()


def launch(database, port):
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--serve",
         "--database", str(database), "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    wait_ready(base, process)
    return process, base


def stop(process, abrupt=False):
    if process.poll() is not None:
        return
    if abrupt:
        process.send_signal(signal.SIGKILL)
    else:
        process.terminate()
    process.wait(timeout=10)


def database_snapshot(path):
    connection = sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        counts = {table: connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                  for table in ("requests", "messages", "coding_postings",
                                "vectors", "code_documents", "code_features")}
        fingerprints = connection.execute(
            "SELECT fingerprint,dimension,COUNT(*) FROM vectors GROUP BY fingerprint,dimension"
        ).fetchall()
        payload_records = connection.execute(
            "SELECT MIN(length(payload_json)),MAX(length(payload_json)) FROM requests"
        ).fetchone()
        return {"quick_check": integrity, "counts": counts,
                "fingerprint_groups": [list(row) for row in fingerprints],
                "request_record_length": list(payload_records)}
    finally:
        connection.close()


def run_verification(database):
    port = free_port()
    process, base = launch(database, port)
    started = time.monotonic()
    try:
        def add(index):
            status, body = call(base, "/add", task_payload(index))
            if status != 200 or body.get("success") is not True:
                raise RuntimeError(f"Add failed at {index}: {status}")
            return body

        add_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
            additions = list(pool.map(add, range(UNIQUE_ADDS)))
        add_seconds = time.monotonic() - add_start
        if len({body["request_id"] for body in additions}) != UNIQUE_ADDS:
            raise RuntimeError("Unique concurrent Add acknowledgements were lost")

        retry_payload = task_payload(10_000)
        retry_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=ADD_WORKERS) as pool:
            retries = list(pool.map(lambda _: call(base, "/add", retry_payload),
                                    range(IDEMPOTENT_RETRIES)))
        retry_seconds = time.monotonic() - retry_start
        if any(status != 200 or body.get("success") is not True for status, body in retries):
            raise RuntimeError("Concurrent idempotent retries did not all acknowledge")

        foreign = task_payload(20_000, user="coding-runtime/other-user")
        status, _ = call(base, "/add", foreign)
        if status != 200:
            raise RuntimeError("Foreign-scope setup failed")

        query = {
            "query": "Repair TaskError23 in Worker23.run from src/module_7/worker_23.py "
                     "and verify tests/test_worker_23.py::test_retry_23",
            "user_id": "coding-runtime/user", "top_k": 100,
        }
        search_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
            searches = list(pool.map(lambda _: call(base, "/search", query), range(SEARCH_CALLS)))
        search_seconds = time.monotonic() - search_start
        if any(status != 200 or not body.get("data") for status, body in searches):
            raise RuntimeError("Concurrent Search failed")
        reference = searches[0][1]
        if any(body != reference for _, body in searches):
            raise RuntimeError("Concurrent Search was not deterministic")
        if "TaskError23" not in reference["data"][0]["content"]:
            raise RuntimeError("Exact engineering experience was not ranked first")
        if any("other-user" in item["content"] or "TaskError20000" in item["content"]
               for item in reference["data"]):
            raise RuntimeError("Search crossed the complete user scope")

        unknown_status, unknown = call(base, "/search", {
            "query": "TaskError20000", "user_id": "coding-runtime/user", "top_k": 100})
        if unknown_status != 200 or any("TaskError20000" in item["content"] for item in unknown["data"]):
            raise RuntimeError("Foreign memory leaked into the primary scope")

        before = database_snapshot(database)
        stop(process, abrupt=True)
        process = None
        restarted, restarted_base = launch(database, port)
        process = restarted
        status, after_response = call(restarted_base, "/search", query)
        if status != 200 or after_response != reference:
            raise RuntimeError("SIGKILL restart changed the persisted Search result")
        after = database_snapshot(database)
        if before != after or after["quick_check"] != "ok":
            raise RuntimeError("Database snapshot changed or failed integrity after restart")
        expected_requests = UNIQUE_ADDS + 1 + 1
        if after["counts"]["requests"] != expected_requests:
            raise RuntimeError("Idempotent retries produced duplicate request rows")
        if after["request_record_length"] != [71, 71]:
            raise RuntimeError("Coding idempotency records are not fixed digests")
        return {
            "schema": "memcore-coding-loopback-runtime-v1",
            "status": "passed", "model_calls": 0, "non_loopback_calls": 0,
            "unique_adds": UNIQUE_ADDS, "add_concurrency": ADD_WORKERS,
            "idempotent_retries": IDEMPOTENT_RETRIES,
            "search_calls": SEARCH_CALLS, "search_concurrency": SEARCH_WORKERS,
            "add_seconds": add_seconds, "retry_seconds": retry_seconds,
            "search_seconds": search_seconds,
            "restart_signal": "SIGKILL", "restart_result_identical": True,
            "database": after, "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        if process is not None:
            stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "competition" / "results" / "CODING_RUNTIME_VERIFICATION.json")
    args = parser.parse_args()
    if args.serve:
        if args.database is None or args.port is None:
            parser.error("--serve requires --database and --port")
        serve(args)
        return
    with tempfile.TemporaryDirectory(prefix="coding-runtime-", dir=ROOT) as directory:
        database = Path(directory) / "memory.sqlite3"
        result = run_verification(database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
