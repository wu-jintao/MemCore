#!/usr/bin/env python3
"""Measured synthetic probe of the participant's authenticated HTTPS API.

Never obtains an AML key, starts an official run, or disables TLS validation.
Reads the service token only from MEMORY_API_TOKEN; reports no credentials.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import gzip
import http.client
import io
import ipaddress
import itertools
import json
import math
import os
from pathlib import Path
import secrets
import socket
import ssl
import statistics
import threading
import time
from urllib.parse import urlsplit


def decode_response(raw, encoding):
    """Bound both wire bytes and decoded bytes for the synthetic probe."""
    limit = 16 * 1024 * 1024
    if len(raw) > limit:
        raise ValueError("Response exceeds the probe byte limit")
    if encoding == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(limit + 1)
    elif encoding not in ("", "identity"):
        raise ValueError("Unsupported response encoding")
    if len(raw) > limit:
        raise ValueError("Decoded response exceeds the probe byte limit")
    return raw


def _read_response(response, row):
    """Count bytes returned before a failure, with the existing wire limit."""
    limit = 16 * 1024 * 1024
    expected_length = response.length
    raw = bytearray()
    while len(raw) <= limit:
        try:
            part = response.read1(min(64 * 1024, limit + 1 - len(raw)))
        except http.client.IncompleteRead as failure:
            row["wire_bytes"] = len(raw) + len(failure.partial)
            raise
        if not part:
            break
        raw.extend(part)
        row["wire_bytes"] = len(raw)
    if len(raw) > limit:
        raise ValueError("Response exceeds the probe byte limit")
    if expected_length is not None and len(raw) != expected_length:
        raise http.client.IncompleteRead(bytes(raw), expected_length - len(raw))
    return bytes(raw)


def probe_request(base, path, payload=None, *, token=None, authenticated=True,
                  expected=200, timeout=1800, response_encoding="gzip", verify=None):
    """One direct HTTPS request; diagnostic rows never contain bodies or secrets."""
    if path not in ("/health", "/add", "/search"):
        raise ValueError("Use a fixed probe API path")
    parsed = urlsplit(base)
    probe_id = "probe-" + secrets.token_hex(12)
    row = {"probe_id": probe_id, "method": "GET" if payload is None else "POST",
           "path": path, "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "ok": False, "bytes": 0, "wire_bytes": 0, "request_bytes": 0,
           "tls_connect_seconds": None, "headers_seconds": None,
           "body_seconds": None, "parse_verify_seconds": None}
    started = time.perf_counter()
    phase = "parse_verify"
    connection = None
    fields = {"connect_tls": "tls_connect_seconds", "upload_headers": "headers_seconds",
              "body": "body_seconds", "parse_verify": "parse_verify_seconds"}

    def timed(name, operation):
        nonlocal phase
        phase = name
        before = time.perf_counter()
        try:
            return operation()
        finally:
            key = fields[name]
            row[key] = (row[key] or 0) + time.perf_counter() - before

    try:
        body = timed("parse_verify", lambda: None if payload is None else json.dumps(payload).encode("utf-8"))
        # request_bytes is the prepared body length, not a claim of bytes sent.
        row["request_bytes"] = len(body) if body is not None else 0
        headers = {"Content-Type": "application/json", "Accept-Encoding": response_encoding,
                   "Connection": "close", "X-Memory-Probe-Id": probe_id}
        if authenticated and payload is not None:
            headers["Authorization"] = "Bearer " + token

        def connect():
            nonlocal connection
            # Direct http.client connections neither use proxies nor follow redirects.
            connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443,
                timeout=timeout, context=ssl.create_default_context())
            connection.connect()

        timed("connect_tls", connect)

        def upload_and_headers():
            connection.request(row["method"], path, body=body, headers=headers)
            return connection.getresponse()

        response = timed("upload_headers", upload_and_headers)
        row["status"] = response.status
        encoding = response.getheader("Content-Encoding", "identity").lower()
        # Do not serialize arbitrary response header values into diagnostic rows.
        row["content_encoding"] = encoding if encoding in ("", "identity", "gzip") else "unsupported"
        raw = timed("body", lambda: _read_response(response, row))

        def parse_and_verify():
            decoded = decode_response(raw, row["content_encoding"])
            row["bytes"] = len(decoded)
            value = json.loads(decoded)
            if row["status"] != expected:
                row.update(error="HTTPStatusError", failure_phase="parse_verify")
            elif verify is not None and not verify(value):
                row.update(error="ResponseVerificationError", failure_phase="parse_verify")
            else:
                row["ok"] = True
            return value

        value = timed("parse_verify", parse_and_verify)
        return value, row
    except Exception as failure:
        error = type(failure).__name__
        if "status" in row and row["status"] != expected:
            row.update(error="HTTPStatusError", phase_error=error)
        else:
            row["error"] = error
        row["failure_phase"] = phase
        return None, row
    finally:
        try:
            if connection is not None:
                connection.close()
        except Exception as failure:
            row["close_error"] = type(failure).__name__
        finally:
            # Includes serialization, all entered phases, verification and cleanup.
            row["seconds"] = time.perf_counter() - started


def summary(rows):
    seconds = sorted(row["seconds"] for row in rows)
    return {"requests": len(rows), "failed": sum(not row["ok"] for row in rows),
            "p50_seconds": statistics.median(seconds) if seconds else None,
            "p95_seconds": seconds[max(0, math.ceil(.95 * len(seconds)) - 1)] if seconds else None,
            "max_response_bytes": max((row["bytes"] for row in rows), default=0),
            "max_wire_response_bytes": max((row.get("wire_bytes", row["bytes"]) for row in rows), default=0),
            "response_encodings": sorted({row.get("content_encoding", "unknown") for row in rows}),
            "errors": [row.get("error", "Unexpected status/response") for row in rows if not row["ok"]][:5]}


def run(args):
    parsed = urlsplit(args.base_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
            parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise ValueError("Use a bare HTTPS origin, without credentials, query or extra path")
    addresses = {row[4][0] for row in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("The target must resolve to public addresses")
    token = os.environ.get("MEMORY_API_TOKEN")
    if not token:
        raise ValueError("Set MEMORY_API_TOKEN; do not place it in a command argument or URL")
    base = args.base_url.rstrip("/")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(6)
    user_id = "synthetic-public-probe/" + run_id + "/user"
    request_records = []
    request_indices = itertools.count()
    records_lock = threading.Lock()

    def call(path, payload=None, authenticated=True, expected=200, *, kind, verify=None):
        with records_lock:
            index = next(request_indices)
        value, row = probe_request(base, path, payload, token=token,
            authenticated=authenticated, expected=expected, timeout=args.timeout,
            response_encoding=args.response_encoding, verify=verify)
        row.update(request_index=index, kind=kind)
        with records_lock:
            request_records.append(row)
        return value, row

    def payload(index):
        marker = "probe" + hashlib.sha256((run_id + str(index)).encode()).hexdigest()[:24]
        prose = "I preserve this original record for a synthetic network capacity check. " * 8
        messages = [{"role": "user" if n % 2 == 0 else "assistant",
                     "content": prose + "Archive identifier " + marker + " sequence " + str(n) + ".",
                     "timestamp": 1700000000000 + index * 20000 + n * 1000}
                    for n in range(20)]
        return {"request_id": "chunk-" + str(index), "user_id": user_id,
                "session_id": "session-" + str(index), "messages": messages}, marker

    def acknowledged(response, data):
        return isinstance(response, dict) and response.get("success") is True and all(
            response.get(key) == data[key] for key in ("request_id", "user_id", "session_id"))

    def valid_evidence(response, top_k):
        return isinstance(response, dict) and isinstance(response.get("data"), list) and 0 < len(
            response["data"]) <= top_k and all(isinstance(item, dict) and isinstance(item.get("id"), str)
            and bool(item["id"].strip()) and isinstance(item.get("content"), str)
            and bool(item["content"].strip()) for item in response["data"])

    report = {"kind": "synthetic-public-https-probe", "official_evaluation": False,
              "run_id": run_id, "base_url": base, "resolved_public_addresses": sorted(addresses),
              "configuration": {"concurrency": args.concurrency, "messages_per_add": 20,
                                "synthetic_messages": args.concurrency * 20, "search_top_k": 100,
                                "request_timeout_seconds": args.timeout,
                                "accepted_response_encoding": args.response_encoding},
              "public_health": False, "unauthenticated_post_rejected": False,
              **{name: summary([]) for name in ("add", "immediate_visibility", "search", "identical_retry")},
              "conflicting_retry_rejected": False, "complete_user_isolation": False,
              "diagnostics": {"connect_tls": "DNS/TCP/TLS combined",
                              "upload_headers": "request header/body send and response-header wait",
                              "parse_verify": "request serialization and response decode/parse/verification",
                              "request_bytes": "prepared body length; not confirmed uploaded bytes",
                              "wire_bytes": "response bytes returned before success or failure",
                              "timeout": "blocking socket operation timeout; not a total request deadline"},
              "limits": ["Independent generated histories only, not official questions or score",
                         "Small fresh corpus; supplement with actual-host long-history load",
                         "Backend/version and restart must be verified separately on the host",
                         "Dedicated synthetic user scope remains in the service database"]}

    def save_report():
        report["request_records"] = sorted(request_records, key=lambda row: row["request_index"])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    _, healthy = call("/health", authenticated=False, kind="public_health",
                      verify=lambda value: value == {"status": "ok"})
    _, unauthenticated = call("/search", {"user_id": user_id, "query": "probe", "top_k": 1},
                              authenticated=False, expected=401, kind="unauthenticated_post")
    report.update(public_health=healthy["ok"], unauthenticated_post_rejected=unauthenticated["ok"])
    if not healthy["ok"] or not unauthenticated["ok"]:
        report["aborted_phase"] = "health_authentication"
        save_report()
        raise RuntimeError("Public health/authentication gate failed; no synthetic Add was sent")
    initial = []

    def add_and_check(index):
        data, marker = payload(index)
        _, row = call("/add", data, kind="add", verify=lambda value: acknowledged(value, data))
        _, visible = call("/search", {"user_id": user_id, "query": marker, "top_k": 5},
            kind="immediate_visibility", verify=lambda value: valid_evidence(value, 5) and any(
                marker in item["content"] for item in value["data"]))
        return row, visible

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        initial = list(pool.map(add_and_check, range(args.concurrency)))
    report.update(add=summary([row[0] for row in initial]),
                  immediate_visibility=summary([row[1] for row in initial]))
    if any(not add["ok"] or not visible["ok"] for add, visible in initial):
        report["aborted_phase"] = "add_visibility"
        save_report()
        raise RuntimeError("Concurrent Add/immediate visibility failed; inspect service health privately")

    def search(index):
        _, row = call("/search", {"user_id": user_id,
                                  "query": "synthetic network capacity archive original record", "top_k": 100},
                      kind="search", verify=lambda value: valid_evidence(value, 100))
        return row

    def repeat(index):
        data, _ = payload(0)
        _, row = call("/add", data, kind="identical_retry", verify=lambda value: acknowledged(value, data))
        return row
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        searches = list(pool.map(search, range(args.concurrency * 2)))
        repeats = list(pool.map(repeat, range(args.concurrency * 2)))
    conflict, _ = payload(0)
    conflict["messages"][0]["content"] = "A changed body must conflict, without replacing the original."
    _, conflicting = call("/add", conflict, expected=409, kind="conflicting_retry")
    _, isolation = call("/search", {"user_id": user_id.replace("/user", "/another-user"),
                                    "query": "archive original record", "top_k": 100},
                        kind="user_isolation", verify=lambda value: value == {"data": []})
    report.update(search=summary(searches), identical_retry=summary(repeats),
                  conflicting_retry_rejected=conflicting["ok"], complete_user_isolation=isolation["ok"])
    save_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all((report["public_health"], report["unauthenticated_post_rejected"],
                     report["conflicting_retry_rejected"], report["complete_user_isolation"])) and all(
        not report[name]["failed"] for name in ("add", "immediate_visibility", "search", "identical_retry")) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--response-encoding", choices=("identity", "gzip"), default="gzip")
    parser.add_argument("--report", type=Path, default=Path(".local/public-https-probe.json"))
    args = parser.parse_args()
    if not 16 <= args.concurrency <= 64 or not 0 < args.timeout <= 1800:
        parser.error("Concurrency must be 16..64; timeout must be positive and at most 1800 seconds")
    try:
        return run(args)
    except (ValueError, RuntimeError) as failure:
        parser.exit(2, str(failure) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
