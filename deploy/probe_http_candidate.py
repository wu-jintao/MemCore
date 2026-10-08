#!/usr/bin/env python3
"""Bounded synthetic probe for an operator's own loopback HTTP candidate.

The default only writes a private plan. --execute sends at most 30 requests.
No official evaluation endpoints, external hosts, gold answers, environment
credentials, shell commands, proxies, redirects, or model services are used by
this client. Run on the candidate host; the operator supplies its private env
file, which is read without executing it and only for MEMORY_API_TOKEN.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import statistics
import tempfile
import threading
import time
from urllib.parse import urlsplit

try:
    from deploy.verify_public_api import decode_response
except ModuleNotFoundError:  # Running this file directly on the candidate host.
    from verify_public_api import decode_response


MAX_REQUESTS = 30
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
PRIMARY_MARKER = "synthetic-primary-evidence-marker"
OTHER_MARKER = "synthetic-other-evidence-marker"
CONFLICT_MARKER = "synthetic-conflicting-body-marker"


class DeadlineExceeded(Exception):
    pass


class RequestTimeout(Exception):
    pass


def validate_origin(value):
    """Accept only bare, literal loopback origins; never resolve a DNS name."""
    if any(char.isspace() or ord(char) < 32 for char in value):
        raise ValueError("Use a bare loopback HTTP origin")
    try:
        parsed = urlsplit(value)
        port = 80 if parsed.port is None else parsed.port
    except ValueError:
        raise ValueError("Use a valid loopback HTTP origin") from None
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or "?" in value or "#" in value
            or not 1 <= port <= 65535):
        raise ValueError("Only bare HTTP localhost, 127.0.0.1, or [::1] origins are allowed")
    # localhost is pinned to the IPv4 loopback address, avoiding DNS changes.
    host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
    return value.rstrip("/"), host, port


def read_private_token(path):
    """Read one literal assignment only. Do not source the file or read os.environ."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("The token env file must be a readable private regular file") from None
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o077 or info.st_size > 65536):
            raise ValueError("The token env file must be owner-only, owned by the current operator, and small")
        try:
            source = stream.read(65537).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("The token env file must contain UTF-8 text") from None
    values = []
    for line in source.splitlines():
        match = re.match(r"^\s*(?:export\s+)?MEMORY_API_TOKEN\s*=\s*(.*?)\s*$", line)
        if match:
            value = match.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values.append(value)
    if (len(values) != 1 or not values[0] or len(values[0]) > 4096
            or any(ord(char) < 32 or ord(char) > 126 for char in values[0])):
        raise ValueError("Provide exactly one nonempty literal MEMORY_API_TOKEN assignment")
    return values[0]


def save_private_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.stat()
    if parent.st_uid != os.geteuid() or parent.st_mode & 0o077:
        raise ValueError("The report directory must be owner-only and owned by the current operator")
    if path.is_symlink():
        raise ValueError("The report path must not be a symlink")
    if path.exists():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("An existing report must be an owner-only regular file")
    descriptor, temporary = tempfile.mkstemp(prefix=".candidate-probe-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_plan(run_id):
    primary = "synthetic-http-probe/" + run_id + "/CaseSensitiveUser"
    other = "synthetic-http-probe/" + run_id + "/OtherUser"
    messages = [{"role": "user", "timestamp": 1704067200000 + index * 60000,
                 "content": (f"{PRIMARY_MARKER}. Synthetic archive item {index:02d}: "
                             "Maya prefers warm oat milk coffee. Maya's synthetic meeting "
                             "is at the library on 2024-01-08. Keep this source evidence.")}
                for index in range(32)]
    initial = {"request_id": "synthetic-chunk-0", "user_id": primary,
               "session_id": "synthetic-session-0", "messages": messages}
    conflict = {**initial, "messages": [dict(item) for item in messages]}
    conflict["messages"][0]["content"] = CONFLICT_MARKER + ". The changed request must be rejected."
    second = {"request_id": initial["request_id"], "user_id": other,
              "session_id": "synthetic-session-0", "messages": [
                  {"role": "user", "timestamp": 1704067200000,
                   "content": OTHER_MARKER + ". Maya prefers cold black coffee in this separate synthetic user."}]}

    def search(user):
        return {"user_id": user, "query": "Maya coffee meeting synthetic archive evidence", "top_k": 100}

    plan = [("health", "/health", None, 200, "health", False),
            ("unauthenticated_search", "/search", search(primary), 401, "status", False),
            ("initial_add", "/add", initial, 200, "ack", True),
            ("immediate_visibility", "/search", search(primary), 200, "primary", True),
            ("identical_add_retry", "/add", initial, 200, "ack", True),
            ("conflicting_add_retry", "/add", conflict, 409, "status", True),
            ("post_retry_integrity", "/search", search(primary), 200, "integrity", True),
            ("separate_user_add", "/add", second, 200, "ack", True),
            ("separate_user_search", "/search", search(other), 200, "other", True)]
    plan.extend((f"concurrent_search_{index:02d}", "/search", search(primary), 200, "primary", True)
                for index in range(16))
    variants = ["synthetic-http-probe/" + run_id + "/UnwrittenUser",
                "synthetic-http-probe/" + run_id,
                primary[:-1], primary.lower(), primary + " "]
    plan.extend((f"complete_user_isolation_{index}", "/search", search(user), 200, "empty", True)
                for index, user in enumerate(variants))
    assert len(plan) == MAX_REQUESTS
    return [{"request_index": index, "kind": kind, "path": path, "payload": payload,
             "expected_status": expected, "verification": verify, "authenticated": authenticated}
            for index, (kind, path, payload, expected, verify, authenticated) in enumerate(plan)]


def request_row(spec):
    return {"request_index": spec["request_index"], "kind": spec["kind"], "path": spec["path"],
            "method": "GET" if spec["payload"] is None else "POST",
            "expected_status": spec["expected_status"], "state": "planned", "attempted": False,
            "ok": False, "status": None, "seconds": None, "response_bytes": 0,
            "wire_response_bytes": 0, "prepared_request_bytes": 0}


def valid_evidence(value, marker):
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        return False
    items = value["data"]
    return 0 < len(items) <= 100 and len({item.get("id") for item in items if isinstance(item, dict)
                                        and isinstance(item.get("id"), str)}) == len(items) and all(
        isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
        and isinstance(item.get("content"), str) and marker in item["content"]
        and CONFLICT_MARKER not in item["content"]
        and (OTHER_MARKER if marker == PRIMARY_MARKER else PRIMARY_MARKER) not in item["content"]
        for item in items)


def verify_response(spec, value, baseline):
    kind = spec["verification"]
    if kind == "status":
        return True
    if kind == "health":
        return isinstance(value, dict) and value.get("status") == "ok"
    if kind == "ack":
        return isinstance(value, dict) and value.get("success") is True and all(
            value.get(key) == spec["payload"][key] for key in ("request_id", "user_id", "session_id"))
    if kind == "empty":
        return isinstance(value, dict) and value.get("data") == []
    if kind == "other":
        return valid_evidence(value, OTHER_MARKER)
    if not valid_evidence(value, PRIMARY_MARKER):
        return False
    if kind == "integrity":
        return {item["id"]: item["content"] for item in value["data"]} == baseline
    return True


def probe_request(host, port, spec, row, *, token, timeout, deadline, baseline):
    """One direct request, with both an operation timeout and a total deadline."""
    started = time.monotonic()
    request_end = min(deadline, started + timeout)
    connection = None
    connected_socket = None
    expired = threading.Event()
    phase = "prepare"
    row.update(attempted=True, state="running", started_at_utc=datetime.now(timezone.utc).isoformat())

    def stop_socket():
        expired.set()
        if connected_socket is not None:
            try:
                connected_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def remaining():
        now = time.monotonic()
        if now >= deadline:
            raise DeadlineExceeded()
        if expired.is_set() or now >= request_end:
            raise RequestTimeout()
        amount = request_end - now
        # HTTP/1.0 may close the socket after delivering the full body. Keep
        # deadline checks active, but update timeouts only on an open socket.
        if connected_socket is not None and connected_socket.fileno() >= 0:
            connected_socket.settimeout(amount)
        return amount

    timer = threading.Timer(max(0, request_end - started), stop_socket)
    timer.daemon = True
    timer.start()
    value = None
    try:
        body = None if spec["payload"] is None else json.dumps(spec["payload"]).encode("utf-8")
        row["prepared_request_bytes"] = 0 if body is None else len(body)
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity", "Connection": "close"}
        if spec["authenticated"]:
            headers["Authorization"] = "Bearer " + token
        phase = "connect"
        # HTTPConnection bypasses proxy environment variables and follows no redirects.
        connection = http.client.HTTPConnection(host, port, timeout=remaining())
        connection.connect()
        connected_socket = connection.sock
        phase = "upload_headers"
        remaining()
        connection.request(row["method"], spec["path"], body=body, headers=headers)
        response = connection.getresponse()
        row["status"] = response.status
        phase = "body"
        raw = bytearray()
        expected_length = response.length
        while len(raw) <= MAX_RESPONSE_BYTES:
            remaining()
            try:
                part = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
            except http.client.IncompleteRead as failure:
                row["wire_response_bytes"] = len(raw) + len(failure.partial)
                raise
            if not part:
                break
            raw.extend(part)
            row["wire_response_bytes"] = len(raw)
        remaining()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeds the probe byte limit")
        if expected_length is not None and len(raw) != expected_length:
            raise http.client.IncompleteRead(bytes(raw), expected_length - len(raw))
        phase = "parse_verify"
        encoding = response.getheader("Content-Encoding", "identity").lower()
        decoded = decode_response(bytes(raw), encoding)
        row["response_bytes"] = len(decoded)
        if row["status"] != spec["expected_status"]:
            row["error"] = "HTTPStatusError"
        elif spec["verification"] == "status":
            row["ok"] = True
        else:
            value = json.loads(decoded)
            row["ok"] = bool(verify_response(spec, value, baseline))
            if not row["ok"]:
                row["error"] = "ResponseVerificationError"
        remaining()
        row["state"] = "completed"
    except Exception as failure:
        if time.monotonic() >= deadline:
            row.update(state="unfinished", error="DeadlineExceeded")
        elif expired.is_set() or time.monotonic() >= request_end or isinstance(failure, (TimeoutError, RequestTimeout)):
            row.update(state="completed", error="RequestTimeout")
        else:
            row.update(state="completed", error=type(failure).__name__)
        row["ok"] = False
        row["failure_phase"] = phase
    finally:
        timer.cancel()
        if connection is not None:
            connection.close()
        row["seconds"] = time.monotonic() - started
    return value


def summarize(rows):
    completed = [row for row in rows if row["state"] == "completed"]
    timings = sorted(row["seconds"] for row in rows if row["seconds"] is not None)
    return {"planned": len(rows), "attempted": sum(row["attempted"] for row in rows),
            "completed": len(completed), "succeeded": sum(row["ok"] for row in completed),
            "failed": sum(not row["ok"] for row in completed), "unfinished": len(rows) - len(completed),
            "p50_seconds": statistics.median(timings) if timings else None,
            "p95_seconds": timings[max(0, math.ceil(.95 * len(timings)) - 1)] if timings else None,
            "max_response_bytes": max((row["response_bytes"] for row in rows), default=0),
            "max_wire_response_bytes": max((row["wire_response_bytes"] for row in rows), default=0)}


def run(args):
    base, host, port = validate_origin(args.base_url)
    if not 0 < args.timeout <= 120 or not 0 < args.deadline_seconds <= 300:
        raise ValueError("Timeout must be 0..120 seconds and wall deadline 0..300 seconds, both exclusive of zero")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(6)
    plan = make_plan(run_id)
    rows = [request_row(spec) for spec in plan]
    report = {"kind": "synthetic-loopback-http-candidate-probe", "run_id": run_id,
              "base_url": base, "connection_host": host, "official_evaluation": False,
              "execution_requested": bool(args.execute), "full_capacity_validated": False,
              "status": "planned", "configuration": {"max_http_requests": MAX_REQUESTS,
                  "initial_add_messages": 32, "separate_user_add_messages": 1,
                  "concurrent_searches": 16, "search_top_k": 100,
                  "request_total_timeout_seconds": args.timeout,
                  "wall_deadline_seconds": args.deadline_seconds},
              "limitations": ["Small synthetic functional/concurrency probe, not Full-scale capacity certification",
                  "No official evaluation, gold answer, real user data, or external network target",
                  "Backend/model/version and long-history resource capacity require separate host evidence",
                  "Fresh dedicated synthetic user scopes remain in the candidate database"],
              "request_records": rows, "summary": summarize(rows)}
    # Validate output privacy and retain the complete plan before any token read or HTTP.
    save_private_report(args.output, report)
    if not args.execute:
        return report, 0
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    report.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
    baseline = {}

    def call(index, barrier=None):
        row = rows[index]
        if time.monotonic() >= deadline:
            row.update(state="unfinished", error="DeadlineExceeded")
            return None
        if barrier is not None:
            try:
                barrier.wait(timeout=max(0, deadline - time.monotonic()))
            except threading.BrokenBarrierError:
                row.update(state="unfinished", error="DeadlineExceeded")
                return None
        return probe_request(host, port, plan[index], row, token=token, timeout=args.timeout,
                             deadline=deadline, baseline=baseline)

    try:
        token = read_private_token(args.token_env_file)
        for index in range(9):
            value = call(index)
            if index == 3 and rows[index]["ok"]:
                baseline.update({item["id"]: item["content"] for item in value["data"]})
            if not rows[index]["ok"]:
                report["stopped_after"] = plan[index]["kind"]
                break
        else:
            barrier = threading.Barrier(16)
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(call, index, barrier) for index in range(9, 25)]
                for future in futures:
                    future.result()
            for index in range(25, MAX_REQUESTS):
                call(index)
    except Exception as failure:
        # Never include an exception message: it might contain a credential or response body.
        report["execution_error"] = type(failure).__name__
    finally:
        for row in rows:
            if row["state"] in ("planned", "running"):
                row.update(state="unfinished", error="DeadlineExceeded" if time.monotonic() >= deadline
                           else "NotExecutedAfterGateFailure")
        report["summary"] = summarize(rows)
        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "passed_small_synthetic_probe" if report["summary"]["succeeded"] == MAX_REQUESTS else "incomplete_or_failed"
        save_private_report(args.output, report)
    return report, 0 if report["summary"]["succeeded"] == MAX_REQUESTS else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token-env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--deadline-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        report, code = run(args)
    except (ValueError, OSError) as failure:
        parser.exit(2, (str(failure) if isinstance(failure, ValueError) else "Private report write failed") + "\n")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "full_capacity_validated": False}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
