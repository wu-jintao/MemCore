"""Synthetic diagnostics tests: no public requests, cloud access or model calls."""

from collections import deque
from contextlib import redirect_stdout
import gzip
import http.client
import io
import itertools
import json
import os
from pathlib import Path
import ssl
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy import verify_public_api as probe


SECRET = "synthetic-credential-never-report"
PRIVATE_BODY = "synthetic-original-body-never-report"
BASE = "https://probe.example.invalid"


class FakeResponse:
    def __init__(self, raw=b'{"status":"ok"}', *, status=200, encoding="identity",
                 parts=None, expected_length=None):
        self.status = status
        self.encoding = encoding
        self.length = len(raw) if expected_length is None else expected_length
        self.parts = deque([raw] if parts is None else parts)

    def getheader(self, name, default=None):
        return self.encoding if name == "Content-Encoding" else default

    def read1(self, amount):
        if not self.parts:
            return b""
        part = self.parts.popleft()
        if isinstance(part, Exception):
            raise part
        if len(part) > amount:
            self.parts.appendleft(part[amount:])
        return part[:amount]


class FakeConnection:
    def __init__(self, response=None, *, errors=None, responder=None):
        self.response = response or FakeResponse()
        self.errors = errors or {}
        self.responder = responder
        self.sent = None
        self.closed = False

    def connect(self):
        if "connect" in self.errors:
            raise self.errors["connect"]

    def request(self, method, path, *, body, headers):
        self.sent = (method, path, body, headers)
        if "request" in self.errors:
            raise self.errors["request"]

    def getresponse(self):
        if "headers" in self.errors:
            raise self.errors["headers"]
        return self.responder(self.sent) if self.responder else self.response

    def close(self):
        self.closed = True
        if "close" in self.errors:
            raise self.errors["close"]


class ProbeRequestTests(unittest.TestCase):
    def invoke(self, connection, **options):
        context = ssl.create_default_context()
        with mock.patch.object(probe.ssl, "create_default_context", return_value=context), \
                mock.patch.object(probe.http.client, "HTTPSConnection", return_value=connection) as factory, \
                mock.patch.object(probe.time, "perf_counter", side_effect=itertools.count()):
            value, row = probe.probe_request(BASE, "/search", {"content": PRIVATE_BODY},
                token=SECRET, timeout=120, **options)
        factory.assert_called_once_with("probe.example.invalid", 443, timeout=120, context=context)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(connection.closed)
        serialized = json.dumps(row)
        for forbidden in (SECRET, PRIVATE_BODY, "Authorization"):
            self.assertNotIn(forbidden, serialized)
        self.assertGreaterEqual(row["seconds"], sum(row[key] or 0 for key in
            ("tls_connect_seconds", "headers_seconds", "body_seconds", "parse_verify_seconds")))
        return value, row

    def test_success_safe_tag_bytes_full_timing_and_tls(self):
        conn = FakeConnection()
        value, row = self.invoke(conn, verify=lambda value: value == {"status": "ok"})
        self.assertEqual(value, {"status": "ok"})
        self.assertTrue(row["ok"])
        self.assertNotIn("failure_phase", row)
        method, path, body, headers = conn.sent
        self.assertEqual((method, path), ("POST", "/search"))
        self.assertEqual(row["request_bytes"], len(body))
        self.assertEqual(row["wire_bytes"], len(b'{"status":"ok"}'))
        self.assertEqual(row["bytes"], row["wire_bytes"])
        self.assertEqual(headers["Authorization"], "Bearer " + SECRET)
        self.assertEqual(headers["Connection"], "close")
        self.assertEqual(headers["X-Memory-Probe-Id"], row["probe_id"])
        self.assertRegex(row["probe_id"], r"^probe-[0-9a-f]{24}$")
        self.assertIsNotNone(row["started_at_utc"])

    def test_connect_tls_upload_and_header_failures(self):
        for where, error, phase in (
            ("connect", TimeoutError(SECRET), "connect_tls"),
            ("connect", ssl.SSLCertVerificationError(SECRET), "connect_tls"),
            ("request", TimeoutError(PRIVATE_BODY), "upload_headers"),
            ("headers", TimeoutError(SECRET), "upload_headers")):
            with self.subTest(where=where, error=type(error).__name__):
                _, row = self.invoke(FakeConnection(errors={where: error}))
                self.assertFalse(row["ok"])
                self.assertEqual(row["failure_phase"], phase)
                self.assertEqual(row["error"], type(error).__name__)
                self.assertEqual(row["wire_bytes"], 0)
                self.assertNotIn("status", row)

    def test_body_timeout_retains_status_and_partial_returned_bytes(self):
        response = FakeResponse(parts=[b'{"', TimeoutError(SECRET)])
        _, row = self.invoke(FakeConnection(response))
        self.assertEqual((row["failure_phase"], row["error"]), ("body", "TimeoutError"))
        self.assertEqual(row["status"], 200)
        self.assertEqual(row["wire_bytes"], 2)
        self.assertEqual(row["bytes"], 0)
        self.assertIsNotNone(row["body_seconds"])

    def test_incomplete_chunk_and_content_length_preserve_byte_counts(self):
        for response, count in (
            (FakeResponse(parts=[b'{"', http.client.IncompleteRead(b"x", 9)]), 3),
            (FakeResponse(raw=b"short", expected_length=20), 5)):
            with self.subTest(count=count):
                _, row = self.invoke(FakeConnection(response))
                self.assertEqual((row["failure_phase"], row["error"]), ("body", "IncompleteRead"))
                self.assertEqual(row["wire_bytes"], count)

    def test_real_stdlib_fixed_length_and_chunked_reads_in_memory(self):
        class MemorySocket:
            def __init__(self, data):
                self.data = data

            def makefile(self, mode):
                return io.BytesIO(self.data)

        for data, incomplete in (
            (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}", False),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{}", True),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n", False),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{}", True)):
            with self.subTest(incomplete=incomplete, chunked=b"chunked" in data):
                response = http.client.HTTPResponse(MemorySocket(data))
                response.begin()
                row = {"wire_bytes": 0}
                if incomplete:
                    with self.assertRaises(http.client.IncompleteRead):
                        probe._read_response(response, row)
                else:
                    self.assertEqual(probe._read_response(response, row), b"{}")
                self.assertEqual(row["wire_bytes"], 2)
                response.close()

    def test_http_errors_are_responses_and_expected_statuses_succeed(self):
        for status in (401, 409):
            with self.subTest(status=status):
                _, row = self.invoke(FakeConnection(FakeResponse(status=status)), expected=status)
                self.assertTrue(row["ok"])
                self.assertEqual(row["status"], status)
                self.assertNotIn("error", row)
        for raw, encoding, phase_error in (
            (b'{"error":"unavailable"}', "identity", None),
            (SECRET.encode(), "identity", "JSONDecodeError"),
            (SECRET.encode(), "gzip", "BadGzipFile")):
            with self.subTest(encoding=encoding, parse_error=phase_error):
                _, row = self.invoke(FakeConnection(FakeResponse(raw, status=503, encoding=encoding)))
                self.assertEqual(row["error"], "HTTPStatusError")
                self.assertEqual(row["failure_phase"], "parse_verify")
                self.assertEqual(row["status"], 503)
                self.assertEqual(row["wire_bytes"], len(raw))
                self.assertEqual(row.get("phase_error"), phase_error)

    def test_http_status_error_also_preserves_body_failure(self):
        _, row = self.invoke(FakeConnection(FakeResponse(status=503,
            parts=[b"x", TimeoutError(SECRET)])))
        self.assertEqual((row["status"], row["error"], row["phase_error"], row["failure_phase"]),
                         (503, "HTTPStatusError", "TimeoutError", "body"))

    def test_gzip_wire_and_decoded_counts(self):
        decoded = json.dumps({"text": PRIVATE_BODY * 80}).encode()
        compressed = gzip.compress(decoded)
        value, row = self.invoke(FakeConnection(FakeResponse(compressed, encoding="gzip")))
        self.assertEqual(value["text"], PRIVATE_BODY * 80)
        self.assertTrue(row["ok"])
        self.assertEqual(row["wire_bytes"], len(compressed))
        self.assertEqual(row["bytes"], len(decoded))
        self.assertEqual(row["content_encoding"], "gzip")

    def test_parse_decode_and_verify_failures_are_classified_without_content(self):
        def bad_verifier(value):
            raise ValueError(SECRET + PRIVATE_BODY)

        for response, verifier, error in (
            (FakeResponse(SECRET.encode()), None, "JSONDecodeError"),
            (FakeResponse(SECRET.encode(), encoding="gzip"), None, "BadGzipFile"),
            (FakeResponse(encoding=SECRET), None, "ValueError"),
            (FakeResponse(), lambda value: False, "ResponseVerificationError"),
            (FakeResponse(), bad_verifier, "ValueError")):
            with self.subTest(error=error):
                _, row = self.invoke(FakeConnection(response), verify=verifier)
                self.assertEqual((row["failure_phase"], row["error"]), ("parse_verify", error))
                self.assertGreater(row["wire_bytes"], 0)

    def test_gzip_decoded_limit_remains_bounded(self):
        compressed = gzip.compress(b"x" * (16 * 1024 * 1024 + 1))
        _, row = self.invoke(FakeConnection(FakeResponse(compressed, encoding="gzip")))
        self.assertEqual((row["failure_phase"], row["error"]), ("parse_verify", "ValueError"))
        self.assertEqual(row["wire_bytes"], len(compressed))
        self.assertEqual(row["bytes"], 0)

    def test_redirect_is_not_followed_and_close_failure_keeps_row(self):
        _, row = self.invoke(FakeConnection(FakeResponse(status=302), errors={"close": OSError(SECRET)}))
        self.assertEqual(row["error"], "HTTPStatusError")
        self.assertEqual(row["close_error"], "OSError")

    def test_business_verification_is_inside_total_and_phase_time(self):
        now = [0.0]

        def verify(value):
            now[0] += 7.0
            return True

        with mock.patch.object(probe.http.client, "HTTPSConnection", return_value=FakeConnection()), \
                mock.patch.object(probe.time, "perf_counter", side_effect=lambda: now[0]):
            _, row = probe.probe_request(BASE, "/health", authenticated=False, verify=verify)
        self.assertTrue(row["ok"])
        self.assertEqual(row["seconds"], 7.0)
        self.assertEqual(row["parse_verify_seconds"], 7.0)


class SyntheticService:
    def __init__(self, *, bad_health=False, bad_add=False):
        self.bad_health = bad_health
        self.bad_add = bad_add
        self.stored = {}
        self.lock = threading.Lock()
        self.paths = []

    def connection(self, *args, **kwargs):
        return FakeConnection(responder=self.respond)

    def respond(self, sent):
        method, path, body, headers = sent
        data = json.loads(body) if body is not None else None
        with self.lock:
            self.paths.append(path)
            status = 200
            if method == "GET":
                value = {"status": "failed" if self.bad_health else "ok"}
            elif "Authorization" not in headers:
                status, value = 401, {"error": "authentication required"}
            elif path == "/add":
                key = (data["user_id"], data["request_id"])
                if self.bad_add:
                    status, value = 503, {"error": "synthetic outage"}
                elif key in self.stored and self.stored[key] != data:
                    status, value = 409, {"error": "synthetic conflict"}
                else:
                    self.stored[key] = data
                    value = {"success": True, **{k: data[k] for k in ("user_id", "request_id", "session_id")}}
            else:
                items = []
                for (user, request_id), record in self.stored.items():
                    if user != data["user_id"]:
                        continue
                    for i, message in enumerate(record["messages"]):
                        if data["query"].startswith("probe") and data["query"] not in message["content"]:
                            continue
                        items.append({"id": request_id + "/" + str(i), "content": message["content"]})
                value = {"data": items[:data["top_k"]]}
        raw = json.dumps(value).encode()
        encoding = "gzip" if headers.get("Accept-Encoding") == "gzip" else "identity"
        return FakeResponse(gzip.compress(raw) if encoding == "gzip" else raw, status=status, encoding=encoding)


class ProbeRunTests(unittest.TestCase):
    def test_cli_defaults_and_timeout_limit_are_preserved(self):
        with mock.patch.object(sys, "argv", ["verify_public_api", "--base-url", BASE]), \
                mock.patch.object(probe, "run", return_value=0) as run:
            self.assertEqual(probe.main(), 0)
        args = run.call_args.args[0]
        self.assertEqual((args.concurrency, args.timeout, args.response_encoding), (16, 1800, "gzip"))
        with mock.patch.object(sys, "argv", ["verify_public_api", "--base-url", BASE, "--timeout", "1801"]), \
                mock.patch.object(sys, "stderr", io.StringIO()), \
                mock.patch.object(probe, "run") as run:
            with self.assertRaises(SystemExit) as failure:
                probe.main()
        self.assertEqual(failure.exception.code, 2)
        run.assert_not_called()

    def run_synthetic(self, service, *, encoding="gzip", abort=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.json"
            args = SimpleNamespace(base_url=BASE, concurrency=16, timeout=120,
                                   response_encoding=encoding, report=path)
            with mock.patch.dict(os.environ, {"MEMORY_API_TOKEN": SECRET}), \
                    mock.patch.object(probe.socket, "getaddrinfo", return_value=[(None, None, None, None, ("1.1.1.1", 443))]), \
                    mock.patch.object(probe.http.client, "HTTPSConnection", side_effect=service.connection), \
                    redirect_stdout(io.StringIO()) as output:
                if abort:
                    with self.assertRaises(RuntimeError):
                        probe.run(args)
                else:
                    self.assertEqual(probe.run(args), 0)
            text = path.read_text()
            for forbidden in (SECRET, "Authorization", "I preserve this original record",
                              "synthetic network capacity archive original record", "synthetic-public-probe/"):
                self.assertNotIn(forbidden, text + output.getvalue())
            return json.loads(text)

    def test_completed_report_keeps_old_fields_and_all_unique_safe_records(self):
        for encoding in ("gzip", "identity"):
            with self.subTest(encoding=encoding):
                report = self.run_synthetic(SyntheticService(), encoding=encoding)
                self.assertTrue(report["public_health"])
                self.assertTrue(report["complete_user_isolation"])
                self.assertTrue(report["conflicting_retry_rejected"])
                self.assertEqual(report["configuration"]["request_timeout_seconds"], 120)
                self.assertEqual(report["configuration"]["accepted_response_encoding"], encoding)
                old_summary_fields = {"requests", "failed", "p50_seconds", "p95_seconds",
                    "max_response_bytes", "max_wire_response_bytes", "response_encodings", "errors"}
                for phase, count in (("add", 16), ("immediate_visibility", 16), ("search", 32), ("identical_retry", 32)):
                    self.assertEqual(set(report[phase]), old_summary_fields)
                    self.assertEqual(report[phase]["requests"], count)
                    self.assertEqual(report[phase]["failed"], 0)
                rows = report["request_records"]
                self.assertEqual(len(rows), 100)
                self.assertEqual([r["request_index"] for r in rows], list(range(100)))
                self.assertEqual(len({r["probe_id"] for r in rows}), 100)
                self.assertTrue(all(r["ok"] and r["seconds"] >= 0 for r in rows))

    def test_gate_failure_saves_safe_diagnostics_without_add(self):
        service = SyntheticService(bad_health=True)
        report = self.run_synthetic(service, abort=True)
        self.assertEqual(report["aborted_phase"], "health_authentication")
        self.assertEqual(len(report["request_records"]), 2)
        self.assertNotIn("/add", service.paths)
        failed = report["request_records"][0]
        self.assertEqual((failed["failure_phase"], failed["error"]), ("parse_verify", "ResponseVerificationError"))

    def test_add_failure_saves_all_completed_initial_request_rows(self):
        report = self.run_synthetic(SyntheticService(bad_add=True), abort=True)
        self.assertEqual(report["aborted_phase"], "add_visibility")
        self.assertEqual(report["add"]["failed"], 16)
        self.assertEqual(report["immediate_visibility"]["failed"], 16)
        self.assertEqual(len(report["request_records"]), 34)
        self.assertTrue(all(r["failure_phase"] == "parse_verify" for r in report["request_records"] if not r["ok"]))


if __name__ == "__main__":
    unittest.main()
