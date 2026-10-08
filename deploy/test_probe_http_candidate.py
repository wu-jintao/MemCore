"""Loopback-only candidate probe tests; no real env, cloud, SSH, or model calls."""
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from deploy import probe_http_candidate as probe


SECRET = "synthetic-private-token-do-not-report"


class MockService:
    def __init__(self, mode="good"):
        self.mode = mode
        self.stored = {}
        self.requests = []
        self.lock = threading.Lock()
        self.active_searches = 0
        self.peak_searches = 0

    def handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def send(self, status, value):
                body = value if isinstance(value, bytes) else json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                with service.lock:
                    service.requests.append((self.command, self.path, None))
                self.send(503 if service.mode == "bad_health" else 200, {"status": "ok"})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with service.lock:
                    service.requests.append((self.command, self.path, payload))
                if self.headers.get("Authorization") != "Bearer " + SECRET:
                    self.send(401, {"error": "Unauthorized"})
                    return
                if self.path == "/add":
                    scope = (payload["user_id"], payload["request_id"])
                    with service.lock:
                        previous = service.stored.get(scope)
                        if previous is not None and previous != payload:
                            conflict = True
                        else:
                            service.stored[scope] = payload
                            conflict = False
                    if conflict:
                        self.send(409, {"error": "Conflict"})
                    else:
                        self.send(200, {"success": True, **{key: payload[key]
                            for key in ("request_id", "user_id", "session_id")}})
                    return
                with service.lock:
                    service.active_searches += 1
                    service.peak_searches = max(service.peak_searches, service.active_searches)
                    count = sum(1 for method, path, item in service.requests if path == "/search"
                                and item is not None and item["user_id"].endswith("CaseSensitiveUser"))
                try:
                    # At the concurrent batch, brief overlap demonstrates all 16 workers.
                    if count >= 4:
                        time.sleep(0.06)
                    if service.mode == "slow_search":
                        time.sleep(1)
                    if (service.mode == "bad_concurrent" and count >= 4
                            and payload["user_id"].endswith("CaseSensitiveUser")):
                        self.send(503, {"error": SECRET})
                        return
                    if service.mode == "redirect":
                        self.send_response(302)
                        self.send_header("Location", "https://agentmemoryleaderboard.ai/evaluation")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    with service.lock:
                        entries = list(service.stored.items())
                    data = [{"id": request_id + "-" + str(index), "content": message["content"]}
                            for (user_id, request_id), document in entries
                            if user_id == payload["user_id"]
                            or (service.mode == "leaky" and payload["user_id"].endswith("OtherUser"))
                            for index, message in enumerate(document["messages"])]
                    self.send(200, {"data": data[:payload["top_k"]]})
                finally:
                    with service.lock:
                        service.active_searches -= 1

        return Handler


@contextmanager
def local_service(mode="good"):
    state = MockService(mode)

    class Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

    server = Server(("127.0.0.1", 0), state.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.env_file = self.root / "candidate.env"
        self.env_file.write_text("IGNORE_THIS=must-not-be-read-as-credential\n"
                                 "export MEMORY_API_TOKEN='" + SECRET + "'\n")
        self.env_file.chmod(0o600)

    def tearDown(self):
        self.directory.cleanup()

    def args(self, base="http://127.0.0.1:9", **changes):
        values = dict(base_url=base, token_env_file=self.env_file, output=self.root / "report.json",
                      execute=True, timeout=2, deadline_seconds=10)
        values.update(changes)
        return SimpleNamespace(**values)

    def assert_private_and_complete(self, report, args):
        saved = json.loads(args.output.read_text())
        self.assertEqual(saved, report)
        self.assertEqual(args.output.stat().st_mode & 0o777, 0o600)
        serialized = json.dumps(report)
        for forbidden in (SECRET, "IGNORE_THIS", "Authorization", "must-not-be-read-as-credential"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(len(report["request_records"]), 30)
        summary = report["summary"]
        self.assertEqual(summary["planned"], 30)
        self.assertEqual(summary["completed"] + summary["unfinished"], 30)
        self.assertEqual(summary["succeeded"] + summary["failed"], summary["completed"])
        self.assertFalse(report["full_capacity_validated"])
        for row in report["request_records"]:
            for field in ("status", "seconds", "response_bytes", "wire_response_bytes"):
                self.assertIn(field, row)

    def test_accept_only_loopback_http_and_pin_localhost_without_dns(self):
        with mock.patch.object(socket, "getaddrinfo", side_effect=AssertionError("No DNS needed")):
            self.assertEqual(probe.validate_origin("http://localhost:8000/"),
                             ("http://localhost:8000", "127.0.0.1", 8000))
        self.assertEqual(probe.validate_origin("http://[::1]:8000")[1:], ("::1", 8000))
        for url in ("https://127.0.0.1", "http://example.com", "http://127.1",
                    "http://127.0.0.2", "http://localhost.evil", "http://localhost./",
                    "http://user:secret@localhost", "http://localhost/add", "http://localhost?",
                    "http://localhost#", " http://localhost", "http://[::ffff:127.0.0.1]",
                    "http://localhost:65536", "http://localhost:0"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                probe.validate_origin(url)

    def test_plan_requires_no_token_read_or_http(self):
        args = self.args(execute=False, token_env_file=self.root / "missing-private-file")
        with mock.patch.object(probe, "read_private_token", side_effect=AssertionError("No token read")), \
                mock.patch.object(probe.http.client, "HTTPConnection", side_effect=AssertionError("No HTTP")):
            report, code = probe.run(args)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["summary"]["attempted"], 0)
        self.assertEqual(report["summary"]["unfinished"], 30)
        self.assert_private_and_complete(report, args)

    def test_full_small_probe_retries_conflict_16_concurrency_and_complete_scope(self):
        with local_service() as (state, base):
            args = self.args(base)
            # A real environment credential must never be consulted.
            with mock.patch.dict(os.environ, {"MEMORY_API_TOKEN": "wrong-real-env"}):
                report, code = probe.run(args)
            self.assertEqual(len(state.requests), 30)
            self.assertGreaterEqual(state.peak_searches, 12)
            additions = [payload for _, path, payload in state.requests if path == "/add"]
            self.assertEqual(len(additions), 4)
            self.assertEqual(len(additions[0]["messages"]), 32)
            self.assertEqual(additions[0], additions[1])
            self.assertEqual(additions[0]["request_id"], additions[2]["request_id"])
            self.assertNotEqual(additions[0], additions[2])
            self.assertEqual(len(state.stored), 2)
            self.assertEqual(sum(len(document["messages"]) for document in state.stored.values()), 33)
        self.assertEqual(code, 0)
        self.assertEqual(report["summary"]["succeeded"], 30)
        self.assertEqual(report["summary"]["unfinished"], 0)
        self.assert_private_and_complete(report, args)

    def test_health_gate_retains_denominator_and_stops_before_add(self):
        with local_service("bad_health") as (state, base):
            args = self.args(base)
            report, code = probe.run(args)
            self.assertEqual(len(state.requests), 1)
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"], {**report["summary"], "completed": 1, "failed": 1,
                                           "succeeded": 0, "unfinished": 29})
        self.assertEqual(report["request_records"][0]["status"], 503)
        self.assert_private_and_complete(report, args)

    def test_concurrent_failures_remain_in_full_denominator(self):
        with local_service("bad_concurrent") as (state, base):
            args = self.args(base)
            report, code = probe.run(args)
            self.assertEqual(len(state.requests), 30)
        self.assertEqual(code, 1)
        batch = report["request_records"][9:25]
        self.assertEqual(len(batch), 16)
        self.assertTrue(all(row["state"] == "completed" for row in batch))
        self.assertTrue(all(row["status"] == 503 for row in batch))
        self.assertEqual(report["summary"]["failed"], 16)
        self.assert_private_and_complete(report, args)

    def test_redirect_never_followed(self):
        with local_service("redirect") as (state, base):
            args = self.args(base)
            report, code = probe.run(args)
            self.assertEqual(len(state.requests), 4)
        self.assertEqual(code, 1)
        self.assertEqual(report["request_records"][3]["status"], 302)
        self.assertEqual(report["summary"]["unfinished"], 26)
        self.assert_private_and_complete(report, args)

    def test_cross_populated_user_leak_is_detected(self):
        with local_service("leaky") as (state, base):
            args = self.args(base)
            report, code = probe.run(args)
        self.assertEqual(code, 1)
        self.assertEqual(report["stopped_after"], "separate_user_search")
        self.assertEqual(report["request_records"][8]["error"], "ResponseVerificationError")
        self.assert_private_and_complete(report, args)

    def test_wall_deadline_closes_request_and_retains_unfinished_items(self):
        with local_service("slow_search") as (state, base):
            args = self.args(base, timeout=2, deadline_seconds=0.15)
            started = time.monotonic()
            report, code = probe.run(args)
            elapsed = time.monotonic() - started
        self.assertEqual(code, 1)
        self.assertLess(elapsed, 0.6)
        self.assertLess(report["elapsed_seconds"], 0.6)
        row = report["request_records"][3]
        self.assertEqual((row["state"], row["error"]), ("unfinished", "DeadlineExceeded"))
        self.assertEqual(report["summary"]["unfinished"], 27)
        self.assert_private_and_complete(report, args)

    def test_request_timeout_is_failed_and_unsent_work_is_unfinished(self):
        with local_service("slow_search") as (state, base):
            args = self.args(base, timeout=0.08, deadline_seconds=2)
            report, code = probe.run(args)
        self.assertEqual(code, 1)
        row = report["request_records"][3]
        self.assertEqual((row["state"], row["error"]), ("completed", "RequestTimeout"))
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["unfinished"], 26)
        self.assert_private_and_complete(report, args)

    def test_private_token_file_rejects_permissions_symlink_and_duplicate(self):
        self.assertEqual(probe.read_private_token(self.env_file), SECRET)
        self.env_file.chmod(0o644)
        with self.assertRaises(ValueError):
            probe.read_private_token(self.env_file)
        self.env_file.chmod(0o600)
        link = self.root / "linked.env"
        link.symlink_to(self.env_file)
        with self.assertRaises(ValueError):
            probe.read_private_token(link)
        self.env_file.write_text("MEMORY_API_TOKEN=a\nMEMORY_API_TOKEN=b\n")
        with self.assertRaises(ValueError):
            probe.read_private_token(self.env_file)

    def test_token_error_report_contains_no_credentials_and_no_requests(self):
        args = self.args(token_env_file=self.root / "missing.env")
        with mock.patch.object(probe.http.client, "HTTPConnection", side_effect=AssertionError("No request")):
            report, code = probe.run(args)
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["attempted"], 0)
        self.assertEqual(report["execution_error"], "ValueError")
        self.assert_private_and_complete(report, args)

    def test_private_report_rejects_symlink_and_public_directory(self):
        target = self.root / "target.json"
        target.write_text("preserve")
        output = self.root / "link.json"
        output.symlink_to(target)
        with self.assertRaises(ValueError):
            probe.run(self.args(output=output, execute=False))
        self.assertEqual(target.read_text(), "preserve")
        public = self.root / "public"
        public.mkdir(mode=0o755)
        with self.assertRaises(ValueError):
            probe.run(self.args(output=public / "report.json", execute=False))

    def test_cli_defaults_to_plan_and_does_not_print_secret(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = probe.main(["--base-url", "http://localhost:9", "--token-env-file",
                               str(self.env_file), "--output", str(self.root / "cli.json")])
        self.assertEqual(code, 0)
        self.assertNotIn(SECRET, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["status"], "planned")


class BodyCompleteSocketTests(unittest.TestCase):
    """HTTP/1.0 EOF may close the underlying socket after a complete body."""

    def invoke_response(self, *, declared_extra=0, read_error=None):
        raw = b'{"status":"ok"}'

        class Socket:
            def __init__(self):
                self.closed = False
                self.timeout_updates = 0
                self.closed_timeout_attempts = 0

            def fileno(self):
                return -1 if self.closed else 1

            def settimeout(self, timeout):
                if self.closed:
                    self.closed_timeout_attempts += 1
                    raise OSError("synthetic closed socket")
                self.timeout_updates += 1

            def shutdown(self, how):
                self.closed = True

        source_socket = Socket()

        class Response:
            status = 200
            length = len(raw) + declared_extra

            def __init__(self):
                self.reads = 0

            def read1(self, amount):
                self.reads += 1
                if read_error is not None:
                    raise read_error
                if self.reads == 1:
                    # Python 3.12 HTTPResponse can close the socket as soon as
                    # the complete Content-Length body has been delivered.
                    source_socket.closed = True
                    return raw
                return b""

            def getheader(self, name, default=None):
                return default

        connection = mock.Mock()
        connection.sock = source_socket
        connection.getresponse.return_value = Response()
        timer = mock.Mock()
        spec = probe.make_plan("synthetic-closed-socket")[0]
        row = probe.request_row(spec)
        with mock.patch.object(probe.http.client, "HTTPConnection", return_value=connection), \
                mock.patch.object(probe.threading, "Timer", return_value=timer):
            value = probe.probe_request("127.0.0.1", 9, spec, row, token=SECRET,
                timeout=5, deadline=time.monotonic() + 10, baseline={})
        timer.start.assert_called_once()
        timer.cancel.assert_called_once()
        connection.close.assert_called_once()
        return value, row, source_socket

    def test_complete_body_closed_socket_succeeds_without_timeout_update(self):
        value, row, source_socket = self.invoke_response()
        self.assertEqual(value, {"status": "ok"})
        self.assertTrue(row["ok"])
        self.assertEqual(row["state"], "completed")
        self.assertEqual(row["response_bytes"], len(b'{"status":"ok"}'))
        self.assertGreater(source_socket.timeout_updates, 0)
        self.assertEqual(source_socket.closed_timeout_attempts, 0)

    def test_closed_socket_does_not_hide_incomplete_body(self):
        _, row, source_socket = self.invoke_response(declared_extra=2)
        self.assertFalse(row["ok"])
        self.assertEqual(row["error"], "IncompleteRead")
        self.assertEqual(source_socket.closed_timeout_attempts, 0)

    def test_actual_read_timeout_and_network_error_remain_failures(self):
        for error, expected in ((TimeoutError("synthetic timeout"), "RequestTimeout"),
                                (OSError("synthetic network failure"), "OSError")):
            with self.subTest(expected=expected):
                _, row, _ = self.invoke_response(read_error=error)
                self.assertFalse(row["ok"])
                self.assertEqual(row["state"], "completed")
                self.assertEqual(row["error"], expected)
                self.assertEqual(row["failure_phase"], "body")


if __name__ == "__main__":
    unittest.main()
