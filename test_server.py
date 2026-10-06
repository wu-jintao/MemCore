"""Independent synthetic checks, not the competition's official Smoke/Full suite."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stderr
import copy
from http.client import HTTPConnection
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib import error, request

import server
import semantic


ROOT = Path(__file__).resolve().parent
TEST_TOKEN = "synthetic-test-token-only"


class SyntheticEmbeddingBackend:
    """Deterministic protocol stub; these tests make no semantic-quality claim."""
    fingerprint = "synthetic-model-v1"
    dimension = 3

    def __init__(self):
        self.fail = False
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.fail:
            raise semantic.EmbeddingError("synthetic model outage")
        return [[(1., 0., 0.)] if "astronomy" in text else [(0., 1., 0.), (0., 0., 1.)]
                for text in texts]

    def embed_query(self, text):
        if self.fail:
            raise semantic.EmbeddingError("synthetic query outage")
        return (1., 0., 0.)


def addition(user="run-A/synthetic-user", request_id="request-1", session="session-1", messages=None):
    return {"request_id": request_id, "user_id": user, "session_id": session,
            "messages": messages if messages is not None else [
                {"role": "user", "content": "我约好在星河书店见面。", "timestamp": 0}]}


def original_text(evidence):
    return evidence.split("\n", 1)[1].split("\n\n[adjacent context]\n", 1)[0]


def source_metadata(evidence):
    prefix = evidence.split("\n", 1)[0]
    return json.loads(prefix[len("[source "):-1])


def evidence_sources(items):
    return [part for item in items
            for part in item["content"].split("\n\n[adjacent context]\n")]


class MemoryAPITests(unittest.TestCase):
    def setUp(self):
        # Keep every temporary artifact inside the authorized starter directory.
        self.temporary = tempfile.TemporaryDirectory(prefix="test-memory-", dir=ROOT)
        self.db = Path(self.temporary.name) / "memory.sqlite3"
        self.store = server.MemoryStore(self.db)
        self._start()

    def _start(self):
        self.http = server.make_server(("127.0.0.1", 0), self.store, TEST_TOKEN)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.http.server_port)

    def _stop(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=5)

    def tearDown(self):
        self._stop()
        self.temporary.cleanup()

    def call(self, path, payload=None, token=TEST_TOKEN, raw=None):
        body = raw if raw is not None else (
            json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        req = request.Request(self.base + path, data=body, headers=headers)
        try:
            with request.urlopen(req, timeout=40) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except error.HTTPError as response:
            return response.code, json.loads(response.read().decode("utf-8"))

    def search(self, user="run-A/synthetic-user", query="星河书店", top_k=5, **extra):
        return self.call("/search", {"query": query, "user_id": user, "top_k": top_k, **extra})

    def message_count(self):
        with sqlite3.connect(str(self.db)) as connection:
            return connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def enable_semantic(self):
        self._stop()
        backend = SyntheticEmbeddingBackend()
        self.store = server.MemoryStore(self.db, semantic_backend=backend)
        self._start()
        return backend

    def enable_emitted_context(self):
        self._stop()
        self.store = server.MemoryStore(self.db, context_grouping="emitted")
        self._start()

    def test_synchronous_search_original_echo_and_restart_persistence(self):
        payload = addition(user=" run-A/完整 User ID ", request_id=" 请求 001 ", session=" 会话 1 ")
        status, response = self.call("/add", payload)
        self.assertEqual(status, 200)
        self.assertEqual(response, {"success": True, "request_id": payload["request_id"],
                                    "user_id": payload["user_id"], "session_id": payload["session_id"]})
        status, found = self.search(user=payload["user_id"])
        self.assertEqual(status, 200)
        self.assertEqual(original_text(found["data"][0]["content"]), payload["messages"][0]["content"])
        identity = found["data"][0]["id"]
        self.assertTrue(identity)
        self._stop()
        self.store = server.MemoryStore(self.db)
        self._start()
        self.assertEqual(self.search(user=payload["user_id"])[1]["data"][0]["id"], identity)

    def test_full_user_ids_isolate_users_and_runs_without_routing(self):
        first = "run-A/textual/LoCoMo/same-person"
        second = "run-B/textual/LoCoMo/same-person"
        third = "run-A/textual/locomo/same-person"
        self.assertEqual(self.call("/add", addition(user=first))[0], 200)
        self.assertEqual(self.search(user=second)[1], {"data": []})
        self.assertEqual(self.search(user=third)[1], {"data": []})
        self.assertEqual(self.search(user="LoCoMo")[1], {"data": []})
        self.assertEqual(self.search(user="same-person")[1], {"data": []})
        self.assertEqual(self.search(user=first + " ")[1], {"data": []})
        self.assertEqual(self.call("/add", addition(user=second, messages=[
            {"role": "assistant", "content": "星河书店的另一位访客带来蓝色书签。"}]))[0], 200)
        result_a = self.search(user=first)[1]["data"]
        result_b = self.search(user=second)[1]["data"]
        self.assertEqual(len(result_a), 1)
        self.assertEqual(len(result_b), 1)
        self.assertNotEqual(result_a[0]["id"], result_b[0]["id"])
        self.assertNotEqual(result_a[0]["content"], result_b[0]["content"])

    def test_idempotent_retry_conflict_including_changed_session(self):
        original = addition()
        self.assertEqual(self.call("/add", original)[0], 200)
        initial = self.search()[1]
        # Key ordering and JSON formatting do not alter the canonical request.
        self.assertEqual(self.call("/add", dict(reversed(list(original.items()))))[0], 200)
        self.assertEqual(self.message_count(), 1)
        self.assertEqual(self.search()[1], initial)
        changed = copy.deepcopy(original)
        changed["messages"][0]["content"] = "星河书店已换成紫色招牌。"
        status, failure = self.call("/add", changed)
        self.assertEqual(status, 409)
        self.assertNotIn("success", failure)
        self.assertEqual(self.search()[1], initial)
        self.assertEqual(self.call("/add", addition(session="session-2"))[0], 409)
        self.assertEqual(self.message_count(), 1)
        self.assertEqual(self.call("/add", addition(session="session-2", request_id="request-2"))[0], 200)
        self.assertEqual(self.message_count(), 2)

    def test_top_k_options_english_and_empty_ranges(self):
        # Check the top_k bound independently of contextual deduplication.
        self.store.context_radius = 0
        messages = [{"role": "user", "content": "The comet notebook has marker " + str(i)}
                    for i in range(6)]
        self.assertEqual(self.call("/add", addition(messages=messages))[0], 200)
        for limit in (1, 2, 4, 10):
            status, response = self.search(query="COMET notebook", top_k=limit)
            self.assertEqual(status, 200)
            self.assertEqual(len(response["data"]), min(limit, 6))
            self.assertEqual(len({v["id"] for v in response["data"]}), min(limit, 6))
        self.assertEqual(self.search(query="unrelated-lexical-token")[1], {"data": []})
        self.assertEqual(self.search(query="")[1], {"data": []})
        self.assertEqual(original_text(self.search(query="", options=["comet"])[1]["data"][0]["content"]),
                         messages[0]["content"])

    def test_original_text_role_and_source_timestamps_preserved(self):
        messages = [
            {"role": "user", "content": "  星河：先输入晚时间\n保留换行！  ", "timestamp": 1735689600123},
            {"role": "assistant", "content": "星河：后输入零时间。", "timestamp": 0},
            {"role": "user", "content": "星河：没有提供时间。"},
        ]
        self.assertEqual(self.call("/add", addition(messages=messages))[0], 200)
        with sqlite3.connect(str(self.db)) as connection:
            rows = connection.execute(
                "SELECT role, content, source_timestamp_ms, session_id, request_id FROM messages ORDER BY sequence"
            ).fetchall()
        self.assertEqual([(r[0], r[1], r[2]) for r in rows],
                         [(v["role"], v["content"], v.get("timestamp")) for v in messages])
        self.assertTrue(all(r[3:] == ("session-1", "request-1") for r in rows))
        evidence = self.search(query="星河")[1]["data"]
        sources = evidence_sources(evidence)
        self.assertEqual({original_text(value) for value in sources},
                         {v["content"] for v in messages})
        metadata = {original_text(value): source_metadata(value) for value in sources}
        for index, message in enumerate(messages):
            source = metadata[message["content"]]
            self.assertEqual(source["role"], message["role"])
            self.assertEqual(source["message_index"], index)
            if "timestamp" in message:
                self.assertEqual(source["timestamp_ms"], message["timestamp"])
                if message["timestamp"] == 0:
                    self.assertEqual(source["timestamp_utc"], "1970-01-01T00:00:00.000Z")
            else:
                self.assertNotIn("timestamp_ms", source)
                self.assertNotIn("timestamp_utc", source)

    def test_reject_bad_add_schema_without_writes(self):
        bad = [None, [], {}, {**addition(), "user_id": 12}, {**addition(), "messages": []},
               {**addition(), "unexpected": True}]
        for message in ({"role": "", "content": "x"}, {"role": "user", "content": "  "},
                        {"role": "system", "content": "x"}, {"role": "developer", "content": "x"},
                        {"role": "user", "content": ["x"]}, {"role": "user"},
                        {"role": "user", "content": "x", "timestamp": True},
                        {"role": "user", "content": "x", "timestamp": 3.5},
                        {"role": "user", "content": "x", "timestamp": None}):
            bad.append(addition(messages=[message]))
        for payload in bad:
            with self.subTest(payload=payload):
                # JSON null needs explicit bytes because None otherwise means GET.
                raw = json.dumps(payload).encode("utf-8")
                status, response = self.call("/add", raw=raw)
                self.assertEqual(status, 400)
                self.assertNotIn("success", response)
        self.assertEqual(self.message_count(), 0)

    def test_reject_bad_search_schema_and_malformed_json(self):
        original = {"query": "星河", "user_id": "run-A", "top_k": 1}
        bad = [[], {}, {**original, "query": 3}, {**original, "user_id": ""},
               {**original, "top_k": 0}, {**original, "top_k": -1},
               {**original, "top_k": True}, {**original, "top_k": 1.5},
               {**original, "top_k": 101},
               {**original, "options": "a"}, {**original, "options": [1]},
               {"query": "x", "user_id": "run-A"}]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertEqual(self.call("/search", payload)[0], 400)
        self.assertEqual(self.call("/add", raw=b"{broken json")[0], 400)

    def test_authentication_and_public_health(self):
        self.assertEqual(self.call("/health", token=None), (200, {"status": "ok"}))
        for token in (None, "wrong-test-token"):
            self.assertEqual(self.call("/add", addition(), token=token)[0], 401)
            self.assertEqual(self.call("/search", {"query": "x", "user_id": "u", "top_k": 1},
                                       token=token)[0], 401)
        self.assertEqual(self.message_count(), 0)
        with self.assertRaises(ValueError):
            server.make_server(("127.0.0.1", 0), self.store)

    def test_authenticated_slow_body_and_timed_out_body_retry(self):
        def send_partial(payload, finish):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            midpoint = len(raw) // 2
            connection = HTTPConnection("127.0.0.1", self.http.server_port, timeout=2)
            try:
                connection.putrequest("POST", "/add")
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", str(len(raw)))
                connection.putheader("Authorization", "Bearer " + TEST_TOKEN)
                connection.endheaders(raw[:midpoint])
                if finish:
                    # The body legitimately pauses longer than the header
                    # deadline, while remaining within its own read deadline.
                    time.sleep(.2)
                    connection.send(raw[midpoint:])
                response = connection.getresponse()
                return response.status, json.loads(response.read().decode("utf-8"))
            finally:
                connection.close()

        with mock.patch.object(server, "REQUEST_HEADER_TIMEOUT_SECONDS", .1), \
                mock.patch.object(server, "AUTHENTICATED_BODY_TIMEOUT_SECONDS", .6):
            valid = addition(request_id="slow-valid", messages=[
                {"role": "user", "content": "The cobaltarchive book is on the shelf."}])
            self.assertEqual(send_partial(valid, finish=True)[0], 200)
            self.assertEqual(self.message_count(), 1)
            self.assertIn("cobaltarchive", self.search(query="cobaltarchive")[1]["data"][0]["content"])

            incomplete = addition(request_id="slow-retry", messages=[
                {"role": "user", "content": "The violetarchive letter is in the drawer."}])
            status, failure = send_partial(incomplete, finish=False)
            self.assertEqual(status, 408)
            self.assertNotIn("success", failure)
            self.assertEqual(self.message_count(), 1)
            self.assertEqual(self.search(query="violetarchive")[1], {"data": []})
            self.assertEqual(self.call("/add", incomplete)[0], 200)
            self.assertEqual(self.message_count(), 2)
            self.assertIn("violetarchive", self.search(query="violetarchive")[1]["data"][0]["content"])

    def test_atomic_failure_does_not_report_success_or_leave_partial_messages(self):
        payload = addition(messages=[{"role": "user", "content": "星河第一条"},
                                     {"role": "user", "content": "force-storage-fail"}])
        original_tokenizer = server.lexical_tokens

        def fail_second(content):
            if content == "force-storage-fail":
                raise sqlite3.OperationalError("synthetic storage failure")
            return original_tokenizer(content)

        with mock.patch.object(server, "lexical_tokens", side_effect=fail_second):
            with self.assertLogs(level="ERROR"):
                status, response = self.call("/add", payload)
        self.assertEqual(status, 500)
        self.assertNotIn("success", response)
        self.assertEqual(self.message_count(), 0)
        self.assertEqual(self.search()[1], {"data": []})
        self.assertEqual(self.call("/add", payload)[0], 200)
        self.assertEqual(self.message_count(), 2)

    def test_16_concurrent_adds_searches_and_idempotent_retries(self):
        def write(index):
            payload = addition(request_id="parallel-" + str(index), messages=[
                {"role": "user", "content": "Concurrency evidence uniqueitem" + str(index),
                 "timestamp": index}])
            status, response = self.call("/add", payload)
            self.assertEqual(status, 200)
            self.assertTrue(response["success"])
            result = self.search(query="uniqueitem" + str(index), top_k=1)
            self.assertEqual(result[0], 200)
            self.assertEqual(original_text(result[1]["data"][0]["content"]), payload["messages"][0]["content"])

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(write, range(48)))
        self.assertEqual(self.message_count(), 48)
        repeated = addition(request_id="parallel-retry", messages=[
            {"role": "user", "content": "retryable single original evidence"}])
        with ThreadPoolExecutor(max_workers=16) as pool:
            replies = list(pool.map(lambda _: self.call("/add", repeated), range(32)))
        self.assertTrue(all(status == 200 for status, _ in replies))
        self.assertEqual(self.message_count(), 49)
        with ThreadPoolExecutor(max_workers=16) as pool:
            found = list(pool.map(lambda _: self.search(query="concurrency", top_k=7), range(48)))
        self.assertTrue(all(status == 200 and 1 <= len(data["data"]) <= 7 for status, data in found))
        self.assertTrue(all(data == found[0][1] for _, data in found))
        sources = evidence_sources(found[0][1]["data"])
        source_ids = [source_metadata(value)["message_id"] for value in sources]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertGreaterEqual(len(source_ids), 7)

    def test_low_ranked_adjacent_anchor_survives_whole_item_prefix_budget(self):
        self.enable_emitted_context()
        question = "Which city did we choose for the coppermaple meeting?"
        reply = "  We chose Kyoto.\nKeep this original answer exactly.  "
        self.assertEqual(self.call("/add", addition(messages=[
            {"role": "user", "content": question, "timestamp": 1000},
            {"role": "assistant", "content": reply, "timestamp": 2000}]))[0], 200)
        self.assertEqual(self.call("/add", addition(request_id="large-distractor", session="elsewhere",
            messages=[{"role": "user", "content": "Irrelevant archive paragraph. " * 200}]))[0], 200)
        with closing(self.store._connect()) as connection:
            rows = list(connection.execute("SELECT * FROM messages ORDER BY sequence"))
        first, answer, distractor = rows
        complete_pair = (self.store._evidence(first) + "\n\n[adjacent context]\n"
                         + self.store._evidence(answer))
        ranking = [(first["id"], 3.), (distractor["id"], 2.), (answer["id"], 1.)]
        # The pair fits, while the intervening distractor cannot fit the prefix.
        with mock.patch.object(self.store, "_lexical_candidates", return_value=ranking), \
                mock.patch.object(server, "MAX_EVIDENCE_CHARACTERS", len(complete_pair) + 10):
            status, found = self.search(query="coppermaple", top_k=3)
        self.assertEqual(status, 200)
        self.assertEqual(len(found["data"]), 1)
        self.assertEqual(found["data"][0]["id"], first["id"])
        self.assertEqual(found["data"][0]["content"], complete_pair)
        sources = evidence_sources(found["data"])
        self.assertEqual([original_text(value) for value in sources], [question, reply])
        self.assertEqual(source_metadata(sources[1])["message_id"], answer["id"])
        self.assertEqual(source_metadata(sources[1])["role"], "assistant")
        self.assertEqual(source_metadata(sources[1])["timestamp_ms"], 2000)

    def test_contextual_anchor_dedup_does_not_expand_original_top_k_pool(self):
        self.enable_emitted_context()
        self.assertEqual(self.call("/add", addition(messages=[
            {"role": "user", "content": "The ivorymaple question."},
            {"role": "assistant", "content": "The original adjacent ivorymaple answer."}]))[0], 200)
        self.assertEqual(self.call("/add", addition(request_id="outside-pool", session="elsewhere",
            messages=[{"role": "user", "content": "OUTSIDE-POOL source must not refill the result."}]))[0], 200)
        with closing(self.store._connect()) as connection:
            ids = [row["id"] for row in connection.execute("SELECT id FROM messages ORDER BY sequence")]
        with mock.patch.object(self.store, "_lexical_candidates",
                               return_value=[(identity, 3. - rank) for rank, identity in enumerate(ids)]):
            status, found = self.search(query="ivorymaple", top_k=2)
        self.assertEqual(status, 200)
        self.assertEqual(len(found["data"]), 1)
        sources = evidence_sources(found["data"])
        self.assertEqual([source_metadata(value)["message_id"] for value in sources], ids[:2])
        self.assertNotIn("OUTSIDE-POOL", json.dumps(found))
        self.assertEqual(found["data"][0]["score"], 3.)

    def test_neighbor_rejected_by_window_cap_remains_available_as_anchor(self):
        self.enable_emitted_context()
        first_text = "Long violetmaple source. " * 40
        answer_text = "The violetmaple answer survives as its own complete anchor."
        self.assertEqual(self.call("/add", addition(messages=[
            {"role": "user", "content": first_text},
            {"role": "assistant", "content": answer_text}]))[0], 200)
        with closing(self.store._connect()) as connection:
            rows = list(connection.execute("SELECT * FROM messages ORDER BY sequence"))
        ranking = [(rows[0]["id"], 2.), (rows[1]["id"], 1.)]
        with mock.patch.object(self.store, "_lexical_candidates", return_value=ranking), \
                mock.patch.object(server, "MAX_WINDOW_CHARACTERS", len(self.store._evidence(rows[0])) + 10):
            status, found = self.search(query="violetmaple", top_k=2)
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in found["data"]], [row["id"] for row in rows])
        self.assertEqual([original_text(item["content"]) for item in found["data"]],
                         [first_text, answer_text])
        self.assertTrue(all("\n\n[adjacent context]\n" not in item["content"] for item in found["data"]))

    def test_oversized_first_context_bundle_is_preserved_without_refill(self):
        self.enable_emitted_context()
        first_text = "  The scarletmaple question preserves its whitespace.  "
        answer_text = "The complete scarletmaple answer.\nNo text is trimmed."
        self.assertEqual(self.call("/add", addition(messages=[
            {"role": "user", "content": first_text},
            {"role": "assistant", "content": answer_text}]))[0], 200)
        self.assertEqual(self.call("/add", addition(request_id="later", session="elsewhere",
            messages=[{"role": "user", "content": "LATER evidence after the whole-item cap."}]))[0], 200)
        with closing(self.store._connect()) as connection:
            ids = [row["id"] for row in connection.execute("SELECT id FROM messages ORDER BY sequence")]
        with mock.patch.object(self.store, "_lexical_candidates",
                               return_value=[(identity, 3. - rank) for rank, identity in enumerate(ids)]), \
                mock.patch.object(server, "MAX_EVIDENCE_CHARACTERS", 1):
            status, found = self.search(query="scarletmaple", top_k=3)
        self.assertEqual(status, 200)
        self.assertEqual(len(found["data"]), 1)
        self.assertGreater(len(found["data"][0]["content"]), 1)
        self.assertEqual([original_text(value) for value in evidence_sources(found["data"])],
                         [first_text, answer_text])
        self.assertNotIn("LATER evidence", json.dumps(found))

    def test_default_reserved_grouping_preserves_separate_anchors_and_old_response(self):
        messages = [
            {"role": "user", "content": "Original preceding context.", "timestamp": 0},
            {"role": "user", "content": "The cobaltmaple question.", "timestamp": 1000},
            {"role": "assistant", "content": "  The cobaltmaple answer.\nOriginal newline.  ", "timestamp": 2000},
            {"role": "assistant", "content": "Original following context.", "timestamp": 3000},
        ]
        self.assertEqual(self.store.context_grouping, "reserved")
        self.assertEqual(self.call("/add", addition(messages=messages))[0], 200)
        with closing(self.store._connect()) as connection:
            rows = list(connection.execute("SELECT * FROM messages ORDER BY sequence"))
        ranking = [(rows[1]["id"], 3.0), (rows[2]["id"], 2.0)]
        expected = {"data": [
            {"id": rows[1]["id"], "content": self.store._evidence(rows[1])
             + "\n\n[adjacent context]\n" + self.store._evidence(rows[0]), "score": 3.0},
            {"id": rows[2]["id"], "content": self.store._evidence(rows[2])
             + "\n\n[adjacent context]\n" + self.store._evidence(rows[3]), "score": 2.0},
        ]}
        explicit = server.MemoryStore(self.db, context_grouping="reserved")
        with mock.patch.object(self.store, "_lexical_candidates", return_value=ranking), \
                mock.patch.object(explicit, "_lexical_candidates", return_value=ranking):
            self.assertEqual(self.search(query="cobaltmaple", top_k=2), (200, expected))
            self.assertEqual(explicit.search({"query": "cobaltmaple", "user_id": "run-A/synthetic-user",
                                             "top_k": 2}), expected)
            # An independently fixed old-response prefix remains identical too.
            cap = len(expected["data"][0]["content"]) + 1
            with mock.patch.object(server, "MAX_EVIDENCE_CHARACTERS", cap):
                self.assertEqual(self.search(query="cobaltmaple", top_k=2),
                                 (200, {"data": expected["data"][:1]}))

    def test_invalid_context_grouping_rejects_before_database_or_model_initialization(self):
        invalid_db = Path(self.temporary.name) / "invalid-grouping.sqlite3"
        for value in ("unknown", "EMITTED", " emitted", "", None, False, 0, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "context_grouping must"):
                    server.MemoryStore(invalid_db, context_grouping=value)
                self.assertFalse(invalid_db.exists())
        for environment, arguments in (("EMITTED", []), ("reserved", ["--context-grouping", "unknown"])):
            with self.subTest(environment=environment, arguments=arguments), \
                    mock.patch.dict(os.environ, {"MEMORY_CONTEXT_GROUPING": environment,
                                                 "MEMORY_API_TOKEN": TEST_TOKEN}, clear=True), \
                    mock.patch.object(sys, "argv", ["server.py", *arguments]), \
                    mock.patch.object(semantic, "build_backend") as backend, \
                    mock.patch.object(server, "MemoryStore") as store, \
                    redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    server.main()
                self.assertEqual(failure.exception.code, 2)
                backend.assert_not_called()
                store.assert_not_called()

    def test_context_grouping_cli_defaults_environment_and_explicit_override(self):
        for environment, arguments, expected in (
                ({}, [], "reserved"),
                ({"MEMORY_CONTEXT_GROUPING": "emitted"}, [], "emitted"),
                ({"MEMORY_CONTEXT_GROUPING": "emitted"}, ["--context-grouping", "reserved"], "reserved")):
            fake_http = mock.Mock(server_port=12345)
            with self.subTest(environment=environment, arguments=arguments), \
                    mock.patch.dict(os.environ, {**environment, "MEMORY_API_TOKEN": TEST_TOKEN}, clear=True), \
                    mock.patch.object(sys, "argv", ["server.py", "--db", str(self.db), *arguments]), \
                    mock.patch.object(semantic, "build_backend", return_value=None), \
                    mock.patch.object(server, "MemoryStore") as store, \
                    mock.patch.object(server, "make_server", return_value=fake_http):
                server.main()
                self.assertEqual(store.call_args.kwargs["context_grouping"], expected)
                fake_http.serve_forever.assert_called_once()
                fake_http.server_close.assert_called_once()

    def test_adjacent_context_cross_add_scoped_and_source_preserved(self):
        user = "run-A/neighbor-user"
        first = addition(user=user, request_id="first-chunk", messages=[
            {"role": "user", "content": "What is the ambermaple rendezvous location?", "timestamp": 1000}])
        second = addition(user=user, request_id="second-chunk", messages=[
            {"role": "assistant", "content": "The meeting point is Pier 4.", "timestamp": 2000}])
        self.assertEqual(self.call("/add", first)[0], 200)
        self.assertEqual(self.call("/add", second)[0], 200)
        # Neither matching user suffixes nor matching session IDs grant scope.
        self.call("/add", addition(user="run-B/neighbor-user", request_id="other-run", messages=[
            {"role": "assistant", "content": "PRIVATE-CROSS-RUN", "timestamp": 1500}]))
        self.call("/add", addition(user=user, request_id="other-session", session="elsewhere", messages=[
            {"role": "assistant", "content": "PRIVATE-OTHER-SESSION", "timestamp": 1500}]))
        found = self.search(user=user, query="ambermaple", top_k=1)[1]["data"]
        self.assertEqual(len(found), 1)
        evidence = found[0]["content"]
        self.assertEqual(original_text(evidence), first["messages"][0]["content"])
        self.assertIn("\nThe meeting point is Pier 4.", evidence)
        self.assertNotIn("PRIVATE-", evidence)
        context = evidence.split("\n\n[adjacent context]\n", 1)[1]
        source = source_metadata(context)
        self.assertEqual(source["request_id"], "second-chunk")
        self.assertEqual(source["message_index"], 0)
        self.assertEqual(source["timestamp_ms"], 2000)
        self.assertNotEqual(source["message_id"], found[0]["id"])

    def test_source_time_orders_neighbors_instead_of_commit_order(self):
        for request_id, content, timestamp in (
                ("late", "The redchapel appointment is now cancelled.", 3000),
                ("early", "The original venue was near the hill.", 1000),
                ("middle", "The redchapel appointment starts at noon.", 2000)):
            self.call("/add", addition(request_id=request_id, messages=[
                {"role": "user", "content": content, "timestamp": timestamp}]))
        found = self.search(query="redchapel starts noon", top_k=1)[1]["data"][0]
        self.assertEqual(source_metadata(found["content"])["request_id"], "middle")
        self.assertIn("The original venue was near the hill.", found["content"])
        self.assertIn("now cancelled", found["content"])
        self.assertLess(found["content"].index("The original venue"),
                        found["content"].index("now cancelled"))

    def test_mixed_source_times_keep_undated_adjacent_reply_within_scoped_radius(self):
        user = "run-A/mixed-neighbor-user"
        self.assertEqual(self.call("/add", addition(user=user, request_id="prefix", messages=[
            {"role": "user", "content": "DISTANT-PAST note", "timestamp": 1999},
            {"role": "user", "content": "SECOND-BEFORE note", "timestamp": 10000},
            {"role": "user", "content": "FIRST-BEFORE note", "timestamp": 9000}]))[0], 200)
        anchor = "What is the ambermaple meeting place?"
        self.assertEqual(self.call("/add", addition(user=user, request_id="question", messages=[
            {"role": "user", "content": anchor, "timestamp": 2000}]))[0], 200)
        # Global receive order may interleave other scopes; they are never neighbors.
        for other_user, other_session in (("run-B/mixed-neighbor-user", "session-1"),
                                           (user, "another-session")):
            self.assertEqual(self.call("/add", addition(
                user=other_user, session=other_session, request_id="other-scope", messages=[
                    {"role": "assistant", "content": "PRIVATE-OTHER-SCOPE reply"}]))[0], 200)
        reply = "The meeting point is Pier 4."
        self.assertEqual(self.call("/add", addition(user=user, request_id="reply", messages=[
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "SECOND-AFTER note", "timestamp": 8000},
            {"role": "user", "content": "DISTANT-FUTURE note", "timestamp": 2001}]))[0], 200)
        for radius in (1, 2):
            with self.subTest(radius=radius):
                self.store.context_radius = radius
                status, found = self.search(user=user, query="ambermaple", top_k=1)
                self.assertEqual(status, 200)
                self.assertEqual(len(found["data"]), 1)
                evidence = found["data"][0]["content"]
                self.assertEqual(original_text(evidence), anchor)
                self.assertIn(reply, evidence)
                self.assertIn("FIRST-BEFORE", evidence)
                self.assertNotIn("DISTANT-", evidence)
                self.assertNotIn("PRIVATE-", evidence)
                contexts = evidence.split("\n\n[adjacent context]\n")[1:]
                expected = (["FIRST-BEFORE note", reply] if radius == 1 else
                            ["SECOND-BEFORE note", "FIRST-BEFORE note", reply, "SECOND-AFTER note"])
                self.assertEqual([original_text(value) for value in contexts], expected)
                reply_source = source_metadata(next(value for value in contexts if reply in value))
                self.assertEqual(reply_source["request_id"], "reply")
                self.assertEqual(reply_source["session_id"], "session-1")
                self.assertNotIn("timestamp_ms", reply_source)
                self.assertNotIn("timestamp_utc", reply_source)

    def test_undated_anchor_orders_dated_neighbors_by_received_sequence(self):
        messages = [
            {"role": "assistant", "content": "The earlier received message.", "timestamp": 9000},
            {"role": "user", "content": "Where is the lilacmaple meeting place?"},
            {"role": "assistant", "content": "The later received message.", "timestamp": 1000},
        ]
        self.assertEqual(self.call("/add", addition(messages=messages))[0], 200)
        status, found = self.search(query="lilacmaple", top_k=1)
        self.assertEqual(status, 200)
        evidence = found["data"][0]["content"]
        contexts = evidence.split("\n\n[adjacent context]\n")[1:]
        self.assertEqual([original_text(value) for value in contexts],
                         [messages[0]["content"], messages[2]["content"]])
        self.assertEqual([source_metadata(value)["timestamp_ms"] for value in contexts], [9000, 1000])

    def test_storage_failure_after_first_insert_rolls_back_all_indexes(self):
        with sqlite3.connect(str(self.db)) as connection:
            connection.execute("""
                CREATE TRIGGER reject_second_message BEFORE INSERT ON messages
                WHEN NEW.content='reject after first insertion'
                BEGIN SELECT RAISE(ABORT, 'synthetic mid-transaction failure'); END
            """)
        payload = addition(messages=[{"role": "user", "content": "ambermaple first insertion"},
                                     {"role": "assistant", "content": "reject after first insertion"}])
        with self.assertLogs(level="ERROR"):
            self.assertEqual(self.call("/add", payload)[0], 500)
        with sqlite3.connect(str(self.db)) as connection:
            for table in ("requests", "messages", "postings", "user_statistics", "term_statistics"):
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0], 0)
            connection.execute("DROP TRIGGER reject_second_message")
        self.assertEqual(self.call("/add", payload)[0], 200)
        self.assertEqual(self.message_count(), 2)

    def test_statistics_are_per_user_and_retry_does_not_change_them(self):
        first = addition(user="scope/A", messages=[{"role": "user", "content": "rareamber common"},
                                                   {"role": "assistant", "content": "common common"}])
        self.call("/add", first)
        before = self.search(user="scope/A", query="rareamber common", top_k=2)[1]
        self.call("/add", first)
        self.call("/add", addition(user="scope/B", messages=[
            {"role": "user", "content": "rareamber " * 100}]))
        self.assertEqual(before, self.search(user="scope/A", query="rareamber common", top_k=2)[1])
        with sqlite3.connect(str(self.db)) as connection:
            self.assertEqual(connection.execute(
                "SELECT message_count, total_length FROM user_statistics WHERE user_id='scope/A'"
            ).fetchone(), (2, 4))
            self.assertEqual(connection.execute(
                "SELECT document_count FROM term_statistics WHERE user_id='scope/A' AND term='common'"
            ).fetchone(), (2,))

    def test_semantic_vectors_commit_with_original_scope_and_restart(self):
        backend = self.enable_semantic()
        self.assertEqual(self.call("/add", addition(messages=[
            {"role": "user", "content": "I study astronomy."},
            {"role": "assistant", "content": "The telescope is on the balcony."}]))[0], 200)
        self.assertEqual(self.call("/add", addition(user="other-user", messages=[
            {"role": "user", "content": "PRIVATE astronomy secret"}]))[0], 200)
        found = self.search(query="celestial", top_k=1)[1]["data"]
        self.assertEqual(original_text(found[0]["content"]), "I study astronomy.")
        self.assertNotIn("PRIVATE", found[0]["content"])
        with sqlite3.connect(str(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 4)
        self._stop()
        self.store = server.MemoryStore(self.db, semantic_backend=backend)
        self._start()
        self.assertEqual(self.search(query="celestial", top_k=1)[1]["data"], found)

    def test_rrf_tie_preserves_unique_lexical_source_outside_dense_candidates(self):
        self.enable_semantic()
        filler = addition(request_id="sky-catalog", messages=[
            {"role": "user", "content": "astronomy observation from the sky catalog " + str(index)}
            for index in range(132)])
        target = addition(request_id="museum-reference", session="museum-session", messages=[
            {"role": "user", "content": "The amethystchronicle reference describes the harbor museum."}])
        self.assertEqual(self.call("/add", filler)[0], 200)
        self.assertEqual(self.call("/add", target)[0], 200)
        self.assertEqual(self.message_count(), 133)
        with sqlite3.connect(str(self.db)) as connection:
            target_id = connection.execute(
                "SELECT id FROM messages WHERE user_id=? AND request_id=?",
                (target["user_id"], target["request_id"])).fetchone()[0]
        observed_dense_ids = []
        original_rank_vectors = semantic.rank_vectors

        def record_candidates(*args, **kwargs):
            ranked = original_rank_vectors(*args, **kwargs)
            observed_dense_ids.extend(identity for identity, _ in ranked)
            return ranked

        with mock.patch.object(semantic, "rank_vectors", side_effect=record_candidates):
            status, found = self.search(query="amethystchronicle", top_k=1)
        self.assertEqual(status, 200)
        self.assertEqual(len(observed_dense_ids), 128)
        self.assertNotIn(target_id, observed_dense_ids)
        self.assertEqual(found["data"][0]["id"], target_id)
        self.assertEqual(original_text(found["data"][0]["content"]), target["messages"][0]["content"])

    def test_model_outage_never_acknowledges_partial_index_and_retries_are_free(self):
        backend = self.enable_semantic()
        backend.fail = True
        self.assertEqual(self.call("/add", addition())[0], 503)
        self.assertEqual(self.message_count(), 0)
        backend.fail = False
        self.assertEqual(self.call("/add", addition())[0], 200)
        calls = backend.document_calls
        backend.fail = True
        self.assertEqual(self.call("/add", addition())[0], 200)
        self.assertEqual(backend.document_calls, calls)
        self.assertEqual(self.call("/add", addition(session="changed"))[0], 409)
        self.assertEqual(self.search()[0], 503)
        self.assertEqual(self.message_count(), 1)

    def test_semantic_capacity_rejects_without_writes_and_outage_stays_retryable(self):
        # Exercise the actual offline adapter's source-offset segmentation and
        # total-segment checks through HTTP, without downloading real weights.
        from test_semantic import FakeModel

        directory = Path(self.temporary.name) / "synthetic-model"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"synthetic-test-only")
        config = semantic.EmbeddingConfig(
            provider="local", model_dir=str(directory), segment_tokens=24,
            overlap_tokens=6, max_document_segments=2, max_query_segments=1,
            max_total_segments=3)
        model = FakeModel()
        with mock.patch.object(semantic.LocalE5Backend, "_load_model", return_value=model):
            backend = semantic.LocalE5Backend(config)
        self._stop()
        self.store = server.MemoryStore(self.db, semantic_backend=backend)
        self._start()

        for request_id, messages in (
                ("document-capacity", [{"role": "user", "content": "x" * 80}]),
                ("batch-capacity", [{"role": "user", "content": "x" * 30},
                                    {"role": "assistant", "content": "y" * 30}])):
            with self.subTest(request_id=request_id):
                status, failure = self.call("/add", addition(request_id=request_id, messages=messages))
                self.assertEqual(status, 422)
                self.assertIn("declared semantic capacity", failure["error"])
                self.assertNotIn("success", failure)
                with sqlite3.connect(str(self.db)) as connection:
                    for table in ("requests", "messages", "postings", "vectors",
                                  "user_statistics", "term_statistics"):
                        self.assertEqual(connection.execute(
                            "SELECT COUNT(*) FROM " + table).fetchone()[0], 0)
        self.assertEqual(model.calls, [])
        status, failure = self.search(query="q" * 40)
        self.assertEqual(status, 422)
        self.assertIn("declared semantic capacity", failure["error"])
        self.assertNotIn("success", failure)
        # A prior rejected request did not reserve its idempotency key.
        valid = addition(request_id="document-capacity", messages=[
            {"role": "user", "content": "cobalt book"}])
        self.assertEqual(self.call("/add", valid)[0], 200)
        self.assertEqual(self.search(query="cobalt")[0], 200)
        with sqlite3.connect(str(self.db)) as connection:
            for table in ("requests", "messages", "vectors"):
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0], 1)
        with mock.patch.object(backend, "_encode_batch", side_effect=semantic.EmbeddingError("model outage")):
            self.assertEqual(self.call("/add", addition(request_id="outage", messages=[
                {"role": "user", "content": "amber book"}]))[0], 503)
            self.assertEqual(self.search(query="cobalt")[0], 503)
        self.assertEqual(self.message_count(), 1)

    def test_semantic_model_switch_or_missing_vectors_rejected_at_startup(self):
        self.call("/add", addition())
        with self.assertRaisesRegex(ValueError, "lack the configured semantic index"):
            server.MemoryStore(self.db, semantic_backend=SyntheticEmbeddingBackend())
        self.assertEqual(self.search()[0], 200)

    def test_backend_candidate_ids_re_scoped_before_reading_text(self):
        self.enable_semantic()
        self.call("/add", addition(messages=[{"role": "user", "content": "My astronomy note"}]))
        self.call("/add", addition(user="unrelated-scope", messages=[
            {"role": "user", "content": "PRIVATE astronomy note"}]))
        with sqlite3.connect(str(self.db)) as connection:
            foreign_id = connection.execute(
                "SELECT id FROM messages WHERE user_id='unrelated-scope'").fetchone()[0]
        with mock.patch.object(semantic, "rank_vectors", return_value=[(foreign_id, 1.)]):
            self.assertEqual(self.search(query="celestial", top_k=1), (200, {"data": []}))

    def test_zero_semantic_weight_keeps_scoped_lexical_search_and_required_add_index(self):
        self._stop()
        backend = SyntheticEmbeddingBackend()
        self.store = server.MemoryStore(self.db, semantic_backend=backend, semantic_weight=0)
        self._start()
        self.assertEqual(self.call("/add", addition(messages=[
            {"role": "user", "content": "My astronomy notebook is green."}]))[0], 200)
        self.assertEqual(self.call("/add", addition(user="other-scope", messages=[
            {"role": "user", "content": "PRIVATE astronomy notebook"}]))[0], 200)
        with sqlite3.connect(str(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 2)
        # An unavailable semantic backend cannot change weight-zero Search or
        # make it return another scope's evidence; Add still requires indexing.
        backend.fail = True
        found = self.search(query="astronomy", top_k=1)
        self.assertEqual(found[0], 200)
        self.assertEqual(original_text(found[1]["data"][0]["content"]), "My astronomy notebook is green.")
        self.assertNotIn("PRIVATE", found[1]["data"][0]["content"])
        self.assertEqual(self.search(query="celestial", top_k=1), (200, {"data": []}))
        self.assertEqual(self.call("/add", addition(request_id="new-request"))[0], 503)
        self.assertEqual(self.message_count(), 2)

    def test_legacy_postings_migrate_without_changing_source_ids(self):
        self.call("/add", addition())
        before = self.search()[1]
        self._stop()
        with sqlite3.connect(str(self.db)) as connection:
            connection.execute("INSERT INTO terms SELECT term,message_id,frequency FROM postings")
            for table in ("postings", "user_statistics", "term_statistics", "schema_migrations"):
                connection.execute("DELETE FROM " + table)
        self.store = server.MemoryStore(self.db)
        self._start()
        self.assertEqual(before, self.search()[1])
        self.assertEqual(self.call("/add", addition())[0], 200)
        self.assertEqual(self.message_count(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
