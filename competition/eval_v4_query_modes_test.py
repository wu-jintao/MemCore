#!/usr/bin/env python3
"""Synthetic tests for frozen-corpus native-v4 replay.

All credentials and prepared data below are invented. Provider transports are
in-process mocks; the only HTTP service used by fixtures is local loopback.
"""
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import eval_v4_query_modes as modes
import eval_v4_budget as budget
from eval_prepare import SOURCE_MAPPING_VERSION, file_hash, json_line
from semantic import EmbeddingConfig, EmbeddingError
from server import MemoryStore


COMPATIBLE = "https://synthetic.maas.aliyuncs.com/compatible-mode/v1/embeddings"
NATIVE = "https://synthetic.maas.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
SECRET = "SYNTHETIC_ONLY_SECRET"
E0 = [1.0] + [0.0] * 2047
E1 = [0.0, 1.0] + [0.0] * 2046


def native_response(request, usage=7, reverse=False, vector=None):
    payload = json.loads(request.data)
    rows = [{"text_index": index, "embedding": list(vector or E0)}
            for index in range(len(payload["input"]["texts"]))]
    if reverse:
        rows.reverse()
    return {"output": {"embeddings": rows}, "usage": {"total_tokens": usage}}


def compatible_response(request, usage=7):
    payload = json.loads(request.data)
    return {"model": budget.MODEL,
            "data": [{"index": index, "embedding": E0}
                     for index in range(len(payload["input"]))],
            "usage": {"prompt_tokens": usage, "total_tokens": usage}}


def json_transport(request, timeout, limit):
    response = (native_response(request) if isinstance(json.loads(request.data)["input"], dict)
                else compatible_response(request))
    return 200, json.dumps(response).encode("utf-8")


class NativeClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def ledger(self, tokens=10000, requests=20, runtime=30):
        number = len(list(self.root.glob("ledger-*")))
        return budget.BudgetLedger(self.root / ("ledger-" + str(number) + ".json"),
                                   tokens, requests, runtime)

    def client(self, ledger, transport):
        return modes.NativeV4Client(COMPATIBLE, NATIVE, SECRET, ledger,
                                    transport=transport, http_timeout=30)

    def test_native_payload_total_only_usage_and_durable_instruction_reservation(self):
        ledger = self.ledger()
        instruction = modes.INSTRUCT
        texts = ["汉字", "question"]
        observed = []

        def transport(request, timeout, limit):
            payload = json.loads(request.data)
            observed.append(payload)
            self.assertEqual(request.full_url, NATIVE)
            self.assertEqual(payload, {"model": "text-embedding-v4",
                "input": {"texts": texts}, "parameters": {
                    "dimension": 2048, "output_type": "dense", "text_type": "query",
                    "instruct": instruction}})
            saved = json.loads(ledger.path.read_text())
            expected = sum(len(text.encode("utf-8")) + 32
                           + len(instruction.encode("utf-8")) + 32 for text in texts)
            self.assertEqual(saved["calls"][0]["reserved_input_tokens"], expected)
            self.assertEqual(saved["calls"][0]["status"], "pending")
            return 200, json.dumps(native_response(request, usage=9, reverse=True)).encode()

        vectors = self.client(ledger, transport).encode(texts, "query", instruction)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(list(vectors[0]), E0)
        self.assertEqual(ledger.state["accounted_input_tokens"], 9)
        self.assertEqual(ledger.state["reported_input_tokens"], 9)
        self.assertEqual(ledger.state["calls"][0]["reported_total_tokens"], 9)
        self.assertEqual(len(observed), 1)
        self.assertNotIn(SECRET, ledger.path.read_text())
        self.assertNotIn(instruction, ledger.path.read_text())
        self.assertNotIn("question", ledger.path.read_text())

    def test_native_query_without_instruction_omits_it_and_document_uses_document(self):
        ledger = self.ledger()
        seen = []

        def transport(request, timeout, limit):
            seen.append(json.loads(request.data))
            return 200, json.dumps(native_response(request)).encode()

        client = self.client(ledger, transport)
        client.encode(["question"], "query")
        client.encode(["history"], "document")
        self.assertNotIn("instruct", seen[0]["parameters"])
        self.assertEqual(seen[0]["parameters"]["text_type"], "query")
        self.assertEqual(seen[1]["parameters"]["text_type"], "document")
        self.assertNotIn("instruct", seen[1]["parameters"])

    def test_text_indices_restore_input_order(self):
        def transport(request, timeout, limit):
            response = native_response(request)
            response["output"]["embeddings"] = [
                {"text_index": 1, "embedding": E1}, {"text_index": 0, "embedding": E0}]
            return 200, json.dumps(response).encode()
        vectors = self.client(self.ledger(), transport).encode(["one", "two"], "query")
        self.assertEqual(list(vectors[0]), E0)
        self.assertEqual(list(vectors[1]), E1)

    def test_instruction_budget_blocks_before_transport(self):
        # One query itself would fit; its per-input instruction reservation does not.
        ledger = self.ledger(tokens=100)
        calls = []
        client = self.client(ledger, lambda *args: calls.append(args))
        with self.assertRaises(EmbeddingError):
            client.encode(["q", "q"], "query", modes.INSTRUCT)
        self.assertEqual(calls, [])
        self.assertTrue(ledger.stopped)
        self.assertEqual(ledger.state["attempted_requests"], 0)

    def test_every_http_or_transport_failure_locks_successor_without_retry(self):
        for failure in ("status", "exception"):
            with self.subTest(failure=failure):
                ledger = self.ledger()
                calls = []

                def transport(request, timeout, limit):
                    calls.append(1)
                    if failure == "exception":
                        raise HTTPError(request.full_url, 429, SECRET, {}, io.BytesIO(b"private response"))
                    return 429, b"private response"

                client = self.client(ledger, transport)
                for _ in range(2):
                    with self.assertRaises(EmbeddingError):
                        client.encode(["q"], "query")
                self.assertEqual(len(calls), 1)
                self.assertEqual(ledger.state["failed_requests"], 1)
                self.assertTrue(ledger.stopped)
                self.assertNotIn(SECRET, ledger.path.read_text())
                self.assertNotIn("private response", ledger.path.read_text())

    def test_invalid_native_responses_lock_successor_and_keep_full_reservation_if_usage_unknown(self):
        cases = ("missing_usage", "boolean_usage", "zero_usage", "negative_usage",
                 "wrong_dimension", "nonfinite", "duplicate_index", "missing_index", "noninteger_index")
        for case in cases:
            with self.subTest(case=case):
                ledger = self.ledger()
                calls = []

                def transport(request, timeout, limit):
                    calls.append(1)
                    response = native_response(request)
                    rows = response["output"]["embeddings"]
                    if case == "missing_usage":
                        response.pop("usage")
                    elif case == "boolean_usage":
                        response["usage"]["total_tokens"] = True
                    elif case == "zero_usage":
                        response["usage"]["total_tokens"] = 0
                    elif case == "negative_usage":
                        response["usage"]["total_tokens"] = -1
                    elif case == "wrong_dimension":
                        rows[0]["embedding"] = [1.0]
                    elif case == "nonfinite":
                        rows[0]["embedding"][0] = float("nan")
                    elif case == "duplicate_index":
                        rows[1]["text_index"] = 0
                    elif case == "missing_index":
                        rows.pop()
                    elif case == "noninteger_index":
                        rows[0]["text_index"] = True
                    return 200, json.dumps(response).encode()

                client = self.client(ledger, transport)
                for _ in range(2):
                    with self.assertRaises(EmbeddingError):
                        client.encode(["one", "two"], "query")
                self.assertEqual(len(calls), 1)
                self.assertTrue(ledger.stopped)
                self.assertEqual(ledger.state["failed_requests"], 1)
                if case in {"missing_usage", "boolean_usage", "zero_usage", "negative_usage"}:
                    self.assertEqual(ledger.state["accounted_input_tokens"], 70)

    def test_usage_over_reservation_locks_second_request(self):
        ledger = self.ledger()
        calls = []

        def transport(request, timeout, limit):
            calls.append(1)
            return 200, json.dumps(native_response(request, usage=500)).encode()

        client = self.client(ledger, transport)
        for _ in range(2):
            with self.assertRaises(EmbeddingError):
                client.encode(["q"], "query")
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger.state["accounted_input_tokens"], 500)
        self.assertEqual(ledger.state["stop_reason"], "usage_exceeds_reservation")

    def test_invalid_fixed_inputs_stop_without_transport(self):
        cases = [([], "query", ""), (["q"] * 11, "query", ""),
                 (["q"], "document", modes.INSTRUCT), (["q"], "query", "another instruction"),
                 (["q"], "compatible", ""), (["   "], "query", "")]
        for texts, text_type, instruction in cases:
            with self.subTest(text_type=text_type, input_count=len(texts), instruction=instruction):
                ledger = self.ledger()
                calls = []
                client = self.client(ledger, lambda *arguments: calls.append(1))
                with self.assertRaises(EmbeddingError):
                    client.encode(texts, text_type, instruction)
                with self.assertRaises(EmbeddingError):
                    client.encode(["valid successor"], "query")
                self.assertEqual(calls, [])
                self.assertTrue(ledger.stopped)
                self.assertEqual(ledger.state["attempted_requests"], 0)

    def test_closed_client_discards_key_and_never_starts_transport(self):
        calls = []
        client = self.client(self.ledger(), lambda *arguments: calls.append(1))
        client.close()
        self.assertEqual(client.key, "")
        with self.assertRaises(EmbeddingError):
            client.encode(["question"], "query")
        self.assertEqual(calls, [])

    def test_query_prefetch_deduplicates_segments_and_normalizes_their_mean(self):
        ledger = self.ledger()
        calls = []

        def transport(request, timeout, limit):
            payload = json.loads(request.data)
            calls.append(payload)
            response = native_response(request)
            response["output"]["embeddings"] = [
                {"text_index": 1, "embedding": E1}, {"text_index": 0, "embedding": E0}]
            return 200, json.dumps(response).encode()

        source = {"unique_segments": ["one", "two"],
                  "query_segments": {"original long question": ["one", "two"],
                                     "overlapping original question": ["one"]}}
        cache = modes.prefetch_queries(self.client(ledger, transport), source, modes.INSTRUCT)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["input"]["texts"], ["one", "two"])
        self.assertEqual(list(cache["overlapping original question"]), E0)
        self.assertAlmostEqual(cache["original long question"][0], 1 / math.sqrt(2), places=7)
        self.assertAlmostEqual(cache["original long question"][1], 1 / math.sqrt(2), places=7)
        self.assertAlmostEqual(sum(value * value for value in cache["original long question"]), 1.0)

    def test_native_route_rejects_another_origin_or_credential_bearing_url(self):
        for endpoint in (NATIVE.replace("synthetic.", "other."),
                         NATIVE.replace("https://", "http://"), NATIVE + "?secret=key",
                         NATIVE + "#fragment", NATIVE.replace("https://", "https://key@"),
                         "https://agentmemoryleaderboard.ai" + modes.NATIVE_PATH):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                modes.validate_native_endpoint(COMPATIBLE, endpoint)


class CachedBackendTests(unittest.TestCase):
    def test_only_source_fingerprint_is_used_for_frozen_vectors(self):
        backend = modes.CachedQueryBackend("source-corpus-fingerprint", {"question": E0}, "query-variant-hash")
        self.assertEqual(backend.dimension, 2048)
        self.assertEqual(backend.fingerprint, "source-corpus-fingerprint")
        self.assertEqual(list(backend.embed_query("question")), E0)

    def test_unknown_queries_and_all_documents_are_rejected(self):
        backend = modes.CachedQueryBackend("source", {"question": E0}, "variant")
        with self.assertRaises(EmbeddingError):
            backend.embed_query(" question")
        with self.assertRaises(EmbeddingError):
            backend.embed_query("unknown question")
        for documents in (["history"], []):
            with self.assertRaises(EmbeddingError):
                backend.embed_documents(documents)


class DatabaseTests(unittest.TestCase):
    def test_readonly_closed_wal_header_copy_requires_no_sidecar_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sqlite3"
            copied = Path(directory) / "copied.sqlite3"
            connection = sqlite3.connect(source)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE sentinel (value TEXT)")
            connection.execute("INSERT INTO sentinel VALUES ('synthetic')")
            connection.commit()
            connection.close()
            modes.backup_database(source, copied)
            before = file_hash(copied)
            self.assertFalse(Path(str(copied) + "-wal").exists())
            self.assertEqual(modes.database_identity(copied)["sentinel"]["rows"], 1)
            self.assertEqual(file_hash(copied), before)
            self.assertFalse(Path(str(copied) + "-wal").exists())
            self.assertFalse(Path(str(copied) + "-shm").exists())

    def test_readonly_database_includes_existing_uncheckpointed_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "with-wal.sqlite3"
            copied = Path(directory) / "wal-snapshot.sqlite3"
            connection = sqlite3.connect(source)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA wal_autocheckpoint=0")
                connection.execute("CREATE TABLE sentinel (value TEXT)")
                connection.execute("INSERT INTO sentinel VALUES ('synthetic')")
                connection.commit()
                self.assertTrue(Path(str(source) + "-wal").exists())
                source_identity = modes.database_identity(source)
                self.assertEqual(source_identity["sentinel"]["rows"], 1)
                before = file_hash(source)
                modes.backup_database(source, copied)
                self.assertEqual(modes.database_identity(copied), source_identity)
                self.assertEqual(file_hash(source), before)
                self.assertTrue(Path(str(source) + "-wal").exists())
            finally:
                connection.close()


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def fixture(self):
        """Build a complete baseline using invented data and local transport."""
        data_root = self.root / "data"
        folder = data_root / "prepared" / "locomo_refined" / "dev"
        folder.mkdir(parents=True)
        history = {"sample_id": "history", "sessions": [{"session_id": "session", "messages": [
            {"role": "user", "content": "HISTORY_ONLY_" + str(index)
                + (" Alice moved to Paris on 2024-02-29; number 13." if index % 2 == 0
                   else " 中文记录：阿丽丝喜欢蓝色，编号十四。")}
            for index in range(8)]}]}
        queries = [{"sample_id": "history", "query_id": "q" + str(index),
                    "query": "QUERY_ONLY_" + str(index) + " Where did Alice move?"}
                   for index in range(2)]
        gold = [{"sample_id": "history", "query_id": query["query_id"],
                 "original_query_id": "native-" + query["query_id"], "category": "synthetic",
                 "answer": "GOLD_PRIVATE_SENTINEL", "gold_session_ids": ["session"],
                 "gold_units": [{"unit_id": "gold-" + query["query_id"], "session_id": "session",
                    "message_position": 0, "text": "Alice moved to Paris",
                    "source_message_content": history["sessions"][0]["messages"][0]["content"]}]}
                for query in queries]
        for name, rows in (("histories", [history]), ("queries", queries), ("gold", gold)):
            (folder / (name + ".jsonl")).write_text("".join(json_line(row) for row in rows))
        specification = {"histories": 1, "questions": 2, "files": {
            name: {"sha256": file_hash(folder / (name + ".jsonl"))}
            for name in ("histories", "queries", "gold")}}
        (data_root / "prepared" / "summary.json").write_text(json.dumps({
            "status": "public_proxy_not_official_aml", "source_mapping_version": SOURCE_MAPPING_VERSION,
            "datasets": {"locomo_refined": {"splits": {"dev": specification}}}}))
        baseline_parent = self.root / "baseline-parent"
        baseline_parent.mkdir(mode=0o700)
        plan_args = SimpleNamespace(data_root=data_root, max_histories=1, max_queries=0,
                                   locomo_max_histories=1, lme_max_histories=1)
        baseline_plan = budget.source_plan(plan_args, "locomo_refined", permitted_roots=[data_root])
        parent_plan = {"status": "public_dev_plan_no_model_calls", "model": budget.MODEL,
            "dimension": 2048, "batch_size": 10, "concurrency": 1, "retries": 0,
            "segment_bytes": 1536, "endpoint": COMPATIBLE, "max_input_tokens": 100000,
            "max_requests": 100, "max_runtime_seconds": 30, "datasets": [baseline_plan],
            "credential_file_read": False}
        (baseline_parent / "plan.json").write_text(json.dumps(parent_plan))
        ledger = budget.BudgetLedger(baseline_parent / "fixture-ledger.json", 100000, 100, 30)
        configuration = EmbeddingConfig(provider="http", model=budget.MODEL, dimension=2048,
            batch_size=10, concurrency=1, http_retries=0, endpoint=COMPATIBLE, api_key=SECRET,
            allow_http=True, http_segment_bytes=1536)
        baseline = baseline_parent / "locomo_refined-v4"
        with patch("builtins.print"):
            budget.run_side(plan_args, baseline_plan, baseline, "frozen-run-id",
                budget.BudgetedV4Backend(configuration, ledger, json_transport))
        args = SimpleNamespace(data_root=data_root, source_run=baseline,
            model_env_file=self.root / "never-read.env",
            endpoint=NATIVE, api_key_name="MEMORY_EMBEDDING_API_KEY",
            max_input_tokens=100000, max_requests=100, max_runtime_seconds=30,
            http_timeout=30, output_dir=self.root / "output", execute=False)
        return args, folder

    def inspect(self, args):
        return modes.inspect_source(args, expected_questions=2, expected_histories=1)

    def test_inspection_never_opens_gold_and_preserves_exact_source_namespace(self):
        args, folder = self.fixture()
        original_open = Path.open

        def no_gold(path, *arguments, **keywords):
            if path.resolve() == (folder / "gold.jsonl").resolve():
                raise AssertionError("Inspection must not consume gold")
            return original_open(path, *arguments, **keywords)

        database_hash = file_hash(args.source_run / "memory.sqlite3")
        with patch.object(Path, "open", new=no_gold):
            source = self.inspect(args)
        self.assertEqual(source["users"], {"history": "proxy/frozen-run-id/history"})
        self.assertEqual(len(source["probes"]), 8)
        self.assertEqual(source["plan"]["questions"], 2)
        self.assertEqual(source["plan"]["histories"], 1)
        self.assertEqual(source["plan"]["planned_model_requests"], 3)
        self.assertEqual(source["plan"]["add_requests"], 0)
        self.assertEqual(source["plan"]["source_run_id"], "frozen-run-id")
        self.assertFalse(source["plan"]["full_capacity_validated"])
        self.assertTrue(all(set(query) == {"query_id", "sample_id", "query"}
                            for query in source["queries"]))
        self.assertEqual(file_hash(args.source_run / "memory.sqlite3"), database_hash)
        self.assertFalse(args.model_env_file.exists())

    def test_inspection_rejects_vectors_with_inexact_user_namespace(self):
        args, _ = self.fixture()
        connection = sqlite3.connect(args.source_run / "memory.sqlite3")
        try:
            connection.execute("UPDATE vectors SET user_id='history'")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(ValueError):
            self.inspect(args)

    def test_complete_mock_replay_reuses_documents_and_reads_gold_only_after_both_arms(self):
        args, prepared = self.fixture()
        source = self.inspect(args)
        before = {name: file_hash(args.source_run / name)
                  for name in ("memory.sqlite3", "summary.json", "retrieval.jsonl")}
        args.output_dir.mkdir(mode=0o700)
        calls, gold_reads = [], []
        original_open = Path.open

        def transport(request, timeout, limit):
            payload = json.loads(request.data)
            calls.append(payload)
            self.assertNotIn("GOLD_PRIVATE_SENTINEL", json.dumps(payload))
            self.assertTrue(all("GOLD" not in text for text in payload["input"]["texts"]))
            self.assertTrue(all("proxy/" not in text for text in payload["input"]["texts"]))
            return 200, json.dumps(native_response(request, usage=7)).encode()

        def audit_gold(path, *arguments, **keywords):
            if path.resolve() == (prepared / "gold.jsonl").resolve():
                gold_reads.append(1)
                self.assertEqual(len(calls), 3)
                ledger = json.loads((args.output_dir / "ledger.json").read_text())
                self.assertTrue(ledger["model_call_phase_closed"])
                for variant in ("B", "C"):
                    rows = modes.read_jsonl(args.output_dir / ("native-query-" + variant) / "retrieval.jsonl")
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(row["status"] == "ok" for row in rows))
            return original_open(path, *arguments, **keywords)

        def fake_key(path, endpoint, name):
            self.assertEqual(path, args.model_env_file)
            self.assertEqual(endpoint, COMPATIBLE)
            self.assertEqual(name, "MEMORY_EMBEDDING_API_KEY")
            return SECRET

        with patch.object(Path, "open", new=audit_gold), \
             patch.object(MemoryStore, "add", side_effect=AssertionError("Replay must not Add")), \
             patch.object(modes.CachedQueryBackend, "embed_documents",
                          side_effect=AssertionError("Replay must not re-embed documents")):
            report = modes.run_experiment(args, source, args.output_dir,
                                          transport=transport, key_reader=fake_key)
        self.assertEqual(report["status"], "completed_query_only_public_dev")
        self.assertEqual(report["budget"]["attempted_requests"], 3)
        self.assertEqual(report["budget"]["accounted_input_tokens"], 21)
        self.assertEqual([payload["parameters"]["text_type"] for payload in calls],
                         ["document", "query", "query"])
        self.assertEqual(len(calls[0]["input"]["texts"]), 8)
        self.assertTrue(all(text.startswith("HISTORY_ONLY_") for text in calls[0]["input"]["texts"]))
        self.assertTrue(all(text.startswith("QUERY_ONLY_") for payload in calls[1:]
                            for text in payload["input"]["texts"]))
        self.assertNotIn("instruct", calls[1]["parameters"])
        self.assertEqual(calls[2]["parameters"]["instruct"], modes.INSTRUCT)
        self.assertTrue(gold_reads)
        self.assertEqual({name: file_hash(args.source_run / name) for name in before}, before)
        self.assertFalse(args.model_env_file.exists())
        self.assertNotIn(SECRET, (args.output_dir / "ledger.json").read_text())
        comparison = json.loads((args.output_dir / "comparison.json").read_text())
        self.assertEqual(comparison["matched_questions"], 2)
        self.assertEqual(len(comparison["runs"]), 3)
        self.assertEqual([run["variant"] for run in report["runs"]], ["B", "C"])
        for run in report["runs"]:
            self.assertEqual(run["planned"], 2)
            self.assertEqual(run["completed"], 2)
            self.assertEqual(run["successful"], 2)
            self.assertEqual(run["failed"], 0)
            self.assertEqual(run["unfinished"], 0)
        for variant in ("B", "C"):
            arm = args.output_dir / ("native-query-" + variant)
            summary = json.loads((arm / "summary.json").read_text())
            self.assertEqual(summary["ingestion_run_id"], "frozen-run-id")
            self.assertEqual(summary["question_count"], 2)
            self.assertEqual(summary["add_request_count"], 0)
            self.assertEqual(summary["configuration"]["source_corpus_fingerprint"],
                             source["plan"]["source_corpus_fingerprint"])
            self.assertEqual(modes.database_identity(arm / "memory.sqlite3"),
                             source["plan"]["source_database_identity"])
            self.assertEqual((arm / "memory.sqlite3").stat().st_mode & 0o777, 0o600)

    def test_failed_document_gate_stops_before_query_replay_and_gold(self):
        args, prepared = self.fixture()
        source = self.inspect(args)
        args.output_dir.mkdir(mode=0o700)
        calls = []
        original_open = Path.open

        def transport(request, timeout, limit):
            calls.append(json.loads(request.data))
            return 200, json.dumps(native_response(request, vector=E1)).encode()

        def no_gold(path, *arguments, **keywords):
            if path.resolve() == (prepared / "gold.jsonl").resolve():
                raise AssertionError("A failed document gate must never read gold")
            return original_open(path, *arguments, **keywords)

        with patch.object(Path, "open", new=no_gold), \
             patch.object(modes, "replay", side_effect=AssertionError("Gate failure must not replay")), \
             patch.object(modes, "postprocess_compare", side_effect=AssertionError("No scoring after gate failure")):
            report = modes.run_experiment(args, source, args.output_dir,
                transport=transport, key_reader=lambda *arguments: SECRET)
        self.assertEqual(report["status"], "stopped_query_only_public_dev")
        self.assertEqual(report["budget"]["stop_reason"], "document_equivalence_failure")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["parameters"]["text_type"], "document")
        self.assertFalse(report["document_gate"]["passed"])
        self.assertEqual(report["runs"], [{"variant": variant, "planned": 2, "completed": 0,
            "successful": 0, "failed": 0, "unfinished": 2} for variant in ("B", "C")])
        self.assertFalse((args.output_dir / "comparison.json").exists())

    def test_query_failure_stops_c_without_replay_or_gold(self):
        args, _ = self.fixture()
        source = self.inspect(args)
        args.output_dir.mkdir(mode=0o700)
        calls = []

        def transport(request, timeout, limit):
            payload = json.loads(request.data)
            calls.append(payload)
            if payload["parameters"]["text_type"] == "query":
                return 429, b"private provider response"
            return 200, json.dumps(native_response(request)).encode()

        with patch.object(modes, "replay", side_effect=AssertionError("No replay after query failure")), \
             patch.object(modes, "postprocess_compare", side_effect=AssertionError("No gold after query failure")):
            report = modes.run_experiment(args, source, args.output_dir,
                transport=transport, key_reader=lambda *arguments: SECRET)
        self.assertEqual(report["status"], "stopped_query_only_public_dev")
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["budget"]["attempted_requests"], 2)
        self.assertEqual(report["budget"]["stop_reason"], "http_429")
        self.assertEqual(report["runs"], [{"variant": variant, "planned": 2, "completed": 0,
            "successful": 0, "failed": 0, "unfinished": 2} for variant in ("B", "C")])

    def test_default_dry_main_reads_no_key_and_starts_no_transport(self):
        args, _ = self.fixture()
        source = self.inspect(args)
        with patch.object(modes, "PRIVATE_ROOT", self.root), \
             patch.object(modes, "inspect_source", return_value=source), \
             patch.object(modes, "read_model_key", side_effect=AssertionError("Dry plan must not read a key")), \
             patch.object(modes, "https_transport", side_effect=AssertionError("Dry plan must not send HTTP")), \
             patch.object(modes, "run_experiment", side_effect=AssertionError("Dry plan must not execute")), \
             patch("builtins.print"):
            code = modes.main(["--source-run", str(args.source_run), "--data-root", str(args.data_root),
                "--model-env-file", str(args.model_env_file), "--endpoint", args.endpoint,
                "--max-input-tokens", "100000", "--max-requests", "100", "--output-dir", str(args.output_dir)])
        self.assertEqual(code, 0)
        plan = json.loads((args.output_dir / "plan.json").read_text())
        self.assertEqual(plan["status"], "query_modes_dry_plan_no_model_calls")
        self.assertFalse(plan["credential_file_read"])
        self.assertFalse(args.model_env_file.exists())
        self.assertFalse((args.output_dir / "ledger.json").exists())
        self.assertFalse((args.output_dir / "source-snapshot.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
