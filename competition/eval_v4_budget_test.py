#!/usr/bin/env python3
"""Synthetic offline tests: no real credential file or external API is used."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import eval_v4_budget as v4
from eval_prepare import SOURCE_MAPPING_VERSION, file_hash, json_line
from semantic import EmbeddingConfig, EmbeddingError


def fake_response(request, usage=3, model=v4.MODEL):
    payload = json.loads(request.data)
    vector = [1.0] + [0.0] * (v4.DIMENSION - 1)
    return {"model": model, "data": [{"index": i, "embedding": vector} for i in range(len(payload["input"]))],
            "usage": {"prompt_tokens": usage, "total_tokens": usage}}


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def ledger(self, tokens=10000, requests=20, runtime=30, clock=time.monotonic):
        return v4.BudgetLedger(self.root / ("ledger-" + str(len(list(self.root.glob('ledger-*')))) + ".json"), tokens, requests, runtime, clock)

    def backend(self, ledger, transport):
        config = EmbeddingConfig(provider="http", model=v4.MODEL, dimension=2048, batch_size=10,
                                 concurrency=1, http_retries=0, endpoint="https://synthetic.invalid/v1/embeddings",
                                 api_key="SYNTHETIC_SECRET", allow_http=True)
        return v4.BudgetedV4Backend(config, ledger, transport)

    def test_unicode_reservation_and_actual_usage_reclaims_unused_budget(self):
        ledger = self.ledger(tokens=75, requests=2)
        first = ledger.begin(["汉"], "hash")
        self.assertEqual(ledger.state["calls"][0]["reserved_input_tokens"], 35)
        ledger.finish(first, "ok", {"input_tokens": 7, "total_tokens": 7})
        second = ledger.begin(["a" * 36], "hash")  # 68 + actual 7 = limit 75.
        ledger.finish(second, "ok", {"input_tokens": 2, "total_tokens": 2})
        with self.assertRaises(v4.BudgetStopped):
            ledger.begin(["x"], "hash")
        saved = json.loads(ledger.path.read_text())
        self.assertEqual(saved["accounted_input_tokens"], 9)
        self.assertEqual(saved["attempted_requests"], 2)
        self.assertEqual(saved["stop_reason"], "request_limit")
        self.assertEqual(ledger.path.stat().st_mode & 0o777, 0o600)

    def test_reservation_blocks_before_any_transport(self):
        ledger = self.ledger(tokens=32)
        calls = []
        backend = self.backend(ledger, lambda *args: calls.append(args))
        with self.assertRaises(EmbeddingError):
            backend.embed_query("x")
        self.assertEqual(calls, [])
        self.assertEqual(ledger.state["attempted_requests"], 0)
        self.assertEqual(ledger.state["stop_reason"], "token_reservation_limit")

    def test_success_uses_fixed_model_and_durable_reservation_at_boundary(self):
        ledger = self.ledger()
        seen = []
        def transport(request, timeout, limit):
            saved = json.loads(ledger.path.read_text())
            self.assertEqual(saved["calls"][0]["status"], "pending")
            payload = json.loads(request.data)
            seen.append(payload)
            self.assertEqual(payload["model"], v4.MODEL)
            self.assertEqual(payload["dimensions"], 2048)
            self.assertEqual(payload["input"], ["HISTORY_ONLY"])
            return 200, json.dumps(fake_response(request)).encode()
        result = self.backend(ledger, transport).embed_documents(["HISTORY_ONLY"])
        self.assertEqual(len(result[0][0]), 2048)
        self.assertEqual(len(seen), 1)
        self.assertEqual(ledger.state["reported_input_tokens"], 3)
        self.assertNotIn("HISTORY_ONLY", ledger.path.read_text())
        self.assertNotIn("SYNTHETIC_SECRET", ledger.path.read_text())

    def test_429_is_terminal_and_never_retried(self):
        ledger = self.ledger()
        calls = []
        def transport(request, timeout, limit):
            calls.append(1)
            raise HTTPError(request.full_url, 429, "SYNTHETIC_SECRET must not be logged", {}, io.BytesIO(b"private body"))
        backend = self.backend(ledger, transport)
        for _ in range(2):
            with self.assertRaises(EmbeddingError):
                backend.embed_query("x")
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger.state["stop_reason"], "http_429")
        self.assertEqual(ledger.state["failed_requests"], 1)
        self.assertEqual(ledger.state["accounted_input_tokens"], 33)
        self.assertNotIn("SYNTHETIC_SECRET", ledger.path.read_text())

    def test_missing_usage_stops_and_accounts_full_reservation(self):
        ledger = self.ledger()
        def transport(request, timeout, limit):
            response = fake_response(request)
            response.pop("usage")
            return 200, json.dumps(response).encode()
        with self.assertRaises(EmbeddingError):
            self.backend(ledger, transport).embed_query("abc")
        self.assertEqual(ledger.state["stop_reason"], "missing_usage")
        self.assertEqual(ledger.state["unknown_usage_requests"], 1)
        self.assertEqual(ledger.state["accounted_input_tokens"], 35)

    def test_wrong_response_model_stops_even_with_valid_vectors(self):
        ledger = self.ledger()
        def transport(request, timeout, limit):
            return 200, json.dumps(fake_response(request, model="different-model")).encode()
        with self.assertRaises(EmbeddingError):
            self.backend(ledger, transport).embed_query("abc")
        self.assertEqual(ledger.state["stop_reason"], "response_model_mismatch")
        self.assertEqual(ledger.state["reported_input_tokens"], 3)

    def test_usage_exceeding_reservation_stops_before_successor(self):
        ledger = self.ledger(tokens=1000)
        calls = []
        def transport(request, timeout, limit):
            calls.append(1)
            return 200, json.dumps(fake_response(request, usage=500)).encode()
        backend = self.backend(ledger, transport)
        for _ in range(2):
            with self.assertRaises(EmbeddingError):
                backend.embed_query("x")
        self.assertEqual(len(calls), 1)
        self.assertEqual(ledger.state["accounted_input_tokens"], 500)
        self.assertEqual(ledger.state["stop_reason"], "usage_exceeds_reservation")

    def test_malformed_vectors_keep_known_usage_and_lock_calls(self):
        ledger = self.ledger()
        def transport(request, timeout, limit):
            response = fake_response(request)
            response["data"][0]["embedding"] = [1.0]
            return 200, json.dumps(response).encode()
        with self.assertRaises(EmbeddingError):
            self.backend(ledger, transport).embed_query("x")
        self.assertEqual(ledger.state["reported_input_tokens"], 3)
        self.assertEqual(ledger.state["failed_requests"], 1)
        self.assertTrue(ledger.stopped)

    def test_hard_deadline_returns_even_if_transport_ignores_timeout(self):
        ledger = self.ledger(runtime=.03)
        release = threading.Event()
        calls = []
        def transport(request, timeout, limit):
            calls.append(1)
            release.wait(1)
            return 200, json.dumps(fake_response(request)).encode()
        started = time.monotonic()
        try:
            with self.assertRaises(EmbeddingError):
                self.backend(ledger, transport).embed_query("x")
            self.assertLess(time.monotonic() - started, .5)
            self.assertEqual(ledger.state["stop_reason"], "runtime_limit")
            self.assertEqual(ledger.state["accounted_input_tokens"], 33)
            self.assertEqual(len(calls), 1)
        finally:
            release.set()

    def test_deadline_expiring_during_reservation_never_starts_transport(self):
        now = [0.0]
        ledger = self.ledger(runtime=10, clock=lambda: now[0])
        original_save = ledger._save
        def slow_save():
            original_save()
            if ledger.state["calls"]:
                now[0] = 11.0
        calls = []
        with patch.object(ledger, "_save", side_effect=slow_save):
            with self.assertRaises(EmbeddingError):
                self.backend(ledger, lambda *args: calls.append(1)).embed_query("x")
        self.assertEqual(calls, [])
        self.assertEqual(ledger.state["attempted_requests"], 1)
        self.assertEqual(ledger.state["accounted_input_tokens"], 33)
        self.assertFalse(ledger.state["calls"][0]["transport_started"])

    def test_query_segment_capacity_failure_blocks_without_transport(self):
        ledger = self.ledger()
        calls = []
        with self.assertRaises(EmbeddingError):
            self.backend(ledger, lambda *args: calls.append(1)).embed_query("x" * (1536 * 9))
        self.assertEqual(calls, [])
        self.assertTrue(ledger.stopped)


class RunnerTests(unittest.TestCase):
    def fixture(self, base):
        root = base / "data"
        folder = root / "prepared" / "locomo_refined" / "dev"
        folder.mkdir(parents=True)
        history_text = "HISTORY_SENTINEL Alice moved to Paris."
        history = {"sample_id": "history", "sessions": [{"session_id": "session", "messages": [
            {"role": "user", "content": history_text}]}]}
        query = {"sample_id": "history", "query_id": "query", "query": "QUERY_SENTINEL Where did Alice move?"}
        gold = {"sample_id": "history", "query_id": "query", "original_query_id": "native",
                "category": "synthetic", "answer": "GOLD_PRIVATE_SENTINEL", "gold_session_ids": ["session"],
                "gold_units": [{"unit_id": "gold", "session_id": "session", "message_position": 0,
                                "text": "Alice moved to Paris.", "source_message_content": history_text}]}
        for name, row in (("histories", history), ("queries", query), ("gold", gold)):
            (folder / (name + ".jsonl")).write_text(json_line(row))
        spec = {"histories": 1, "questions": 1, "files": {name: {"sha256": file_hash(folder / (name + ".jsonl"))}
                    for name in ("histories", "queries", "gold")}}
        (root / "prepared" / "summary.json").write_text(json.dumps({"status": "public_proxy_not_official_aml",
            "source_mapping_version": SOURCE_MAPPING_VERSION, "datasets": {"locomo_refined": {"splits": {"dev": spec}}}}))
        args = SimpleNamespace(data_root=root, max_histories=None, max_queries=0, locomo_max_histories=6,
                               lme_max_histories=5, max_input_tokens=1000, max_requests=20,
                               max_runtime_seconds=30, model_env_file=base / "never-read.env",
                               endpoint="https://synthetic.invalid/v1/embeddings", api_key_name="MEMORY_EMBEDDING_API_KEY")
        return args

    def test_complete_mock_loopback_run_compares_all_questions_without_gold_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = self.fixture(base)
            plan = v4.source_plan(args, "locomo_refined", permitted_roots=[args.data_root])
            output = base / "output"
            output.mkdir(mode=0o700)
            inputs = []
            def transport(request, timeout, limit):
                inputs.extend(json.loads(request.data)["input"])
                return 200, json.dumps(fake_response(request)).encode()
            old_mask = os.umask(0o077)
            try:
                with patch("builtins.print"):
                    report = v4.execute(args, [plan], output, transport=transport,
                                        key_reader=lambda *args: "SYNTHETIC_SECRET")
            finally:
                os.umask(old_mask)
            self.assertEqual(report["status"], "completed_public_dev_proxy")
            self.assertEqual(report["budget"]["attempted_requests"], 2)
            self.assertEqual(inputs, ["HISTORY_SENTINEL Alice moved to Paris.", "QUERY_SENTINEL Where did Alice move?"])
            self.assertFalse(args.model_env_file.exists())
            comparison = json.loads((output / "locomo_refined-comparison.json").read_text())
            self.assertEqual(comparison["matched_questions"], 1)
            self.assertEqual(comparison["runs"][1]["metrics"]["k100/characters0"]["full_turn_recall_all"], 1)
            self.assertNotIn("SYNTHETIC_SECRET", (output / "ledger.json").read_text())
            summary = json.loads((output / "locomo_refined-v4/summary.json").read_text())
            self.assertEqual(summary["configuration"]["core_source_sha256_at_start"], summary["configuration"]["core_source_sha256_at_end"])
            self.assertEqual((output / "locomo_refined-v4/memory.sqlite3").stat().st_mode & 0o777, 0o600)

    def test_failed_mock_run_keeps_denominator_and_stops_all_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = self.fixture(base)
            plan = v4.source_plan(args, "locomo_refined", permitted_roots=[args.data_root])
            output = base / "output"
            output.mkdir(mode=0o700)
            calls = []
            def transport(request, timeout, limit):
                calls.append(1)
                return 429, b"private quota response"
            with patch("builtins.print"):
                report = v4.execute(args, [plan], output, transport=transport, key_reader=lambda *args: "fake")
            self.assertEqual(report["status"], "stopped_public_dev_proxy")
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["budget"]["failed_requests"], 1)
            summary = json.loads((output / "locomo_refined-v4/summary.json").read_text())
            self.assertEqual(summary["question_count"], 1)
            self.assertEqual(summary["metrics_by_k"]["100"]["failed_questions"], 1)

    def test_source_guard_rejects_unapproved_root_before_opening_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(data_root=Path(directory) / "official-hidden-data")
            with self.assertRaises(ValueError):
                v4.source_plan(args, "locomo_refined")

    def test_dry_run_never_reads_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = self.fixture(base)
            planned = v4.source_plan(args, "locomo_refined", permitted_roots=[args.data_root])
            with patch.object(v4, "PRIVATE_ROOT", base), patch.object(v4, "source_plan", return_value=planned), \
                 patch.object(v4, "read_model_key", side_effect=AssertionError("Must not read credentials")), patch("builtins.print"):
                code = v4.main(["--dataset", "locomo_refined", "--model-env-file", str(args.model_env_file),
                    "--endpoint", args.endpoint, "--max-input-tokens", "750000", "--max-requests", "2000",
                    "--output-dir", str(base / "dry")])
            self.assertEqual(code, 0)
            self.assertFalse(args.model_env_file.exists())
            self.assertFalse((base / "dry/ledger.json").exists())

    def test_endpoint_rejects_http_credentials_query_and_official_evaluator(self):
        for endpoint in ("http://synthetic.invalid/embeddings", "https://key@synthetic.invalid/embeddings",
                         "https://synthetic.invalid/embeddings?key=secret", "https://agentmemoryleaderboard.ai/embeddings"):
            with self.assertRaises(ValueError):
                v4.validate_endpoint(endpoint)

    def test_synthetic_private_env_parser_rejects_symlink_and_unsafe_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            path = base / "fake.env"
            path.write_text("MEMORY_EMBEDDING_API_KEY='fake-literal-$()'\nMEMORY_EMBEDDING_MODEL=text-embedding-v4\n")
            path.chmod(0o600)
            self.assertEqual(v4.read_model_key(path, "https://synthetic.invalid/embeddings", "MEMORY_EMBEDDING_API_KEY"), "fake-literal-$()")
            link = base / "link.env"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                v4.read_model_key(link, "https://synthetic.invalid/embeddings", "MEMORY_EMBEDDING_API_KEY")
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                v4.read_model_key(path, "https://synthetic.invalid/embeddings", "MEMORY_EMBEDDING_API_KEY")


if __name__ == "__main__":
    unittest.main()
