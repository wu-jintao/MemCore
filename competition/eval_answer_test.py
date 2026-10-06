#!/usr/bin/env python3
"""Synthetic offline boundary tests; these are not benchmark results."""
import argparse
import io
import json
import os
from pathlib import Path
import ssl
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import eval_answer as evaluator


class CharacterEncoding:
    name = "synthetic-character-encoding"

    def encode(self, text, disallowed_special=()):
        return list(text)


def response(text="Paris", prompt=20, completion=5, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, model, messages, max_output, judge=False):
        self.calls.append((model, messages, max_output, judge))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


class AnswerProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.private = self.root / ".local"
        self.private.mkdir()
        self.encoding = CharacterEncoding()
        self.patch_private = patch.object(evaluator, "PRIVATE_ROOT", self.private)
        self.patch_private.start()
        self.data_root = self.private / "data"
        prepared = self.data_root / "prepared" / "locomo_refined" / "dev"
        prepared.mkdir(parents=True)
        self.queries = [{"sample_id": "history", "query_id": "q1", "query": "Where did Alice move?"},
                        {"sample_id": "history", "query_id": "q2", "query": "Which pet?", "options": ["A. Cat", "B. Dog"]}]
        self.gold = [{"sample_id": "history", "query_id": row["query_id"], "answer": "GOLD_ONLY_SENTINEL",
                      "gold_units": ["FORBIDDEN_EVIDENCE_SENTINEL"]} for row in self.queries]
        for name, rows in (("histories", [{"sample_id": "history", "messages": []}]),
                           ("queries", self.queries), ("gold", self.gold)):
            self.write_jsonl(prepared / (name + ".jsonl"), rows)
        self.spec = {"files": {name: {"sha256": evaluator.file_hash(prepared / (name + ".jsonl"))}
                    for name in ("histories", "queries", "gold")}}
        self.baseline = self.create_run("baseline")
        self.candidate = self.create_run("candidate", reverse=True)

    def tearDown(self):
        self.patch_private.stop()
        self.temp.cleanup()

    def write_jsonl(self, path, rows):
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def create_run(self, name, reverse=False):
        folder = self.root / "runs" / name
        folder.mkdir(parents=True)
        summary = {"run_id": name, "dataset": "locomo_refined", "split": "dev", "question_count": 2,
                   "server_source_changed_during_run": False,
                   "configuration": {"top_k": 100, "character_budget": 0, "prepared_data": self.spec,
                    "source_code_sha256": {"server.py_at_start": "1" * 64, "server.py_at_end": "1" * 64}}}
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        records = [{"sample_id": "history", "query_id": row["query_id"], "status": "ok",
                    "retrieved": [{"id": "source1", "content": "[source original]\nAlice moved to Paris.\u2028Complete source."},
                                  {"id": "source2", "content": "The pet is a cat."}]} for row in self.queries]
        self.write_jsonl(folder / "retrieval.jsonl", list(reversed(records)) if reverse else records)
        return folder

    def inputs(self):
        return evaluator.matched_inputs(self.baseline, self.candidate, self.data_root, 2)

    def jobs(self, settings=None):
        runs, queries, gold, _ = self.inputs()
        return evaluator.build_plan(runs, queries, gold, settings or evaluator.Settings(), self.encoding)[0]

    def args(self, extra=()):
        return evaluator.parser().parse_args(["--baseline-run", str(self.baseline), "--candidate-run", str(self.candidate),
            "--data-root", str(self.data_root), "--output-dir", str(self.private / "output"), *extra])

    def test_reject_non_dev_before_reading_retrieval(self):
        summary_path = self.candidate / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["split"] = "heldout"
        summary_path.write_text(json.dumps(summary))
        (self.candidate / "retrieval.jsonl").unlink()
        with patch.object(evaluator, "read_jsonl", side_effect=AssertionError("must not read raw records")):
            with self.assertRaisesRegex(evaluator.EvaluationError, "Only public dev"):
                evaluator.frozen_run(self.candidate)

    def test_fixed_query_order_exact_matching_and_unicode(self):
        runs, queries, _, _ = self.inputs()
        self.assertEqual(runs[0][0]["dataset"], "locomo_refined")
        self.assertEqual([row["query_id"] for row in queries], ["q1", "q2"])
        self.assertIn("\u2028", runs[0][1]["q1"]["retrieved"][0]["content"])
        rows = evaluator.read_jsonl(self.candidate / "retrieval.jsonl")
        rows[0]["query_id"] = "other"
        self.write_jsonl(self.candidate / "retrieval.jsonl", rows)
        with self.assertRaisesRegex(evaluator.EvaluationError, "exact same"):
            self.inputs()

    def test_source_sha_changes_are_rejected(self):
        path = self.data_root / "prepared" / "locomo_refined" / "dev" / "gold.jsonl"
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(evaluator.EvaluationError, "SHA"):
            self.inputs()

    def test_retrieval_symlink_target_is_rejected_without_open(self):
        sentinel = self.root / "validation" / "retrieval.jsonl"
        sentinel.parent.mkdir()
        sentinel.write_text("SYNTHETIC_FORBIDDEN_SPLIT_SENTINEL")
        records = self.candidate / "retrieval.jsonl"
        records.unlink()
        records.symlink_to(sentinel)
        with patch.object(evaluator, "read_jsonl", side_effect=AssertionError("must not open a symlink target")):
            with self.assertRaises(evaluator.EvaluationError):
                evaluator.frozen_run(self.candidate)

    def test_prepared_dev_directory_and_raw_file_links_are_rejected(self):
        # Exercise the final resolved directory itself, without opening bytes.
        dataset = self.data_root.resolve() / "prepared" / "locomo_refined"
        sentinel = self.root / "raw"
        sentinel.mkdir()
        (sentinel / "histories.jsonl").write_text("SYNTHETIC_RAW_SENTINEL")
        moved = dataset / "original-dev"
        (dataset / "dev").rename(moved)
        (dataset / "dev").symlink_to(sentinel, target_is_directory=True)
        original_hash = evaluator.file_hash
        def hash_guard(path):
            if path.name == "histories.jsonl":
                self.fail("must not hash a prepared symlink target")
            return original_hash(path)
        with patch.object(evaluator, "file_hash", side_effect=hash_guard):
            with self.assertRaises(evaluator.EvaluationError):
                self.inputs()

    def test_ingestion_summary_link_is_rejected_without_target_open(self):
        path = self.candidate / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(ingestion_run_id="alias", ingestion_provenance={})
        path.write_text(json.dumps(summary))
        forbidden = self.root / "heldout"
        forbidden.mkdir()
        (forbidden / "summary.json").write_text("SYNTHETIC_FORBIDDEN_INGESTION_SENTINEL")
        (self.candidate.parent / "alias").symlink_to(forbidden, target_is_directory=True)
        original_read = evaluator.read_json
        def read_guard(file):
            if file.parent.name == "alias":
                self.fail("must not open an ingestion symlink target")
            return original_read(file)
        with patch.object(evaluator, "read_json", side_effect=read_guard):
            with self.assertRaises(evaluator.EvaluationError):
                evaluator.frozen_run(self.candidate)

    def test_gold_only_judge_no_evidence_or_label_in_answer(self):
        job = self.jobs()[0]
        answer_prompt = json.dumps(job["messages"])
        self.assertNotIn("GOLD_ONLY_SENTINEL", answer_prompt)
        self.assertNotIn("FORBIDDEN_EVIDENCE_SENTINEL", answer_prompt)
        judge = json.dumps(evaluator.judge_messages(job["query"], "Candidate", job["reference"]))
        self.assertIn("GOLD_ONLY_SENTINEL", judge)
        self.assertNotIn("FORBIDDEN_EVIDENCE_SENTINEL", judge)

    def configure_longmemeval(self, dates=("2023/04/01 (Sat) 16:08", "2023/04/02 (Sun) 16:08")):
        original = self.data_root / "prepared" / "locomo_refined" / "dev"
        folder = self.data_root / "prepared" / "longmemeval_s" / "dev"
        folder.mkdir(parents=True, exist_ok=True)
        queries = [dict(row) for row in self.queries]
        queries[0]["query"] = "What did I do two months ago?"
        labels = [dict(row) for row in self.gold]
        for row, value in zip(labels, dates):
            row["category"] = "CATEGORY_ONLY_SENTINEL"
            if value is not None:
                row["question_date"] = value
        self.write_jsonl(folder / "histories.jsonl", evaluator.read_jsonl(original / "histories.jsonl"))
        self.write_jsonl(folder / "queries.jsonl", queries)
        self.write_jsonl(folder / "gold.jsonl", labels)
        spec = {"files": {name: {"sha256": evaluator.file_hash(folder / (name + ".jsonl"))}
                          for name in ("histories", "queries", "gold")}}
        for run in (self.baseline, self.candidate):
            path = run / "summary.json"
            summary = json.loads(path.read_text())
            summary["dataset"] = "longmemeval_s"
            summary["configuration"]["prepared_data"] = spec
            path.write_text(json.dumps(summary))
        return folder

    def test_longmemeval_public_date_enters_answer_and_judge_without_labels(self):
        folder = self.configure_longmemeval()
        _, queries, gold, _ = self.inputs()
        query = queries[0]
        self.assertEqual(query["question_date"], "2023/04/01 (Sat) 16:08")
        self.assertIn("two months ago", query["query"])
        answer = evaluator.answer_messages(query, "Historical public memory")
        serialized = json.dumps(answer)
        self.assertIn(query["question_date"], serialized)
        for sentinel in ("GOLD_ONLY_SENTINEL", "FORBIDDEN_EVIDENCE_SENTINEL", "CATEGORY_ONLY_SENTINEL"):
            self.assertNotIn(sentinel, serialized)
        judge = evaluator.judge_messages(query, "A candidate", gold[query["query_id"]]["answer"])
        judge_data = json.loads(judge[1]["content"])
        self.assertEqual(judge_data["question_data"], json.loads(evaluator.question_payload(query)))
        self.assertEqual(set(judge_data["question_data"]), {"question", "question_date"})
        self.assertIn("GOLD_ONLY_SENTINEL", judge_data["reference_answer"])
        self.assertNotIn("FORBIDDEN_EVIDENCE_SENTINEL", json.dumps(judge_data))
        self.assertNotIn("CATEGORY_ONLY_SENTINEL", json.dumps(judge_data))
        # Enrichment is in memory only; neither the prepared question file nor
        # its frozen SHA/provenance is rewritten.
        self.assertNotIn("question_date", evaluator.read_jsonl(folder / "queries.jsonl")[0])

    def test_longmemeval_date_manifest_and_dry_run_still_make_no_request(self):
        self.configure_longmemeval()
        with patch.object(evaluator, "load_key", side_effect=AssertionError("dry-run key read")):
            report = evaluator.run(self.args(), self.encoding, client_factory=lambda *_: self.fail("dry-run network"))
        metadata = report["packing"]["public_question_metadata"]
        self.assertTrue(metadata["question_date_required"])
        self.assertEqual(metadata["copied_count"], 2)
        self.assertEqual(metadata["copied_field_whitelist"], ["question_date"])
        self.assertIn("prepared gold", metadata["source"])
        self.assertEqual(len(metadata["metadata_sha256"]), 64)
        self.assertEqual(report["usage_and_cost"]["attempted_requests"], 0)
        self.assertNotIn("GOLD_ONLY_SENTINEL", json.dumps(report))

    def test_longmemeval_missing_invalid_or_inconsistent_date_rejected_before_execution(self):
        invalid = (None, 123, "2023/4/01 (Sat) 16:08", "2023/02/29 (Wed) 16:08",
                   "2023/04/31 (Mon) 16:08", "2023/04/01 (Fri) 16:08", "2023/04/01 (Sat) 25:08",
                   "0000/04/01 (Sat) 16:08", "2023/04/01 (Sat) 16:08\n")
        for date in invalid:
            with self.subTest(date=date):
                self.configure_longmemeval((date, "2023/04/02 (Sun) 16:08"))
                with patch.object(evaluator, "load_key", side_effect=AssertionError("invalid metadata key read")):
                    with self.assertRaises(evaluator.EvaluationError):
                        evaluator.run(self.args(["--execute", "--max-cost-usd", "1"]), self.encoding,
                                      client_factory=lambda *_: self.fail("invalid metadata network"))
        folder = self.configure_longmemeval()
        queries = evaluator.read_jsonl(folder / "queries.jsonl")
        queries[0]["question_date"] = "2023/04/02 (Sun) 16:08"
        self.write_jsonl(folder / "queries.jsonl", queries)
        for run in (self.baseline, self.candidate):
            path = run / "summary.json"
            summary = json.loads(path.read_text())
            summary["configuration"]["prepared_data"]["files"]["queries"]["sha256"] = evaluator.file_hash(folder / "queries.jsonl")
            path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(evaluator.EvaluationError, "differs"):
            self.inputs()

    def test_question_date_sha_verified_before_metadata_is_used(self):
        folder = self.configure_longmemeval()
        labels = evaluator.read_jsonl(folder / "gold.jsonl")
        labels[0]["question_date"] = "2023/04/02 (Sun) 16:08"
        self.write_jsonl(folder / "gold.jsonl", labels)
        with self.assertRaisesRegex(evaluator.EvaluationError, "SHA"):
            self.inputs()

    def test_locomo_question_payload_remains_without_date(self):
        _, queries, _, _ = self.inputs()
        self.assertNotIn("question_date", queries[0])
        self.assertEqual(json.loads(evaluator.question_payload(queries[0])), {"question": self.queries[0]["query"]})
        report = evaluator.run(self.args(), self.encoding)
        metadata = report["packing"]["public_question_metadata"]
        self.assertFalse(metadata["question_date_required"])
        self.assertEqual(metadata["copied_count"], 0)
        self.assertEqual(metadata["copied_field_whitelist"], [])

    def test_legacy_locomo_dataset_identifier_remains_supported(self):
        summary = json.loads((self.baseline / "summary.json").read_text())
        summary["dataset"] = "locomo"
        evaluator.validate_summary(summary)

    def test_pack_exact_whole_source_prefix_no_skip(self):
        items = [{"id": "a", "content": "A full source\n[source-shaped quote]"},
                 {"id": "b", "content": "B" * 400}, {"id": "c", "content": "tiny but later"}]
        first = "Memory 1:\n" + items[0]["content"] + "\n\n"
        messages, details = evaluator.pack_memories(self.queries[0], items, self.encoding, len(first), 5000, 64)
        self.assertEqual(details["selected_items"], 1)
        self.assertIn(first, messages[1]["content"])
        self.assertNotIn("tiny but later", messages[1]["content"])
        _, empty = evaluator.pack_memories(self.queries[0], items, self.encoding, len(first) - 1, 5000, 64)
        self.assertEqual(empty["selected_items"], 0)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.pack_memories(self.queries[0], items, self.encoding, 100, 20, 64)

    def test_dry_run_does_not_read_key_or_construct_http(self):
        args = self.args(["--api-key-file", str(self.root / "nonexistent-secret-key")])
        with patch.object(evaluator, "load_key", side_effect=AssertionError("dry-run key read")):
            report = evaluator.run(args, self.encoding, client_factory=lambda *_: self.fail("dry-run HTTP"))
        self.assertEqual(report["usage_and_cost"]["attempted_requests"], 0)
        self.assertIsNone(report["metrics"])
        self.assertEqual(report["matched_question_denominator"], 2)
        serialized = json.dumps(report)
        for secret in ("GOLD_ONLY_SENTINEL", "Where did Alice", "[source original]", str(self.root), '"q1"'):
            self.assertNotIn(secret, serialized)
        self.assertEqual((self.private / "output" / "summary.json").stat().st_mode & 0o777, 0o600)

    def test_execute_preflight_limits_before_key_and_network(self):
        for flags in (("--execute",), ("--execute", "--max-cost-usd", "0.000000001"),
                      ("--execute", "--max-cost-usd", "1", "--max-requests", "1")):
            with patch.object(evaluator, "load_key", side_effect=AssertionError("preflight key read")):
                with self.assertRaises(evaluator.EvaluationError):
                    evaluator.run(self.args(flags), self.encoding, client_factory=lambda *_: self.fail("preflight HTTP"))

    def test_custom_provider_requires_explicit_proxy_price_rates(self):
        args = self.args(["--execute", "--max-cost-usd", "1", "--endpoint", "https://example.invalid/v1/chat/completions"])
        with patch.object(evaluator, "load_key", side_effect=AssertionError("unpriced provider key read")):
            with self.assertRaisesRegex(evaluator.EvaluationError, "price rates"):
                evaluator.run(args, self.encoding)

    def test_retrieval_answer_judge_failures_stay_in_matched_denominator(self):
        jobs = self.jobs()
        jobs[0]["status"] = "retrieval_error"
        client = FakeClient([response(), response('{"correct":true,"reason":"Equivalent"}'),
            evaluator.ProviderError("http_429"), response(), response('{"correct":"true","reason":"bad bool"}')])
        rows, ledger = evaluator.execute_jobs(jobs, evaluator.Settings(), self.encoding, client, lambda _: None)
        metrics = evaluator.aggregate_results(rows, 2)
        self.assertEqual(metrics["baseline"]["denominator"], 2)
        self.assertEqual(metrics["baseline"]["failed_questions"], 2)
        self.assertEqual(metrics["candidate"]["correct"], 1)
        self.assertEqual(metrics["candidate"]["accuracy_including_failures"], .5)
        self.assertEqual(len(client.calls), 5)  # No automatic retries.
        self.assertEqual(sum(row["usage_status"] == "unknown" for row in ledger.calls), 1)
        self.assertGreater(ledger.calls[2]["accounted_cost_usd"], 0)

    def test_usage_excess_stops_further_requests(self):
        jobs = self.jobs()
        client = FakeClient([response(prompt=jobs[0]["answer_input_bound"] + 1)])
        rows, ledger = evaluator.execute_jobs(jobs, evaluator.Settings(), self.encoding, client, lambda _: None)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(ledger.stopped)
        self.assertEqual(len(rows), 4)
        self.assertEqual(evaluator.aggregate_results(rows, 2)["candidate"]["failed_questions"], 2)

    def test_missing_usage_reserves_full_cost(self):
        ledger = evaluator.Ledger(evaluator.Settings())
        row = ledger.reserve("answer", 1000, 512)
        reserved = ledger.accounted_usd
        ledger.settle(row, {"choices": []})
        self.assertEqual(ledger.accounted_usd, reserved)
        self.assertEqual(row["usage_status"], "unknown")

    def test_error_body_and_unexpected_exception_never_saved(self):
        jobs = self.jobs()
        client = FakeClient([{"error": {"message": "SECRET_KEY_PROVIDER_ERROR"}},
                             RuntimeError("SECRET_KEY_PROVIDER_ERROR"),
                             evaluator.ProviderError("http_500"), evaluator.ProviderError("http_500")])
        rows, _ = evaluator.execute_jobs(jobs, evaluator.Settings(), self.encoding, client, lambda _: None)
        self.assertNotIn("SECRET_KEY_PROVIDER_ERROR", json.dumps(rows))
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(client.calls), 4)

    def test_private_key_permissions_and_symlink_rejected(self):
        key_path = self.private / "key"
        key_path.write_text("SYNTHETIC_KEY")
        key_path.chmod(0o644)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.load_key("OPENAI_API_KEY", key_path)
        key_path.chmod(0o600)
        self.assertEqual(evaluator.load_key("OPENAI_API_KEY", key_path), "SYNTHETIC_KEY")
        link = self.private / "link"
        link.symlink_to(key_path)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.load_key("OPENAI_API_KEY", link)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.private_directory(self.root / "public-output")

    def test_verified_tls_no_redirects_no_provider_error_body_read(self):
        for endpoint in ("http://127.0.0.1/v1/chat/completions", "https://key@host/v1/chat/completions",
                         "https://host/v1/chat/completions?key=SECRET", "https://host/other"):
            with self.assertRaises(evaluator.EvaluationError):
                evaluator.validate_endpoint(endpoint)
        client = evaluator.ChatClient(evaluator.Settings(), "SYNTHETIC_KEY")
        handlers = client.opener.handlers
        context = next(row._context for row in handlers if isinstance(row, evaluator.urllib.request.HTTPSHandler))
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertIsNone(evaluator.NoRedirects().redirect_request(None, None, 302, "", {}, "https://other/"))
        body = io.BytesIO(b"SECRET_PROVIDER_BODY")
        with patch.object(body, "read", side_effect=AssertionError("Do not read error body")):
            error = urllib.error.HTTPError(evaluator.DEFAULT_ENDPOINT, 302, "SECRET_PROVIDER_REASON", {}, body)
            with patch.object(client.opener, "open", side_effect=error) as mocked:
                with self.assertRaisesRegex(evaluator.ProviderError, "redirect_blocked"):
                    client.complete(evaluator.DEFAULT_MODEL, [], 10)
            self.assertEqual(mocked.call_count, 1)

    def test_ingestion_namespace_sha_verified_for_search_only(self):
        summary_path = self.candidate / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary.update(ingestion_run_id="baseline", ingestion_provenance={
            "source_summary_sha256": evaluator.file_hash(self.baseline / "summary.json"),
            "source_retrieval_sha256": evaluator.file_hash(self.baseline / "retrieval.jsonl")})
        summary_path.write_text(json.dumps(summary))
        lineage = evaluator.frozen_run(self.candidate)[2]
        self.assertEqual(lineage["ingestion_run_id"], "baseline")
        self.assertEqual(lineage["ingestion_summary_sha256"], evaluator.file_hash(self.baseline / "summary.json"))
        summary["ingestion_provenance"]["source_summary_sha256"] = "f" * 64
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(evaluator.EvaluationError, "lineage SHA"):
            evaluator.frozen_run(self.candidate)

    def test_offline_tokenizer_fetch_blocked(self):
        try:
            import tiktoken
            import tiktoken.load
        except ImportError:
            self.skipTest("Official tokenizer optional in standard-library test environment")
        def implicit_download(_):
            return tiktoken.load.read_file("https://example.invalid/vocabulary")
        with patch.object(tiktoken, "get_encoding", side_effect=implicit_download):
            with self.assertRaisesRegex(evaluator.EvaluationError, "never downloads"):
                evaluator.offline_encoding("o200k_base")

    def local_settings(self, **extra):
        values = {"endpoint": "http://127.0.0.1:8000/v1/chat/completions", "local_loopback": True,
                  "input_usd_per_million": 0, "output_usd_per_million": 0,
                  "chat_template_kwargs": {"enable_thinking": False}}
        values.update(extra)
        return evaluator.Settings(**values)

    def test_literal_loopback_http_requires_explicit_local_mode(self):
        for endpoint in ("http://127.0.0.1:8000/v1/chat/completions",
                         "http://[::1]:8000/v1/chat/completions"):
            with self.assertRaises(evaluator.EvaluationError):
                evaluator.validate_endpoint(endpoint)
            self.assertEqual(evaluator.validate_endpoint(endpoint, True), endpoint)
        for endpoint in ("http://localhost:8000/v1/chat/completions", "http://127.0.0.2/v1/chat/completions",
                         "http://remote.example/v1/chat/completions", "https://remote.example/v1/chat/completions",
                         "http://user@127.0.0.1/v1/chat/completions", "http://127.0.0.1/v1/chat/completions?q=x",
                         "http://127.0.0.1/v1/chat/completions#x", "http://[::ffff:127.0.0.1]/v1/chat/completions"):
            with self.assertRaises(evaluator.EvaluationError):
                evaluator.validate_endpoint(endpoint, True)

    def test_local_zero_api_fee_does_not_relax_request_or_usage_limits(self):
        settings = self.local_settings(max_requests=1)
        settings.validate()
        ledger = evaluator.Ledger(settings)
        row = ledger.reserve("answer", 10, 10)
        self.assertEqual(ledger.accounted_usd, 0)
        with self.assertRaisesRegex(evaluator.ProviderError, "budget_exhausted"):
            ledger.reserve("judge", 10, 10)
        ledger.settle(row, response(prompt=11))
        self.assertTrue(ledger.stopped)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.Settings(input_usd_per_million=0, output_usd_per_million=0).validate()
        with self.assertRaises(evaluator.EvaluationError):
            self.local_settings(input_usd_per_million=.15).validate()

    def test_local_preflight_requires_tokenizer_and_explicit_execution_ceiling(self):
        local = ["--local-loopback", "--endpoint", "http://127.0.0.1/v1/chat/completions"]
        with self.assertRaisesRegex(evaluator.EvaluationError, "tokenizer path"):
            evaluator.run(self.args(local), self.encoding)
        with patch.object(evaluator, "load_key", side_effect=AssertionError("preflight key read")):
            with self.assertRaises(evaluator.EvaluationError):
                evaluator.run(self.args([*local, "--tokenizer-path", str(self.root), "--execute"]), self.encoding)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.run(self.args(["--tokenizer-path", str(self.root)]), self.encoding)

    def test_local_execute_has_optional_auth_and_truthful_zero_api_cost(self):
        flags = ["--local-loopback", "--endpoint", "http://127.0.0.1/v1/chat/completions",
                 "--tokenizer-path", str(self.root), "--execute", "--max-cost-usd", ".1"]
        fake = FakeClient([value for _ in range(4) for value in
            (response(), response('{"correct":true,"reason":"Equivalent"}'))])
        with patch.dict(os.environ, {"AML_LOCAL_LLM_API_KEY": "", "OPENAI_API_KEY": "SYNTHETIC_REMOTE_KEY"}):
            with patch.object(evaluator, "load_key", side_effect=AssertionError("unauthenticated local key read")):
                def factory(settings, key):
                    self.assertIsNone(key)
                    self.assertEqual(settings.chat_template_kwargs, {"enable_thinking": False})
                    return fake
                report = evaluator.run(self.args(flags), self.encoding, factory)
        self.assertEqual(report["usage_and_cost"]["attempted_requests"], 8)
        self.assertEqual(report["usage_and_cost"]["api_fee_usd"], 0)
        self.assertFalse(report["usage_and_cost"]["compute_cost_accounted"])
        self.assertFalse(report["network"]["tls_verification"])
        self.assertIn("GPU", report["cost_plan"]["price_source"])

    def test_local_body_real_loopback_and_unchanged_remote_body(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append({"body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                                 "authorization": self.headers.get("Authorization")})
                body = json.dumps(response('{"correct":true,"reason":"Equivalent"}')).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = self.local_settings(endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions", temperature=.2)
            result = evaluator.ChatClient(settings, None).complete(settings.answer_model, [], 25, judge=True)
            self.assertTrue(evaluator.parse_judgment(evaluator.completion_text(result))["correct"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        body = captured[0]["body"]
        self.assertEqual(body["max_tokens"], 25)
        self.assertEqual(body["temperature"], .2)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertNotIn("store", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertIsNone(captured[0]["authorization"])
        remote = evaluator.ChatClient(evaluator.Settings(), "SYNTHETIC_KEY")
        opened = unittest.mock.MagicMock()
        opened.__enter__.return_value.read.return_value = json.dumps(response()).encode()
        with patch.object(remote.opener, "open", return_value=opened) as mocked:
            remote.complete(evaluator.DEFAULT_MODEL, [], 25)
        remote_body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(remote_body["max_completion_tokens"], 25)
        self.assertEqual(remote_body["temperature"], 0)
        self.assertFalse(remote_body["store"])
        self.assertNotIn("chat_template_kwargs", remote_body)

    def test_local_profile_rejects_invalid_temperature_kwargs_and_mixed_tokenizers(self):
        for kwargs in ({"temperature": float("nan")}, {"temperature": -1}, {"temperature": 3},
                       {"chat_template_kwargs": {"enable_thinking": "false"}},
                       {"chat_template_kwargs": {"add_generation_prompt": False}},
                       {"judge_model": "different-model"}):
            with self.assertRaises(evaluator.EvaluationError):
                self.local_settings(**kwargs).validate()

    def tokenizer_fixture(self):
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from tokenizers.pre_tokenizers import Whitespace
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            self.skipTest("Local HF tokenizer dependencies are optional in standard-library tests")
        core = Tokenizer(WordLevel({"[UNK]": 0, "user": 1, "assistant": 2, "system": 3,
                                  ":": 4, "Answer": 5, "direct": 6, "thinking": 7}, unk_token="[UNK]"))
        core.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
        tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}{% if enable_thinking %}thinking {% else %}direct {% endif %}")
        folder = (self.root / "tokenizer").resolve()
        tokenizer.save_pretrained(str(folder))
        return folder, tokenizer

    def test_hf_tokenizer_loads_offline_and_counts_actual_chat_template(self):
        folder, original = self.tokenizer_fixture()
        from transformers import AutoTokenizer
        with patch.object(AutoTokenizer, "from_pretrained", wraps=AutoTokenizer.from_pretrained) as load:
            encoding = evaluator.offline_hf_encoding(folder, {"enable_thinking": False})
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        self.assertFalse(load.call_args.kwargs["trust_remote_code"])
        messages = [{"role": "user", "content": "Answer"}]
        expected = len(original.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=False, enable_thinking=False))
        self.assertEqual(evaluator.message_tokens(encoding, messages, 64), expected + 64)
        self.assertGreater(expected, evaluator.tokens(encoding, "Answer"))
        self.assertIn("tokenizer.json", encoding.metadata["files_sha256"])
        self.assertEqual(len(encoding.metadata["chat_template_sha256"]), 64)
        self.assertNotIn(str(folder), json.dumps(encoding.metadata))
        self.assertEqual(encoding.template_kwargs, {"enable_thinking": False})

    def test_hf_tokenizer_rejects_symlinks_missing_template_and_mutation(self):
        folder, _ = self.tokenizer_fixture()
        link = self.root / "tokenizer-link"
        link.symlink_to(folder, target_is_directory=True)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.local_tokenizer_files(link)
        source = folder / "tokenizer.json"
        saved = self.root / "saved-tokenizer.json"
        source.rename(saved)
        source.symlink_to(saved)
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.local_tokenizer_files(folder)
        source.unlink()
        saved.rename(source)
        from transformers import AutoTokenizer
        original_load = AutoTokenizer.from_pretrained
        def mutating_load(*args, **kwargs):
            tokenizer = original_load(*args, **kwargs)
            with (folder / "tokenizer_config.json").open("a") as stream:
                stream.write("\n")
            return tokenizer
        with patch.object(AutoTokenizer, "from_pretrained", side_effect=mutating_load):
            with self.assertRaisesRegex(evaluator.EvaluationError, "changed"):
                evaluator.offline_hf_encoding(folder, {"enable_thinking": False})
        config_path = folder / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        config.pop("chat_template", None)
        config_path.write_text(json.dumps(config))
        template_file = folder / "chat_template.jinja"
        if template_file.exists():
            template_file.unlink()
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.offline_hf_encoding(folder, {"enable_thinking": False})

    def test_hf_template_token_count_controls_packing(self):
        folder, _ = self.tokenizer_fixture()
        encoding = evaluator.offline_hf_encoding(folder, {"enable_thinking": False})
        query = {"query": "Answer"}
        minimum = evaluator.message_tokens(encoding, evaluator.answer_messages(query, ""), 64)
        items = [{"id": "a", "content": "Answer " * 500}]
        with self.assertRaises(evaluator.EvaluationError):
            evaluator.pack_memories(query, items, encoding, 10000, minimum - 1, 64)
        _, packed = evaluator.pack_memories(query, items, encoding, 10000, minimum, 64)
        self.assertEqual(packed["selected_items"], 0)


if __name__ == "__main__":
    unittest.main()
