import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval_answer as core
import eval_prepare as preparation
import eval_retrieval as retrieval
import eval_validate_answers as validation


class Encoding:
    def encode(self, value, **kwargs):
        return value.split()


class FakeClient:
    def __init__(self, settings, key, fail_first=False, mutate=None):
        self.calls = []
        self.fail_first = fail_first
        self.mutate = mutate

    def complete(self, model, messages, cap, judge=False):
        self.calls.append((judge, messages))
        if len(self.calls) == 1 and self.mutate:
            self.mutate()
        if len(self.calls) == 1 and self.fail_first:
            raise core.ProviderError("test_failure")
        return {"choices": [{"message": {"content": '{"label":"CORRECT"}' if judge else "Answer only"},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": sum(len(m["content"].split()) for m in messages), "completion_tokens": 2}}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.data = root / "data"
        self.folder = self.data / "prepared/locomo_refined/validation"
        self.baseline = root / "runs/baseline_validation"
        self.candidate = root / "runs/candidate_validation"
        self.manifest = root / "governance.json"
        self.histories = [{"sample_id": sample, "sessions": [{"session_id": "session_" + sample,
            "messages": [{"role": "user", "content": "First history " + sample, "timestamp": 0},
                         {"role": "assistant", "content": "Second history " + sample, "timestamp": 0}]}]}
            for sample in ("sample_a", "sample_b")]
        self.queries = [{"query_id": "q1", "sample_id": "sample_a", "query": "First question?", "answer": "QUERY_ANSWER_SENTINEL", "category": "CATEGORY_SENTINEL", "evidence": "EVIDENCE_SENTINEL"},
                        {"query_id": "q2", "sample_id": "sample_a", "query": "Second question?"},
                        {"query_id": "q3", "sample_id": "sample_b", "query": "Third question?"}]
        self.gold = [{"query_id": q["query_id"], "sample_id": q["sample_id"], "answer": "REFERENCE_SENTINEL"} for q in self.queries]
        self.write_rows(self.folder / "histories.jsonl", self.histories)
        self.write_rows(self.folder / "queries.jsonl", self.queries)
        self.write_rows(self.folder / "gold.jsonl", self.gold)
        files = {n: {"sha256": core.file_hash(self.folder / (n + ".jsonl")),
                     "size": (self.folder / (n + ".jsonl")).stat().st_size} for n in ("histories", "queries", "gold")}
        self.spec = {"histories": 2, "questions": 3, "messages": 4, "gold_turn_annotations": 3, "files": files}
        self.write_json(self.data / "prepared/summary.json", {"datasets": {"locomo_refined": {"splits": {"validation": self.spec}}}})
        groups = []
        for index, history in enumerate(self.histories):
            qs = [q for q in self.queries if q["sample_id"] == history["sample_id"]]
            groups.append({"group_id": "group_" + str(index), "partition": "validation", "reason": "preserved_prepared_split",
                           "original_prepared_splits": ["validation"],
                           "members": [{"dataset": "locomo_refined", "sample_id": history["sample_id"]}],
                           "queries": [{"dataset": "locomo_refined", "query_id": q["query_id"]} for q in qs]})
        self.governance = {"schema": "memory-validation-governance-v1", "allocation": "preserved_prepared_splits",
            "selection_reads_labels": False, "public_not_blind": True, "groups": groups,
            "question_counts": {"validation": 3},
            "sources": [{"path": str(self.folder / (n + ".jsonl")), "sha256": files[n]["sha256"]} for n in ("histories", "queries")]}
        self.write_json(self.manifest, self.governance)
        config = {"top_k": 100, "character_budget": 0, "launch_local": True, "add_concurrency": 1,
                  "source_mapping_version": preparation.SOURCE_MAPPING_VERSION, "prepared_data": self.spec,
                  "source_code_sha256": {"server.py_at_start": "a" * 64, "server.py_at_end": "a" * 64,
                    "eval_prepare.py": core.file_hash(Path(preparation.__file__)),
                    "eval_retrieval.py": core.file_hash(Path(retrieval.__file__))}}
        self.summary = {"status": "public_proxy_not_official_aml", "run_id": "baseline_validation", "split": "validation",
            "dataset": "locomo_refined", "question_count": 3, "history_count": 2,
            "ingestion_failed_histories": 0, "server_source_changed_during_run": False, "configuration": config}
        self.items = {}
        sequence = 1
        for history in self.histories:
            _, payloads, sources, _ = retrieval.requests_for_history(history, self.summary["run_id"])
            items = []
            for payload in payloads:
                for index, message in enumerate(payload["messages"]):
                    original = sources[payload["session_id"], payload["request_id"], index]
                    header = {"role": message["role"], "session_id": payload["session_id"], "request_id": payload["request_id"],
                              "message_index": index, "message_id": original["message_id"], "received_sequence": sequence,
                              "timestamp_ms": message["timestamp"], "timestamp_utc": "1970-01-01T00:00:00.000Z"}
                    content = "[source " + json.dumps(header, separators=(",", ":")) + "]\n" + message["content"]
                    items.append({"id": original["message_id"], "content": content})
                    sequence += 1
            self.items[history["sample_id"]] = items
        self.baseline_rows = [{"query_id": q["query_id"], "sample_id": q["sample_id"], "status": "ok",
                               "retrieved": copy.deepcopy(self.items[q["sample_id"]])} for q in self.queries]
        self.write_json(self.baseline / "summary.json", self.summary)
        self.write_rows(self.baseline / "retrieval.jsonl", self.baseline_rows)
        self.candidate_rows = copy.deepcopy(self.baseline_rows)
        for row in self.candidate_rows:
            row["retrieved"].reverse()
        self.candidate_summary = copy.deepcopy(self.summary)
        self.candidate_summary.update(run_id="candidate_validation", source_run_id="baseline_validation", ingestion_run_id="baseline_validation",
            candidate_set_and_content_unchanged=True,
            rerank_provenance={"source_summary_sha256": core.file_hash(self.baseline / "summary.json"),
                "source_retrieval_sha256": core.file_hash(self.baseline / "retrieval.jsonl"),
                "rank_engine_sha256": "b" * 64, "tool_sha256": "c" * 64})
        self.write_json(self.candidate / "summary.json", self.candidate_summary)
        self.write_rows(self.candidate / "retrieval.jsonl", self.candidate_rows)

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n")

    @staticmethod
    def write_rows(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value))

    def args(self, *extra):
        return validation.parser().parse_args(["--validation-manifest", str(self.manifest), "--baseline-run", str(self.baseline),
            "--candidate-run", str(self.candidate), "--data-root", str(self.data), "--local-loopback",
            "--tokenizer-path", str(self.root / "unused-tokenizer"), "--endpoint", "http://127.0.0.1:18080/v1/chat/completions",
            "--memory-tokens", "500", "--input-tokens", "5000", "--context-tokens", "65536",
            "--answer-output-tokens", "100", "--judge-output-tokens", "50", "--max-requests", "12",
            "--output-dir", str(self.root / "private/output"), *extra])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="validation-runner-")
        self.root = Path(self.temp.name).resolve()
        self.fixture = Fixture(self.root)
        self.private_patch = patch.object(core, "PRIVATE_ROOT", self.root / "private")
        self.private_patch.start()

    def tearDown(self):
        self.private_patch.stop()
        self.temp.cleanup()

    def test_complete_manifest_and_order_only_permutation_are_accepted(self):
        runs, queries, gold, hashes, governance = validation.matched_validation_inputs(self.fixture.args())
        self.assertEqual([q["query_id"] for q in queries], ["q1", "q2", "q3"])
        self.assertEqual(governance["histories"], 2)
        self.assertEqual(set(queries[0]), {"query_id", "sample_id", "query"})
        self.assertEqual(set(gold["q1"]), {"answer"})
        self.assertTrue(all(core.file_hash(Path(p)) == h for p, h in hashes.items()))

    def test_default_plan_never_constructs_client_or_reads_key(self):
        with patch.object(core, "load_key", side_effect=AssertionError("key access")):
            report = validation.run(self.fixture.args(), Encoding(), lambda *a: self.fail("client constructed"))
        self.assertEqual(report["status"], "dry_run_no_network")
        self.assertEqual(report["split"], "validation")
        self.assertEqual(report["cost_plan"]["planned_requests_maximum"], 12)
        self.assertIsNone(report["metrics"])
        self.assertTrue(report["all_inputs_and_tools_unchanged"])

    def test_fresh_execution_keeps_gold_and_poisoned_query_fields_out_of_answer(self):
        clients = []
        def factory(*args):
            client = FakeClient(*args); clients.append(client); return client
        report = validation.run(self.fixture.args("--execute", "--max-cost-usd", "0.10"), Encoding(), factory)
        self.assertEqual(report["metrics"]["baseline"]["denominator"], 3)
        self.assertEqual(report["usage"]["attempted_requests"], 12)
        for judge, messages in clients[0].calls:
            text = str(messages)
            self.assertEqual("REFERENCE_SENTINEL" in text, judge)
            for sentinel in ("QUERY_ANSWER_SENTINEL", "CATEGORY_SENTINEL", "EVIDENCE_SENTINEL"):
                self.assertNotIn(sentinel, text)

    def test_answer_failure_retains_full_matched_denominators(self):
        report = validation.run(self.fixture.args("--execute", "--max-cost-usd", "0.10"), Encoding(),
            lambda *a: FakeClient(*a, fail_first=True))
        self.assertEqual(report["metrics"]["baseline"]["denominator"], 3)
        self.assertEqual(report["metrics"]["baseline"]["failed_questions"], 1)
        self.assertEqual(report["metrics"]["candidate"]["denominator"], 3)
        self.assertEqual(report["usage"]["attempted_requests"], 11)
        self.assertEqual(report["usage"]["usage_unknown_requests"], 1)

    def test_insufficient_requests_and_zero_cost_reject_before_client(self):
        for extra in [("--max-requests", "11", "--max-cost-usd", "0.10"), ("--max-cost-usd", "0")]:
            with self.subTest(extra=extra), self.assertRaises(core.EvaluationError):
                validation.run(self.fixture.args("--execute", *extra), Encoding(), lambda *a: self.fail("client constructed"))

    def test_question_prefix_is_rejected(self):
        with self.assertRaises(core.EvaluationError):
            validation.matched_validation_inputs(self.fixture.args("--max-questions", "2"))

    def test_missing_or_duplicate_question_is_rejected(self):
        for duplicate in (False, True):
            rows = copy.deepcopy(self.fixture.candidate_rows)
            if duplicate: rows[-1]["query_id"] = rows[0]["query_id"]
            else: rows.pop()
            self.fixture.write_rows(self.fixture.candidate / "retrieval.jsonl", rows)
            with self.subTest(duplicate=duplicate), self.assertRaises(core.EvaluationError):
                validation.matched_validation_inputs(self.fixture.args())

    def test_changed_bytes_or_duplicate_returned_id_reject_before_gold(self):
        original_read = core.read_jsonl
        def no_gold(path):
            self.assertNotEqual(Path(path).name, "gold.jsonl")
            return original_read(path)
        for duplicate in (False, True):
            rows = copy.deepcopy(self.fixture.candidate_rows)
            if duplicate: rows[0]["retrieved"][1] = copy.deepcopy(rows[0]["retrieved"][0])
            else: rows[0]["retrieved"][0]["content"] += "changed"
            self.fixture.write_rows(self.fixture.candidate / "retrieval.jsonl", rows)
            with self.subTest(duplicate=duplicate), patch.object(core, "read_jsonl", side_effect=no_gold), self.assertRaises(core.EvaluationError):
                validation.matched_validation_inputs(self.fixture.args())

    def test_cross_conversation_source_is_rejected(self):
        self.fixture.baseline_rows[0]["retrieved"] = copy.deepcopy(self.fixture.items["sample_b"])
        self.fixture.candidate_rows[0]["retrieved"] = copy.deepcopy(self.fixture.items["sample_b"])
        self.fixture.write_rows(self.fixture.baseline / "retrieval.jsonl", self.fixture.baseline_rows)
        self.fixture.candidate_summary["rerank_provenance"]["source_retrieval_sha256"] = core.file_hash(self.fixture.baseline / "retrieval.jsonl")
        self.fixture.write_json(self.fixture.candidate / "summary.json", self.fixture.candidate_summary)
        self.fixture.write_rows(self.fixture.candidate / "retrieval.jsonl", self.fixture.candidate_rows)
        with self.assertRaises(core.EvaluationError): validation.matched_validation_inputs(self.fixture.args())

    def test_unverified_source_header_field_is_rejected_before_gold(self):
        for rows in (self.fixture.baseline_rows, self.fixture.candidate_rows):
            content = rows[0]["retrieved"][0]["content"]
            header, body = content.split("\n", 1)
            metadata = json.loads(header[len("[source "):-1]); metadata["answer"] = "UNPROVEN_ANSWER"
            rows[0]["retrieved"][0]["content"] = "[source " + json.dumps(metadata, separators=(",", ":")) + "]\n" + body
        self.fixture.write_rows(self.fixture.baseline / "retrieval.jsonl", self.fixture.baseline_rows)
        # Preserve the exact permutation after changing both source lists.
        self.fixture.candidate_rows[0]["retrieved"] = list(reversed(copy.deepcopy(self.fixture.baseline_rows[0]["retrieved"])))
        self.fixture.candidate_summary["rerank_provenance"]["source_retrieval_sha256"] = core.file_hash(self.fixture.baseline / "retrieval.jsonl")
        self.fixture.write_rows(self.fixture.candidate / "retrieval.jsonl", self.fixture.candidate_rows)
        self.fixture.write_json(self.fixture.candidate / "summary.json", self.fixture.candidate_summary)
        original_read = core.read_jsonl
        def no_gold(path):
            self.assertNotEqual(Path(path).name, "gold.jsonl")
            return original_read(path)
        with patch.object(core, "read_jsonl", side_effect=no_gold), self.assertRaises(core.EvaluationError):
            validation.matched_validation_inputs(self.fixture.args())

    def test_forged_provenance_changed_configuration_or_question_text_reject(self):
        cases = ("source_sha", "config", "typed_config", "query")
        for case in cases:
            summary = copy.deepcopy(self.fixture.candidate_summary); rows = copy.deepcopy(self.fixture.candidate_rows)
            if case == "source_sha": summary["rerank_provenance"]["source_summary_sha256"] = "d" * 64
            elif case == "config": summary["configuration"]["character_budget"] = 100
            elif case == "typed_config": summary["configuration"]["add_concurrency"] = True
            else: rows[0]["query"] = "Different question?"
            self.fixture.write_json(self.fixture.candidate / "summary.json", summary)
            self.fixture.write_rows(self.fixture.candidate / "retrieval.jsonl", rows)
            with self.subTest(case=case), self.assertRaises(core.EvaluationError):
                validation.matched_validation_inputs(self.fixture.args())

    def test_duplicate_header_keys_or_forged_sequence_are_rejected(self):
        original_baseline = copy.deepcopy(self.fixture.baseline_rows)
        original_candidate = copy.deepcopy(self.fixture.candidate_rows)
        for duplicate in (False, True):
            baseline = copy.deepcopy(original_baseline); candidate = copy.deepcopy(original_candidate)
            content = baseline[0]["retrieved"][0]["content"]
            replacement = '"received_sequence":999,"received_sequence":1' if duplicate else '"received_sequence":999'
            baseline[0]["retrieved"][0]["content"] = content.replace('"received_sequence":1', replacement)
            candidate[0]["retrieved"] = list(reversed(copy.deepcopy(baseline[0]["retrieved"])))
            self.fixture.write_rows(self.fixture.baseline / "retrieval.jsonl", baseline)
            self.fixture.write_rows(self.fixture.candidate / "retrieval.jsonl", candidate)
            summary = copy.deepcopy(self.fixture.candidate_summary)
            summary["rerank_provenance"]["source_retrieval_sha256"] = core.file_hash(self.fixture.baseline / "retrieval.jsonl")
            self.fixture.write_json(self.fixture.candidate / "summary.json", summary)
            with self.subTest(duplicate=duplicate), self.assertRaises(core.EvaluationError):
                validation.matched_validation_inputs(self.fixture.args())

    def test_dev_or_heldout_summary_cannot_be_relabelled(self):
        for split in ("dev", "heldout"):
            summary = copy.deepcopy(self.fixture.candidate_summary); summary["split"] = split
            self.fixture.write_json(self.fixture.candidate / "summary.json", summary)
            with self.subTest(split=split), self.assertRaises(core.EvaluationError):
                validation.matched_validation_inputs(self.fixture.args())

    def test_symlink_is_rejected_without_reading_its_target(self):
        path = self.fixture.candidate / "retrieval.jsonl"
        path.unlink(); target = self.root / "heldout/must-not-open.jsonl"; target.parent.mkdir(); target.write_text("not JSON")
        path.symlink_to(target)
        with self.assertRaises(core.EvaluationError): validation.matched_validation_inputs(self.fixture.args())

    def test_lme_governance_cannot_claim_independent_validation(self):
        self.fixture.governance["groups"][0]["members"][0]["dataset"] = "longmemeval_s"
        self.fixture.write_json(self.fixture.manifest, self.fixture.governance)
        with self.assertRaises(core.EvaluationError): validation.matched_validation_inputs(self.fixture.args())

    def test_changed_input_at_execution_end_is_recorded_and_rejected(self):
        def mutate():
            p = self.fixture.baseline / "summary.json"; p.write_text(p.read_text() + " ")
        with self.assertRaises(core.EvaluationError):
            validation.run(self.fixture.args("--execute", "--max-cost-usd", "0.10"), Encoding(),
                lambda *a: FakeClient(*a, mutate=mutate))
        summary = json.loads((self.root / "private/output/summary.json").read_text())
        self.assertFalse(summary["all_inputs_and_tools_unchanged"])


if __name__ == "__main__":
    unittest.main()
