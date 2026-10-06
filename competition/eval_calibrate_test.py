import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
import eval_answer as core
import eval_calibrate as calibration


class Encoding:
    def encode(self, value, **kwargs):
        return value.split()


class Client:
    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def complete(self, model, messages, cap, judge=False):
        self.calls.append((judge, messages))
        value = '{"label":"CORRECT"}' if judge else "The park"
        if self.invalid:
            value = "bad JSON"
        return {"choices": [{"message": {"content": value}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def inputs():
    query = {"query_id": "q1", "query": "Where?", "question_date": "DATE_SENTINEL"}
    record = {"status": "ok", "retrieved": [{"id": "m1", "content": "The park"}]}
    runs = [({"dataset": "longmemeval_s"}, {"q1": record}, {"run_id": "a"}),
            ({"dataset": "longmemeval_s"}, {"q1": record}, {"run_id": "b"})]
    return runs, [query], {"q1": {"answer": "The park"}}


class CalibrationTests(unittest.TestCase):
    def settings(self, **kwargs):
        values = dict(local_loopback=True, endpoint="http://127.0.0.1:18080/v1/chat/completions",
                      input_usd_per_million=0, output_usd_per_million=0,
                      input_tokens=5000, context_tokens=65536, memory_tokens=50,
                      answer_output_tokens=100, max_requests=4)
        values.update(kwargs)
        return core.Settings(**values)

    def test_whole_item_prefix_never_skips_over_large_first_item(self):
        items = [{"id": "big", "content": "large " * 60}, {"id": "small", "content": "small"}]
        messages, packing = calibration.pack_memories("longmemeval_s", inputs()[1][0], items, Encoding(), self.settings())
        self.assertEqual(packing["selected_items"], 0)
        self.assertNotIn("small", messages[0]["content"])
        self.assertNotIn("DATE_SENTINEL", messages[0]["content"])

    def test_rejudge_never_calls_answer_and_preserves_failure_denominator(self):
        runs, queries, gold = inputs()
        previous = {("q1", "baseline"): {"status": "answer_error"},
                    ("q1", "candidate"): {"status": "ok", "answer": "The park"}}
        jobs, plan = calibration.build_plan(runs, queries, gold, self.settings(), Encoding(), previous)
        self.assertEqual(plan["planned_requests_maximum"], 1)
        client = Client()
        saved = []
        results, ledger = calibration.execute_jobs(jobs, self.settings(), Encoding(), client, saved.append)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0][0])
        metrics = core.aggregate_results(results, 1)
        self.assertEqual(metrics["baseline"]["failed_questions"], 1)
        self.assertEqual(metrics["baseline"]["denominator"], 1)
        self.assertEqual(metrics["candidate"]["correct"], 1)
        self.assertEqual(len(saved), 2)

    def test_fresh_answer_has_no_reference_or_date_and_four_calls(self):
        runs, queries, gold = inputs()
        gold["q1"]["answer"] = "GOLD_SENTINEL"
        jobs, plan = calibration.build_plan(runs, queries, gold, self.settings(), Encoding())
        client = Client()
        results, ledger = calibration.execute_jobs(jobs, self.settings(), Encoding(), client, lambda row: None)
        self.assertEqual(plan["planned_requests_maximum"], 4)
        self.assertEqual(len(client.calls), 4)
        for judge, messages in client.calls:
            text = str(messages)
            self.assertNotIn("DATE_SENTINEL", text)
            self.assertEqual("GOLD_SENTINEL" in text, judge)

    def test_invalid_judge_is_counted_as_failure_without_retry(self):
        runs, queries, gold = inputs()
        previous = {(q["query_id"], s): {"status": "ok", "answer": "A"}
                    for q in queries for s in ("baseline", "candidate")}
        jobs, _ = calibration.build_plan(runs, queries, gold, self.settings(), Encoding(), previous)
        client = Client(invalid=True)
        results, _ = calibration.execute_jobs(jobs, self.settings(), Encoding(), client, lambda row: None)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(r["status"] == "judge_error" for r in results))
        self.assertEqual(core.aggregate_results(results, 1)["candidate"]["failed_questions"], 1)

    def test_token_excess_stops_following_network_calls(self):
        runs, queries, gold = inputs()
        jobs, _ = calibration.build_plan(runs, queries, gold, self.settings(), Encoding())
        jobs[0]["packing"]["estimated_input_tokens"] = 1
        client = Client()
        results, ledger = calibration.execute_jobs(jobs, self.settings(), Encoding(), client, lambda row: None)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(ledger.stopped)
        self.assertEqual(len(results), 2)

    def test_stably_tampered_historical_answers_rejected_by_original_digest(self):
        runs, queries, gold = inputs()
        spec = {"files": {"queries": {"sha256": "a" * 64}}}
        rows = [{"query_id": "q1", "side": side, "run_id": run_id,
                 "status": "ok", "answer": "original"}
                for side, run_id in (("baseline", "a"), ("candidate", "b"))]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            raw = folder / "answers-judgments.jsonl"
            raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
            summary = {"version": core.VERSION, "tool_sha256": core.file_hash(Path(core.__file__)),
                       "status": "executed_public_dev_proxy_not_official_aml",
                       "source_runs": [r[2] for r in runs], "prepared_file_sha256": {"queries": "a" * 64},
                       "private_raw_records_sha256": core.file_hash(raw), "matched_question_denominator": 1,
                       "selected_query_ids_sha256": core.canonical_hash(["q1"])}
            (folder / "summary.json").write_text(json.dumps(summary))
            with patch.object(core, "PRIVATE_ROOT", folder):
                calibration.historical_answers(folder, runs, queries, spec)
                rows[0]["answer"] = "tampered"
                raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaisesRegex(core.EvaluationError, "original summary"):
                    calibration.historical_answers(folder, runs, queries, spec)


if __name__ == "__main__":
    unittest.main()
