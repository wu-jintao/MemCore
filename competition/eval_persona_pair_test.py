#!/usr/bin/env python3
"""Synthetic Persona replay boundary tests; no corpus, model or network calls."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import eval_persona as persona
import eval_persona_pair as pair


class Encoding:
    name = "synthetic-byte-encoding"
    def encode(self, text, disallowed_special=()):
        return text.encode("utf-8")


def response(text="<final_answer>(a)", usage=True):
    result = {"model": pair.MODEL, "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    if usage:
        result["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
    return result


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, model, messages, max_output):
        self.calls.append((model, messages, max_output))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class PersonaPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.private = self.root / ".local"
        self.private.mkdir()
        self.private_patch = patch.object(pair, "PRIVATE_ROOT", self.private)
        self.private_patch.start()
        self.data = self.private / "persona-prepared"
        self.folder = self.data / "prepared" / persona.DATASET / "dev"
        self.folder.mkdir(parents=True)
        self.histories, self.lineage = [], []
        native = [{"role": "system", "content": "historical persona note"},
                  {"role": "user", "content": "The historical preference is blue.\u2028Original text."},
                  {"role": "assistant", "content": "FUTURE_SENTINEL beyond the earlier cutoff"}]
        for sid, group, cutoff in (("a1", "group-A", 2), ("a2", "group-A", 3),
                                    ("b1", "group-B", 2), ("c1", "group-C", 2)):
            history, sources = persona.history_prefix(native, cutoff, sid)
            self.histories.append(history)
            self.lineage.append({"sample_id": sid, "group_id": group, "exclusive_end_index": cutoff,
                "source_messages": sources, "history_record_sha256": persona.digest(history)})
        options = repr(["(a) OPTION_A", "(b) OPTION_B", "(c) OPTION_C", "(d) OPTION_D"])
        # Unselected C question comes first: question order cannot affect group selection.
        self.queries = [{"sample_id": sid, "query_id": qid, "query": "PUBLIC_QUERY_" + qid,
            "all_options": options, "options": persona.parse_options(options),
            "answer": "QUERY_GOLD_SENTINEL", "gold_units": ["QUERY_EVIDENCE_SENTINEL"]}
            for qid, sid in (("q0", "c1"), ("q1", "a1"), ("q2", "a2"), ("q3", "b1"))]
        self.gold = [{"query_id": q["query_id"], "sample_id": q["sample_id"], "answer": "(a)",
                      "gold_option": "a", "topic": "PRIVATE_GRADER_SENTINEL"} for q in self.queries]
        self.write_prepared()
        self.baseline = self.make_run("baseline", False)
        self.candidate = self.make_run("candidate", True)

    def tearDown(self):
        self.private_patch.stop()
        self.temp.cleanup()

    def jsonl(self, path, rows):
        path.write_text("".join(persona.canonical_json(r) + "\n" for r in rows), encoding="utf-8")

    def write_prepared(self):
        for name, rows in (("histories", self.histories), ("lineage", self.lineage),
                           ("queries", self.queries), ("gold", self.gold)):
            self.jsonl(self.folder / (name + ".jsonl"), rows)
        self.spec = {"histories": len(self.histories), "questions": len(self.queries),
            "files": {n: {"size": (self.folder / (n + ".jsonl")).stat().st_size,
                          "sha256": persona.file_hash(self.folder / (n + ".jsonl"))}
                      for n in ("histories", "lineage", "queries", "gold")}}
        summary = {"version": persona.VERSION, "status": "prepared_public_native_proxy_not_official_aml",
            "upstream": persona.UPSTREAM, "revision": persona.REVISION, "dataset_license": "mit",
            "source_mapping_version": "persona-native-exclusive-prefix-v1",
            "datasets": {persona.DATASET: {"splits": {"dev": self.spec}}}}
        (self.data / "prepared" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def make_run(self, name, reverse):
        directory = self.private / name
        directory.mkdir()
        summary = {"dataset": persona.DATASET, "split": "dev", "run_id": name,
            "model_label": name + "-declared-retrieval-only", "question_count": len(self.queries),
            "server_source_changed_during_run": False,
            "configuration": {"top_k": 100, "character_budget": 0, "prepared_data": self.spec,
                "source_code_sha256": {"server.py_at_start": "1" * 64, "server.py_at_end": "1" * 64}}}
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        histories = {h["sample_id"]: h for h in self.histories}
        rows = [{"query_id": q["query_id"], "sample_id": q["sample_id"], "status": "ok",
            "answer": "RETRIEVAL_GOLD_SENTINEL", "retrieved": [{"id": "source-" + q["sample_id"],
                "content": histories[q["sample_id"]]["sessions"][0]["messages"][1]["content"]}]}
                for q in self.queries]
        self.jsonl(directory / "retrieval.jsonl", list(reversed(rows)) if reverse else rows)
        return directory

    def args(self, extra=(), arms=True):
        values = ["replay-answer", "--data-root", str(self.data)]
        if arms:
            values += ["--baseline-run", str(self.baseline), "--candidate-run", str(self.candidate)]
        return pair.parser().parse_args([*values, *extra])

    def execute(self, responses, extra=()):
        client = Client(responses)
        args = self.args(["--execute", "--output-dir", str(self.private / "output"), *extra])
        report = pair.run(args, encoding=Encoding(), client_factory=lambda settings, key: client)
        return report, client

    def edit_rows(self, folder, mutate):
        path = folder / "retrieval.jsonl"
        rows = pair.qa.read_jsonl(path)
        mutate(rows)
        self.jsonl(path, rows)

    def test_default_plan_complete_groups_never_opens_gold_or_key_or_writes(self):
        gold = self.folder / "gold.jsonl"
        gold.unlink()  # Dry-run must not even require opening it.
        before = sorted(str(p) for p in self.private.rglob("*"))
        with patch.object(pair.qa, "load_key", side_effect=AssertionError("no key read")), \
                patch.object(pair.qa, "ChatClient", side_effect=AssertionError("no client")), \
                patch.object(pair, "private_output", side_effect=AssertionError("no output")):
            report = pair.run(self.args(["--api-key-file", str(self.private / "absent-key")]), encoding=Encoding())
        self.assertEqual(report["selection"]["group_ids"], ["group-A", "group-B"])
        self.assertEqual(report["selection"]["histories"], 3)
        self.assertEqual(report["selection"]["questions_per_arm"], 3)
        self.assertEqual(report["budget"]["planned_answer_requests"], 6)
        self.assertIsNone(report["metrics"])
        self.assertEqual(before, sorted(str(p) for p in self.private.rglob("*")))

    def test_plan_without_arms_reports_complete_selection_and_no_requests(self):
        report = pair.run(self.args(arms=False))
        self.assertEqual(report["selection"]["groups"], 2)
        self.assertEqual(report["budget"]["maximum_paired_requests"], 6)
        self.assertEqual(report["attempted_requests"], 0)

    def test_successful_paired_mcq_no_gold_in_prompts_and_private_output(self):
        report, client = self.execute([response("(b)"), response(), response(), response(), response(), response()])
        self.assertEqual(report["metrics"]["baseline"]["correct"], 2)
        self.assertEqual(report["metrics"]["candidate"]["correct"], 3)
        self.assertEqual(report["paired"], {"gained": 1, "lost": 0})
        self.assertEqual(len(client.calls), 6)
        prompts = json.dumps(client.calls)
        for sentinel in ("PRIVATE_GRADER_SENTINEL", "QUERY_GOLD_SENTINEL", "QUERY_EVIDENCE_SENTINEL",
                         "RETRIEVAL_GOLD_SENTINEL", "FUTURE_SENTINEL", "PUBLIC_QUERY_q0"):
            self.assertNotIn(sentinel, prompts)
        for option in ("OPTION_A", "OPTION_B", "OPTION_C", "OPTION_D"):
            self.assertIn(option, prompts)
        self.assertEqual(stat.S_IMODE((self.private / "output").stat().st_mode), 0o700)
        for path in (self.private / "output").iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(report["usage"]["reported_requests"], 6)
        self.assertEqual(report["metrics"]["baseline"]["denominator"], 3)
        self.assertIn("eval_persona.py", report["source_code_sha256"])

    def test_future_retrieved_text_rejected_before_key_or_client(self):
        def future(rows):
            next(r for r in rows if r["query_id"] == "q1")["retrieved"][0]["content"] = "FUTURE_SENTINEL beyond the earlier cutoff"
        self.edit_rows(self.baseline, future)
        with patch.object(pair, "scoring_references", side_effect=AssertionError("no gold before provenance")):
            with self.assertRaisesRegex(pair.Error, "cutoff"):
                pair.run(self.args(), encoding=Encoding())

    def test_wrapped_adjacent_history_verified_and_role_time_preserved(self):
        history = self.histories[1]  # Later eligible cutoff includes an assistant message.
        session = history["sessions"][0]
        def wrapped(rows):
            row = next(r for r in rows if r["query_id"] == "q2")
            bodies = []
            for i, message in enumerate(session["messages"]):
                source = {"role": message["role"], "session_id": session["session_id"],
                    "request_id": "request", "message_index": i, "message_id": "message", "received_sequence": i,
                    "timestamp_ms": 1700000000000, "timestamp_utc": "2023-11-14T22:13:20.000Z"}
                bodies.append("[source " + persona.canonical_json(source) + "]\n" + message["content"])
            row["retrieved"][0]["content"] = "\n\n[adjacent context]\n".join(bodies)
        self.edit_rows(self.baseline, wrapped)
        report, client = self.execute([response() for _ in range(6)])
        self.assertEqual(report["metrics"]["baseline"]["correct"], 3)
        self.assertIn("received_sequence", json.dumps(client.calls))
        self.assertIn("timestamp_utc", json.dumps(client.calls))
        excerpts = json.loads(client.calls[2][1][0]["content"].split("\n\n", 1)[1])["historical_excerpts"]
        self.assertIn('"role":"assistant"', excerpts)
        self.assertIn("historical persona note", json.dumps(client.calls))

    def test_wrong_source_session_or_native_index_rejected(self):
        for field, value in (("session_id", "unknown"), ("native_message_index", 100)):
            with self.subTest(field=field):
                row = {"retrieved": [{"id": "id", "content": "[source " + persona.canonical_json({
                    "role": "user", "session_id": self.histories[0]["sessions"][0]["session_id"], field: value}) + "]\n" +
                    self.histories[0]["sessions"][0]["messages"][1]["content"]}]}
                with self.assertRaises(pair.Error):
                    pair.verified_items(row, self.histories[0], self.lineage[0])

    def test_missing_duplicate_and_mismatched_question_sets_rejected(self):
        original = (self.candidate / "retrieval.jsonl").read_text()
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[0])),
                         lambda rows: rows.pop(0)):
            with self.subTest(mutation=mutation):
                self.edit_rows(self.candidate, mutation)
                with self.assertRaises(pair.Error):
                    pair.run(self.args(), encoding=Encoding())
                (self.candidate / "retrieval.jsonl").write_text(original)

    def test_failed_retrieval_parse_failure_and_429_remain_in_full_denominator(self):
        self.edit_rows(self.baseline, lambda rows: next(r for r in rows if r["query_id"] == "q2").update(status="ingestion_error"))
        report, client = self.execute([response("unparseable"), response(), pair.qa.ProviderError("http_429")])
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(report["stopped"])
        baseline, candidate = report["metrics"]["baseline"], report["metrics"]["candidate"]
        self.assertEqual(baseline["parse_failures"], 1)
        self.assertEqual(baseline["status_counts"], {"invalid_answer": 1, "retrieval_error": 1, "run_stopped": 1})
        self.assertEqual(candidate["status_counts"], {"ok": 1, "answer_error": 1, "run_stopped": 1})
        self.assertEqual(baseline["denominator"], 3)
        self.assertEqual(candidate["result_rows"], 3)
        self.assertEqual(report["usage"]["unknown_requests"], 1)

    def test_unknown_usage_stops_remaining_attempts(self):
        report, client = self.execute([response(usage=False)])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(report["usage"]["unknown_requests"], 1)
        self.assertEqual(report["metrics"]["candidate"]["status_counts"], {"run_stopped": 3})

    def test_non_dev_summary_and_symlink_refused_before_text_open(self):
        path = self.candidate / "summary.json"
        summary = json.loads(path.read_text())
        summary["split"] = "validation"
        path.write_text(json.dumps(summary))
        (self.candidate / "retrieval.jsonl").unlink()
        with self.assertRaises(pair.Error):
            pair.run(self.args(), encoding=Encoding())
        outside = self.root / "heldout" / "forbidden.jsonl"
        outside.parent.mkdir()
        outside.write_text("FORBIDDEN")
        (self.folder / "queries.jsonl").unlink()
        (self.folder / "queries.jsonl").symlink_to(outside)
        with self.assertRaises(pair.Error):
            pair.selection(self.data)

    def test_changed_cutoff_or_source_hash_is_rejected(self):
        self.lineage[0]["exclusive_end_index"] += 1
        self.write_prepared()
        with self.assertRaisesRegex(pair.Error, "cutoff"):
            pair.selection(self.data)

    def test_remote_endpoints_and_formal_models_have_no_interface(self):
        for endpoint in ("https://api.openai.com/v1/chat/completions", "http://192.0.2.1:18080/v1/chat/completions",
                         "http://localhost:18080/v1/chat/completions", "http://127.0.0.1:18080/v1/chat/completions?key=x",
                         "http://127.0.0.1:18080/other/chat/completions"):
            with self.subTest(endpoint=endpoint), self.assertRaises(pair.Error):
                pair.run(self.args(["--endpoint", endpoint]), encoding=Encoding())
        self.assertFalse(any(a.dest in {"model", "answer_model", "allow_remote"} for a in pair.parser()._actions))

    def test_request_ceiling_and_execute_gate_checked_before_credentials(self):
        args = self.args(["--execute", "--output-dir", str(self.private / "output"), "--max-requests", "5"])
        with patch.object(pair, "scoring_references", side_effect=AssertionError("no gold for insufficient budget")):
            with self.assertRaisesRegex(pair.Error, "ceiling"):
                pair.run(args, encoding=Encoding())
        args.command = "plan"
        with self.assertRaises(pair.Error):
            pair.run(args, encoding=Encoding())

    def test_gold_checksum_checked_only_for_execute(self):
        (self.folder / "gold.jsonl").write_text("invalid")
        pair.run(self.args(), encoding=Encoding())
        with self.assertRaisesRegex(pair.Error, "SHA"):
            self.execute([])

    def test_private_output_cannot_overwrite_or_escape(self):
        with self.assertRaises(pair.Error):
            pair.private_output(self.root / "public")
        existing = self.private / "already"
        existing.mkdir()
        with self.assertRaises(pair.Error):
            pair.private_output(existing)
        target = self.root / "external"
        target.mkdir()
        link = self.private / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(pair.Error):
            pair.private_output(link / "output")


if __name__ == "__main__":
    unittest.main()
