#!/usr/bin/env python3
"""Meaningful offline checks for leakage boundaries and evidence verification.

These fixtures are synthetic correctness tests, never reported as benchmark scores.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

from eval_prepare import LEGACY_SOURCE_MAPPING_VERSION, SOURCE_MAPPING_VERSION, allocate_splits, build_locomo, build_longmemeval, evidence_ids, file_hash, json_line, migrate_dev_sources, migrate_history_sources, normalized
from eval_retrieval import aggregate, evidence_hits, local_embedding_environment, requests_for_history, validate_search
from eval_compare import compare
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import MemoryStore


class ComparisonProvenanceTests(unittest.TestCase):
    def test_search_only_comparison_verifies_original_ingestion_namespace(self):
        history = {"sample_id": "history", "sessions": [{"session_id": "session", "messages": [
            {"role": "user", "content": "Repeated source text."},
            {"role": "assistant", "content": "Repeated source text."}]}]}
        _, payloads, _, ids = requests_for_history(history, "ingestion")
        source = {"session_id": "session", "request_id": payloads[0]["request_id"], "message_index": 1}
        message_id = next(mid for mid, original in ids.items() if original["position"] == ("session", 1))
        record = {"query_id": "query", "sample_id": "history", "status": "ok",
                  "retrieved": [{"id": message_id, "content": "[source " + json.dumps(source) + "]\nRepeated source text."}],
                  "metrics": {"5": {"hit_turns": 1, "source_verified_hit_turns": 1}}}
        gold = {"query_id": "query", "sample_id": "history", "gold_session_ids": ["session"],
                "gold_units": [{"unit_id": "gold", "session_id": "session", "message_position": 1,
                                "text": "Repeated source text."}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared" / "locomo_refined" / "dev"
            prepared.mkdir(parents=True)
            for name, row in (("histories", history), ("queries", {"query_id": "query"}), ("gold", gold)):
                (prepared / (name + ".jsonl")).write_text(json_line(row), encoding="utf-8")
            specification = {"files": {name: {"sha256": file_hash(prepared / (name + ".jsonl"))}
                                       for name in ("histories", "queries", "gold")}}
            runs = [root / "ingestion", root / "search-only"]
            for folder in runs:
                folder.mkdir()
                summary = {"run_id": folder.name, "dataset": "locomo_refined", "split": "dev",
                           "question_count": 1, "model_label": folder.name, "elapsed_seconds": 0,
                           "add_latency": {}, "search_latency": {}, "configuration": {
                               "top_k": 100, "character_budget": 0, "prepared_data": specification,
                               "source_mapping_version": SOURCE_MAPPING_VERSION}}
                if folder.name == "search-only":
                    summary["ingestion_run_id"] = "ingestion"
                (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                (folder / "retrieval.jsonl").write_text(json_line(record), encoding="utf-8")
            result = compare(runs, root, [5], [0])
        for run in result["runs"]:
            metrics = run["metrics"]["k5/characters0"]
            self.assertEqual(metrics["full_turn_recall_all"], 1)
            self.assertEqual(metrics["source_verified_turn_recall_macro"], 1)
            self.assertEqual(metrics["regrading_changes_vs_original_saved_metrics"][
                "questions_with_changed_verified_hit_turn_count"], 0)
        self.assertEqual(result["runs"][1]["metrics"]["k5/characters0"][
            "paired_complete_coverage_changes_vs_first_run"], {"gained_questions": 0, "lost_questions": 0})


class EvidenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.history = {"sample_id": "opaque-history", "sessions": [{"session_id": "opaque-session", "messages": [
            {"role": "user", "content": "[session_date: 2023/05/20 (Sat) 02:21]\nAlice moved to Paris.", "timestamp": 0},
            {"role": "assistant", "content": "[session_date: 2023/05/20 (Sat) 02:21]\nI understand.", "timestamp": 0}]}]}
        self.user, self.payloads, self.sources, self.ids = requests_for_history(self.history, "test-run")
        self.gold = {"gold_units": [{"unit_id": "gold-unit", "session_id": "opaque-session", "message_position": 0,
                                    "text": "Alice moved to Paris."}], "gold_session_ids": ["opaque-session"]}

    def test_metadata_alone_does_not_count_as_evidence(self):
        anchor = next(iter(self.ids))
        result = evidence_hits([{"id": anchor, "content": "Unrelated facts about London."}], self.gold, self.sources, self.ids,
                               Counter({"Alice moved to Paris.": 1}))
        self.assertEqual(result["hit_turns"], 0)
        self.assertEqual(result["hit_sessions"], 0)

    def test_only_whole_turn_counts(self):
        result = evidence_hits([{"id": "unknown", "content": "Paris"}], self.gold, self.sources, self.ids,
                               Counter({"Alice moved to Paris.": 1}))
        self.assertEqual(result["hit_turns"], 0)

    def short_text_fixture(self, text):
        history = {"sample_id": "short-history", "sessions": [{"session_id": "gold-session", "messages": [
            {"role": "assistant", "content": text, "timestamp": 0}]}]}
        _, payloads, sources, ids = requests_for_history(history, "short-run")
        gold = {"gold_units": [{"unit_id": "gold", "session_id": "gold-session", "message_position": 0, "text": text}],
                "gold_session_ids": ["gold-session"]}
        return payloads, sources, ids, gold

    def test_role_metadata_cannot_trigger_unique_text_fallback(self):
        _, _, _, gold = self.short_text_fixture("assistant")
        content = '[source {"role":"assistant","session_id":"other","request_id":"unknown","message_index":0}]\nEntirely unrelated source text.'
        result = evidence_hits([{"id": "unrelated", "content": content}], gold, {}, {}, Counter({"assistant": 1}))
        self.assertEqual(result["hit_turns"], 0)
        self.assertFalse(result["recall_all"])

    def test_known_anchor_id_with_metadata_only_is_not_verified(self):
        _, sources, ids, gold = self.short_text_fixture("assistant")
        content = '[source {"role":"assistant","session_id":"other","request_id":"unknown","message_index":0}]\nEntirely unrelated source text.'
        result = evidence_hits([{"id": next(iter(ids)), "content": content}], gold, sources, ids, Counter({"assistant": 1}))
        self.assertEqual(result["hit_turns"], 0)
        self.assertEqual(result["source_verified_hit_turns"], 0)

    def test_adjacent_marker_is_not_message_evidence(self):
        _, _, _, gold = self.short_text_fixture("[adjacent context]")
        header = '[source {"role":"assistant","session_id":"other","request_id":"unknown","message_index":0}]\n'
        content = header + "Unrelated.\n\n[adjacent context]\n" + header + "Also unrelated."
        result = evidence_hits([{"id": "unrelated", "content": content}], gold, {}, {}, Counter({"[adjacent context]": 1}))
        self.assertEqual(result["hit_turns"], 0)

    def test_real_body_short_text_still_verifies(self):
        payloads, sources, ids, gold = self.short_text_fixture("assistant")
        source = {"session_id": "gold-session", "request_id": payloads[0]["request_id"], "message_index": 0}
        content = "[source " + json.dumps(source) + "]\nassistant"
        result = evidence_hits([{"id": next(iter(ids)), "content": content}], gold, sources, ids, Counter({"assistant": 1}))
        self.assertEqual(result["source_verified_hit_turns"], 1)

    def test_literal_source_and_adjacent_quotes_in_original_are_preserved(self):
        text = 'A literal example:\n[source {"role":"assistant"}]\nQuoted text.\n\n[adjacent context]\n[source {"role":"user"}]\nAnother quoted line.'
        payloads, sources, ids, gold = self.short_text_fixture(text)
        source = {"session_id": "gold-session", "request_id": payloads[0]["request_id"], "message_index": 0}
        content = "[source " + json.dumps(source) + "]\n" + text
        for returned_id, returned_content in (("unknown", content), (next(iter(ids)), text), ("unknown", text)):
            result = evidence_hits([{"id": returned_id, "content": returned_content}], gold, sources, ids, Counter({normalized(text): 1}))
            self.assertEqual(result["hit_turns"], 1)
            self.assertEqual(result["source_verified_hit_turns"], 0 if returned_id == "unknown" and returned_content == text else 1)

    def test_inherited_embedding_settings_and_keys_cannot_change_local_experiment(self):
        parent = {"MEMORY_EMBEDDING_PROVIDER": "http", "MEMORY_EMBEDDING_ALLOW_HTTP": "1",
                  "MEMORY_EMBEDDING_API_KEY": "SECRET_SENTINEL", "MEMORY_EMBEDDING_SEGMENT_TOKENS": "17",
                  "MEMORY_EMBEDDING_QUERY_PREFIX": "WRONG", "OTHER_SETTING": "preserved"}
        env = local_embedding_environment(parent, "local", Path("/already-downloaded-model"))
        self.assertNotIn("SECRET_SENTINEL", env.values())
        self.assertEqual(env["MEMORY_EMBEDDING_PROVIDER"], "local")
        self.assertEqual(env["MEMORY_EMBEDDING_ALLOW_HTTP"], "0")
        self.assertEqual(env["MEMORY_EMBEDDING_SEGMENT_TOKENS"], "384")
        self.assertEqual(env["MEMORY_EMBEDDING_QUERY_PREFIX"], "query: ")
        self.assertEqual(env["OTHER_SETTING"], "preserved")

    def test_full_turn_cannot_be_constructed_across_two_source_bodies(self):
        _, _, _, gold = self.short_text_fixture("First half. Second half.")
        header = '[source {"role":"assistant","session_id":"other","request_id":"unknown","message_index":0}]\n'
        content = header + "First half.\n\n[adjacent context]\n" + header + "Second half."
        result = evidence_hits([{"id": "unrelated", "content": content}], gold, {}, {}, Counter({"First half. Second half.": 1}))
        self.assertEqual(result["hit_turns"], 0)

    def test_adjacent_source_is_verified_separately(self):
        first = self.payloads[0]
        source = {"session_id": first["session_id"], "request_id": first["request_id"], "message_index": 0}
        content = "anchor\n\n[adjacent context]\n[source " + json.dumps(source) + "]\n" + first["messages"][0]["content"]
        result = evidence_hits([{"id": "window-id", "content": content}], self.gold, self.sources, self.ids,
                               Counter({"Alice moved to Paris.": 2}))
        self.assertEqual(result["source_verified_hit_turns"], 1)

    def test_repeated_text_without_location_is_not_a_hit(self):
        result = evidence_hits([{"id": "unknown", "content": "Alice moved to Paris."}], self.gold, self.sources, self.ids,
                               Counter({"Alice moved to Paris.": 2}))
        self.assertEqual(result["hit_turns"], 0)
        self.assertEqual(result["ambiguous_without_source"], 1)

    def test_contract_excess_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            validate_search({"data": [{"id": "a", "content": "text"}, {"id": "b", "content": "text"}]}, 1)

    def test_request_failure_remains_in_metric_denominator(self):
        record = {"status": "ok", "metrics": {"5": {"recall_any": True, "recall_all": True, "turn_recall": 1,
                  "source_verified_turn_recall": 1, "session_recall": 1, "session_recall_all": True,
                  "gold_turns": 1, "hit_turns": 1, "ambiguous_without_source": 0, "returned_characters": 20, "returned_items": 1}}}
        result = aggregate([record, {"status": "search_error"}], 5, 2)
        self.assertEqual(result["full_turn_recall_all"], .5)
        self.assertEqual(result["failed_questions"], 1)


class SplitAndLabelTests(unittest.TestCase):
    def test_group_splits_are_order_independent(self):
        groups = ["group-" + str(i) for i in range(10)]
        a = allocate_splits("fixture", groups, "fixed")
        b = allocate_splits("fixture", list(reversed(groups)), "fixed")
        self.assertEqual(a, b)
        self.assertEqual(Counter(a.values()), {"dev": 6, "validation": 2, "heldout": 2})

    def test_evidence_parser_does_not_guess_unknown_locations(self):
        self.assertEqual(evidence_ids(["D8:6; D9:17", "D30:05"]), ["D30:5", "D8:6", "D9:17"])
        self.assertIsNone(evidence_ids(["D"]))
        self.assertIsNone(evidence_ids(["D1:1 or perhaps elsewhere"]))

    def test_longmemeval_answer_and_turn_label_never_enter_add(self):
        rows = []
        for i in range(8):
            sid = "shared-evidence" if i < 2 else "session-" + str(i)
            rows.append({"question_id": "q" + str(i), "question_type": "multi-session", "question": "FUTURE_QUESTION_SENTINEL",
                         "answer": "GOLD_ANSWER_SENTINEL", "question_date": "2023/05/21 (Sun) 12:00",
                         "answer_session_ids": [sid], "haystack_session_ids": [sid],
                         "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                         "haystack_sessions": [[{"role": "user", "content": "History only.", "has_answer": True}]]})
        prepared, _ = build_longmemeval(rows, "fixture", "fixed")
        split_for_query = {}
        for split, entries in prepared.items():
            for history, queries, gold in entries:
                _, requests, _, _ = requests_for_history(history, "fixture")
                body = json.dumps(requests)
                self.assertNotIn("FUTURE_QUESTION_SENTINEL", body)
                self.assertNotIn("GOLD_ANSWER_SENTINEL", body)
                self.assertNotIn("has_answer", body)
                self.assertNotIn("answer_session_ids", body)
                split_for_query[gold[0]["original_query_id"]] = split
        self.assertEqual(split_for_query["q0"], split_for_query["q1"])

    def test_locomo_entire_conversation_stays_in_one_split(self):
        rows = []
        for i in range(5):
            rows.append({"sample_id": "conversation-" + str(i), "conversation": {"speaker_a": "A", "speaker_b": "B",
                         "session_1_date_time": "1:56 pm on 8 May, 2023", "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "History only."}]},
                         "qa": [{"question": "FUTURE_Q", "answer": ["GOLD_ANSWER"], "evidence": ["D1:1"], "category": 1,
                                 "is_multi_modality": False} for _ in range(3)]})
        prepared, _ = build_locomo(rows, "fixture", "fixed")
        observed = {}
        for split, entries in prepared.items():
            for history, queries, _ in entries:
                self.assertEqual(len(queries), 3)
                self.assertNotIn(history["sample_id"], observed)
                observed[history["sample_id"]] = split
                body = json.dumps(requests_for_history(history, "fixture")[1])
                self.assertNotIn("FUTURE_Q", body)
                self.assertNotIn("GOLD_ANSWER", body)
                self.assertNotIn("evidence", body)


class SessionOccurrenceTests(unittest.TestCase):
    def legacy_fixture(self, same_text=False):
        first = {"role": "user", "content": "First date: distractoralpha.", "timestamp": 0}
        second = {"role": "user", "content": "Second date: evidencebeta.", "timestamp": 1}
        if same_text:
            second = dict(first)
        history = {"sample_id": "history", "sessions": [
            {"session_id": "repeated-session", "messages": [first]},
            {"session_id": "repeated-session", "messages": [second]}]}
        label = {"sample_id": "history", "query_id": "query", "answer": "FUTURE_ANSWER_SENTINEL",
                 "gold_units": [{"unit_id": "legacy-unit", "session_id": "repeated-session", "message_position": 0,
                                 "text": second["content"], "source_message_content": second["content"]}],
                 "gold_session_ids": ["repeated-session"]}
        return history, label

    def test_duplicate_session_migration_add_retry_and_source_grading(self):
        original, label = self.legacy_fixture()
        history, labels, repeats = migrate_history_sources(original, [label])
        self.assertEqual(repeats, 1)
        self.assertEqual(history["sessions"][0], original["sessions"][0])
        self.assertNotEqual(history["sessions"][0]["session_id"], history["sessions"][1]["session_id"])
        self.assertEqual(original["sessions"][0]["session_id"], original["sessions"][1]["session_id"])
        gold = labels[0]
        self.assertEqual(gold["gold_units"][0]["session_id"], history["sessions"][1]["session_id"])
        self.assertNotEqual(gold["gold_units"][0]["unit_id"], label["gold_units"][0]["unit_id"])
        self.assertEqual(set(gold["gold_session_ids"]), {s["session_id"] for s in history["sessions"]})
        user, payloads, sources, ids = requests_for_history(history, "stable-run")
        self.assertEqual(payloads, requests_for_history(history, "stable-run")[1])
        self.assertEqual(len({p["request_id"] for p in payloads}), 2)
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(ids), 2)
        self.assertNotIn("FUTURE_ANSWER_SENTINEL", json.dumps(payloads))
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite3", context_radius=0)
            for payload in payloads:
                acknowledgement = store.add(payload)
                self.assertEqual(store.add(payload), acknowledgement)
            first = store.search({"user_id": user, "query": "distractoralpha", "top_k": 5})["data"]
            second = store.search({"user_id": user, "query": "evidencebeta", "top_k": 5})["data"]
        frequencies = Counter({normalized(label["gold_units"][0]["text"]): 1})
        self.assertEqual(evidence_hits(first, gold, sources, ids, frequencies)["hit_turns"], 0)
        self.assertEqual(evidence_hits(second, gold, sources, ids, frequencies)["source_verified_hit_turns"], 1)

    def test_unmigrated_duplicate_history_is_rejected_but_legacy_regrading_is_explicit(self):
        history, _ = self.legacy_fixture()
        with self.assertRaisesRegex(ValueError, "migrate dev"):
            requests_for_history(history, "run")
        legacy = requests_for_history(history, "run", LEGACY_SOURCE_MAPPING_VERSION)[1]
        self.assertEqual(legacy[0]["request_id"], legacy[1]["request_id"])

    def test_ambiguous_legacy_gold_occurrence_is_not_guessed(self):
        history, label = self.legacy_fixture(same_text=True)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            migrate_history_sources(history, [label])

    def test_unique_source_ids_and_labels_keep_legacy_rules(self):
        history, label = self.legacy_fixture()
        history["sessions"] = history["sessions"][1:]
        upgraded, labels, repeats = migrate_history_sources(history, [label])
        self.assertEqual(upgraded, history)
        self.assertEqual(labels, [label])
        self.assertEqual(repeats, 0)
        self.assertEqual(requests_for_history(upgraded, "run"),
                         requests_for_history(history, "run", LEGACY_SOURCE_MAPPING_VERSION))

    def test_raw_builder_and_legacy_migration_agree_for_repeated_native_sessions(self):
        rows = []
        for i in range(5):
            sid = "native-" + str(i)
            rows.append({"question_id": "q" + str(i), "question_type": "single-session-user",
                         "question": "FUTURE_QUESTION_SENTINEL", "answer": "FUTURE_ANSWER_SENTINEL",
                         "question_date": "2023/05/22 (Mon) 12:00", "answer_session_ids": [sid],
                         "haystack_session_ids": [sid, sid],
                         "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 02:21"],
                         "haystack_sessions": [[{"role": "user", "content": "Earlier source."}],
                                               [{"role": "user", "content": "Later evidence.", "has_answer": True}]]})
        prepared, _ = build_longmemeval(rows, "fixture", "fixed")
        for entries in prepared.values():
            for history, queries, labels in entries:
                sid = history["sessions"][0]["session_id"]
                legacy_history = json.loads(json.dumps(history))
                legacy_history["sessions"][1]["session_id"] = sid
                legacy_labels = json.loads(json.dumps(labels))
                unit = legacy_labels[0]["gold_units"][0]
                # The first-occurrence legacy unit ID comes from the same native turn.
                from eval_prepare import digest
                native = rows[int(labels[0]["original_query_id"][1:])]["haystack_session_ids"][0]
                unit["unit_id"] = digest(history["sample_id"] + "\0" + native + "\0" + "0")
                unit["session_id"] = sid
                legacy_labels[0]["gold_session_ids"] = [sid]
                actual_history, actual_labels, _ = migrate_history_sources(legacy_history, legacy_labels)
                self.assertEqual(actual_history, history)
                self.assertEqual(actual_labels, labels)
                body = json.dumps(requests_for_history(history, "run")[1])
                self.assertNotIn(queries[0]["query"], body)
                self.assertNotIn(labels[0]["answer"], body)

    def test_dev_migration_preserves_old_files_queries_split_and_avoids_other_splits(self):
        history, label = self.legacy_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory) / "old"
            folder = source_root / "prepared" / "fixture" / "dev"
            folder.mkdir(parents=True)
            files = {"histories": history, "gold": label,
                     "queries": {"sample_id": "history", "query_id": "query", "query": "FUTURE_QUESTION_SENTINEL"}}
            for name, row in files.items():
                (folder / (name + ".jsonl")).write_text(json_line(row))
            specifications = {name: {"size": (folder / (name + ".jsonl")).stat().st_size,
                                    "sha256": file_hash(folder / (name + ".jsonl"))} for name in files}
            original = {"manifest_sha256": "fixed-manifest", "split_seed": "fixed-seed", "timezone_convention": "fixed-UTC",
                        "datasets": {"fixture": {"splits": {"dev": {"histories": 1, "questions": 1, "messages": 2,
                                                                    "gold_turn_annotations": 1, "files": specifications},
                                                            "validation": {}, "heldout": {}}}}}
            (source_root / "prepared" / "summary.json").write_text(json.dumps(original))
            target_root = Path(directory) / "new"
            upgraded = migrate_dev_sources(source_root, target_root, ["fixture"])
            self.assertEqual(upgraded["split_seed"], original["split_seed"])
            self.assertEqual(upgraded["source_mapping_version"], SOURCE_MAPPING_VERSION)
            self.assertEqual(set(upgraded["datasets"]["fixture"]["splits"]), {"dev"})
            target = target_root / "prepared" / "fixture" / "dev"
            self.assertEqual((target / "queries.jsonl").read_bytes(), (folder / "queries.jsonl").read_bytes())
            for name in files:
                self.assertEqual(file_hash(folder / (name + ".jsonl")), specifications[name]["sha256"])
            self.assertFalse((target.parent / "validation").exists())
            self.assertFalse((target.parent / "heldout").exists())
            with self.assertRaisesRegex(ValueError, "new, absent"):
                migrate_dev_sources(source_root, target_root, ["fixture"])


if __name__ == "__main__":
    unittest.main()
