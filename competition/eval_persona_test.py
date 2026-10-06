"""Meaningful offline checks for PersonaMem prefix, split and MCQ boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import eval_persona as persona


def options():
    return repr(["(a) OPTION_A", "(b) OPTION_B", "(c) OPTION_C", "(d) OPTION_D"])


def row(index, context=None, cutoff="4", persona_id=None):
    return {"persona_id": persona_id or "p" + str(index), "question_id": "q" + str(index),
            "shared_context_id": context or "c" + str(index),
            "end_index_in_shared_context": cutoff,
            "user_question_or_message": "CSV_QUERY_" + str(index),
            "all_options": options(), "correct_answer": "(b)",
            "question_type": "PRIVATE_GRADER_CATEGORY", "topic": "PRIVATE_GRADER_TOPIC"}


def fixture():
    rows = [row(i) for i in range(4)]
    contexts = {"c" + str(i): [
        {"role": "system", "content": "Persona description " + str(i)},
        {"role": "user", "content": "Native remembered fact " + str(i)},
        {"role": "assistant", "content": "Native assistant turn " + str(i)},
        {"role": "user", "content": "Native final user turn " + str(i)},
        {"role": "assistant", "content": "FUTURE_ANSWER_SENTINEL_" + str(i)},
    ] for i in range(4)}
    return rows, contexts


def flatten(records, field):
    result = {}
    for split in persona.SPLITS:
        result.update(records[split][field])
    return result


class MultipleChoiceTests(unittest.TestCase):
    def test_preserves_original_option_text_and_order(self):
        raw = "[\n '(a) Alpha', '(b) Beta', '(c) Gamma', '(d) Delta'\n]"
        query = {"query": "Public question", "all_options": raw, "options": persona.parse_options(raw),
                 "answer": "DO_NOT_SERIALIZE_GOLD", "category": "DO_NOT_SERIALIZE_CATEGORY"}
        messages = persona.build_answer_messages(query, "Recorded evidence")
        body = json.loads(messages[0]["content"].split("\n\n", 1)[1])
        self.assertEqual(body["all_options"], raw)
        self.assertEqual(set(body), {"historical_excerpts", "question", "all_options"})
        self.assertNotIn("DO_NOT_SERIALIZE", messages[0]["content"])
        self.assertEqual(messages[0]["role"], "user")

    def test_options_reject_code_ambiguity_and_reordering(self):
        for raw in ("__import__('os').system('false')", "('a', 'b', 'c', 'd')", "['(a) X']",
                    "['(b) X', '(a) Y', '(c) Z', '(d) W']", "['(a) ', '(b) Y', '(c) Z', '(d) W']"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                persona.parse_options(raw)

    def test_answer_builder_rejects_options_inconsistent_with_original(self):
        query = {"query": "Question", "all_options": options(), "options": ["replacement"]}
        with self.assertRaises(ValueError):
            persona.build_answer_messages(query, "history")

    def test_strict_final_choice_and_reasoning_prefix(self):
        examples = {"a": "a", " (B) ": "b", "<final_answer>(c)": "c",
                    "Discuss (a) and (b).\n<final_answer>(d)</final_answer>": "d"}
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(persona.parse_choice(text), expected)

    def test_no_ambiguous_or_whole_response_fallback(self):
        examples = [None, 1, "", "(a) or (b)", "The best is (a)", '{"choice":"a"}',
                    "<final_answer>(a) because of evidence", "<final_answer>(a)</final_answer> (b)",
                    "<final_answer>(b)<final_answer>(a)", "<final_answer>(a) (a)", "(e)",
                    "</final_answer>reasoning<final_answer>(a)",
                    "<final_answer>(a)</final_answer></final_answer>"]
        for text in examples:
            with self.subTest(text=text), self.assertRaises(ValueError):
                persona.parse_choice(text)

    def test_invalid_answers_remain_wrong_not_omitted(self):
        self.assertEqual(persona.score_choice("uncertain", "(a)"),
                         {"status": "invalid_answer", "selected_option": None, "correct": False})
        self.assertTrue(persona.score_choice("<final_answer>(A)", "a")["correct"])
        self.assertFalse(persona.score_choice("(b)", "(a)")["correct"])
        with self.assertRaises(ValueError):
            persona.score_choice("(a)", "(a) or (b)")


class HistoricalBoundaryTests(unittest.TestCase):
    def test_exclusive_cutoff_never_pairs_with_future_assistant(self):
        rows, contexts = fixture()
        records, audit = persona.build_records(rows, contexts, "fixture-seed")
        self.assertEqual(audit["questions"], 4)
        for history in flatten(records, "histories").values():
            payloads = persona.add_payloads(history, "isolated-user", "fixture-run")
            body = json.dumps(payloads)
            self.assertNotIn("FUTURE_ANSWER_SENTINEL", body)
            self.assertNotIn("CSV_QUERY", body)
            self.assertNotIn("OPTION_", body)
            self.assertNotIn("PRIVATE_GRADER", body)
            self.assertEqual(sum(len(p["messages"]) for p in payloads), 4)
            self.assertTrue(all(p["user_id"] == "isolated-user" for p in payloads))

    def test_native_terminal_query_kept_but_next_answer_excluded(self):
        rows, contexts = fixture()
        contexts["c0"][3]["content"] = rows[0]["user_question_or_message"]
        records, audit = persona.build_records(rows, contexts, "fixture-seed")
        self.assertEqual(audit["native_question_is_last_history_user_turn"], 1)
        for history in flatten(records, "histories").values():
            self.assertNotIn("FUTURE_ANSWER_SENTINEL", json.dumps(history))
        self.assertTrue(any(rows[0]["user_question_or_message"] in json.dumps(h)
                            for h in flatten(records, "histories").values()))

    def test_system_descriptions_preserved_as_history_with_original_positions(self):
        native = [{"role": "system", "content": "Description one"},
                  {"role": "user", "content": "User fact"},
                  {"role": "system", "content": "Description two"},
                  {"role": "assistant", "content": "Assistant fact"}]
        history, sources = persona.history_prefix(native, 4, "sample")
        safe = [m for session in history["sessions"] for m in session["messages"]]
        self.assertEqual([x["source_role"] for x in sources], [m["role"] for m in native])
        self.assertEqual([x["native_message_index"] for x in sources], [0, 1, 2, 3])
        self.assertEqual([m["role"] for m in safe], ["user", "user", "user", "assistant"])
        self.assertEqual(safe[0]["content"], "[historical system note]\nDescription one")
        self.assertEqual(safe[2]["content"], "[historical system note]\nDescription two")
        self.assertTrue(all(set(m) == {"role", "content"} for m in safe))
        # A cutoff directly after system must not pull its following user into history.
        short, _ = persona.history_prefix(native, 3, "sample")
        self.assertNotIn("Assistant fact", json.dumps(short))

    def test_both_user_and_assistant_ended_prefixes(self):
        _, contexts = fixture()
        for cutoff, role in ((3, "assistant"), (4, "user")):
            history, _ = persona.history_prefix(contexts["c0"], cutoff, "sample")
            self.assertEqual(history["sessions"][-1]["messages"][-1]["role"], role)
        for cutoff in (0, -1, 6, True, 3.0):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                persona.history_prefix(contexts["c0"], cutoff, "sample")

    def test_add_whitelist_and_chunk_identity(self):
        _, contexts = fixture()
        history, _ = persona.history_prefix(contexts["c0"], 4, "sample")
        history["gold"] = "SECRET_REFERENCE"
        history["sessions"][0]["messages"][0]["correct_answer"] = "SECRET_REFERENCE"
        first = persona.add_payloads(history, "u1", "run", max_messages=2)
        second = persona.add_payloads(history, "u2", "run", max_messages=2)
        self.assertNotIn("SECRET_REFERENCE", json.dumps(first))
        self.assertTrue(all(set(p) == {"request_id", "user_id", "session_id", "messages"} for p in first))
        self.assertTrue(all(len(p["messages"]) <= 2 for p in first))
        self.assertFalse({p["request_id"] for p in first} & {p["request_id"] for p in second})


class GroupingTests(unittest.TestCase):
    def test_transitive_complete_user_context_and_content_grouping(self):
        log = [{"role": "user", "content": "Same complete log"}]
        contexts = {"a": [{"role": "user", "content": "Different log"}], "b": log, "d": copy.deepcopy(log)}
        rows = [row(0, "a", "1", "pa"), row(1, "b", "1", "pa"),
                row(2, "b", "1", "pc"), row(3, "d", "1", "pd")]
        groups, _ = persona.history_groups(rows, contexts)
        self.assertEqual(len(set(groups)), 1)

    def test_question_gold_and_category_do_not_control_history_or_split(self):
        rows, contexts = fixture()
        first, _ = persona.build_records(rows, contexts, "fixture-seed")
        changed = copy.deepcopy(rows)
        for r in changed:
            r.update(user_question_or_message="Changed question", correct_answer="(d)",
                     all_options=repr(["(a) Different A", "(b) Different B", "(c) Different C", "(d) Different D"]),
                     question_type="Changed grader category")
        second, _ = persona.build_records(changed, contexts, "fixture-seed")
        for split in persona.SPLITS:
            self.assertEqual(first[split]["histories"], second[split]["histories"])
            self.assertEqual(first[split]["lineage"], second[split]["lineage"])
            self.assertEqual(set(first[split]["queries"]), set(second[split]["queries"]))

    def test_row_order_and_prefixes_preserve_split_membership(self):
        rows, contexts = fixture()
        rows.append({**rows[0], "question_id": "additional", "end_index_in_shared_context": "3"})
        first, _ = persona.build_records(rows, contexts, "fixture-seed")
        second, _ = persona.build_records(list(reversed(rows)), contexts, "fixture-seed")
        self.assertEqual(first, second)
        person_splits = {}
        for split in persona.SPLITS:
            for record in first[split]["lineage"].values():
                native = record["native_persona_id"]
                self.assertEqual(person_splits.setdefault(native, split), split)

    def test_shared_prefix_prepared_once_and_all_questions_kept(self):
        rows, contexts = fixture()
        rows.append({**rows[0], "question_id": "additional"})
        records, audit = persona.build_records(rows, contexts, "fixture-seed")
        self.assertEqual(len(flatten(records, "histories")), 4)
        self.assertEqual(len(flatten(records, "queries")), 5)
        self.assertEqual(len(flatten(records, "gold")), 5)
        self.assertEqual(audit["questions"], 5)

    def test_duplicate_id_bad_cutoff_and_unknown_context_fail_closed(self):
        rows, contexts = fixture()
        for change in ({"end_index_in_shared_context": "4.0"}, {"shared_context_id": "missing"}):
            changed = copy.deepcopy(rows)
            changed[0].update(change)
            with self.assertRaises(ValueError):
                persona.build_records(changed, contexts, "fixture-seed")
        with self.assertRaises(ValueError):
            persona.build_records(rows + [rows[0]], contexts, "fixture-seed")


class PreparationTests(unittest.TestCase):
    def test_real_manifest_pins_revision_size_and_sha(self):
        manifest = persona.read_manifest(persona.DEFAULT_MANIFEST)
        self.assertEqual(manifest["revision"], persona.REVISION)
        if not any((persona.DEFAULT_RAW / item["name"]).exists() for item in manifest["files"]):
            self.skipTest("Optional pinned PersonaMem dataset has not been downloaded")
        verified = persona.verify_files(persona.DEFAULT_RAW, manifest["files"])
        self.assertEqual(set(verified), set(persona.SOURCE_FILES))

    def test_same_size_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.txt"
            path.write_bytes(b"abc")
            spec = [{"name": path.name, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}]
            persona.verify_files(Path(tmp), spec)
            path.write_bytes(b"abd")
            with self.assertRaises(ValueError):
                persona.verify_files(Path(tmp), spec)

    def test_network_free_separate_output_hashes_and_no_overwrite(self):
        rows, contexts = fixture()
        with patch.object(socket, "socket", side_effect=AssertionError("Network forbidden")):
            records, _ = persona.build_records(rows, contexts, "fixture-seed")
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(persona, "PRIVATE_ROOT", Path(tmp)):
                root = Path(tmp) / "prepared-fixture"
                report = persona.write_prepared(root, records, {"version": "fixture"})
                before = (root / "prepared" / "summary.json").read_bytes()
                for split, detail in report["datasets"][persona.DATASET]["splits"].items():
                    folder = root / "prepared" / persona.DATASET / split
                    for name, spec in detail["files"].items():
                        self.assertEqual(persona.file_hash(folder / (name + ".jsonl")), spec["sha256"])
                    self.assertNotIn("PRIVATE_GRADER", (folder / "histories.jsonl").read_text())
                    self.assertNotIn('"gold_option"', (folder / "queries.jsonl").read_text())
                with self.assertRaises(FileExistsError):
                    persona.write_prepared(root, records, {"version": "overwrite"})
                self.assertEqual((root / "prepared" / "summary.json").read_bytes(), before)

    def test_output_must_be_private_and_profile_has_unconfirmed_production_boundary(self):
        rows, contexts = fixture()
        records, _ = persona.build_records(rows, contexts, "fixture-seed")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            persona.write_prepared(Path(tmp) / "public-output", records, {})
        profile = persona.profile()
        self.assertFalse(profile["production_models_confirmed"])
        self.assertFalse(profile["production_tokenizer_budget_mapping_confirmed"])
        self.assertFalse(profile["source_code_redistribution_license_verified"])


if __name__ == "__main__":
    unittest.main()
