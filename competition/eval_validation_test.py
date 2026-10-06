#!/usr/bin/env python3
"""Identity isolation and actual local-store governance regressions."""
from contextlib import redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_validation as validation


def history(sample, dataset="public-test", content=None, **identities):
    return {"dataset": dataset, "sample_id": sample, **identities,
            "sessions": [{"session_id": "sample-scoped-" + sample,
                          "messages": [{"role": "user", "content": content or "Unique fact " + sample}]}]}


def questions(*histories):
    return [{"dataset": row["dataset"], "sample_id": row["sample_id"],
             "query_id": row["sample_id"] + "-q", "query": "An independently worded question"} for row in histories]


def partition(manifest, sample):
    return next(group["partition"] for group in manifest["groups"]
                if any(row["sample_id"] == sample for row in group["members"]))


class IsolationTests(unittest.TestCase):
    def test_preserve_original_partitions_and_quarantine_cross_split_overlap(self):
        histories = [history("dev", original_split="dev"), history("validation", original_split="validation"),
                     history("holdout", original_split="heldout"),
                     history("cross-val", original_split="validation", conversation_id="shared"),
                     history("cross-held", original_split="heldout", conversation_id="shared"),
                     history("used-val", original_split="validation", user_id="already-used")]
        used = [{"dataset": "public-test", "user_ids": ["already-used"]}]
        result = validation.build_validation_manifest(histories, questions(*histories), used, preserve_prepared_splits=True)
        self.assertEqual({key: partition(result, key) for key in ("dev", "validation", "holdout", "cross-val", "cross-held", "used-val")},
                         {"dev": "development", "validation": "validation", "holdout": "holdout", "cross-val": "development", "cross-held": "development", "used-val": "development"})
        self.assertEqual(result["allocation"], "preserved_prepared_splits")
        self.assertIsNone(result["seed"])
        self.assertIsNone(result["fractions"])
        self.assertEqual(result["forced_development_groups"], 2)
        self.assertEqual(result, validation.build_validation_manifest(histories[::-1], questions(*histories)[::-1], used,
                                                                      seed="A different hash seed is irrelevant", preserve_prepared_splits=True))
        # Repeated exact sample identity across folders must also be quarantined.
        duplicated = [history("same", original_split="validation"), history("same", original_split="heldout")]
        result = validation.build_validation_manifest(duplicated, questions(duplicated[0]), preserve_prepared_splits=True)
        self.assertEqual(partition(result, "same"), "development")
        with self.assertRaises(ValueError):
            validation.build_validation_manifest([history("missing")], questions(history("missing")), preserve_prepared_splits=True)

    def test_pinned_native_session_reuse_links_date_changed_histories_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"question_id": "native-a", "haystack_session_ids": ["source-one"], "answer": "unused answer", "has_answer": True},
                    {"question_id": "native-b", "haystack_session_ids": ["source-one", "source-two"], "answer": "another unused answer"},
                    {"question_id": "native-c", "haystack_session_ids": ["source-two"], "has_answer": False}]
            histories = [history(hashlib.sha256(("longmemeval_s\0" + row["question_id"]).encode()).hexdigest()[:32],
                                 dataset="longmemeval_s", content="A date-modified transcript " + str(index)) for index, row in enumerate(rows)]
            data = root / "native.json"
            manifest = root / "manifest.json"

            def pin():
                data.write_text(json.dumps(rows))
                manifest.write_text(json.dumps({"datasets": [{"id": "longmemeval_s", "files": [{"name": "native.json", "sha256": validation.file_hash(data), "size": data.stat().st_size}]}]}))

            pin()
            identities, sources = validation.native_identity_map(histories, root, manifest)
            self.assertEqual(len(identities), 3)
            self.assertTrue(sources[0]["pin_verified"])
            self.assertFalse(sources[0]["answer_or_evidence_labels_used"])
            self.assertFalse(sources[0]["real_user_identity"])
            self.assertEqual(sources[0]["fields_selected"], ["question_id", "haystack_session_ids"])
            result = validation.build_validation_manifest(histories, questions(*histories),
                                                          [{"dataset": "longmemeval_s", "sample_ids": [histories[0]["sample_id"]]}], identities)
            self.assertEqual(result["independent_groups"], 1)
            self.assertEqual(result["question_counts"], {"development": 3, "validation": 0, "holdout": 0})
            rows[0]["answer"], rows[0]["has_answer"] = "Changed labels do not choose identities", False
            pin()
            same, _ = validation.native_identity_map(histories, root, manifest)
            self.assertEqual(same, identities)
            data.write_text(data.read_text() + " ")
            with self.assertRaises(ValueError):
                validation.native_identity_map(histories, root, manifest)

    def test_identity_map_merges_native_conversation_and_external_real_user(self):
        native = [{"dataset": "public-test", "sample_id": "a", "conversation_ids": ["native-conversation"]}]
        real_user = [{"dataset": "public-test", "sample_id": "a", "user_id": "actual-person"}]
        merged = validation.merge_identity_maps(native, real_user, native)
        self.assertEqual(merged, [{"dataset": "public-test", "sample_id": "a", "conversation_ids": ["native-conversation"], "user_ids": ["actual-person"]}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_text(json.dumps({"identities": merged, "metadata": "No label fields"}))
            self.assertEqual(validation.load_identity_map(path), merged)
            path.write_text(json.dumps(native[0]) + "\n" + json.dumps(real_user[0]) + "\n")
            self.assertEqual(validation.load_identity_map(path), native + real_user)
        with self.assertRaises(ValueError):
            validation.merge_identity_maps([{"dataset": "public-test", "sample_id": "a", "user_ids": "not a list"}])

    def test_stable_hash_and_transitive_user_conversation_grouping(self):
        histories = [history("a", user_id="person-A", conversation_id="conversation-1"),
                     history("b", user_id="person-A", conversation_id="conversation-2"),
                     history("c", user_id="person-C", conversation_id="conversation-2")]
        histories += [history(str(index), user_id="person-" + str(index)) for index in range(40)]
        queries = questions(*histories)
        queries.append({**queries[0], "query_id": "second-question-same-person"})
        original = validation.build_validation_manifest(histories, queries)
        random.Random(9127).shuffle(histories)
        random.Random(1479).shuffle(queries)
        reordered = validation.build_validation_manifest(histories, queries)
        self.assertEqual(original, reordered)
        self.assertEqual(original["independent_groups"], 41)
        self.assertEqual(partition(original, "a"), partition(original, "c"))
        self.assertTrue(original["real_user_identity_available_for_all_histories"])
        self.assertTrue(all(original["question_counts"][name] > 0 for name in ("development", "validation", "holdout")))

    def test_used_identity_forces_entire_shared_source_component_to_development(self):
        a = history("a", content="Shared original session")
        b = history("b", content="Shared original session", conversation_id="linked-conversation")
        c = history("c", conversation_id="linked-conversation")
        untouched = history("unused")
        histories, queries = [a, b, c, untouched], questions(a, b, c, untouched)
        # No label/answer is needed, and even 100% non-dev allocation cannot
        # reclassify any query belonging to a previously used component.
        for usage in ({"dataset": "public-test", "query_ids": ["a-q"]},
                      {"dataset": "public-test", "sample_ids": ["a"]},
                      {"dataset": "public-test", "conversation_ids": ["linked-conversation"]}):
            with self.subTest(usage=usage):
                result = validation.build_validation_manifest(histories, queries, [usage],
                                                              validation_fraction=0.5, holdout_fraction=0.5)
                self.assertEqual({partition(result, value) for value in ("a", "b", "c")}, {"development"})
                self.assertNotEqual(partition(result, "unused"), "development")
                self.assertEqual(result["question_counts"]["development"], 3)
        self.assertFalse(result["real_user_identity_available_for_all_histories"])
        self.assertTrue(result["public_not_blind"])
        self.assertFalse(result["selection_reads_labels"])

    def test_source_identity_preserves_role_time_and_exact_user_identifiers(self):
        original = history("a", content="Same wording", user_id="Person")
        independent = history("b", content="Same wording", user_id=" Person ")
        independent["sessions"][0]["messages"][0]["role"] = "assistant"
        dated = history("c", content="Same wording", user_id="person")
        dated["sessions"][0]["messages"][0]["timestamp"] = 1000
        result = validation.build_validation_manifest([original, independent, dated], questions(original, independent, dated),
                                                      [{"dataset": "public-test", "user_ids": ["Person"]}],
                                                      validation_fraction=0.5, holdout_fraction=0.5)
        self.assertEqual(result["independent_groups"], 3)
        self.assertEqual(partition(result, "a"), "development")
        self.assertNotEqual(partition(result, "b"), "development")
        self.assertNotEqual(partition(result, "c"), "development")

    def test_labels_scope_conflicts_and_external_identity_conflicts_are_rejected(self):
        row = history("a")
        query = questions(row)[0]
        for field in ("answer", "gold", "gold_units", "category"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validation.build_validation_manifest([row], [{**query, field: "Not a selection input"}])
        with self.assertRaises(ValueError):
            validation.build_validation_manifest([row], [{**query, "sample_id": "absent"}])
        conflicted = copy.deepcopy(row)
        conflicted["sessions"][0]["messages"].append(copy.deepcopy(conflicted["sessions"][0]["messages"][0]))
        with self.assertRaises(ValueError):
            validation.build_validation_manifest([row, conflicted], [query])
        with self.assertRaises(ValueError):
            validation.build_validation_manifest([row], [query], identities=[{"dataset": "public-test", "sample_id": "absent", "user_id": "person"}])
        for fraction in (-0.1, True, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                validation.build_validation_manifest([row], [query], validation_fraction=fraction)

    def test_prepared_loader_never_reads_gold_and_preserves_embedded_unicode_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "public-test" / "heldout"
            path.mkdir(parents=True)
            row = history("a", content="Exact original\u2028text\u2029and newlines\nare preserved")
            (path / "histories.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
            (path / "queries.jsonl").write_text(json.dumps(questions(row)[0]) + "\n")
            (path / "gold.jsonl").write_bytes(b"\xffnot readable gold")
            reads = []
            actual_open = Path.open

            def capture_open(target, *args, **kwargs):
                reads.append(target.name)
                self.assertNotEqual(target.name, "gold.jsonl")
                return actual_open(target, *args, **kwargs)

            with patch.object(Path, "open", capture_open):
                histories, queries, sources = validation.load_prepared([path])
            self.assertEqual(histories[0]["sessions"], row["sessions"])
            self.assertEqual(len(queries), 1)
            self.assertEqual({item["path"].split("/")[-1] for item in sources}, {"histories.jsonl", "queries.jsonl"})
            self.assertNotIn("gold.jsonl", reads)

    def test_discovery_only_completed_public_dev_retrieval_and_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, changes, retrieval in (("completed", {}, True), ("heldout", {"split": "heldout"}, True),
                                             ("unfinished", {"elapsed_seconds": None}, True), ("aggregate", {}, False),
                                             ("formal", {"status": "official"}, True)):
                path = root / name
                path.mkdir()
                (path / "summary.json").write_text(json.dumps({"status": "public_proxy_not_official_aml", "split": "dev",
                                                              "dataset": "public-test", "question_count": 1, "elapsed_seconds": 0.2, **changes}))
                if retrieval:
                    (path / "retrieval.jsonl").write_text(json.dumps({"query_id": "a-q", "sample_id": "a"}) + "\n")
            paths, audit = validation.discover_used_runs(root)
            self.assertEqual([path.name for path in paths], ["completed"])
            self.assertEqual(sum(audit["skipped_summary_counts"].values()), 4)
            used = validation.used_run(paths[0])
            self.assertEqual(used["sample_ids"], ["a"])
            self.assertEqual(used["query_ids"], ["a-q"])
            output = root / "frozen.json"
            validation.write_new(output, {"frozen": True})
            with self.assertRaises(FileExistsError):
                validation.write_new(output, {"frozen": False})
            self.assertEqual(json.loads(output.read_text()), {"frozen": True})
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validation.main(["split", "--prepared-dir", "unused"])


class ActualStoreGovernanceTests(unittest.TestCase):
    def test_actual_store_streaming_update_scope_idempotency_restart_and_grounding(self):
        fixture = validation.synthetic_fixtures()
        self.assertEqual(fixture, validation.synthetic_fixtures())
        self.assertEqual(len(fixture["semantic_cases"]), 10)
        self.assertEqual(len({case["capability"] for case in fixture["semantic_cases"]}), 10)
        self.assertEqual(sum(step["action"] == "search" for step in fixture["steps"]), 14)
        for grouping in ("reserved", "emitted"):
            with self.subTest(grouping=grouping), tempfile.TemporaryDirectory() as directory:
                adapter = validation.MemoryStoreAdapter(Path(directory) / "memory.sqlite3", context_grouping=grouping)
                result = validation.run_synthetic(adapter, fixture)
                failed = [row for row in result["behavior"]["steps"] if not row["passed"]]
                self.assertTrue(result["behavior"]["all_passed"], failed)
                self.assertEqual(result["behavior"]["passed"], 26)
                self.assertEqual(result["source_grounding"], {"passed_searches": 14, "searches": 14})
                self.assertIsNone(result["semantic_accuracy"])
                self.assertEqual(result["model_requests"], 0)
                self.assertEqual(len(result["answer_inputs"]), 10)
                self.assertTrue(all(set(value) == {"case_id", "query", "search_response"} for value in result["answer_inputs"]))
                self.assertTrue(all(case["correct"] is None for case in result["semantic_cases"]))

    def test_semantic_exact_metric_is_separate_and_checks_unordered_complete_lists(self):
        fixture = validation.synthetic_fixtures()
        answers = {case["case_id"]: case["reference_answer"] for case in fixture["semantic_cases"]}
        answers["complete-list"] = [" Pencil ", "NOTEBOOK", "ruler"]
        with tempfile.TemporaryDirectory() as directory:
            adapter = validation.MemoryStoreAdapter(Path(directory) / "memory.sqlite3")
            complete = validation.run_synthetic(adapter, fixture, answers)
        self.assertEqual(complete["semantic_accuracy"], 1.0)
        answers["fact-current"] = "Maple"
        answers["complete-list"] = ["notebook", "ruler"]
        with tempfile.TemporaryDirectory() as directory:
            partial = validation.run_synthetic(validation.MemoryStoreAdapter(Path(directory) / "memory.sqlite3"), fixture, answers)
        self.assertEqual(partial["semantic_accuracy"], 0.8)
        self.assertTrue(partial["behavior"]["all_passed"])
        with self.assertRaises(ValueError):
            validation.run_synthetic(None, fixture, answers=[])

    def test_grounding_rejects_fabrication_roles_timestamps_scope_and_future_sources(self):
        fixture = validation.synthetic_fixtures()
        first = fixture["steps"][0]
        query = fixture["steps"][1]
        visible = {mid for mid, source in fixture["sources"].items() if source["request_id"] == first["payload"]["request_id"]}
        with tempfile.TemporaryDirectory() as directory:
            adapter = validation.MemoryStoreAdapter(Path(directory) / "memory.sqlite3")
            adapter.add(first["payload"])
            _, response = adapter.search(query["payload"])
        self.assertEqual(validation.verify_grounding(response, fixture["sources"], query["payload"]["user_id"], visible), visible)
        text = response["data"][0]["content"]
        for changed in (text.replace("Maple", "Fabricated"), text + " extra claim", text[:-1],
                        text.replace('"role":"user"', '"role":"assistant"'),
                        text.replace('"timestamp_ms":1767225600000', '"timestamp_ms":1767225600001')):
            altered = copy.deepcopy(response)
            altered["data"][0]["content"] = changed
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validation.verify_grounding(altered, fixture["sources"], query["payload"]["user_id"], visible)
        for user, visibility in (("synthetic-validation/person-B", visible), (query["payload"]["user_id"], set())):
            with self.subTest(user=user, visibility=visibility), self.assertRaises(ValueError):
                validation.verify_grounding(response, fixture["sources"], user, visibility)
        duplicate = copy.deepcopy(response)
        duplicate["data"][0]["content"] += "\n\n[adjacent context]\n" + text
        with self.assertRaises(ValueError):
            validation.verify_grounding(duplicate, fixture["sources"], query["payload"]["user_id"], visible)
        empty = copy.deepcopy(response)
        empty["data"][0]["content"] = ""
        with self.assertRaises(ValueError):
            validation.verify_grounding(empty, fixture["sources"], query["payload"]["user_id"], visible)

    def test_original_source_containing_marker_is_not_misparsed(self):
        # An original turn may itself quote source-looking text. Parse by the
        # known original byte length, not by regex/splitting that marker.
        fixture = validation.synthetic_fixtures()
        first = fixture["steps"][0]
        source = next(source for source in fixture["sources"].values() if source["request_id"] == first["payload"]["request_id"])
        content = 'Maple marker example\n\n[adjacent context]\n[source {"message_id":"fake"}]\nnot a real source'
        first["payload"]["messages"][0]["content"] = content
        source["content"] = content
        visible = {source["message_id"]}
        with tempfile.TemporaryDirectory() as directory:
            adapter = validation.MemoryStoreAdapter(Path(directory) / "memory.sqlite3")
            adapter.add(first["payload"])
            _, response = adapter.search({"user_id": source["user_id"], "query": "Maple marker example", "top_k": 100})
        self.assertEqual(validation.verify_grounding(response, fixture["sources"], source["user_id"], visible), visible)


if __name__ == "__main__":
    unittest.main()
