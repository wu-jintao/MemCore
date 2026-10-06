import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import eval_rerank as rerank


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return [ord(c) for c in text]

    def num_special_tokens_to_add(self, pair):
        return 4

    def prepare_for_model(self, first, pair_ids, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [1] + first + [2, 2] + pair_ids + [2],
                "attention_mask": [1] * (len(first) + len(pair_ids) + 4)}


class RerankTests(unittest.TestCase):
    def test_token_intervals_cover_long_tail_and_unicode(self):
        for length in (0, 1, 448, 449, 10000):
            spans = rerank.intervals(length, 448, 64)
            self.assertEqual(set(range(length)), {i for a, b in spans for i in range(a, b)})
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], length)
        text = "甲🙂尾巴" * 1000
        features, trace = rerank.item_pairs(Tokenizer(), "查询", text)
        self.assertGreater(trace["pairs"], 1)
        self.assertTrue(all(len(f["input_ids"]) <= 512 for f in features))
        covered = set()
        for feature in features:
            covered.update(feature["input_ids"][5:-1])
        self.assertTrue({ord(c) for c in text} <= covered)

    def test_long_query_is_windowed_without_truncation(self):
        query, text = "Q" * 500 + "尾", "D" * 800 + "末"
        features, trace = rerank.item_pairs(Tokenizer(), query, text)
        self.assertGreater(trace["query_windows"], 1)
        self.assertTrue(all(len(f["input_ids"]) <= 512 for f in features))
        self.assertTrue(any(ord("尾") in f["input_ids"] for f in features))
        self.assertTrue(any(ord("末") in f["input_ids"] for f in features))

    def test_pair_capacity_rejects_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            rerank.item_pairs(Tokenizer(), "question", "x" * 10000, max_pairs=2)

    def test_invalid_window_rejected(self):
        for args in ((10, 0, 0), (10, 5, 5), (-1, 5, 0)):
            with self.assertRaises(ValueError):
                rerank.intervals(*args)

    def test_exact_permutation_allows_reorder_only(self):
        before = [{"id": "a", "content": "[source {}]\n原文", "score": .1},
                  {"id": "b", "content": "Other", "score": .2}]
        rerank.assert_permutation(before, [{**before[1], "score": -3}, {**before[0], "score": 9}])
        for after in ([before[0]], [before[0], before[0]],
                      [{**before[0], "content": "原文"}, before[1]],
                      [{**before[0], "id": "another-user"}, before[1]]):
            with self.assertRaises(ValueError):
                rerank.assert_permutation(before, after)

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            rerank.validate_items([{"id": "x", "content": "source", "score": float("nan")}])
        with self.assertRaises(ValueError):
            rerank.validate_items([{"id": "x", "content": "source", "score": True}])

    def test_bundle_rejects_labels_duplicates_and_wrong_scope(self):
        row = {"query_id": "q", "sample_id": "s", "query": "question", "retrieved": [{"id": "m", "content": "original", "score": .1}]}
        manifest = {"split": "dev", "dataset": "longmemeval_s", "source_question_count": 60,
                    "question_count": 1, "query_ids": ["q"], "exported_gold": False}
        rerank.validate_bundle(manifest, [row])
        for bad in ([{**row, "gold": "not allowed"}], [{**row, "query_id": "other"}], [row, row]):
            with self.assertRaises(ValueError):
                rerank.validate_bundle(manifest, bad)
        for bad in ({**manifest, "split": "heldout"}, {**manifest, "exported_gold": True},
                    {**manifest, "source_question_count": 283}):
            with self.assertRaises(ValueError):
                rerank.validate_bundle(bad, [row])

    def test_validation_rejected_before_raw_read(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "summary.json").write_text(json.dumps({"split": "validation"}))
            with self.assertRaisesRegex(ValueError, "dev"):
                rerank.load_source(folder)
            self.assertFalse((folder / "retrieval.jsonl").exists())

    def test_unapproved_source_size_rejected_before_raw_read(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "summary.json").write_text(json.dumps({"split": "dev", "dataset": "longmemeval_s", "question_count": 283}))
            with self.assertRaisesRegex(ValueError, "Lo547"):
                rerank.load_source(folder)

    def test_prepare_drops_metrics_gold_and_category_from_ranking_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            query_dir = root / "prepared/longmemeval_s/dev"
            query_dir.mkdir(parents=True)
            queries = [{"query_id": "q" + str(i), "sample_id": "h", "query": "Question", "answer": "DO NOT EXPORT"} for i in range(60)]
            query_file = query_dir / "queries.jsonl"
            query_file.write_text("".join(json.dumps(q) + "\n" for q in queries))
            source = root / "source"
            source.mkdir()
            rows = [{"query_id": q["query_id"], "sample_id": "h", "status": "ok", "category": "secret-category",
                     "metrics": {"100": {"gold_turns": 9}}, "retrieved": [{"id": "m", "content": "Unmodified source", "score": .1}]} for q in queries]
            summary = {"run_id": "original", "model_label": "E5", "dataset": "longmemeval_s", "split": "dev", "question_count": 60,
                       "configuration": {"top_k": 100, "character_budget": 0, "prepared_data": {"files": {"queries": {"sha256": rerank.file_hash(query_file)}}}}}
            (source / "summary.json").write_text(json.dumps(summary))
            (source / "retrieval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            args = mock.Mock(source_run=source, data_root=root, output=root / "bundle", max_queries=5)
            rerank.prepare(args)
            raw = (args.output / "input.jsonl").read_text()
            self.assertNotIn("DO NOT EXPORT", raw)
            self.assertNotIn("secret-category", raw)
            self.assertNotIn("metrics", raw)
            self.assertNotIn("gold_turns", raw)
            exported = list(rerank.jsonl(args.output / "input.jsonl"))
            self.assertEqual(len(exported), 5)
            self.assertTrue(all(set(r) == {"query_id", "sample_id", "query", "retrieved"} for r in exported))
            # Selection is repeatable without consulting category or labels.
            args.output = root / "bundle2"
            rerank.prepare(args)
            self.assertEqual(raw, (args.output / "input.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
