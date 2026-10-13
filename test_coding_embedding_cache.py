from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from competition import cache_coding_embeddings as cache


class FakeBackend:
    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[self.vector(index), self.vector(index + 1)] for index, _ in enumerate(texts)]

    def _embed(self, texts, query):
        self.query_calls += 1
        return [[self.vector(index)] for index, _ in enumerate(texts)]

    @staticmethod
    def vector(index):
        value = np.zeros(cache.DIMENSION, dtype="float32")
        value[index % cache.DIMENSION] = 1
        return value.tolist()


class CodingEmbeddingCacheTests(unittest.TestCase):
    def test_atomic_npz_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            cache.atomic_npz(path, keys=np.asarray(["a"]), vectors=np.zeros((1, cache.DIMENSION), dtype="float32"))
            self.assertTrue(path.is_file())

    def test_execute_resumes_without_repeating_model_operations(self):
        manifest = {"plan_sha256": "plan", "model": cache.MODEL, "dimension": cache.DIMENSION}
        documents = [("d1", "document one"), ("d2", "document two")]
        queries = [("q1", "query one")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / "env"
            env.write_text("MEMORY_EMBEDDING_URL=https://example.com/embeddings\nMEMORY_EMBEDDING_API_KEY=fake\n")
            args = mock.Mock(cache=root / "cache", env=env)
            first = FakeBackend()
            with mock.patch.object(cache, "backend_from_env", return_value=first):
                receipt = cache.execute(args, manifest, documents, queries)
            self.assertEqual(receipt["model_operations_this_run"], 2)
            self.assertEqual(first.document_calls, 1)
            self.assertEqual(first.query_calls, 1)
            second = FakeBackend()
            with mock.patch.object(cache, "backend_from_env", return_value=second):
                receipt = cache.execute(args, manifest, documents, queries)
            self.assertEqual(receipt["model_operations_this_run"], 0)
            self.assertEqual(second.document_calls, 0)
            self.assertEqual(second.query_calls, 0)

    def test_partial_shard_execution_resumes(self):
        manifest = {"plan_sha256": "sharded-plan", "model": cache.MODEL, "dimension": cache.DIMENSION}
        documents = [(f"d{i}", "document " + str(i)) for i in range(45)]
        queries = [("q1", "query one")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); env = root / "env"
            env.write_text("MEMORY_EMBEDDING_URL=https://example.com/embeddings\nMEMORY_EMBEDDING_API_KEY=fake\n")
            args = mock.Mock(cache=root / "cache", env=env)
            first = FakeBackend()
            with mock.patch.object(cache, "backend_from_env", return_value=first):
                receipt = cache.execute(args, manifest, documents, queries, maximum_shards=1)
            self.assertEqual(receipt["status"], "partial")
            self.assertEqual(receipt["model_operations_this_run"], 1)
            second = FakeBackend()
            with mock.patch.object(cache, "backend_from_env", return_value=second):
                receipt = cache.execute(args, manifest, documents, queries)
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["model_operations_this_run"], 3)
            self.assertTrue((args.cache / "documents.npz").is_file())
            self.assertTrue((args.cache / "queries.npz").is_file())

    def test_plan_contains_only_hashes_not_raw_text(self):
        manifest, documents, queries = cache.plan(cache.DEFAULT_DATA)
        rendered = str(manifest)
        self.assertEqual(len(documents), cache.EXPECTED_DOCUMENTS)
        self.assertEqual(len(queries), cache.EXPECTED_QUERIES)
        self.assertNotIn(documents[0][1], rendered)
        self.assertNotIn(queries[0][1], rendered)
        self.assertLessEqual(manifest["combined_segments"], cache.EXPECTED_MAX_COMBINED_SEGMENTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
