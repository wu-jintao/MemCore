"""Synthetic Coding profile checks; no official CAMBench inputs are used."""

from pathlib import Path
import sqlite3
import tempfile
import unittest

import coding
import server


ROOT = Path(__file__).resolve().parent


class CodingFeatureTests(unittest.TestCase):
    def test_extracts_paths_symbols_exceptions_tests_and_task_type(self):
        text = """Fix AttributeError in src/cache/loader.py inside CacheLoader.load_item.
Root cause: stale cache_key was reused. Updated tests/test_loader.py::test_cache_refresh.
Verified with: pytest tests/test_loader.py -q; all tests passed."""
        result = coding.extract_features(text)
        self.assertIn("src/cache/loader.py", result.paths)
        self.assertIn("cacheloader.load_item", result.symbols)
        self.assertIn("bugfix", result.task_types)
        self.assertIn("root_cause", result.stages)
        self.assertIn("verification", result.stages)
        self.assertIn("symbol:attributeerror", result.weighted)
        self.assertIn("command:pytest tests/test_loader.py -q; all tests passed.", result.weighted)
        self.assertGreater(coding.quality_prior(result), 0)

    def test_filters_environment_urls_and_generic_identifiers(self):
        text = "https://github.com/org/repo/blob/main/src/api.py /home/me/.venv/lib/python3.11/site-packages/pkg/core.py 1/2 generic_cache_key"
        result = coding.extract_features(text)
        self.assertIn("pkg/core.py", result.paths)
        self.assertNotIn("https://github.com/org/repo/blob/main/src/api.py", result.paths)
        self.assertNotIn("1/2", result.paths)
        self.assertNotIn("generic_cache_key", result.symbols)

    def test_confidence_damps_broad_environment_dumps(self):
        precise = coding.extract_features("Fix TypeError in src/api.py with pytest tests/test_api.py")
        broad = coding.extract_features(" ".join(f"pkg/module_{i}.py Class{i}.method" for i in range(80)))
        self.assertGreater(coding.query_confidence(precise), coding.query_confidence(broad))

    def test_semantic_representation_keeps_task_patch_tests_and_features(self):
        text = """HISTORICAL TASK:
Fix AttributeError in src/cache.py Cache.load.

HISTORICAL PATCH:
diff --git a/src/cache.py b/src/cache.py
--- a/src/cache.py
+++ b/src/cache.py
@@ -1 +1 @@
-old()
+new()

HISTORICAL TEST PATCH:
diff --git a/tests/test_cache.py b/tests/test_cache.py
+def test_cache(): pass

FAIL_TO_PASS: tests/test_cache.py::test_cache

PASS_TO_PASS: tests/test_api.py::test_ok"""
        document = coding.semantic_document(text)
        query = coding.semantic_query("Repair AttributeError in src/cache.py Cache.load")
        self.assertIn("Task description", document)
        self.assertIn("+new()", document)
        self.assertIn("test_cache", document)
        self.assertIn("path:src/cache.py", document)
        self.assertIn("Software engineering task", query)
        self.assertIn("symbol:attributeerror", query)

    def test_failed_attempt_has_lower_prior_than_verified_fix(self):
        failure = coding.extract_features("Attempted a patch but it did not work; tests still fail.")
        success = coding.extract_features("Root cause identified; patch applied; regression tests passed.")
        self.assertLess(coding.quality_prior(failure), coding.quality_prior(success))


class CaptureEmbeddingBackend:
    fingerprint = "capture-coding-v1"
    dimension = 3

    def __init__(self):
        self.documents = []
        self.queries = []
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        self.documents.extend(texts)
        return [[(1.0, 0.0, 0.0)] for _ in texts]

    def embed_query(self, text):
        self.query_calls += 1
        self.queries.append(text)
        return (1.0, 0.0, 0.0)


class DenseRerankBackend:
    fingerprint = "dense-rerank-synthetic-v1"
    dimension = 3

    def __init__(self):
        self.query_operations = 0

    def embed_documents(self, texts):
        output = []
        for text in texts:
            marker = text.split("candidate-", 1)[1][:2]
            index = int(marker)
            if index == 0:
                vector = (1.0, 0.0, 0.0)
            elif index == 1:
                vector = (0.0, 1.0, 0.0)
            else:
                vector = (1.0, (12 - index) / 10, 0.0)
            output.append([vector])
        return output

    def embed_query_segments(self, text):
        self.query_operations += 1
        segments = ((0.0, 1.0, 0.0), (0.0, 0.99, 0.1))
        mean = (0.0, 0.998733495, 0.050313074)
        return mean, segments

    def embed_query(self, text):
        raise AssertionError("Dense reranking must reuse embed_query_segments")


class CodingRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="coding-memory-", dir=ROOT)
        self.db = Path(self.temp.name) / "memory.sqlite3"
        self.store = server.MemoryStore(
            self.db, context_radius=0, profile="coding", coding_weight=1.4,
            coding_result_limit=8, coding_character_limit=120000)

    def tearDown(self):
        self.temp.cleanup()

    def add(self, request_id, text, session="history"):
        return self.store.add({
            "request_id": request_id,
            "user_id": "repo-scope",
            "session_id": session,
            "messages": [{"role": "assistant", "content": text}],
        })

    def search(self, query, top_k=100):
        return self.store.search({"query": query, "user_id": "repo-scope", "top_k": top_k})["data"]

    def test_coding_profile_uses_compact_semantic_text_but_returns_raw_evidence(self):
        backend = CaptureEmbeddingBackend()
        path = Path(self.temp.name) / "semantic.db"
        store = server.MemoryStore(path, semantic_backend=backend, context_radius=0, profile="coding")
        raw = "HISTORICAL TASK:\nFix TypeError in src/api.py Api.fetch.\n\nHISTORICAL PATCH:\n" + (" context" * 5000) + "\n+return fixed"
        store.add({"request_id": "semantic", "user_id": "repo", "session_id": "s",
                   "messages": [{"role": "assistant", "content": raw}]})
        found = store.search({"query": "Fix TypeError in src/api.py Api.fetch",
                              "user_id": "repo", "top_k": 8})["data"]
        self.assertTrue(backend.documents)
        self.assertLess(len(backend.documents[0]), len(raw))
        self.assertIn("Engineering memory for reuse", backend.documents[0])
        self.assertIn("Software engineering task", backend.queries[0])
        self.assertIn(raw, found[0]["content"])

    def test_dense_post_rerank_reorders_only_the_packed_set_with_one_query_operation(self):
        backend = DenseRerankBackend()
        path = Path(self.temp.name) / "dense-rerank.db"
        baseline = server.MemoryStore(
            path, semantic_backend=backend, context_radius=0, profile="coding",
            semantic_weight=0.0, coding_weight=0.0,
            coding_dense_rerank_weight=0.0, coding_result_limit=13,
            coding_character_limit=500000,
        )
        names = ["candidate-" + str(index).zfill(2) for index in range(13)]
        for name in names:
            baseline.add({
                "request_id": name, "user_id": "dense-user", "session_id": name,
                "messages": [{"role": "assistant", "content": name + " synthetic evidence"}],
            })
        with sqlite3.connect(path) as connection:
            rows = list(connection.execute(
                "SELECT request_id,id,role,content,user_id,session_id,request_id,message_index,"
                "source_timestamp_ms,ingested_at_ms,doc_length,sequence FROM messages "
                "WHERE user_id='dense-user' ORDER BY request_id"))
            identities = {row[0]: row[1] for row in rows}
            connection.row_factory = sqlite3.Row
            documents = list(connection.execute(
                "SELECT * FROM messages WHERE user_id='dense-user' ORDER BY request_id"))
        baseline.coding_character_limit = sum(
            len(server.MemoryStore._evidence(document)) for document in documents[:12])
        fixed = [(identities[name], len(names) - index)
                 for index, name in enumerate(names)]
        payload = {"query": "synthetic segmented query", "user_id": "dense-user", "top_k": 100}
        original = baseline._lexical_candidates
        baseline._lexical_candidates = lambda connection, value, limit: fixed
        try:
            before = baseline.search(payload)["data"]
        finally:
            baseline._lexical_candidates = original
        self.assertEqual(backend.query_operations, 0)
        self.assertEqual([item["id"] for item in before],
                         [identities[name] for name in names[:12]])

        candidate = server.MemoryStore(
            path, semantic_backend=backend, context_radius=0, profile="coding",
            semantic_weight=0.0, coding_weight=0.0,
            coding_dense_rerank_weight=0.5, coding_result_limit=13,
            coding_character_limit=baseline.coding_character_limit,
        )
        original = candidate._lexical_candidates
        candidate._lexical_candidates = lambda connection, value, limit: fixed
        try:
            after = candidate.search(payload)["data"]
        finally:
            candidate._lexical_candidates = original
        self.assertEqual(backend.query_operations, 1)
        self.assertEqual(after[0]["id"], identities["candidate-01"])
        self.assertEqual({item["id"] for item in before}, {item["id"] for item in after})
        self.assertEqual({item["id"]: item["content"] for item in before},
                         {item["id"]: item["content"] for item in after})
        self.assertNotIn(identities["candidate-12"], {item["id"] for item in after})
        self.assertEqual(sum(len(item["content"]) for item in before),
                         sum(len(item["content"]) for item in after))

    def test_dense_post_rerank_requires_coding_segment_backend(self):
        path = Path(self.temp.name) / "invalid-dense-rerank.db"
        with self.assertRaisesRegex(ValueError, "requires the coding profile"):
            server.MemoryStore(path, semantic_backend=CaptureEmbeddingBackend(),
                               coding_dense_rerank_weight=0.5)
        with self.assertRaisesRegex(ValueError, "query segment embeddings"):
            server.MemoryStore(path, semantic_backend=CaptureEmbeddingBackend(),
                               profile="coding", coding_dense_rerank_weight=0.5)

    def test_cross_user_embedding_cache_reuses_vectors_without_cross_scope_search(self):
        backend = CaptureEmbeddingBackend()
        path = Path(self.temp.name) / "cache-reuse.db"
        store = server.MemoryStore(path, semantic_backend=backend, context_radius=0,
                                   profile="coding")
        content = "Root cause in src/cache.py Cache.run; regression tests passed."
        for user, request in (("user-a", "request-a"), ("user-b", "request-b")):
            store.add({"request_id": request, "user_id": user, "session_id": "s",
                       "messages": [{"role": "assistant", "content": content}]})
        self.assertEqual(backend.document_calls, 1)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM coding_embedding_cache").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 2)
        result_a = store.search({"query": "Cache.run src/cache.py", "user_id": "user-a", "top_k": 8})["data"]
        result_b = store.search({"query": "Cache.run src/cache.py", "user_id": "user-b", "top_k": 8})["data"]
        self.assertEqual(len(result_a), 1)
        self.assertEqual(len(result_b), 1)
        self.assertNotEqual(result_a[0]["id"], result_b[0]["id"])

    def test_embedding_cache_survives_restart_and_new_backend_makes_no_document_call(self):
        first = CaptureEmbeddingBackend()
        path = Path(self.temp.name) / "cache-restart.db"
        content = "Fix TypeError in src/api.py Api.fetch; tests passed."
        store = server.MemoryStore(path, semantic_backend=first, context_radius=0, profile="coding")
        store.add({"request_id": "a", "user_id": "user-a", "session_id": "s",
                   "messages": [{"role": "assistant", "content": content}]})
        second = CaptureEmbeddingBackend()
        restarted = server.MemoryStore(path, semantic_backend=second, context_radius=0, profile="coding")
        restarted.add({"request_id": "b", "user_id": "user-b", "session_id": "s",
                       "messages": [{"role": "assistant", "content": content}]})
        self.assertEqual(first.document_calls, 1)
        self.assertEqual(second.document_calls, 0)

    def test_embedding_cache_rolls_back_with_failed_message_transaction(self):
        backend = CaptureEmbeddingBackend()
        path = Path(self.temp.name) / "cache-rollback.db"
        store = server.MemoryStore(path, semantic_backend=backend, context_radius=0, profile="coding")
        with sqlite3.connect(path) as connection:
            connection.execute("""
                CREATE TRIGGER reject_cached_message BEFORE INSERT ON messages
                WHEN NEW.content='reject cached message'
                BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END
            """)
        payload = {"request_id": "failed", "user_id": "u", "session_id": "s",
                   "messages": [{"role": "assistant", "content": "reject cached message"}]}
        with self.assertRaises(sqlite3.IntegrityError):
            store.add(payload)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM coding_embedding_cache").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 0)

    def test_exact_file_symbol_and_exception_beat_high_volume_noise(self):
        relevant = """Bug fix for src/cache/loader.py CacheLoader.load_item AttributeError.
Root cause: cache_key was None after eviction. Guarded the lookup and added
pytest tests/test_loader.py::test_cache_refresh. All regression tests passed."""
        self.add("relevant", relevant)
        for index in range(80):
            self.add("noise-" + str(index),
                     "Generic cache documentation update with loader examples and feature notes " + str(index),
                     session="noise-" + str(index))
        found = self.search(
            "Fix AttributeError in CacheLoader.load_item from src/cache/loader.py; "
            "the cache_key becomes None after eviction")
        self.assertTrue(found)
        self.assertEqual(found[0]["id"], self.message_id("relevant"))
        self.assertIn("regression tests passed", found[0]["content"])

    def test_bugfix_query_prefers_verified_fix_over_failed_attempt(self):
        self.add("failed", "Tried changing parser.py but it did not work; tests still fail with ParseError.")
        self.add("fixed", "Root cause of ParseError in parser.py identified. Fixed parse_node and tests passed.")
        found = self.search("Repair ParseError in parser.py parse_node")
        self.assertEqual(found[0]["id"], self.message_id("fixed"))

    def test_coding_profile_caps_count_and_characters_at_whole_evidence_boundaries(self):
        for index in range(20):
            self.add("item-" + str(index),
                     "Fix module/file_%d.py function_%d regression. Tests passed. %s" %
                     (index, index, "evidence " * 300), session="s-" + str(index))
        found = self.search("Fix module regression tests", top_k=100)
        self.assertLessEqual(len(found), 8)
        self.assertLessEqual(sum(len(item["content"]) for item in found[:-1]), 120000)

    def test_coding_request_idempotency_stores_only_digest(self):
        payload = {"request_id": "digest", "user_id": "repo-scope", "session_id": "s",
                   "messages": [{"role": "user", "content": "x" * 5000}]}
        self.store.add(payload)
        with sqlite3.connect(self.db) as connection:
            value = connection.execute("SELECT payload_json FROM requests WHERE request_id='digest'").fetchone()[0]
        self.assertRegex(value, r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("x" * 100, value)
        self.assertEqual(self.store.add(payload)["success"], True)
        changed = {**payload, "messages": [{"role": "user", "content": "y" * 5000}]}
        with self.assertRaises(server.ConflictError):
            self.store.add(changed)

    def test_coding_lexical_index_uses_compact_representation_and_returns_raw(self):
        raw = "HISTORICAL TASK:\nFix CacheError in src/cache.py.\n\nHISTORICAL PATCH:\n" + ("unchanged context\n" * 2000) + "+fixed_line\n"
        self.add("compact-lexical", raw)
        with sqlite3.connect(self.db) as connection:
            stored = connection.execute("SELECT content FROM messages WHERE request_id='compact-lexical'").fetchone()[0]
            posting_count = connection.execute("SELECT COUNT(*) FROM coding_postings p JOIN messages m ON m.sequence=p.message_sequence WHERE m.request_id='compact-lexical'").fetchone()[0]
            textual_postings = connection.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        self.assertEqual(stored, raw)
        self.assertLess(posting_count, len(server.lexical_tokens(raw)))
        self.assertEqual(textual_postings, 0)
        found = self.search("CacheError src/cache.py fixed_line")
        self.assertIn(raw, found[0]["content"])

    def test_textual_request_record_preserves_canonical_body(self):
        path = Path(self.temp.name) / "textual-request.db"
        store = server.MemoryStore(path)
        payload = {"request_id": "raw", "user_id": "u", "session_id": "s",
                   "messages": [{"role": "user", "content": "original"}]}
        store.add(payload)
        with sqlite3.connect(path) as connection:
            value = connection.execute("SELECT payload_json FROM requests").fetchone()[0]
        self.assertIn("original", value)
        self.assertFalse(value.startswith("sha256:"))

    def test_compact_schema_uses_integer_message_sequence_without_reverse_index(self):
        with sqlite3.connect(self.db) as connection:
            feature_columns = [row[1] for row in connection.execute("PRAGMA table_info(code_features)")]
            document_columns = [row[1] for row in connection.execute("PRAGMA table_info(code_documents)")]
            posting_columns = [row[1] for row in connection.execute("PRAGMA table_info(coding_postings)")]
            cache_columns = [row[1] for row in connection.execute("PRAGMA table_info(coding_embedding_cache)")]
            indexes = {row[1] for row in connection.execute("PRAGMA index_list(code_features)")}
            markers = {row[0] for row in connection.execute("SELECT name FROM schema_migrations")}
        self.assertEqual(feature_columns, ["feature", "message_sequence", "weight"])
        self.assertEqual(document_columns, ["message_sequence", "quality"])
        self.assertEqual(posting_columns, ["term", "message_sequence", "frequency"])
        self.assertEqual(cache_columns, ["fingerprint", "content_sha256", "segment_index", "dimension", "vector"])
        self.assertIn("coding-postings-v1", markers)
        self.assertIn("coding-embedding-cache-v1", markers)
        self.assertNotIn("code_features_message", indexes)

    def test_restart_preserves_compact_coding_index(self):
        self.add("restart", "Root cause in src/api.py Api.fetch; regression tests passed.")
        before = self.search("Api.fetch src/api.py")
        restarted = server.MemoryStore(self.db, context_radius=0, profile="coding",
                                       coding_weight=1.4, coding_result_limit=8,
                                       coding_character_limit=120000)
        self.assertEqual(restarted.search({"query": "Api.fetch src/api.py",
                                          "user_id": "repo-scope", "top_k": 100})["data"], before)

    def test_incompatible_old_coding_schema_requires_fresh_database(self):
        path = Path(self.temp.name) / "old-coding.db"
        textual = server.MemoryStore(path)
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TABLE code_documents (user_id TEXT, message_id TEXT, quality REAL);
                CREATE TABLE code_features (user_id TEXT, feature TEXT, message_id TEXT, weight INTEGER, sequence INTEGER);
                CREATE TABLE coding_postings (term TEXT, message_sequence INTEGER, frequency INTEGER);
            """)
        with self.assertRaisesRegex(ValueError, "fresh database"):
            server.MemoryStore(path, profile="coding")

    def test_default_textual_profile_has_no_coding_tables_or_caps(self):
        other = Path(self.temp.name) / "textual.sqlite3"
        store = server.MemoryStore(other, context_radius=0)
        with sqlite3.connect(other) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("code_features", tables)
        self.assertEqual(store.profile, "textual")

    def message_id(self, request_id):
        with sqlite3.connect(self.db) as connection:
            return connection.execute(
                "SELECT id FROM messages WHERE user_id=? AND request_id=?",
                ("repo-scope", request_id)).fetchone()[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
