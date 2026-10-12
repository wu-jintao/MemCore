"""Synthetic query-capacity recovery checks; no network or model service."""

from dataclasses import replace
from http.client import IncompleteRead
import json
import math
import unittest
from unittest.mock import patch

from semantic import (EmbeddingCapacityError, EmbeddingConfig,
                      EmbeddingConfigError, HttpEmbeddingBackend,
                      normalize_vector)


class QueryCapacityRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.network_guard = patch("semantic.urlopen", side_effect=AssertionError("Unexpected network"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)
        self.config = EmbeddingConfig(
            provider="http", model="text-embedding-v4", dimension=2048,
            batch_size=10, concurrency=1, allow_http=True,
            endpoint="https://example.invalid/embeddings", api_key="offline-stub",
            http_segment_bytes=1536, max_query_segments=128, http_retries=2,
        )

    @staticmethod
    def stub_backend(config):
        backend = HttpEmbeddingBackend(config)
        calls, normalized = [], []

        def encode(texts, deadline):
            calls.append(list(texts))
            vectors = []
            for text in texts:
                vector = [float(len(text)), float(sum(map(ord, text)) % 97 + 1)] + [0.0] * 2046
                vectors.append(vector)
                normalized.append(normalize_vector(vector, 2048))
            return vectors

        backend._encode_batch = encode
        return backend, calls, normalized

    def test_environment_selects128_default_stays8_and129_is_invalid(self):
        self.assertEqual(EmbeddingConfig().max_query_segments, 8)
        self.assertEqual(EmbeddingConfig.from_env(
            {"MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS": "128"}).max_query_segments, 128)
        with self.assertRaises(EmbeddingConfigError):
            replace(self.config, max_query_segments=129)

    def test_long_query_preserves_every_character_and_mean_pools_all_batches(self):
        # Deliberately synthetic; exercise both Unicode boundaries and a final tail.
        text = "A中🙂" * 23690 + "TAIL"
        backend, calls, normalized = self.stub_backend(self.config)
        vector = backend.embed_query(text)
        segments = [value for batch in calls for value in batch]
        self.assertEqual("".join(segments), text)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(len(batch) <= 10 for batch in calls))
        self.assertTrue(all(len(value.encode("utf-8")) <= 1536 for value in segments))
        expected = normalize_vector(tuple(
            math.fsum(value[index] for value in normalized) / len(normalized)
            for index in range(2048)), 2048)
        self.assertEqual(vector, expected)

    def test128_ascii_boundary_is_inclusive_and_overflow_fails_before_encoding(self):
        backend, calls, _ = self.stub_backend(self.config)
        self.assertEqual(len(backend._segments("A" * 196608, True, 128)), 128)
        with self.assertRaises(EmbeddingCapacityError):
            backend.embed_query("A" * 196609)
        self.assertEqual(calls, [])
        with self.assertRaises(EmbeddingCapacityError):
            HttpEmbeddingBackend(replace(self.config, max_query_segments=8)).embed_query("A" * 12289)

    def test_short_query_and_fingerprint_remain_unchanged(self):
        old, old_calls, _ = self.stub_backend(replace(self.config, max_query_segments=8, http_retries=0))
        new, new_calls, _ = self.stub_backend(self.config)
        self.assertEqual(old.fingerprint, new.fingerprint)
        self.assertEqual(old.embed_query("synthetic short query"), new.embed_query("synthetic short query"))
        self.assertEqual(old_calls, new_calls)


class HttpIncompleteReadRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = EmbeddingConfig(
            provider="http", model="synthetic-embedding", dimension=3,
            batch_size=2, allow_http=True,
            endpoint="https://example.invalid/embeddings", api_key="offline-stub",
            http_segment_bytes=32, http_retries=2, http_timeout=30,
            operation_timeout=600,
        )
        self.now = 100.0
        self.sleeps = []
        clock = patch("semantic.time.monotonic", side_effect=lambda: self.now)
        sleeper = patch("semantic.time.sleep", side_effect=self.advance)
        clock.start()
        sleeper.start()
        self.addCleanup(clock.stop)
        self.addCleanup(sleeper.stop)

    def advance(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    @staticmethod
    def response(inputs, truncate=False):
        class Response:
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def read(self, limit):
                if truncate:
                    raise IncompleteRead(b'{"data":', 100)
                return json.dumps({"data": [
                    {"index": index, "embedding": [1.0, 2.0, float(index + 3)]}
                    for index in range(len(inputs))]}).encode()

        return Response()

    def test_truncated_read_retries_the_identical_complete_batch(self):
        calls, responses = [], []

        def open_response(request, timeout):
            payload = json.loads(request.data)
            calls.append((payload, timeout))
            response = self.response(payload["input"], truncate=len(calls) == 1)
            responses.append(response)
            return response

        backend = HttpEmbeddingBackend(self.config)
        with patch("semantic.urlopen", side_effect=open_response):
            output = backend.embed_documents(["synthetic first", "synthetic second"])
        self.assertEqual(len(output), 2)
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(calls[0][0]["input"], ["synthetic first", "synthetic second"])
        self.assertEqual([timeout for _, timeout in calls], [30, 30])
        self.assertEqual(self.sleeps, [0.5])
        self.assertTrue(all(response.closed for response in responses))

    def test_exhausted_retries_fail_whole_operation_without_partial_vectors(self):
        from semantic import EmbeddingError
        calls, responses = [], []

        def open_response(request, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            response = self.response(payload["input"], truncate=len(calls) > 1)
            responses.append(response)
            return response

        backend = HttpEmbeddingBackend(self.config)
        with patch("semantic.urlopen", side_effect=open_response):
            with self.assertRaisesRegex(EmbeddingError, "Embedding service could not be reached"):
                backend.embed_documents(["first", "second", "third"])
        self.assertEqual(len(calls), 4)  # Completed first batch, then three attempts at second.
        self.assertEqual(calls[0]["input"], ["first", "second"])
        self.assertTrue(all(call["input"] == ["third"] for call in calls[1:]))
        self.assertEqual(self.sleeps, [0.5, 1.0])
        self.assertTrue(all(response.closed for response in responses))

    def test_operation_deadline_caps_read_timeout_and_retry_sleep(self):
        from semantic import EmbeddingError
        timeouts = []

        def open_response(request, timeout):
            timeouts.append(timeout)
            return self.response(["synthetic"], truncate=True)

        backend = HttpEmbeddingBackend(self.config)
        with patch("semantic.urlopen", side_effect=open_response):
            with self.assertRaisesRegex(EmbeddingError, "operation exceeded its time limit"):
                backend._encode_batch(["synthetic"], deadline=self.now + 0.25)
        self.assertEqual(timeouts, [0.25])
        self.assertEqual(self.sleeps, [0.25])


if __name__ == "__main__":
    unittest.main()
