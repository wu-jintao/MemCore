"""Offline tests: generated vectors and a loopback HTTP fake, no downloaded model."""

import contextlib
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from semantic import (EmbeddingConfig, EmbeddingConfigError, EmbeddingError,
                      HttpEmbeddingBackend, LocalE5Backend, build_backend,
                      decode_vector, encode_vector, normalize_vector, rank_vectors)


class FakeTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False, **kwargs):
        if truncation:
            raise AssertionError("Silent truncation is forbidden")
        offsets = [(index, index + 1) for index in range(len(text))]
        value = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            value["offset_mapping"] = offsets
        return value


class FakeArray:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.max_seq_length = 48
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        result = []
        for text in texts:
            values = [0.0] * 384
            for character in text:
                values[ord(character) % 384] += 1.0
            result.append(list(normalize_vector(values)))
        return FakeArray(result)


@contextlib.contextmanager
def fake_service(mode="normal"):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            vectors = [{"index": index, "embedding": [float(index + 1), 2.0, 3.0]}
                       for index in range(len(payload["input"]))]
            if mode == "text_embedding_v4":
                dimension = payload.get("dimensions", 1024)
                vectors = [{"index": index,
                            "embedding": [float(index + 1)] + [2.0] * (dimension - 1)}
                           for index in range(len(payload["input"]))]
            elif mode == "wrong_dimension":
                vectors[0]["embedding"] = [1.0, 2.0]
            elif mode == "duplicate":
                vectors[-1]["index"] = 0
            elif mode == "nan":
                vectors[0]["embedding"] = [float("nan"), 2.0, 3.0]
            elif mode == "failure":
                self.send_response(429)
                self.end_headers()
                return
            raw = json.dumps({"data": list(reversed(vectors))}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port) + "/embeddings", calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class ConfigurationTests(unittest.TestCase):
    def test_default_is_disabled_and_does_not_contact_network(self):
        with patch("semantic.urlopen", side_effect=AssertionError("Unexpected network")):
            self.assertIsNone(build_backend(EmbeddingConfig.from_env({})))

    def test_external_calls_require_explicit_opt_in_and_safe_url(self):
        for changes in ({}, {"allow_http": True, "endpoint": "http://example.com/embeddings"},
                        {"allow_http": True, "endpoint": "https://user:password@example.com/embeddings"}):
            params = {"provider": "http", "endpoint": "https://example.com/embeddings", "api_key": "fake"}
            params.update(changes)
            with self.assertRaises(EmbeddingConfigError):
                EmbeddingConfig(**params)

    def test_invalid_bounds_fail(self):
        for changes in ({"batch_size": 0}, {"threads": 16}, {"operation_timeout": float("inf")},
                        {"overlap_tokens": 384}, {"dimension": 0}):
            with self.assertRaises(EmbeddingConfigError):
                EmbeddingConfig(**changes)


class VectorTests(unittest.TestCase):
    def test_round_trip_and_invalid_values(self):
        self.assertEqual(decode_vector(encode_vector([3.0, 4.0]), 2),
                         struct.unpack("<2f", struct.pack("<2f", 0.6, 0.8)))
        for vector in ([0.0, 0.0], [math.nan, 1.0], [True, 1.0], []):
            with self.assertRaises(EmbeddingError):
                encode_vector(vector)
        for blob in (b"wrong", struct.pack("<2f", 0.0, 0.0), struct.pack("<2f", math.nan, 1.0)):
            with self.assertRaises(EmbeddingError):
                decode_vector(blob, 2)

    def test_segment_max_pooling_and_exact_top_k(self):
        rng = random.Random(18)
        rows, expected = [], {}
        for index in range(3000):
            identity = "m" + str(rng.randrange(500)).zfill(4)
            vector = normalize_vector([rng.uniform(-1, 1) for _ in range(4)])
            blob = encode_vector(vector)
            rows.append((identity, blob))
            expected[identity] = max(expected.get(identity, -2), decode_vector(blob, 4)[0])
        wanted = sorted(expected.items(), key=lambda item: (-item[1], item[0]))[:13]
        for chunk_size in (1, 7, 256):
            actual = rank_vectors([1.0, 0.0, 0.0, 0.0], iter(rows), limit=13, chunk_size=chunk_size)
            self.assertEqual([identity for identity, _ in actual], [identity for identity, _ in wanted])
            for (_, score), (_, expected_score) in zip(actual, wanted):
                self.assertAlmostEqual(score, expected_score, places=6)

    def test_ties_are_deterministic_and_python_fallback_works(self):
        rows = [(identity, encode_vector([1.0, 0.0])) for identity in ("b", "aa", "z", "a", "b")]
        with patch.dict(sys.modules, {"numpy": None}):
            self.assertEqual(rank_vectors([1.0, 0.0], rows, limit=2), [("a", 1.0), ("aa", 1.0)])

    def test_corrupt_vector_does_not_silently_disappear(self):
        with self.assertRaises(EmbeddingError):
            rank_vectors([1.0, 0.0], [("m", b"broken")])


class LocalBackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"synthetic-test-only")
        self.config = EmbeddingConfig(provider="local", model_dir=str(root),
                                      batch_size=3, segment_tokens=24, overlap_tokens=6)
        self.model = FakeModel()
        self.load_patch = patch.object(LocalE5Backend, "_load_model", return_value=self.model)
        self.load_patch.start()
        self.backend = LocalE5Backend(self.config)

    def tearDown(self):
        self.load_patch.stop()
        self.directory.cleanup()

    def test_all_source_positions_and_tail_are_encoded_without_truncation(self):
        text = "旧历史" * 70 + "TAIL关键证据"
        output = self.backend.embed_documents([text, "short"])
        self.assertEqual(len(output), 2)
        self.assertGreater(len(output[0]), 1)
        self.assertTrue(all(len(call) <= 3 for call in self.model.calls))
        passages = [value for call in self.model.calls for value in call]
        self.assertTrue(all(value.startswith("passage: ") for value in passages))
        self.assertTrue(any("TAIL关键证据" in value for value in passages))
        # Unique characters make coverage and source offsets observable.
        unique = "".join(chr(0x4e00 + index) for index in range(120))
        parts = self.backend._segments(unique, False, 512)
        recovered = set("".join(part[len("passage: "):] for part in parts))
        self.assertEqual(recovered, set(unique))

    def test_query_all_segments_are_mean_pooled_and_capacity_is_explicit(self):
        vector = self.backend.embed_query("what changed " * 5)
        self.assertEqual(len(vector), 384)
        self.assertAlmostEqual(sum(value * value for value in vector), 1)
        self.assertTrue(all(value.startswith("query: ") for call in self.model.calls for value in call))
        with self.assertRaises(EmbeddingError):
            self.backend.embed_query("long" * 1000)

    def test_capacity_failure_returns_no_partial_vectors(self):
        backend = LocalE5Backend(replace(self.config, max_document_segments=1))
        with self.assertRaises(EmbeddingError):
            backend.embed_documents(["short", "long" * 100])
        self.assertEqual(self.model.calls, [])

    def test_fingerprint_tracks_weights_and_segmentation(self):
        changed = LocalE5Backend(replace(self.config, segment_tokens=20))
        self.assertNotEqual(self.backend.fingerprint, changed.fingerprint)
        (Path(self.directory.name) / "model.safetensors").write_bytes(b"different-test-weights")
        changed = LocalE5Backend(self.config)
        self.assertNotEqual(self.backend.fingerprint, changed.fingerprint)

    def test_queue_is_bounded(self):
        backend = LocalE5Backend(replace(self.config, acquire_timeout=0.01))
        backend._slots.acquire()
        try:
            with self.assertRaises(EmbeddingError):
                backend.embed_documents(["text"])
        finally:
            backend._slots.release()


class HttpBackendTests(unittest.TestCase):
    @staticmethod
    def configuration(endpoint, **changes):
        params = dict(provider="http", model="synthetic-embedding", endpoint=endpoint,
                      api_key="test-key", allow_http=True, dimension=3, batch_size=2,
                      http_segment_bytes=32)
        params.update(changes)
        return EmbeddingConfig(**params)

    def test_batch_bound_response_reordering_and_source_coverage(self):
        with fake_service() as (endpoint, calls):
            backend = HttpEmbeddingBackend(self.configuration(endpoint))
            text = "记忆覆盖与隔离" * 7
            output = backend.embed_documents([text, "short", "another"])
            self.assertEqual(len(output), 3)
            self.assertTrue(all(set(call) == {"model", "input", "encoding_format"} for call in calls))
            self.assertTrue(all(len(call["input"]) <= 2 for call in calls))
            flattened = [value for call in calls for value in call["input"]]
            self.assertEqual("".join(flattened[:-2]), text)
            self.assertTrue(all(len(value.encode()) <= 32 for value in flattened))
            self.assertAlmostEqual(sum(value * value for value in output[0][0]), 1)
            self.assertLess(output[0][0][0], output[0][1][0])

    def test_text_embedding_v4_requests_and_parses_explicit_2048_dimensions(self):
        with fake_service("text_embedding_v4") as (endpoint, calls):
            backend = HttpEmbeddingBackend(self.configuration(
                endpoint, model="text-embedding-v4", dimension=2048, batch_size=10))
            texts = ["document " + str(index) for index in range(11)]
            output = backend.embed_documents(texts)
            self.assertEqual([len(call["input"]) for call in calls], [10, 1])
            self.assertEqual([text for call in calls for text in call["input"]], texts)
            self.assertTrue(all(call["dimensions"] == 2048 for call in calls))
            self.assertTrue(all(call["encoding_format"] == "float" for call in calls))
            self.assertEqual(len(output), len(texts))
            self.assertTrue(all(len(segments) == 1 and len(segments[0]) == 2048
                                for segments in output))
            self.assertAlmostEqual(sum(value * value for value in output[0][0]), 1)
            self.assertLess(output[0][0][0], output[1][0][0])

    def test_text_embedding_v4_invalid_configuration_fails_before_request(self):
        with fake_service("text_embedding_v4") as (endpoint, calls):
            for changes in ({"dimension": 384}, {"batch_size": 11}, {"batch_size": 64}):
                with self.subTest(**changes):
                    parameters = {"model": "text-embedding-v4", "dimension": 2048,
                                  "batch_size": 10}
                    parameters.update(changes)
                    with self.assertRaises(EmbeddingConfigError):
                        build_backend(self.configuration(endpoint, **parameters))
            self.assertEqual(calls, [])

    def test_malformed_service_data_and_rate_limit_fail_whole_operation(self):
        for mode in ("wrong_dimension", "duplicate", "nan", "failure"):
            with fake_service(mode) as (endpoint, calls):
                backend = HttpEmbeddingBackend(self.configuration(endpoint))
                with self.assertRaises(EmbeddingError):
                    backend.embed_documents(["first", "second"])
                self.assertEqual(len(calls), 1)

    def test_redirects_never_forward_requests_or_credentials(self):
        target_calls, source_calls = [], []
        synthetic_key = "synthetic-redirect-test-key"

        class TargetHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                target_calls.append((self.command, self.headers.get("Authorization")))
                raw = json.dumps({"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_POST = do_GET

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
        target_url = "http://127.0.0.1:" + str(target.server_port) + "/redirected"

        class RedirectHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                source_calls.append(self.headers.get("Authorization"))
                raw = (synthetic_key + " synthetic redirect response body").encode()
                self.send_response(self.server.redirect_status)
                self.send_header("Location", target_url)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        workers = [threading.Thread(target=service.serve_forever, daemon=True)
                   for service in (target, source)]
        for worker in workers:
            worker.start()
        try:
            endpoint = "http://127.0.0.1:" + str(source.server_port) + "/embeddings"
            backend = HttpEmbeddingBackend(self.configuration(
                endpoint, api_key=synthetic_key, http_retries=2))
            for expected_calls, status in enumerate((301, 302, 303, 307, 308), 1):
                with self.subTest(status=status):
                    source.redirect_status = status
                    with self.assertRaises(EmbeddingError) as failure:
                        backend.embed_documents(["synthetic document"])
                    self.assertEqual(str(failure.exception), "Embedding service returned HTTP " + str(status))
                    self.assertIsNone(failure.exception.__cause__)
                    self.assertTrue(failure.exception.__suppress_context__)
                    self.assertEqual(len(source_calls), expected_calls)
                    self.assertEqual(source_calls[-1], "Bearer " + synthetic_key)
                    self.assertEqual(target_calls, [])
        finally:
            for service, worker in zip((target, source), workers):
                service.shutdown()
                service.server_close()
                worker.join(timeout=2)

    def test_keys_are_not_in_fingerprint_and_configuration_changes_are(self):
        config = self.configuration("https://example.com/embeddings")
        backend = HttpEmbeddingBackend(config)
        self.assertNotIn("test-key", backend.fingerprint)
        self.assertEqual(backend.fingerprint, HttpEmbeddingBackend(replace(config, api_key="different")).fingerprint)
        self.assertNotEqual(backend.fingerprint, HttpEmbeddingBackend(replace(config, dimension=4)).fingerprint)


class HttpQueueBudgetTests(unittest.TestCase):
    """Instant slot/clock doubles verify the larger queue without sleeping."""

    class Slot:
        def __init__(self, available=True):
            self.available = available
            self.held = False
            self.waits = []
            self.releases = 0

        def acquire(self, timeout):
            self.waits.append(timeout)
            if not self.available:
                return False
            if self.held:
                raise AssertionError("Prior embedding operation leaked its slot")
            self.held = True
            return True

        def release(self):
            if not self.held:
                raise AssertionError("Unacquired slot was released")
            self.held = False
            self.releases += 1

    def backend(self):
        config = HttpBackendTests.configuration("https://synthetic.invalid/embeddings",
            acquire_timeout=900, operation_timeout=600)
        backend = HttpEmbeddingBackend(config)
        backend._slots = self.Slot()
        return backend

    def test_http_900_is_valid_but_excess_or_nonfinite_waits_fail(self):
        config = replace(self.backend().config, model="text-embedding-v4",
                         dimension=2048, batch_size=10)
        self.assertEqual(config.acquire_timeout, 900)
        self.assertEqual(config.operation_timeout, 600)
        self.assertEqual(HttpEmbeddingBackend(config).fingerprint,
                         HttpEmbeddingBackend(replace(config, acquire_timeout=120)).fingerprint)
        for wait in (900.0001, 901, float("inf"), float("-inf"), float("nan"), 0, -1, True):
            with self.subTest(wait=wait), self.assertRaises(EmbeddingConfigError):
                replace(config, acquire_timeout=wait)
        for provider in ("disabled", "local"):
            with self.subTest(provider=provider), self.assertRaises(EmbeddingConfigError):
                replace(config, provider=provider, acquire_timeout=900)
        self.assertEqual(EmbeddingConfig(acquire_timeout=300).acquire_timeout, 300)

    def test_whole_multibatch_operation_keeps_one_slot_and_600_deadline(self):
        backend = self.backend()
        batches = []

        def encode(texts, deadline):
            self.assertTrue(backend._slots.held)
            self.assertEqual(backend._slots.releases, 0)
            self.assertEqual(deadline, 700)
            batches.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

        with patch("semantic.time.monotonic", return_value=100), \
                patch.object(backend, "_encode_batch", side_effect=encode):
            self.assertEqual(len(backend.embed_documents(["first", "second", "third"])), 3)
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(backend._slots.waits, [900])
        self.assertEqual(backend._slots.releases, 1)
        self.assertFalse(backend._slots.held)

    def test_queue_timeout_does_not_encode_or_release_unacquired_slot(self):
        backend = self.backend()
        backend._slots.available = False
        with patch.object(backend, "_encode_batch", side_effect=AssertionError("No model call")):
            with self.assertRaisesRegex(EmbeddingError, "queue wait"):
                backend.embed_documents(["source"])
        self.assertEqual(backend._slots.waits, [900])
        self.assertEqual(backend._slots.releases, 0)
        self.assertFalse(backend._slots.held)

    def test_operation_timeout_releases_slot_and_next_operation_can_succeed(self):
        backend = self.backend()
        now = [100]

        def exceed_deadline(texts, deadline):
            self.assertEqual(deadline, 700)
            now[0] = 701
            return [[1.0, 0.0, 0.0] for _ in texts]

        with patch("semantic.time.monotonic", side_effect=lambda: now[0]), \
                patch.object(backend, "_encode_batch", side_effect=exceed_deadline):
            with self.assertRaisesRegex(EmbeddingError, "time limit"):
                backend.embed_documents(["source"])
        self.assertEqual(backend._slots.releases, 1)
        self.assertFalse(backend._slots.held)
        with patch("semantic.time.monotonic", return_value=100), \
                patch.object(backend, "_encode_batch", return_value=[[1.0, 0.0, 0.0]]):
            self.assertEqual(len(backend.embed_documents(["next source"])), 1)
        self.assertEqual(backend._slots.waits, [900, 900])
        self.assertEqual(backend._slots.releases, 2)

    def test_model_failure_releases_slot_without_successful_result(self):
        backend = self.backend()
        with patch("semantic.time.monotonic", return_value=100), \
                patch.object(backend, "_encode_batch", side_effect=OSError("synthetic network error")):
            with self.assertRaisesRegex(EmbeddingError, "Embedding operation failed"):
                backend.embed_documents(["source"])
        self.assertEqual(backend._slots.releases, 1)
        self.assertFalse(backend._slots.held)


if __name__ == "__main__":
    unittest.main()
