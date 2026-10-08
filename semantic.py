"""Optional, synchronous embedding adapters and bounded vector ranking.

No model is downloaded, no network is contacted, and no database is opened by
importing this module. The caller owns user isolation and the transaction that
persists source messages together with all returned segment vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import struct
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Bearer credentials may only be sent to the configured endpoint.
        return None


def urlopen(request, timeout=None):
    """Keep a patchable request boundary while refusing every redirect."""
    return build_opener(_RejectRedirects()).open(request, timeout=timeout)


class EmbeddingError(RuntimeError):
    """An entire embedding operation failed; never acknowledge partial indexing."""


class EmbeddingCapacityError(EmbeddingError):
    """Input exceeds a declared, deterministic segment capacity; retry cannot help."""


class EmbeddingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "disabled"
    model: str = "intfloat/multilingual-e5-small"
    model_dir: str = ""
    dimension: int = 384
    batch_size: int = 8
    threads: int = 2
    concurrency: int = 1
    acquire_timeout: float = 120.0
    operation_timeout: float = 600.0
    segment_tokens: int = 384
    overlap_tokens: int = 64
    max_document_segments: int = 512
    max_total_segments: int = 4096
    max_query_segments: int = 8
    endpoint: str = ""
    api_key: str = ""
    allow_http: bool = False
    http_timeout: float = 30.0
    http_retries: int = 0
    http_segment_bytes: int = 1536
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self):
        if self.provider not in ("disabled", "local", "http"):
            raise EmbeddingConfigError("provider must be disabled, local, or http")
        bounds = {
            "dimension": (1, 8192), "batch_size": (1, 64), "threads": (1, 8),
            "concurrency": (1, 8), "segment_tokens": (16, 480),
            "max_document_segments": (1, 4096), "max_total_segments": (1, 16384),
            "max_query_segments": (1, 64), "http_retries": (0, 2),
            "http_segment_bytes": (32, 16384),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise EmbeddingConfigError(name + " is outside its supported bounds")
        if type(self.overlap_tokens) is not int or not 0 <= self.overlap_tokens < self.segment_tokens:
            raise EmbeddingConfigError("overlap_tokens must be smaller than segment_tokens")
        for name in ("acquire_timeout", "operation_timeout", "http_timeout"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise EmbeddingConfigError(name + " must be finite and positive")
        queue_limit = 900 if self.provider == "http" else 300
        if self.operation_timeout > 1500 or self.acquire_timeout > queue_limit or self.http_timeout > 120:
            raise EmbeddingConfigError("Configured wait/operation timeout exceeds module bounds")
        if self.provider == "local":
            if self.model != "intfloat/multilingual-e5-small" or self.dimension != 384:
                raise EmbeddingConfigError("The local adapter currently supports multilingual-e5-small, dimension 384")
            if not self.model_dir:
                raise EmbeddingConfigError("A pre-downloaded local model directory is required")
            if self.concurrency != 1:
                raise EmbeddingConfigError("The local CPU adapter permits one inference operation at a time")
            if self.query_prefix not in ("", "query: ") or self.document_prefix not in ("", "passage: "):
                raise EmbeddingConfigError("The local e5 adapter uses its fixed training prefixes")
        if self.provider == "http":
            if not self.allow_http:
                raise EmbeddingConfigError("External embedding calls require explicit allow_http opt-in")
            parsed = urlsplit(self.endpoint)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise EmbeddingConfigError("Embedding endpoint must not contain credentials, query, or fragment")
            if not parsed.hostname or (parsed.scheme != "https" and not (
                    parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"))):
                raise EmbeddingConfigError("Use HTTPS for external embeddings; HTTP is limited to loopback tests")
            if not self.api_key or not self.model:
                raise EmbeddingConfigError("External embedding model and API key are required")
            if self.model == "text-embedding-v4":
                if self.dimension not in (64, 128, 256, 512, 768, 1024, 1536, 2048):
                    raise EmbeddingConfigError("text-embedding-v4 requires a supported dimension")
                if self.batch_size > 10:
                    raise EmbeddingConfigError("text-embedding-v4 permits at most 10 inputs per batch")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        provider = env.get("MEMORY_EMBEDDING_PROVIDER", "disabled").strip().lower()
        fields = {
            "model": "MEMORY_EMBEDDING_MODEL", "model_dir": "MEMORY_EMBEDDING_MODEL_DIR",
            "endpoint": "MEMORY_EMBEDDING_URL", "api_key": "MEMORY_EMBEDDING_API_KEY",
            "query_prefix": "MEMORY_EMBEDDING_QUERY_PREFIX",
            "document_prefix": "MEMORY_EMBEDDING_DOCUMENT_PREFIX",
        }
        values = {"provider": provider}
        values.update((field, env[key]) for field, key in fields.items() if key in env)
        numeric = {
            "dimension": ("MEMORY_EMBEDDING_DIMENSION", int),
            "batch_size": ("MEMORY_EMBEDDING_BATCH_SIZE", int),
            "threads": ("MEMORY_EMBEDDING_THREADS", int),
            "concurrency": ("MEMORY_EMBEDDING_CONCURRENCY", int),
            "acquire_timeout": ("MEMORY_EMBEDDING_QUEUE_TIMEOUT", float),
            "operation_timeout": ("MEMORY_EMBEDDING_OPERATION_TIMEOUT", float),
            "segment_tokens": ("MEMORY_EMBEDDING_SEGMENT_TOKENS", int),
            "overlap_tokens": ("MEMORY_EMBEDDING_OVERLAP_TOKENS", int),
            "max_document_segments": ("MEMORY_EMBEDDING_MAX_DOCUMENT_SEGMENTS", int),
            "max_total_segments": ("MEMORY_EMBEDDING_MAX_TOTAL_SEGMENTS", int),
            "max_query_segments": ("MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS", int),
            "http_timeout": ("MEMORY_EMBEDDING_HTTP_TIMEOUT", float),
            "http_retries": ("MEMORY_EMBEDDING_HTTP_RETRIES", int),
            "http_segment_bytes": ("MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES", int),
        }
        try:
            values.update((field, cast(env[key])) for field, (key, cast) in numeric.items() if key in env)
        except (TypeError, ValueError) as error:
            raise EmbeddingConfigError("Invalid numeric embedding configuration") from error
        values["allow_http"] = env.get("MEMORY_EMBEDDING_ALLOW_HTTP", "0") == "1"
        return cls(**values)


def normalize_vector(vector, dimension=None):
    try:
        values = tuple(vector)
    except TypeError as error:
        raise EmbeddingError("Vector is not an array") from error
    if not values or (dimension is not None and len(values) != dimension):
        raise EmbeddingError("Embedding dimension mismatch")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        raise EmbeddingError("Embedding contains invalid numeric values")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise EmbeddingError("Embedding has zero or invalid norm")
    return tuple(float(value / norm) for value in values)


def encode_vector(vector):
    """Portable little-endian float32 BLOB; unit-normalized at the boundary."""
    values = normalize_vector(vector)
    return struct.pack("<" + str(len(values)) + "f", *values)


def decode_vector(blob, dimension):
    if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != dimension * 4:
        raise EmbeddingError("Stored embedding length mismatch")
    values = struct.unpack("<" + str(dimension) + "f", blob)
    if any(not math.isfinite(value) for value in values):
        raise EmbeddingError("Stored embedding is non-finite")
    norm = math.fsum(value * value for value in values)
    if abs(norm - 1.0) > 1e-3:
        raise EmbeddingError("Stored embedding is not unit-normalized")
    return values


def _fingerprint(settings):
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "semantic-v1:" + hashlib.sha256(encoded).hexdigest()


class _Backend:
    def __init__(self, config):
        self.config = config
        self.dimension = config.dimension
        self._slots = threading.BoundedSemaphore(config.concurrency)

    def embed_documents(self, texts):
        """Return one list of segment vectors per input; fail the whole call."""
        return self._embed(texts, query=False)

    def embed_query(self, text):
        """Long queries use a normalized mean of all bounded query segments."""
        vectors = self._embed([text], query=True)[0]
        return normalize_vector(tuple(math.fsum(v[index] for v in vectors) / len(vectors)
                                      for index in range(self.dimension)), self.dimension)

    def _embed(self, texts, query):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must be nonempty strings")
        if not texts:
            return []
        if not self._slots.acquire(timeout=self.config.acquire_timeout):
            raise EmbeddingError("Embedding queue wait exceeded its limit")
        deadline = time.monotonic() + self.config.operation_timeout
        try:
            chunks, counts = [], []
            for text in texts:
                maximum = self.config.max_query_segments if query else self.config.max_document_segments
                segments = self._segments(text, query, maximum)
                if not segments:
                    raise EmbeddingError("Embedding segmentation produced no source segments")
                if len(segments) > maximum:
                    raise EmbeddingCapacityError("Input exceeds the declared segment capacity")
                if len(chunks) + len(segments) > self.config.max_total_segments:
                    raise EmbeddingCapacityError("Embedding operation exceeds its declared total segment capacity")
                counts.append(len(segments))
                chunks.extend(segments)
            vectors = []
            for start in range(0, len(chunks), self.config.batch_size):
                if time.monotonic() >= deadline:
                    raise EmbeddingError("Embedding operation exceeded its time limit")
                batch = chunks[start:start + self.config.batch_size]
                output = self._encode_batch(batch, deadline)
                if len(output) != len(batch):
                    raise EmbeddingError("Embedding response count mismatch")
                vectors.extend(normalize_vector(vector, self.dimension) for vector in output)
                if time.monotonic() >= deadline:
                    raise EmbeddingError("Embedding operation exceeded its time limit")
            results, start = [], 0
            for count in counts:
                results.append(vectors[start:start + count])
                start += count
            return results
        except EmbeddingError:
            raise
        except Exception as error:
            # Do not expose input text, credentials, HTTP bodies, or model internals.
            raise EmbeddingError("Embedding operation failed") from error
        finally:
            self._slots.release()


def _model_digest(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise EmbeddingConfigError("Local model directory is missing config.json")
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and
                   path.suffix in (".json", ".txt", ".model", ".safetensors", ".bin"))
    if not any(path.suffix in (".safetensors", ".bin") for path in paths):
        raise EmbeddingConfigError("Local model weights are missing; downloading is never automatic")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        with path.open("rb") as source:
            for part in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(part)
        digest.update(b"\0")
    return str(directory), digest.hexdigest()


class LocalE5Backend(_Backend):
    """Offline, CPU-only multilingual-e5-small with verified token windows."""
    def __init__(self, config):
        super().__init__(config)
        directory, model_hash = _model_digest(config.model_dir)
        self._model = self._load_model(directory, config.threads)
        self._tokenizer = self._model.tokenizer
        if not getattr(self._tokenizer, "is_fast", False):
            raise EmbeddingConfigError("A fast tokenizer is required for lossless source-offset segmentation")
        dimension_getter = getattr(self._model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self._model.get_sentence_embedding_dimension
        if dimension_getter() != self.dimension:
            raise EmbeddingConfigError("Local model dimension does not match configuration")
        self._maximum_tokens = min(int(self._model.max_seq_length), 512)
        self.fingerprint = _fingerprint({
            "provider": "local-cpu", "model": config.model, "weights": model_hash,
            "dimension": self.dimension, "normalization": "l2", "document_prefix": "passage: ",
            "query_prefix": "query: ", "segmentation": "source-offset-tokens-v1",
            "segment_tokens": config.segment_tokens, "overlap_tokens": config.overlap_tokens,
            "max_seq_length": self._maximum_tokens, "long_query": "normalized-segment-mean-v1",
        })

    @staticmethod
    def _load_model(directory, threads):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingConfigError("Local embeddings require torch and sentence-transformers") from error
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting this only before the first parallel operation.
            if torch.get_num_interop_threads() != 1:
                raise EmbeddingConfigError("Set PyTorch interop threads to 1 before starting the service")
        return SentenceTransformer(directory, device="cpu", local_files_only=True, trust_remote_code=False)

    def _segments(self, text, query, maximum):
        prefix = "query: " if query else "passage: "
        tokens = self._tokenizer(text, add_special_tokens=False, truncation=False,
                                 return_offsets_mapping=True, verbose=False)
        offsets = tokens["offset_mapping"]
        if not offsets:
            raise EmbeddingError("Tokenizer produced no source tokens")
        prefix_length = len(self._tokenizer(prefix, add_special_tokens=True, truncation=False)["input_ids"])
        width = min(self.config.segment_tokens, self._maximum_tokens - prefix_length - 4)
        if width <= self.config.overlap_tokens:
            raise EmbeddingError("Token window is too small for the configured overlap")
        step = width - self.config.overlap_tokens
        count = 1 if len(offsets) <= width else 1 + math.ceil((len(offsets) - width) / step)
        if count > maximum:
            raise EmbeddingCapacityError("Input exceeds the declared token-window capacity")
        segments = []
        for start in range(0, len(offsets), step):
            end = min(start + width, len(offsets))
            left = 0 if start == 0 else offsets[start][0]
            right = len(text) if end == len(offsets) else offsets[end - 1][1]
            value = prefix + text[left:right]
            actual = self._tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"]
            if len(actual) > self._maximum_tokens:
                raise EmbeddingError("Retokenized source window exceeds model capacity; nothing was truncated")
            segments.append(value)
            if end == len(offsets):
                break
        return segments

    def _encode_batch(self, texts, deadline):
        values = self._model.encode(texts, batch_size=len(texts), show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True)
        return values.tolist()


def _utf8_segments(text, maximum_bytes, maximum_segments):
    segments, start, size = [], 0, 0
    for index, character in enumerate(text):
        length = len(character.encode("utf-8"))
        if size + length > maximum_bytes:
            segments.append(text[start:index])
            if len(segments) >= maximum_segments:
                raise EmbeddingCapacityError("Input exceeds the declared external segmentation capacity")
            start, size = index, 0
        size += length
    if start < len(text):
        segments.append(text[start:])
    return segments


class HttpEmbeddingBackend(_Backend):
    """Explicitly opted-in OpenAI-compatible /embeddings endpoint.

    Byte windows preserve all source characters. The chosen provider must be
    separately verified not to truncate those windows; its tokenizer is unknown.
    """
    def __init__(self, config):
        super().__init__(config)
        self.fingerprint = _fingerprint({
            "provider": "http", "model": config.model, "endpoint": config.endpoint,
            "dimension": self.dimension, "normalization": "l2",
            "query_prefix": config.query_prefix, "document_prefix": config.document_prefix,
            "segmentation": "utf8-bytes-v1", "segment_bytes": config.http_segment_bytes,
            "long_query": "normalized-segment-mean-v1",
        })

    def _segments(self, text, query, maximum):
        prefix = self.config.query_prefix if query else self.config.document_prefix
        size = self.config.http_segment_bytes - len(prefix.encode("utf-8"))
        if size < 4:
            raise EmbeddingError("External embedding prefix leaves no source capacity")
        return [prefix + part for part in _utf8_segments(text, size, maximum)]

    def _encode_batch(self, texts, deadline):
        payload = {"model": self.config.model, "input": texts, "encoding_format": "float"}
        if self.config.model == "text-embedding-v4":
            payload["dimensions"] = self.config.dimension
        body = json.dumps(payload, ensure_ascii=False).encode()
        request = Request(self.config.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.config.api_key,
        })
        limit = min(16 * 1024 * 1024, len(texts) * self.dimension * 40 + 65536)
        for attempt in range(self.config.http_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingError("External embedding operation exceeded its time limit")
            try:
                with urlopen(request, timeout=min(self.config.http_timeout, remaining)) as response:
                    raw = response.read(limit + 1)
                if len(raw) > limit:
                    raise EmbeddingError("Embedding response exceeds its size limit")
                payload = json.loads(raw)
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list) or len(data) != len(texts):
                    raise EmbeddingError("Embedding response count mismatch")
                ordered = [None] * len(texts)
                for item in data:
                    if not isinstance(item, dict) or type(item.get("index")) is not int:
                        raise EmbeddingError("Embedding response lacks explicit input indexes")
                    index = item["index"]
                    if not 0 <= index < len(texts) or ordered[index] is not None:
                        raise EmbeddingError("Embedding response indexes are invalid or duplicated")
                    ordered[index] = item.get("embedding")
                if any(value is None for value in ordered):
                    raise EmbeddingError("Embedding response omitted an input")
                return ordered
            except HTTPError as error:
                retryable = error.code in (429, 500, 502, 503, 504)
                error.close()
                if not retryable or attempt == self.config.http_retries:
                    raise EmbeddingError("Embedding service returned HTTP " + str(error.code)) from None
            except (URLError, TimeoutError, OSError):
                if attempt == self.config.http_retries:
                    raise EmbeddingError("Embedding service could not be reached") from None
            except (ValueError, UnicodeDecodeError, TypeError) as error:
                raise EmbeddingError("Malformed embedding response") from error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingError("External embedding operation exceeded its time limit")
            time.sleep(min(0.5 * (2 ** attempt), remaining))
        raise EmbeddingError("Embedding service failed")


def build_backend(config=None):
    config = EmbeddingConfig.from_env() if config is None else config
    if config.provider == "disabled":
        return None
    return LocalE5Backend(config) if config.provider == "local" else HttpEmbeddingBackend(config)


_RANK_SLOTS = threading.BoundedSemaphore(2)


class _ReverseIdentity(str):
    def __lt__(self, other):
        return str.__gt__(self, other)


def rank_vectors(query_vector, rows, limit=100, chunk_size=256):
    """Rank scoped (source_id, BLOB) rows, max-pooling duplicate segment IDs.

    The caller must SELECT with the complete user_id and fingerprint. The module
    intentionally has no global memory cache. Only chunk_size vectors and limit
    distinct source IDs are retained, even when rows contains millions of items.
    """
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise EmbeddingError("Invalid semantic candidate limit")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 512:
        raise EmbeddingError("Invalid vector scoring chunk size")
    query = normalize_vector(query_vector)
    dimension = len(query)
    try:
        import numpy as np
    except ImportError:
        np = None
    if not _RANK_SLOTS.acquire(timeout=120):
        raise EmbeddingError("Semantic ranking queue wait exceeded its limit")
    best, heap = {}, []

    def compact():
        while heap and best.get(heap[0][2]) != heap[0][0]:
            heapq.heappop(heap)
        if len(heap) > max(2 * limit, 16):
            heap[:] = [(score, _ReverseIdentity(identity), identity) for identity, score in best.items()]
            heapq.heapify(heap)

    def offer(identity, score):
        if not isinstance(identity, str) or not identity:
            raise EmbeddingError("Stored source identity is invalid")
        score = max(-1.0, min(1.0, float(score)))
        if identity in best:
            if score <= best[identity]:
                return
        elif len(best) >= limit:
            compact()
            weakest_score, _, weakest_id = heap[0]
            if score < weakest_score or (score == weakest_score and identity >= weakest_id):
                return
            heapq.heappop(heap)
            del best[weakest_id]
        best[identity] = score
        heapq.heappush(heap, (score, _ReverseIdentity(identity), identity))
        compact()

    try:
        iterator = iter(rows)
        while True:
            chunk = list(itertools.islice(iterator, chunk_size))
            if not chunk:
                break
            if np is None:
                scores = [math.fsum(a * b for a, b in zip(query, decode_vector(blob, dimension)))
                          for _, blob in chunk]
            else:
                vectors = []
                for _, blob in chunk:
                    if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != dimension * 4:
                        raise EmbeddingError("Stored embedding length mismatch")
                    vector = np.frombuffer(blob, dtype="<f4")
                    if not np.isfinite(vector).all() or abs(float(np.sum(vector * vector)) - 1.0) > 1e-3:
                        raise EmbeddingError("Stored embedding is invalid or not normalized")
                    vectors.append(vector)
                matrix = np.stack(vectors)
                # Elementwise multiply/reduce avoids a multi-threaded BLAS call.
                scores = np.sum(matrix * np.asarray(query, dtype="float32"), axis=1)
            for (identity, _), score in zip(chunk, scores):
                offer(identity, score)
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))
    finally:
        _RANK_SLOTS.release()
