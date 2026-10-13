#!/usr/bin/env python3
"""Cache text-embedding-v4 vectors for the public Coding proxy corpus.

Dry-run by default. ``--execute`` reads an existing private embedding environment,
encodes only the pinned public SWEContextBench Lite inputs, and writes a resumable
private cache. No AML task, Add/Search endpoint, private evaluation data, answer
model, or judge is involved. Raw source text and credentials are never written to
receipts or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

try:
    import numpy as np
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("numpy and pyarrow are required for this public proxy cache") from error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import coding  # noqa: E402
from semantic import EmbeddingConfig, HttpEmbeddingBackend, normalize_vector  # noqa: E402
from competition import eval_coding_proxy as proxy  # noqa: E402

DEFAULT_DATA = ROOT / "competition" / "coding-public-data"
DEFAULT_CACHE = ROOT / ".local" / "coding-embeddings-v2"
MODEL = "text-embedding-v4"
DIMENSION = 2048
SEGMENT_BYTES = 8192
BATCH_SIZE = 10
MAX_DOCUMENT_SEGMENTS = 4
MAX_QUERY_SEGMENTS = 4
MAX_TOTAL_SEGMENTS = 4096
EXPECTED_DOCUMENTS = 300
EXPECTED_QUERIES = 99
EXPECTED_MAX_COMBINED_BYTES = 3_000_000
EXPECTED_MAX_COMBINED_SEGMENTS = 600
SOURCE_VARIANT = "coding-semantic-v2-segment8192"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_sha(value: str) -> str:
    return sha(value.encode("utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    path.chmod(0o600)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key.strip()] = value
    return values


def source_file_hashes() -> dict[str, str]:
    names = ("coding.py", "semantic.py", "competition/eval_coding_proxy.py",
             "competition/cache_coding_embeddings.py")
    return {name: sha((ROOT / name).read_bytes()) for name in names}


def public_file_hashes(data: Path) -> dict[str, str]:
    names = ("SWEContextBench_Lite_Experience.parquet",
             "SWEContextBench_Related_Lite.parquet",
             "SWEContextBench_Relationship.parquet")
    return {name: sha((data / name).read_bytes()) for name in names}


def planned_inputs(data: Path):
    experience, targets, _ = proxy.load_rows(data)
    documents = [(row["instance_id"], coding.semantic_document(proxy.experience_text(row)))
                 for row in sorted(experience, key=lambda item: item["instance_id"])]
    queries = [(row["instance_id"], coding.semantic_query(proxy.query_text(row)))
               for row in sorted(targets, key=lambda item: item["instance_id"])]
    if len(documents) != EXPECTED_DOCUMENTS or len(queries) != EXPECTED_QUERIES:
        raise ValueError("Unexpected public proxy row count")
    if len({key for key, _ in documents}) != len(documents) or len({key for key, _ in queries}) != len(queries):
        raise ValueError("Duplicate public proxy input identity")
    return documents, queries


def segment_count(value: str) -> int:
    return max(1, math.ceil(len(value.encode("utf-8")) / SEGMENT_BYTES))


def plan(data: Path):
    documents, queries = planned_inputs(data)
    all_values = [value for _, value in documents + queries]
    total_bytes = sum(len(value.encode("utf-8")) for value in all_values)
    total_segments = sum(segment_count(value) for value in all_values)
    if total_bytes > EXPECTED_MAX_COMBINED_BYTES or total_segments > EXPECTED_MAX_COMBINED_SEGMENTS:
        raise ValueError("Public proxy embedding scope exceeds its fixed bound")
    if max(segment_count(value) for _, value in documents) > MAX_DOCUMENT_SEGMENTS:
        raise ValueError("A public document exceeds the fixed segment bound")
    if max(segment_count(value) for _, value in queries) > MAX_QUERY_SEGMENTS:
        raise ValueError("A public query exceeds the fixed segment bound")
    manifest = {
        "schema": "memcore-coding-embedding-plan-v1",
        "source_variant": SOURCE_VARIANT,
        "model": MODEL,
        "dimension": DIMENSION,
        "segment_bytes": SEGMENT_BYTES,
        "batch_size": BATCH_SIZE,
        "documents": len(documents),
        "queries": len(queries),
        "combined_utf8_bytes": total_bytes,
        "combined_segments": total_segments,
        "maximum_document_segments": max(segment_count(value) for _, value in documents),
        "maximum_query_segments": max(segment_count(value) for _, value in queries),
        "input_hashes": {
            "documents": {key: text_sha(value) for key, value in documents},
            "queries": {key: text_sha(value) for key, value in queries},
        },
        "public_data_sha256": public_file_hashes(data),
        "source_sha256": source_file_hashes(),
        "private_evaluation_inputs_used": False,
        "aml_add_search_calls": 0,
        "answer_or_judge_calls": 0,
    }
    manifest["plan_sha256"] = sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    return manifest, documents, queries


def backend_from_env(env: dict[str, str]):
    selected = dict(env)
    selected.update({
        "MEMORY_EMBEDDING_PROVIDER": "http",
        "MEMORY_EMBEDDING_MODEL": MODEL,
        "MEMORY_EMBEDDING_DIMENSION": str(DIMENSION),
        "MEMORY_EMBEDDING_BATCH_SIZE": str(BATCH_SIZE),
        "MEMORY_EMBEDDING_CONCURRENCY": "1",
        "MEMORY_EMBEDDING_HTTP_RETRIES": "2",
        "MEMORY_EMBEDDING_HTTP_TIMEOUT": "30",
        "MEMORY_EMBEDDING_OPERATION_TIMEOUT": "180",
        "MEMORY_EMBEDDING_QUEUE_TIMEOUT": "30",
        "MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES": str(SEGMENT_BYTES),
        "MEMORY_EMBEDDING_MAX_DOCUMENT_SEGMENTS": str(MAX_DOCUMENT_SEGMENTS),
        "MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS": str(MAX_QUERY_SEGMENTS),
        "MEMORY_EMBEDDING_MAX_TOTAL_SEGMENTS": str(MAX_TOTAL_SEGMENTS),
        "MEMORY_EMBEDDING_QUERY_PREFIX": "",
        "MEMORY_EMBEDDING_DOCUMENT_PREFIX": "",
        "MEMORY_EMBEDDING_ALLOW_HTTP": "1",
    })
    config = EmbeddingConfig.from_env(selected)
    if config.model != MODEL or config.dimension != DIMENSION or not config.endpoint or not config.api_key:
        raise ValueError("The private environment does not provide the fixed embedding configuration")
    return HttpEmbeddingBackend(config)


def validate_vector_rows(rows, expected_count: int):
    if len(rows) != expected_count:
        raise ValueError("Embedding cache row count mismatch")
    output = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("Embedding cache contains no segment vectors")
        output.append(np.asarray([normalize_vector(vector, DIMENSION) for vector in row], dtype="float32"))
    return output


def document_arrays(keys, rows):
    offsets = [0]
    vectors = []
    for row in rows:
        vectors.extend(row)
        offsets.append(len(vectors))
    return {
        "keys": np.asarray(keys, dtype="U128"),
        "offsets": np.asarray(offsets, dtype="int64"),
        "vectors": np.asarray(vectors, dtype="float32"),
    }


def query_arrays(keys, rows):
    values = []
    for segments in rows:
        mean = tuple(math.fsum(vector[index] for vector in segments) / len(segments)
                     for index in range(DIMENSION))
        values.append(normalize_vector(mean, DIMENSION))
    return {"keys": np.asarray(keys, dtype="U128"),
            "vectors": np.asarray(values, dtype="float32")}


def partition_inputs(items, maximum_items=20, maximum_segments=40):
    shards, current, segments = [], [], 0
    for item in items:
        count = segment_count(item[1])
        if current and (len(current) >= maximum_items or segments + count > maximum_segments):
            shards.append(current)
            current, segments = [], 0
        current.append(item)
        segments += count
    if current:
        shards.append(current)
    return shards


def shard_path(root: Path, kind: str, index: int) -> Path:
    return root / "shards" / kind / (f"{index:04d}.npz")


def shard_valid(path: Path, plan_sha256: str, kind: str, index: int, keys) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text())
        with np.load(path, allow_pickle=False) as arrays:
            saved_keys = arrays["keys"].tolist()
            shape = tuple(arrays["vectors"].shape)
        return (value.get("plan_sha256") == plan_sha256 and value.get("kind") == kind
                and value.get("shard_index") == index and value.get("npz_sha256") == sha(path.read_bytes())
                and saved_keys == list(keys) and shape[-1] == DIMENSION)
    except Exception:
        return False


def write_shard(path: Path, plan_sha256: str, kind: str, index: int, keys, rows):
    arrays = document_arrays(keys, rows)
    atomic_npz(path, **arrays)
    atomic_json(path.with_suffix(".json"), {
        "kind": kind, "shard_index": index, "plan_sha256": plan_sha256,
        "items": len(keys), "segments": sum(len(row) for row in rows),
        "npz_sha256": sha(path.read_bytes()), "raw_text_saved": False,
        "credentials_saved": False,
    })


def merge_shards(root: Path, kind: str, shard_count: int, output: Path, plan_sha256: str):
    keys, rows = [], []
    for index in range(shard_count):
        path = shard_path(root, kind, index)
        with np.load(path, allow_pickle=False) as arrays:
            shard_keys = arrays["keys"].tolist()
            offsets = arrays["offsets"].tolist()
            vectors = arrays["vectors"]
            keys.extend(shard_keys)
            rows.extend([vectors[offsets[pos]:offsets[pos + 1]] for pos in range(len(shard_keys))])
    if kind == "documents":
        atomic_npz(output, **document_arrays(keys, rows))
    else:
        atomic_npz(output, **query_arrays(keys, rows))
    atomic_json(output.with_suffix(".json"), {
        "kind": kind, "plan_sha256": plan_sha256, "items": len(keys),
        "segments": sum(len(row) for row in rows), "npz_sha256": sha(output.read_bytes()),
        "shards": shard_count, "raw_text_saved": False, "credentials_saved": False,
    })


def cache_valid(path: Path, plan_sha256: str, kind: str) -> bool:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text())
        with np.load(path, allow_pickle=False) as arrays:
            vector_shape = tuple(arrays["vectors"].shape)
        return (value.get("plan_sha256") == plan_sha256 and value.get("kind") == kind
                and value.get("npz_sha256") == sha(path.read_bytes()) and vector_shape[-1] == DIMENSION)
    except Exception:
        return False


def execute(args, manifest, documents, queries, maximum_shards=None):
    args.cache.mkdir(parents=True, exist_ok=True)
    args.cache.chmod(0o700)
    plan_path = args.cache / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if previous.get("plan_sha256") != manifest["plan_sha256"]:
            raise ValueError("Existing cache plan differs; use a separate empty cache directory")
    else:
        atomic_json(plan_path, manifest)
    environment = read_env(args.env)
    backend = backend_from_env(environment)
    inputs = {"documents": documents, "queries": queries}
    results = {"documents": args.cache / "documents.npz", "queries": args.cache / "queries.npz"}
    started = time.monotonic()
    model_operations = 0
    completed_shards = 0
    total_shards = 0
    for kind, items in inputs.items():
        shards = partition_inputs(items)
        total_shards += len(shards)
        for index, shard in enumerate(shards):
            path = shard_path(args.cache, kind, index)
            keys = [key for key, _ in shard]
            if shard_valid(path, manifest["plan_sha256"], kind, index, keys):
                completed_shards += 1
                continue
            if maximum_shards is not None and model_operations >= maximum_shards:
                receipt = {
                    "schema": "memcore-coding-embedding-cache-v2", "status": "partial",
                    "plan_sha256": manifest["plan_sha256"], "model": MODEL,
                    "dimension": DIMENSION, "model_operations_this_run": model_operations,
                    "completed_shards": completed_shards, "total_shards": total_shards,
                    "elapsed_seconds": time.monotonic() - started,
                    "private_evaluation_inputs_used": False, "aml_add_search_calls": 0,
                    "answer_or_judge_calls": 0, "raw_text_saved": False,
                    "credentials_saved": False,
                }
                atomic_json(args.cache / "receipt.json", receipt)
                return receipt
            values = [value for _, value in shard]
            raw = backend.embed_documents(values) if kind == "documents" else backend._embed(values, query=True)
            rows = validate_vector_rows(raw, len(shard))
            write_shard(path, manifest["plan_sha256"], kind, index, keys, rows)
            model_operations += 1
            completed_shards += 1
            atomic_json(args.cache / "progress.json", {
                "plan_sha256": manifest["plan_sha256"], "last_kind": kind,
                "last_shard_index": index, "completed_shards": completed_shards,
                "model_operations_this_run": model_operations,
                "raw_text_saved": False, "credentials_saved": False,
            })
    for kind, items in inputs.items():
        shards = partition_inputs(items)
        merge_shards(args.cache, kind, len(shards), results[kind], manifest["plan_sha256"])
    receipt = {
        "schema": "memcore-coding-embedding-cache-v2", "status": "complete",
        "plan_sha256": manifest["plan_sha256"], "model": MODEL, "dimension": DIMENSION,
        "documents_cache_sha256": sha(results["documents"].read_bytes()),
        "queries_cache_sha256": sha(results["queries"].read_bytes()),
        "model_operations_this_run": model_operations,
        "completed_shards": completed_shards, "total_shards": total_shards,
        "elapsed_seconds": time.monotonic() - started,
        "private_evaluation_inputs_used": False, "aml_add_search_calls": 0,
        "answer_or_judge_calls": 0, "raw_text_saved": False, "credentials_saved": False,
    }
    atomic_json(args.cache / "receipt.json", receipt)
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--env", type=Path,
                        help="Private literal embedding environment; required only with --execute")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--maximum-shards", type=int,
                        help="Bound new model operations in this invocation; completed shards are reused")
    args = parser.parse_args()
    manifest, documents, queries = plan(args.data)
    summary = {key: manifest[key] for key in (
        "plan_sha256", "model", "dimension", "documents", "queries",
        "combined_utf8_bytes", "combined_segments", "maximum_document_segments",
        "maximum_query_segments", "private_evaluation_inputs_used", "aml_add_search_calls")}
    if not args.execute:
        print(json.dumps({"status": "dry_run", **summary}, indent=2))
        return
    if args.env is None:
        parser.error("--execute requires --env pointing to a private literal environment file")
    receipt = execute(args, manifest, documents, queries, args.maximum_shards)
    print(json.dumps({"status": receipt["status"], **summary,
                      "model_operations_this_run": receipt["model_operations_this_run"],
                      "elapsed_seconds": receipt["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
