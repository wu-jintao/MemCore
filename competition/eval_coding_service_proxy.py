#!/usr/bin/env python3
"""End-to-end public Coding proxy through the real MemoryStore Add/Search path.

The backend replays immutable cached text-embedding-v4 vectors for the exact
public semantic representations. It makes no model or AML calls. The evaluation
rebuilds a fresh high-noise database through MemoryStore.add and evaluates via
MemoryStore.search, proving the deployable service path rather than a detached
rank-fusion implementation.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import coding  # noqa: E402
import server  # noqa: E402
from competition import eval_coding_proxy as proxy  # noqa: E402
from competition import eval_coding_dense_proxy as dense  # noqa: E402

DEFAULT_CACHE = ROOT / ".local" / "coding-embeddings-v2"
DEFAULT_DATABASE = ROOT / ".local" / "coding-service-proxy-v1" / "memory.sqlite3"
DEFAULT_OUTPUT = ROOT / "competition" / "results" / "CODING_PROXY_SERVICE_E2E.json"
NOISE = 96
SEMANTIC_WEIGHT = 3.0
CODING_WEIGHT = 2.0
RESULT_LIMIT = 12
CHARACTER_LIMIT = 180_000
_ID_RE = re.compile(r"EXPERIENCE_INSTANCE_ID: ([^\n]+)")


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CachedCodingBackend:
    dimension = 2048

    def __init__(self, cache: Path):
        plan = json.loads((cache / "plan.json").read_text())
        self.fingerprint = "coding-public-cache:" + dense.sha(cache / "documents.npz")
        document_hashes = plan["input_hashes"]["documents"]
        query_hashes = plan["input_hashes"]["queries"]
        with np.load(cache / "documents.npz", allow_pickle=False) as values:
            keys = values["keys"].tolist(); offsets = values["offsets"].tolist(); vectors = values["vectors"].copy()
        with np.load(cache / "queries.npz", allow_pickle=False) as values:
            query_keys = values["keys"].tolist(); query_vectors = values["vectors"].copy()
        self.documents = {
            document_hashes[key]: [tuple(float(value) for value in vector)
                                   for vector in vectors[offsets[index]:offsets[index + 1]]]
            for index, key in enumerate(keys)
        }
        self.queries = {
            query_hashes[key]: tuple(float(value) for value in query_vectors[index])
            for index, key in enumerate(query_keys)
        }
        if len(self.documents) != 300 or len(self.queries) != 99:
            raise ValueError("Cached public vector cardinality mismatch")
        self.document_calls = 0
        self.document_inputs = 0
        self.query_calls = 0
        self.query_inputs = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        self.document_inputs += len(texts)
        output = []
        for text in texts:
            vectors = self.documents.get(text_sha(text))
            if vectors is None:
                raise ValueError("Unplanned public document semantic representation")
            output.append(vectors)
        return output

    def embed_query(self, text):
        self.query_calls += 1
        self.query_inputs += 1
        vector = self.queries.get(text_sha(text))
        if vector is None:
            raise ValueError("Unplanned public query semantic representation")
        return vector


def experience_id(content):
    match = _ID_RE.search(content)
    if not match:
        raise ValueError("Experience identity marker missing")
    return match.group(1)


def expected_message_ids(target_id, relevant_ids):
    user = "swe-lite/" + target_id
    request_id = "public-proxy/" + target_id
    return {
        proxy.message_id(user, request_id, index)
        for index in range(len(relevant_ids))
    }


def prepare_corpora():
    coding.extract_features = lru_cache(maxsize=4096)(coding.extract_features)
    experience, targets, links = proxy.load_rows(proxy.DEFAULT_DATA)
    by_id = {row["instance_id"]: row for row in experience}
    prepared = proxy.prepare_experience(experience)
    corpora = []
    for target in targets:
        relevant_ids = set(links[target["instance_id"]])
        relevant = [by_id[item] for item in sorted(relevant_ids)]
        distractors = proxy.choose_distractors(target, prepared, relevant_ids, NOISE)
        corpora.append((target, relevant + distractors, relevant_ids))
    return targets, links, corpora


def build_database(path: Path, cache: Path, corpora):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    backend = CachedCodingBackend(cache)
    store = server.MemoryStore(path, semantic_backend=backend, context_radius=0,
                               semantic_weight=SEMANTIC_WEIGHT, profile="coding",
                               coding_weight=CODING_WEIGHT, coding_result_limit=RESULT_LIMIT,
                               coding_character_limit=CHARACTER_LIMIT)
    started = time.monotonic()
    for target, corpus, _ in corpora:
        target_id = target["instance_id"]
        store.add({
            "request_id": "public-proxy/" + target_id,
            "user_id": "swe-lite/" + target_id,
            "session_id": "history",
            "messages": [{"role": "assistant", "content": proxy.experience_text(row)} for row in corpus],
        })
    return store, backend, time.monotonic() - started


def evaluate(store, targets, links):
    rows = []
    started = time.monotonic()
    for target in targets:
        target_id = target["instance_id"]
        data = store.search({"query": proxy.query_text(target),
                             "user_id": "swe-lite/" + target_id, "top_k": 100})["data"]
        ids = [item["id"] for item in data]
        gold = expected_message_ids(target_id, sorted(links[target_id]))
        ranks = [ids.index(identity) + 1 for identity in gold if identity in ids]
        first = min(ranks) if ranks else None
        rows.append({
            "instance_id": target_id, "repo": target["repo"],
            "task_type": proxy.task_type(target), "first_relevant_rank": first,
            "recall1": int(first is not None and first <= 1),
            "recall3": int(first is not None and first <= 3),
            "recall8": int(first is not None and first <= 8),
            "recall_limit": int(first is not None and first <= store.coding_result_limit),
            "reciprocal_rank": 1 / first if first else 0,
            "all_relevant_recall8": sum(identity in ids[:8] for identity in gold) / len(gold),
            "all_relevant_recall_limit": sum(identity in ids for identity in gold) / len(gold),
            "returned": len(data),
            "returned_characters": sum(len(item["content"]) for item in data),
        })
    summary = proxy.summarize(rows, time.monotonic() - started)
    summary["recall_at_result_limit"] = sum(row["recall_limit"] for row in rows) / len(rows)
    summary["all_relevant_recall_at_result_limit"] = sum(
        row["all_relevant_recall_limit"] for row in rows) / len(rows)
    summary["complete_relevant_tasks_at_result_limit"] = sum(
        row["all_relevant_recall_limit"] == 1.0 for row in rows)
    return summary, rows


def database_counts(path):
    with sqlite3.connect(path) as connection:
        return {table: connection.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                for table in ("requests", "messages", "postings", "vectors",
                              "code_documents", "code_features")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = json.loads((args.cache / "receipt.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("private_evaluation_inputs_used") is not False:
        raise ValueError("A complete public-only embedding cache is required")
    targets, links, corpora = prepare_corpora()
    store, backend, add_seconds = build_database(args.database, args.cache, corpora)
    summary, rows = evaluate(store, targets, links)
    output = {
        "schema": "memcore-coding-service-public-proxy-v1",
        "source": "SWEContextBench Lite public relations through real MemoryStore Add/Search",
        "not_official_cambench": True,
        "private_evaluation_inputs_used": False,
        "model_calls": 0,
        "noise": NOISE,
        "semantic_weight": SEMANTIC_WEIGHT,
        "coding_weight": CODING_WEIGHT,
        "result_limit": RESULT_LIMIT,
        "character_limit": CHARACTER_LIMIT,
        "embedding_cache_plan_sha256": receipt["plan_sha256"],
        "embedding_cache_documents_sha256": receipt["documents_cache_sha256"],
        "embedding_cache_queries_sha256": receipt["queries_cache_sha256"],
        "cached_backend_document_calls": backend.document_calls,
        "cached_backend_document_inputs": backend.document_inputs,
        "cached_backend_query_calls": backend.query_calls,
        "cached_backend_query_inputs": backend.query_inputs,
        "add_seconds": add_seconds,
        "database_bytes": args.database.stat().st_size,
        "database_counts": database_counts(args.database),
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "schema", "noise", "semantic_weight", "coding_weight", "model_calls",
        "cached_backend_document_calls", "cached_backend_query_calls", "add_seconds",
        "database_bytes", "database_counts", "summary")}, indent=2))


if __name__ == "__main__":
    main()
