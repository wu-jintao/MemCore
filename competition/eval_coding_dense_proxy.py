#!/usr/bin/env python3
"""Offline BM25 + coding-feature + cached-v4 proxy evaluation.

Uses only public SWEContextBench Lite rows and private cached vectors produced by
``cache_coding_embeddings.py``. It performs no model, AML, answer, or judge calls.
The result is a retrieval proxy, not an official CAMBench score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
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

DEFAULT_CACHE = ROOT / ".local" / "coding-embeddings-v2"
DEFAULT_OUTPUT = ROOT / "competition" / "results" / "CODING_PROXY_SWE_LITE_DENSE.json"
NOISE_LEVELS = (4, 16, 48, 96)
CODING_WEIGHTS = (0.0, 0.5, 1.0, 1.4, 2.0, 3.0, 4.0)
DENSE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
RESULT_LIMIT = 8
CHARACTER_LIMIT = 120_000
_ID_RE = re.compile(r"EXPERIENCE_INSTANCE_ID: ([^\n]+)")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(root: Path):
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("model") != "text-embedding-v4" \
            or receipt.get("dimension") != 2048 or receipt.get("private_evaluation_inputs_used") is not False \
            or receipt.get("aml_add_search_calls") != 0:
        raise ValueError("A complete public-only v4 cache is required")
    doc_path, query_path = root / "documents.npz", root / "queries.npz"
    if receipt.get("documents_cache_sha256") != sha(doc_path) \
            or receipt.get("queries_cache_sha256") != sha(query_path):
        raise ValueError("Embedding cache hash mismatch")
    with np.load(doc_path, allow_pickle=False) as values:
        keys = values["keys"].tolist()
        offsets = values["offsets"].tolist()
        vectors = values["vectors"].copy()
    documents = {key: vectors[offsets[index]:offsets[index + 1]]
                 for index, key in enumerate(keys)}
    with np.load(query_path, allow_pickle=False) as values:
        queries = dict(zip(values["keys"].tolist(), values["vectors"].copy()))
    if len(documents) != 300 or len(queries) != 99:
        raise ValueError("Unexpected embedding cache cardinality")
    return receipt, documents, queries


def experience_id(content: str) -> str:
    match = _ID_RE.search(content)
    if not match:
        raise ValueError("Public proxy memory lacks its source identity marker")
    return match.group(1)


def dense_ranking(documents, query, message_rows, limit=512):
    scores = []
    for row in message_rows:
        key = experience_id(row["content"])
        vectors = documents.get(key)
        if vectors is None:
            raise ValueError("Public proxy experience is absent from the vector cache")
        score = float(np.max(vectors @ query))
        scores.append((row["id"], score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:limit]


def fuse(rankings):
    scores = Counter()
    lexical_rank = {}
    for index, (weight, rows) in enumerate(rankings):
        if weight <= 0:
            continue
        for rank, (identity, _) in enumerate(rows, 1):
            scores[identity] += weight / (60 + rank)
            if index == 0:
                lexical_rank.setdefault(identity, rank)
    return sorted(scores.items(), key=lambda item: (
        -item[1], lexical_rank.get(item[0], math.inf), item[0]))


def selected_ids(ranking, documents_by_id):
    output, used = [], 0
    for identity, _ in ranking[:RESULT_LIMIT]:
        document = documents_by_id.get(identity)
        if document is None:
            continue
        content_length = len(server.MemoryStore._evidence(document))
        if output and used + content_length > CHARACTER_LIMIT:
            break
        output.append(identity)
        used += content_length
    return output, used


def metrics(rows):
    def mean(key):
        return sum(row[key] for row in rows) / len(rows)
    groups = {}
    for task_type in ("bugfix", "feature", "other"):
        subset = [row for row in rows if row["task_type"] == task_type]
        if subset:
            groups[task_type] = {
                "count": len(subset),
                "recall1": sum(row["recall1"] for row in subset) / len(subset),
                "recall3": sum(row["recall3"] for row in subset) / len(subset),
                "recall8": sum(row["recall8"] for row in subset) / len(subset),
                "mrr": sum(row["mrr"] for row in subset) / len(subset),
            }
    return {
        "count": len(rows), "recall1": mean("recall1"), "recall3": mean("recall3"),
        "recall8": mean("recall8"), "mrr": mean("mrr"),
        "all_relevant_recall8": mean("all_relevant_recall8"),
        "mean_returned": mean("returned"),
        "mean_returned_characters": mean("returned_characters"),
        "by_task_type": groups,
    }


def evaluate_configuration(cases, coding_weight, dense_weight):
    rows = []
    for case in cases:
        coding_effective = coding_weight * case["coding_confidence"]
        ranking = fuse(((1.0, case["lexical"]),
                        (coding_effective, case["coding"]),
                        (dense_weight, case["dense"])))
        ids, characters = selected_ids(ranking, case["documents"])
        ranks = [ids.index(identity) + 1 for identity in case["gold"] if identity in ids]
        first = min(ranks) if ranks else None
        rows.append({
            "task_type": case["task_type"],
            "recall1": int(first is not None and first <= 1),
            "recall3": int(first is not None and first <= 3),
            "recall8": int(first is not None and first <= 8),
            "mrr": 1 / first if first else 0,
            "all_relevant_recall8": sum(identity in ids[:8] for identity in case["gold"])
                                     / len(case["gold"]),
            "returned": len(ids), "returned_characters": characters,
        })
    return metrics(rows)


def build_cases(database: Path, targets, links, document_vectors, query_vectors):
    store = server.MemoryStore(database, context_radius=0, profile="coding",
                               coding_weight=1.0, coding_result_limit=RESULT_LIMIT,
                               coding_character_limit=CHARACTER_LIMIT)
    cases = []
    connection = store._connect()
    try:
        connection.execute("BEGIN")
        for target in targets:
            payload = {"query": proxy.query_text(target),
                       "user_id": "swe-lite/" + target["instance_id"], "top_k": 100}
            rows = list(connection.execute(
                "SELECT * FROM messages WHERE user_id=? ORDER BY sequence", (payload["user_id"],)))
            by_id = {row["id"]: row for row in rows}
            relevant = set(links[target["instance_id"]])
            gold = {row["id"] for row in rows if experience_id(row["content"]) in relevant}
            lexical = store._lexical_candidates(connection, payload, 512)
            coding_result = store._coding_candidates(connection, payload, 512)
            if coding_result:
                coding_rows, confidence = coding_result
            else:
                coding_rows, confidence = [], 0.0
            query = query_vectors[target["instance_id"]]
            dense = dense_ranking(document_vectors, query, rows)
            cases.append({
                "target": target["instance_id"], "task_type": proxy.task_type(target),
                "gold": gold, "documents": by_id, "lexical": lexical,
                "coding": coding_rows, "coding_confidence": confidence, "dense": dense,
            })
    finally:
        connection.close()
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--databases", type=Path,
                        help="Private directory containing coding-4/16/48/96.db proxy databases")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.databases is None:
        parser.error("--databases is required; proxy databases are private reproducible artifacts")
    started = time.monotonic()
    receipt, document_vectors, query_vectors = load_cache(args.cache)
    _, targets, links = proxy.load_rows(proxy.DEFAULT_DATA)
    cases_by_noise = {}
    for noise in NOISE_LEVELS:
        database = args.databases / f"coding-{noise}.db"
        if not database.is_file():
            raise ValueError("Missing public proxy database for noise " + str(noise))
        cases_by_noise[noise] = build_cases(database, targets, links,
                                            document_vectors, query_vectors)
    grid = {}
    for coding_weight in CODING_WEIGHTS:
        for dense_weight in DENSE_WEIGHTS:
            key = f"c{coding_weight:g}-d{dense_weight:g}"
            grid[key] = {
                str(noise): evaluate_configuration(cases_by_noise[noise], coding_weight, dense_weight)
                for noise in NOISE_LEVELS
            }
    def objective(item):
        _, results = item
        high = results["96"]
        means = {
            metric: sum(results[str(noise)][metric] for noise in NOISE_LEVELS) / len(NOISE_LEVELS)
            for metric in ("recall1", "mrr", "recall3", "recall8")
        }
        return (high["recall1"], high["mrr"], means["recall1"], means["mrr"],
                high["recall3"], high["recall8"], means["recall8"])
    selected_key, selected = max(grid.items(), key=objective)
    output = {
        "schema": "memcore-coding-dense-public-proxy-v1",
        "source": "SWEContextBench Lite public relations with cached text-embedding-v4",
        "not_official_cambench": True,
        "private_evaluation_inputs_used": False,
        "model_calls_this_evaluation": 0,
        "embedding_cache_plan_sha256": receipt["plan_sha256"],
        "embedding_cache_documents_sha256": receipt["documents_cache_sha256"],
        "embedding_cache_queries_sha256": receipt["queries_cache_sha256"],
        "result_limit": RESULT_LIMIT, "character_limit": CHARACTER_LIMIT,
        "coding_weights": list(CODING_WEIGHTS), "dense_weights": list(DENSE_WEIGHTS),
        "selected_configuration": selected_key,
        "selected_results": selected,
        "selection_objective": [
            "high_noise_recall1", "high_noise_mrr", "mean_recall1", "mean_mrr",
            "high_noise_recall3", "high_noise_recall8", "mean_recall8"],
        "grid": grid,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "selected_configuration": selected_key,
                      "selected_results": {noise: {key: value for key, value in selected[noise].items()
                                                    if key in ("recall1", "recall3", "recall8", "mrr",
                                                               "all_relevant_recall8", "mean_returned_characters")}
                                           for noise in selected}}, indent=2))


if __name__ == "__main__":
    main()
