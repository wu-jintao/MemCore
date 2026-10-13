#!/usr/bin/env python3
"""Evaluate Coding-profile retrieval on the public SWEContextBench Lite relations.

This is a public proxy for CAMBench's relevant/noisy engineering-memory shape.
It does not use CAMBench data, private annotations, evaluation requests, or an
answer/solve model. Each related task receives its linked historical experience
plus adversarial distractors selected by same-repository priority and lexical
overlap. The report measures retrieval only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from functools import lru_cache
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

try:
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit("pyarrow is required only for this public proxy evaluator") from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import coding  # noqa: E402
import server  # noqa: E402


DEFAULT_DATA = ROOT / "competition" / "coding-public-data"
DEFAULT_OUTPUT = ROOT / "competition" / "results" / "CODING_PROXY_SWE_LITE.json"
NOISE_LEVELS = (4, 16, 48, 96)
CODING_WEIGHTS = (0.5, 1.0, 1.4, 2.0, 3.0, 4.0)


def load_rows(data: Path):
    experience = pq.read_table(data / "SWEContextBench_Lite_Experience.parquet").to_pylist()
    related = pq.read_table(data / "SWEContextBench_Related_Lite.parquet").to_pylist()
    relationships = pq.read_table(data / "SWEContextBench_Relationship.parquet").to_pylist()
    by_experience = {row["instance_id"]: row for row in experience}
    by_related = {row["instance_id"]: row for row in related}
    links: dict[str, list[str]] = defaultdict(list)
    for row in relationships:
        if row["related_instance_id"] in by_related and row["experience_instance_id"] in by_experience:
            links[row["related_instance_id"]].append(row["experience_instance_id"])
    targets = [by_related[key] for key in sorted(links)]
    return experience, targets, links


def compact(value, limit):
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[public proxy field clipped at a whole configured character bound]"


def experience_text(row):
    return "\n\n".join(part for part in (
        "EXPERIENCE_INSTANCE_ID: " + row["instance_id"],
        "REPOSITORY: " + row["repo"],
        "HISTORICAL TASK:\n" + compact(row.get("problem_statement"), 12000),
        "HISTORICAL PATCH:\n" + compact(row.get("patch"), 16000),
        "HISTORICAL TEST PATCH:\n" + compact(row.get("test_patch"), 10000),
        "FAIL_TO_PASS: " + compact(row.get("FAIL_TO_PASS"), 5000),
        "PASS_TO_PASS: " + compact(row.get("PASS_TO_PASS"), 5000),
    ) if part.split(":", 1)[-1].strip())


def query_text(row):
    return "\n\n".join(part for part in (
        compact(row.get("problem_statement"), 16000),
        "FAIL_TO_PASS: " + compact(row.get("FAIL_TO_PASS"), 5000),
    ) if part.split(":", 1)[-1].strip())


def task_type(row):
    features = coding.extract_features(str(row.get("problem_statement") or ""))
    if "bugfix" in features.task_types:
        return "bugfix"
    if "feature" in features.task_types:
        return "feature"
    return "other"


def candidate_content(candidate):
    return "\n".join((str(candidate.get("problem_statement") or ""),
                           str(candidate.get("patch") or ""),
                           str(candidate.get("test_patch") or "")))


def prepare_experience(all_experience):
    prepared = {}
    for candidate in all_experience:
        content = candidate_content(candidate)
        prepared[candidate["instance_id"]] = {
            "row": candidate,
            "tokens": Counter(server.lexical_tokens(content)),
            "features": frozenset(coding.extract_features(content).weighted),
        }
    return prepared


def overlap_score(query_tokens, query_features, prepared, same_repo):
    shared = sum(min(count, 3) for token, count in prepared["tokens"].items()
                 if token in query_tokens)
    feature_overlap = len(prepared["features"] & query_features)
    return (100000 if same_repo else 0) + shared * 10 + feature_overlap


def choose_distractors(target, prepared_experience, relevant, maximum):
    query = query_text(target)
    query_tokens = set(server.lexical_tokens(query))
    query_features = frozenset(coding.extract_features(query).weighted)
    seed = int(hashlib.sha256(target["instance_id"].encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    scored = []
    for instance_id, prepared in prepared_experience.items():
        row = prepared["row"]
        if instance_id in relevant:
            continue
        score = overlap_score(query_tokens, query_features, prepared,
                              row["repo"] == target["repo"])
        scored.append((score, rng.random(), row))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]["instance_id"]))
    return [row for _, _, row in scored[:maximum]]


def message_id(user_id, request_id, index):
    identity = json.dumps([user_id, request_id, index], ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
    return "msg_" + hashlib.sha256(identity).hexdigest()


def build_store(path, profile, corpora):
    store = server.MemoryStore(
        path, context_radius=0, profile=profile,
        coding_weight=1.4, coding_result_limit=8, coding_character_limit=120000)
    expected: dict[str, set[str]] = {}
    start = time.monotonic()
    for target_id, corpus, relevant_ids in corpora:
        user = "swe-lite/" + target_id
        request_id = "public-proxy/" + target_id
        messages = [{"role": "assistant", "content": experience_text(row)} for row in corpus]
        store.add({"request_id": request_id, "user_id": user,
                   "session_id": "history", "messages": messages})
        expected[target_id] = {
            message_id(user, request_id, index)
            for index, row in enumerate(corpus) if row["instance_id"] in relevant_ids
        }
    return store, expected, time.monotonic() - start


def evaluate(store, expected, targets, noise, weight=None):
    if weight is not None:
        store.coding_weight = weight
    rows = []
    start = time.monotonic()
    for target in targets:
        data = store.search({"query": query_text(target),
                             "user_id": "swe-lite/" + target["instance_id"],
                             "top_k": 100})["data"]
        ids = [item["id"] for item in data]
        gold = expected[target["instance_id"]]
        ranks = [ids.index(item) + 1 for item in gold if item in ids]
        first = min(ranks) if ranks else None
        rows.append({
            "instance_id": target["instance_id"], "repo": target["repo"],
            "task_type": task_type(target), "noise": noise,
            "gold_count": len(gold), "returned": len(data),
            "returned_characters": sum(len(item["content"]) for item in data),
            "first_relevant_rank": first,
            "recall1": int(first is not None and first <= 1),
            "recall3": int(first is not None and first <= 3),
            "recall8": int(first is not None and first <= 8),
            "all_relevant_recall8": sum(item in ids[:8] for item in gold) / len(gold),
            "reciprocal_rank": 1 / first if first else 0,
        })
    elapsed = time.monotonic() - start
    return summarize(rows, elapsed)


def summarize(rows, elapsed):
    def mean(key, source=rows):
        return sum(row[key] for row in source) / len(source) if source else None
    groups = {}
    for label in ("bugfix", "feature", "other"):
        subset = [row for row in rows if row["task_type"] == label]
        if subset:
            groups[label] = {"count": len(subset), "recall1": mean("recall1", subset),
                             "recall3": mean("recall3", subset), "recall8": mean("recall8", subset),
                             "mrr": mean("reciprocal_rank", subset)}
    return {
        "count": len(rows), "recall1": mean("recall1"), "recall3": mean("recall3"),
        "recall8": mean("recall8"), "mrr": mean("reciprocal_rank"),
        "all_relevant_recall8": mean("all_relevant_recall8"),
        "mean_returned": mean("returned"),
        "mean_returned_characters": mean("returned_characters"),
        "p95_returned_characters": sorted(row["returned_characters"] for row in rows)[
            max(0, int(.95 * len(rows)) - 1)],
        "search_seconds": elapsed, "by_task_type": groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-directory", type=Path,
                        help="Reuse complete textual-N.db/coding-N.db proxy databases from an interrupted run")
    args = parser.parse_args()
    # Cache identical public-history feature extraction across four noise levels.
    coding.extract_features = lru_cache(maxsize=4096)(coding.extract_features)
    experience, targets, links = load_rows(args.data)
    by_id = {row["instance_id"]: row for row in experience}
    prepared_experience = prepare_experience(experience)
    report = {
        "schema": "memcore-coding-public-proxy-v1",
        "source": "jiayuanz3/SWEContextBench Lite public parquet files",
        "not_official_cambench": True,
        "private_evaluation_inputs_used": False,
        "model_calls": 0,
        "targets": len(targets), "experience_rows": len(experience),
        "noise_definition": "linked experience plus hardest 4/16/48/96 distractors selected by same-repository priority and lexical overlap",
        "noise_levels": {},
    }
    temporary = None
    if args.reuse_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="coding-proxy-", dir=ROOT)
        directory = Path(temporary.name)
    else:
        directory = args.reuse_directory.resolve()
    try:
        for noise in NOISE_LEVELS:
            corpora = []
            for target in targets:
                relevant_ids = set(links[target["instance_id"]])
                relevant = [by_id[item] for item in sorted(relevant_ids)]
                distractors = choose_distractors(target, prepared_experience, relevant_ids, noise)
                corpus = relevant + distractors
                corpora.append((target["instance_id"], corpus, relevant_ids))
            if args.reuse_directory is None:
                textual, textual_expected, textual_add = build_store(directory / f"textual-{noise}.db", "textual", corpora)
                coding_store, coding_expected, coding_add = build_store(directory / f"coding-{noise}.db", "coding", corpora)
            else:
                textual_path = directory / f"textual-{noise}.db"
                coding_path = directory / f"coding-{noise}.db"
                if not textual_path.is_file() or not coding_path.is_file():
                    raise SystemExit(f"Incomplete reuse directory for noise {noise}")
                textual = server.MemoryStore(textual_path, context_radius=0)
                coding_store = server.MemoryStore(coding_path, context_radius=0, profile="coding",
                                                   coding_weight=1.4, coding_result_limit=8,
                                                   coding_character_limit=120000)
                textual_expected = coding_expected = {
                    target_id: {
                        message_id("swe-lite/" + target_id, "public-proxy/" + target_id, index)
                        for index in range(len(relevant_ids))
                    }
                    for target_id, _, relevant_ids in corpora
                }
                textual_add = coding_add = None
            textual_result = evaluate(textual, textual_expected, targets, noise)
            weights = {}
            for weight in CODING_WEIGHTS:
                weights[str(weight)] = evaluate(coding_store, coding_expected, targets, noise, weight)
            best_weight = max(weights, key=lambda key: (weights[key]["recall1"], weights[key]["mrr"], weights[key]["recall3"]))
            report["noise_levels"][str(noise)] = {
                "textual": {**textual_result, "add_seconds": textual_add},
                "coding_weights": weights,
                "best_weight": float(best_weight),
                "coding_best": {**weights[best_weight], "add_seconds": coding_add},
            }
    finally:
        if temporary is not None:
            temporary.cleanup()
    # Select one weight robustly across all four levels, prioritizing high noise.
    aggregate = {}
    for weight in CODING_WEIGHTS:
        results = [report["noise_levels"][str(noise)]["coding_weights"][str(weight)] for noise in NOISE_LEVELS]
        aggregate[str(weight)] = {
            "mean_recall1": statistics.mean(item["recall1"] for item in results),
            "mean_mrr": statistics.mean(item["mrr"] for item in results),
            "high_noise_recall1": results[-1]["recall1"],
            "high_noise_mrr": results[-1]["mrr"],
        }
    report["weight_selection"] = aggregate
    report["selected_weight"] = float(max(aggregate, key=lambda key: (
        aggregate[key]["high_noise_recall1"], aggregate[key]["high_noise_mrr"],
        aggregate[key]["mean_recall1"], aggregate[key]["mean_mrr"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "selected_weight": report["selected_weight"],
                      "noise_levels": {k: {"textual_r1": v["textual"]["recall1"],
                                             "coding_r1": v["coding_weights"][str(report["selected_weight"])]["recall1"],
                                             "textual_mrr": v["textual"]["mrr"],
                                             "coding_mrr": v["coding_weights"][str(report["selected_weight"])]["mrr"]}
                                       for k, v in report["noise_levels"].items()}}, indent=2))


if __name__ == "__main__":
    main()
