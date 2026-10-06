#!/usr/bin/env python3
"""Compare completed PUBLIC dev runs at matched item and character budgets.

This postprocessing only reads already retrieved evidence: no HTTP/model request.
All runs must have exactly the same query IDs and prepared data hashes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from eval_prepare import DEFAULT_ROOT, LEGACY_SOURCE_MAPPING_VERSION, file_hash, normalized
from eval_retrieval import GRADER_VERSION, aggregate, evidence_hits, prefix_budget, read_jsonl, requests_for_history


def compare(run_directories, data_root, ks, budgets):
    runs = []
    for folder in run_directories:
        summary = json.loads((folder / "summary.json").read_text())
        if summary["split"] == "heldout":
            raise ValueError("Development comparison must not consume heldout results")
        if summary.get("server_source_changed_during_run"):
            raise ValueError("A run's server source changed during execution; audit before comparing")
        records = read_jsonl(folder / "retrieval.jsonl")
        if len(records) != summary["question_count"]:
            raise ValueError("Incomplete retrieval result file")
        runs.append((folder, summary, records))
    reference = runs[0][1]
    ids = {r["query_id"] for r in runs[0][2]}
    for _, summary, records in runs[1:]:
        if summary["dataset"] != reference["dataset"] or summary["split"] != reference["split"]:
            raise ValueError("Compare the same dataset and split only")
        if {r["query_id"] for r in records} != ids:
            raise ValueError("Compared runs must evaluate exactly the same public questions")
        if summary["configuration"]["prepared_data"] != reference["configuration"]["prepared_data"]:
            raise ValueError("Compared runs use different prepared data")
    root = data_root / "prepared" / reference["dataset"] / reference["split"]
    labels = {g["query_id"]: g for g in read_jsonl(root / "gold.jsonl") if g["query_id"] in ids}
    sample_ids = {g["sample_id"] for g in labels.values()}
    histories = {h["sample_id"]: h for h in read_jsonl(root / "histories.jsonl") if h["sample_id"] in sample_ids}
    for name in ("histories", "queries", "gold"):
        expected = reference["configuration"]["prepared_data"]["files"][name]["sha256"]
        if file_hash(root / (name + ".jsonl")) != expected:
            raise ValueError("Prepared data has changed")
    frequencies = {}
    for sid, history in histories.items():
        texts = {normalized(u["text"]) for g in labels.values() if g["sample_id"] == sid for u in g["gold_units"]}
        frequencies[sid] = Counter({text: sum(text in normalized(m["content"]) for s in history["sessions"] for m in s["messages"]) for text in texts})
    result = {"status": "public_proxy_not_official_aml", "grader_version": GRADER_VERSION,
              "grader_source_sha256": file_hash(Path(__file__).parent / "eval_retrieval.py"),
              "dataset": reference["dataset"], "split": reference["split"],
              "matched_questions": len(ids), "matched_histories": len(histories), "runs": [],
              "limitations": ["Character budgets are characters, not official Answer tokens. Entire returned item prefixes are preserved.",
                              "These are paired small-development-set retrieval diagnostics, not AML scores or answer accuracy.",
                              "Question observations within one conversation are correlated; this comparison is not a statistically validated general improvement.",
                              "No Answer/Judge or additional service calls are made by comparison."]}
    base_by_budget = {}
    for index, (folder, summary, records) in enumerate(runs):
        source_mapping_version = summary["configuration"].get("source_mapping_version", LEGACY_SOURCE_MAPPING_VERSION)
        ingestion_run_id = summary.get("ingestion_run_id", summary["run_id"])
        source_maps = {sid: requests_for_history(history, ingestion_run_id, source_mapping_version)[2:]
                       for sid, history in histories.items()}
        scored = {}
        for k in ks:
            if k > summary["configuration"]["top_k"]:
                raise ValueError("Comparison k exceeds a run's requested top_k")
            for budget in budgets:
                key = "k" + str(k) + "/characters" + str(budget)
                rows = []
                per_question = {}
                for record in records:
                    sid = record["sample_id"]
                    if record["status"] != "ok":
                        rows.append({"status": record["status"]})
                        per_question[record["query_id"]] = False
                        continue
                    prefix, chars = prefix_budget(record["retrieved"], k, budget)
                    sources, ids_by_message = source_maps[sid]
                    metrics = {**evidence_hits(prefix, labels[record["query_id"]], sources, ids_by_message, frequencies[sid]),
                               "returned_items": len(prefix), "returned_characters": chars}
                    rows.append({"status": "ok", "metrics": {str(k): metrics}})
                    per_question[record["query_id"]] = metrics["recall_all"]
                scored[key] = aggregate(rows, k, len(records))
                if budget == 0 and summary["configuration"]["character_budget"] == 0:
                    successful = [(record, row) for record, row in zip(records, rows) if record["status"] == "ok"]
                    scored[key]["regrading_changes_vs_original_saved_metrics"] = {
                        "questions_with_changed_hit_turn_count": sum(record["metrics"][str(k)]["hit_turns"] != row["metrics"][str(k)]["hit_turns"] for record, row in successful),
                        "questions_with_changed_verified_hit_turn_count": sum(record["metrics"][str(k)]["source_verified_hit_turns"] != row["metrics"][str(k)]["source_verified_hit_turns"] for record, row in successful)}
                if index == 0:
                    base_by_budget[key] = per_question
                else:
                    baseline = base_by_budget[key]
                    scored[key]["paired_complete_coverage_changes_vs_first_run"] = {
                        "gained_questions": sum(per_question[q] and not baseline[q] for q in per_question),
                        "lost_questions": sum(baseline[q] and not per_question[q] for q in per_question)}
        result["runs"].append({"run_id": summary["run_id"], "directory": str(folder), "model_label": summary["model_label"],
                               "original_grader_version": summary.get("grader_version", "source-inclusive-v1-superseded"),
                               "original_report_preserved": True,
                               "configuration": summary["configuration"], "elapsed_seconds": summary["elapsed_seconds"],
                               "add_latency": summary["add_latency"], "search_latency": summary["search_latency"],
                               "metrics": scored})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ks", default="5,10,20,50,100")
    parser.add_argument("--character-budgets", default="0,8000,16000,32000,64000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ks, budgets = [int(x) for x in args.ks.split(",")], [int(x) for x in args.character_budgets.split(",")]
    if not ks or not budgets or min(ks) < 1 or min(budgets) < 0:
        parser.error("Invalid comparison budgets")
    result = compare(args.runs, args.data_root, ks, budgets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "matched_questions": result["matched_questions"], "runs": [r["model_label"] for r in result["runs"]]}))


if __name__ == "__main__":
    main()
