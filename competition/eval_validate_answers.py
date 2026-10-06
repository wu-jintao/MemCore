#!/usr/bin/env python3
"""Default-off local Answer/Judge over the complete original public validation.

This separate loader accepts only manifest-approved LoCoMo validation histories
and questions. It never relabels validation as dev, supports no heldout/prefix
selection, and leaves frozen dev tools unchanged. Source verification uses only
histories; reference answers are opened afterwards for Judge alone. Default is
a no-network plan. Explicit local loopback, offline tokenizer, execution and
request/cost limits are required for generation. This independently worded
local proxy is not an official AML score or an untouched blind benchmark.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

import eval_answer as core
import eval_calibrate as calibration
import eval_contracts as contract
import eval_retrieval as retrieval
import eval_prepare as preparation

VERSION = "complete-public-validation-answer-proxy-v1"
DATASET = "locomo_refined"
FORBIDDEN_PARTS = {"heldout", "held-out", "holdout", "raw", "dev", "development"}


def require(value, message):
    if not value:
        raise core.EvaluationError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def input_file(path):
    path = Path(path).absolute()
    require(not any(p.lower() in FORBIDDEN_PARTS for p in path.parts),
            "Validation inputs cannot use dev, raw or heldout paths")
    require(path.resolve() == path and path.is_file(), "Validation input must be a regular file without links")
    return path


def frozen_json(path, hashes):
    path = input_file(path)
    expected = core.file_hash(path)
    value = core.read_json(path)
    require(core.file_hash(path) == expected, "Validation JSON changed while being read")
    hashes[str(path)] = expected
    return value


def frozen_rows(path, expected, hashes):
    path = input_file(path)
    require(valid_sha(expected) and core.file_hash(path) == expected, "Validation file SHA differs")
    value = core.read_jsonl(path)
    require(all(isinstance(r, dict) for r in value), "Validation rows must be JSON objects")
    require(core.file_hash(path) == expected, "Validation rows changed while being read")
    hashes[str(path)] = expected
    return value


def unique_rows(rows, field):
    ids = [r.get(field) for r in rows]
    require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids),
            "Validation contains missing or duplicate identities")
    return dict(zip(ids, rows))


def manifest_membership(path, hashes):
    manifest = frozen_json(path, hashes)
    require(isinstance(manifest, dict) and manifest.get("schema") == "memory-validation-governance-v1" and
            manifest.get("allocation") == "preserved_prepared_splits" and
            manifest.get("selection_reads_labels") is False and manifest.get("public_not_blind") is True,
            "Require a preserved public identity-governance manifest")
    groups = manifest.get("groups")
    require(isinstance(groups, list) and groups, "Missing validation identity groups")
    seen_queries, seen_samples, expected, group_ids = set(), set(), {}, []
    for group in groups:
        require(isinstance(group, dict) and group.get("partition") in {"development", "validation", "holdout"},
                "Invalid governance group")
        members, queries = group.get("members"), group.get("queries")
        require(isinstance(members, list) and members and isinstance(queries, list) and queries,
                "Empty governance identity group")
        group_samples = set()
        for member in members:
            require(isinstance(member, dict) and isinstance(member.get("dataset"), str) and
                    isinstance(member.get("sample_id"), str) and member["sample_id"], "Invalid sample membership")
            identity = (member["dataset"], member["sample_id"])
            require(identity not in seen_samples, "Sample identity crosses governance groups")
            seen_samples.add(identity)
            group_samples.add(identity)
        for query in queries:
            require(isinstance(query, dict) and isinstance(query.get("dataset"), str) and
                    isinstance(query.get("query_id"), str) and query["query_id"], "Invalid question membership")
            identity = (query["dataset"], query["query_id"])
            require(identity not in seen_queries, "Question identity crosses governance groups")
            seen_queries.add(identity)
        if group["partition"] != "validation":
            continue
        require(group.get("original_prepared_splits") == ["validation"] and
                group.get("reason") == "preserved_prepared_split", "Validation must preserve its original partition")
        require(len(group_samples) == 1 and next(iter(group_samples))[0] == DATASET and
                all(q["dataset"] == DATASET for q in queries),
                "Only complete original LoCoMo validation conversations are supported; no independent LME validation")
        sample = next(iter(group_samples))[1]
        for query in queries:
            expected[query["query_id"]] = sample
        require(isinstance(group.get("group_id"), str) and group["group_id"] not in group_ids,
                "Duplicate or invalid validation group ID")
        group_ids.append(group["group_id"])
    require(expected and type(manifest.get("question_counts", {}).get("validation")) is int and
            manifest["question_counts"]["validation"] == len(expected), "Incomplete manifest validation partition")
    pinned = {}
    for entry in manifest.get("sources", []):
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str), "Invalid governance source metadata")
        parts = Path(entry["path"]).parts
        for name in ("histories", "queries"):
            if parts[-4:] == ("prepared", DATASET, "validation", name + ".jsonl"):
                require(name not in pinned and valid_sha(entry.get("sha256")), "Duplicate/invalid validation source pin")
                pinned[name] = entry["sha256"]
    require(set(pinned) == {"histories", "queries"}, "Original validation histories and questions must be pinned")
    return manifest, expected, group_ids, pinned


def load_run(folder, hashes):
    folder = Path(folder).absolute()
    summary = frozen_json(folder / "summary.json", hashes)
    require(isinstance(summary, dict) and summary.get("split") == "validation" and summary.get("dataset") == DATASET,
            "Only real LoCoMo split=validation runs are permitted")
    require(type(summary.get("question_count")) is int and summary["question_count"] > 0,
            "Invalid validation question denominator")
    require(isinstance(summary.get("run_id"), str) and re.fullmatch(r"[a-zA-Z0-9_-]+", summary["run_id"]),
            "Invalid validation run identity")
    config = summary.get("configuration", {})
    require(isinstance(config, dict) and type(config.get("top_k")) is int and config["top_k"] == 100 and
            type(config.get("character_budget")) is int and config["character_budget"] == 0 and
            config.get("source_mapping_version") == preparation.SOURCE_MAPPING_VERSION,
            "Require untrimmed top100 occurrence-v2 validation retrieval")
    source = config.get("source_code_sha256", {})
    require(isinstance(source, dict) and all(h is None or valid_sha(h) for h in source.values()) and
            valid_sha(source.get("server.py_at_start")) and source["server.py_at_start"] == source.get("server.py_at_end") and
            summary.get("server_source_changed_during_run") is False, "Require frozen retrieval server source")
    path = input_file(folder / "retrieval.jsonl")
    rows = frozen_rows(path, core.file_hash(path), hashes)
    records = unique_rows(rows, "query_id")
    require(len(rows) == summary["question_count"], "Missing validation retrieval rows")
    lineage = {"run_id": summary["run_id"], "split": "validation",
               "summary_sha256": hashes[str(folder / "summary.json")],
               "retrieval_sha256": hashes[str(path)], "source_code_sha256": source,
               "source_mapping_version": config["source_mapping_version"]}
    return summary, records, lineage


def item_counter(row):
    items = row.get("retrieved", [])
    require(isinstance(items, list) and len(items) <= 100, "Invalid returned validation item list")
    for item in items:
        require(isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] and
                isinstance(item.get("content"), str) and item["content"], "Invalid returned source item")
    require(len({i["id"] for i in items}) == len(items), "Duplicate returned source IDs")
    if row.get("status") != "ok":
        require(not items, "Failed retrieval must not carry stale source items")
    return Counter((i["id"], i["content"]) for i in items)


def source_index(history, run_id, first_sequence=1):
    require(isinstance(history.get("sessions"), list) and history["sessions"], "Empty validation history")
    session_ids = []
    for session in history["sessions"]:
        require(isinstance(session, dict) and isinstance(session.get("session_id"), str) and
                session["session_id"] and isinstance(session.get("messages"), list) and session["messages"],
                "Invalid validation session")
        session_ids.append(session["session_id"])
        for message in session["messages"]:
            require(isinstance(message, dict) and message.get("role") in {"user", "assistant"} and
                    isinstance(message.get("content"), str) and message["content"] and
                    (message.get("timestamp") is None or type(message["timestamp"]) is int),
                    "Invalid validation history message")
    require(len(set(session_ids)) == len(session_ids), "Repeated source session identity")
    _, payloads, sources, ids = retrieval.requests_for_history(history, run_id, preparation.SOURCE_MAPPING_VERSION)
    metadata = {}
    sequence = first_sequence
    for payload in payloads:
        for index, message in enumerate(payload["messages"]):
            original = sources[payload["session_id"], payload["request_id"], index]
            header = {"role": message["role"], "session_id": payload["session_id"],
                "request_id": payload["request_id"], "message_index": index,
                "message_id": original["message_id"], "received_sequence": sequence}
            if message.get("timestamp") is not None:
                header["timestamp_ms"] = message["timestamp"]
                try:
                    header["timestamp_utc"] = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=message["timestamp"])).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                except OverflowError:
                    pass  # Same documented service behavior for out-of-range UTC dates.
            metadata[original["message_id"]] = header
            sequence += 1
    return sources, ids, metadata


def verify_sources(items, index):
    sources, ids, metadata = index
    texts = {s["content"] for s in sources.values()}
    for item in items:
        anchor = ids.get(item["id"])
        require(anchor is not None, "Returned anchor is outside its exact validation conversation")
        segments = list(retrieval.source_segments(item["content"], sources, anchor, texts))
        require(segments, "Returned source has no verifiable original body")
        reconstructed = []
        for number, (source, body) in enumerate(segments):
            if source is None:
                require(len(segments) == 1 and body == anchor["content"] and item["content"] == body,
                        "Unmapped returned source text")
                reconstructed.append(body)
                continue
            original = retrieval.source_original(source, sources)
            require(original is not None and body == original["content"], "Returned body differs from original history")
            require(number != 0 or original["message_id"] == item["id"], "Returned anchor/header differs")
            required = {"role", "session_id", "request_id", "message_index", "message_id", "received_sequence"}
            require(required <= set(source) and set(source) <= required | {"timestamp_ms", "timestamp_utc"} and
                    source["message_id"] == original["message_id"] and
                    type(source["received_sequence"]) is int and source["received_sequence"] > 0,
                    "Returned source header contains unverified fields or identity metadata")
            expected = metadata[original["message_id"]]
            require(core.canonical_hash(source) == core.canonical_hash(expected), "Returned source metadata differs from original serial history ingestion")
            reconstructed.append("[source " + json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "]\n" + body)
        # Byte comparison rejects duplicate JSON keys and overwritten values
        # which ordinary json.loads would hide but an Answer model would see.
        require("\n\n[adjacent context]\n".join(reconstructed) == item["content"], "Returned source wrapper bytes differ from the verified service format")


def matched_validation_inputs(args):
    hashes = {}
    manifest, membership, groups, pinned = manifest_membership(args.validation_manifest, hashes)
    runs = [load_run(args.baseline_run, hashes), load_run(args.candidate_run, hashes)]
    baseline, candidate = runs
    expected_ids, expected_samples = set(membership), set(membership.values())
    require(all(set(r[1]) == expected_ids and r[0]["question_count"] == len(expected_ids) for r in runs),
            "Both validation runs must retain every manifest question")
    require(baseline[0].get("status") == "public_proxy_not_official_aml" and
            baseline[0].get("ingestion_run_id", baseline[2]["run_id"]) == baseline[2]["run_id"] and
            baseline[0].get("ingestion_failed_histories") == 0 and
            baseline[0].get("history_count") == len(expected_samples) and
            baseline[0]["configuration"].get("launch_local") is True and
            type(baseline[0]["configuration"].get("add_concurrency")) is int and
            baseline[0]["configuration"]["add_concurrency"] == 1,
            "Baseline must be the complete original local validation ingestion")
    require(candidate[0].get("source_run_id") == baseline[2]["run_id"] and
            candidate[0].get("ingestion_run_id") == baseline[2]["run_id"] and
            candidate[0].get("candidate_set_and_content_unchanged") is True and
            core.canonical_hash(candidate[0]["configuration"]) == core.canonical_hash(baseline[0]["configuration"]),
            "Candidate must preserve original validation ingestion and configuration")
    provenance = candidate[0].get("rerank_provenance", {})
    require(isinstance(provenance, dict) and provenance.get("source_summary_sha256") == baseline[2]["summary_sha256"] and
            provenance.get("source_retrieval_sha256") == baseline[2]["retrieval_sha256"] and
            valid_sha(provenance.get("rank_engine_sha256")) and valid_sha(provenance.get("tool_sha256")),
            "Reranker must separately bind its engine/tool and original source files")
    for run in runs:
        run[2].update(ingestion_run_id=baseline[2]["run_id"],
            ingestion_summary_sha256=baseline[2]["summary_sha256"], ingestion_retrieval_sha256=baseline[2]["retrieval_sha256"])
    candidate[2]["rerank_provenance"] = provenance
    specification = baseline[0]["configuration"].get("prepared_data")
    require(isinstance(specification, dict) and isinstance(specification.get("files"), dict), "Missing prepared validation manifest")
    data_root = Path(args.data_root).absolute()
    prepared = frozen_json(data_root / "prepared/summary.json", hashes)
    require(prepared.get("datasets", {}).get(DATASET, {}).get("splits", {}).get("validation") == specification,
            "Validation preparation differs from frozen retrieval input")
    require(specification.get("histories") == len(expected_samples) and specification.get("questions") == len(expected_ids),
            "Prepared validation conversation/question counts differ")
    files = specification["files"]
    require(all(isinstance(files.get(n), dict) and valid_sha(files[n].get("sha256")) for n in ("histories", "queries", "gold")),
            "Missing frozen prepared validation file hashes")
    require(all(files[n]["sha256"] == pinned[n] for n in pinned), "Validation source differs from original manifest-pinned split")
    for module in (retrieval, preparation):
        path = Path(module.__file__)
        require(baseline[0]["configuration"]["source_code_sha256"].get(path.name) == core.file_hash(path),
                "Source reconstruction helper differs from original ingestion")
    folder = data_root / "prepared" / DATASET / "validation"
    histories = unique_rows(frozen_rows(folder / "histories.jsonl", files["histories"]["sha256"], hashes), "sample_id")
    queries = frozen_rows(folder / "queries.jsonl", files["queries"]["sha256"], hashes)
    query_map = unique_rows(queries, "query_id")
    require(set(histories) == expected_samples and set(query_map) == expected_ids, "Validation must contain complete manifest conversations")
    require(getattr(args, "max_questions", None) in (None, len(queries)), "Validation cannot select a question prefix")
    # The explicit fresh original --launch-local run uses an empty per-run DB.
    # Serial Add assigns one global sequence in prepared history order; do not
    # reset it per conversation or sort it by question or source IDs.
    indexes, next_sequence = {}, 1
    for sample, history in histories.items():
        indexes[sample] = source_index(history, baseline[2]["run_id"], next_sequence)
        next_sequence += sum(len(s["messages"]) for s in history["sessions"])
    source_checks = []
    for query in queries:
        qid = query["query_id"]
        require(query.get("sample_id") == membership[qid] and isinstance(query.get("query"), str) and query["query"].strip() and
                not query.get("options"), "Invalid validation question/conversation mapping")
        before, after = baseline[1][qid], candidate[1][qid]
        require(before.get("sample_id") == after.get("sample_id") == membership[qid] and
                isinstance(before.get("status"), str) and before["status"] == after.get("status"), "Rerank row scope/status differs")
        require(all("query" not in row or row["query"] == query["query"] for row in (before, after)),
                "Rerank question text differs from the original prepared question")
        a, b = item_counter(before), item_counter(after)
        require(a == b, "Reranker changed the original returned id/content multiset")
        for row in (before, after):
            verify_sources(row.get("retrieved", []), indexes[membership[qid]])
        source_checks.append({"query_id": qid, "original_items": sum(a.values()),
            "multiset_sha256": core.canonical_hash(sorted((i, text, count) for (i, text), count in a.items())),
            "baseline_order_sha256": core.canonical_hash([(i["id"], i["content"]) for i in before.get("retrieved", [])]),
            "candidate_order_sha256": core.canonical_hash([(i["id"], i["content"]) for i in after.get("retrieved", [])])})
    # No reference, category or evidence field participates in any source check.
    gold_rows = frozen_rows(folder / "gold.jsonl", files["gold"]["sha256"], hashes)
    references = unique_rows(gold_rows, "query_id")
    require(set(references) == expected_ids, "Judge references must retain every manifest question")
    gold = {}
    for qid, row in references.items():
        answer = row.get("answer")
        require(row.get("sample_id") == membership[qid] and not isinstance(answer, bool) and
                isinstance(answer, (str, int, float, list)) and
                (not isinstance(answer, float) or math.isfinite(answer)), "Invalid validation Judge reference")
        gold[qid] = {"answer": answer}
    public_queries = [{"query_id": q["query_id"], "sample_id": q["sample_id"], "query": q["query"]} for q in queries]
    require(all(core.file_hash(Path(p)) == h for p, h in hashes.items()), "Frozen validation input changed during loading")
    return runs, public_queries, gold, hashes, {"schema": manifest["schema"], "groups": groups,
        "histories": len(expected_samples), "questions": len(queries), "public_not_blind": True,
        "membership_sha256": core.canonical_hash(sorted(membership.items())), "source_checks": source_checks}


def run(args, encoding=None, client_factory=core.ChatClient):
    require(args.local_loopback and args.tokenizer_path is not None, "Validation requires explicit local loopback and offline tokenizer")
    require(args.input_usd_per_million in (None, 0) and args.output_usd_per_million in (None, 0), "Local API fee rates must be zero")
    kwargs = {"enable_thinking": False} if args.chat_template_kwargs is None else json.loads(args.chat_template_kwargs)
    settings = core.Settings(answer_model=args.answer_model, judge_model=args.judge_model, endpoint=args.endpoint,
        memory_tokens=args.memory_tokens, input_tokens=args.input_tokens, context_tokens=args.context_tokens,
        chat_overhead=args.chat_overhead, answer_output_tokens=args.answer_output_tokens, judge_output_tokens=args.judge_output_tokens,
        max_requests=args.max_requests, max_cost_usd=.10 if args.max_cost_usd is None else args.max_cost_usd,
        input_usd_per_million=0, output_usd_per_million=0, timeout=args.timeout,
        local_loopback=True, temperature=args.temperature, chat_template_kwargs=kwargs)
    settings.validate()
    runs, queries, gold, input_hashes, governance = matched_validation_inputs(args)
    encoding = encoding or core.offline_hf_encoding(args.tokenizer_path, kwargs)
    tools = {str(Path(module.__file__)): core.file_hash(Path(module.__file__))
             for module in (core, calibration, contract, retrieval, preparation)}
    tools[str(Path(__file__))] = core.file_hash(Path(__file__))
    jobs, estimate = calibration.build_plan(runs, queries, gold, settings, encoding)
    require(len(jobs) == 2 * len(queries) and all(j["mode"] == "answer" for j in jobs), "Validation plan must retain the full fresh Answer denominator")
    require(not args.execute or args.max_cost_usd is not None and estimate["request_limit_sufficient"],
            "Execution needs sufficient explicit request and cost limits")
    require(all(core.file_hash(Path(p)) == h for p, h in {**input_hashes, **tools}.items()), "Frozen inputs changed before execution")
    ident = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-validation-answer-" + uuid.uuid4().hex[:8]
    output = core.private_directory(args.output_dir or core.PRIVATE_ROOT / "validation-answer-evals" / ident)
    report = {"version": VERSION, "evaluation_id": ident, "split": "validation", "dataset": DATASET,
        "status": "executed_public_validation_local_proxy_not_official_aml" if args.execute else "dry_run_no_network",
        "mode": "answer", "matched_question_denominator": len(queries), "history_count": governance["histories"],
        "selection": "Complete manifest-approved original validation conversations and question set; fixed prepared file order; no prefix selection",
        "selected_query_ids_sha256": core.canonical_hash([q["query_id"] for q in queries]),
        "governance": {k: v for k, v in governance.items() if k != "source_checks"},
        "source_validation": {"history_only": True, "questions_checked": len(queries),
            "candidate_original_id_content_multisets_exact": True, "scope": "Exact source bodies and anchor IDs inside the same original conversation",
            "received_sequence": "Global prepared-history/message order for the declared fresh serial per-run local ingestion; original DB is not reopened",
            "checks_sha256": core.canonical_hash(governance["source_checks"])},
        "source_runs": [r[2] for r in runs], "input_files_sha256": input_hashes, "tools_sha256": tools,
        "contract": contract.profile(DATASET), "answer_rules_sha256": core.digest_bytes(contract.ANSWER_RULES.encode()),
        "judge_rules_sha256": core.digest_bytes(contract.JUDGE_RULES.encode()),
        "model": settings.answer_model, "temperature": settings.temperature,
        "request_limits": {"endpoint": settings.endpoint, "max_requests": settings.max_requests,
            "max_cost_usd": settings.max_cost_usd, "answer_output_tokens": settings.answer_output_tokens,
            "judge_output_tokens": settings.judge_output_tokens, "timeout_seconds": settings.timeout,
            "chat_template_kwargs": settings.chat_template_kwargs, "attempts_per_call": 1},
        "packing": {"memory_tokens": settings.memory_tokens, "input_tokens": settings.input_tokens,
            "context_tokens": settings.context_tokens, "chat_reserve": settings.chat_overhead,
            "tokenizer": getattr(encoding, "metadata", {}), "question_date_included": False,
            "rule": "Ordered complete item prefix, original item bytes, independent single-user-message prompt",
            "reference": "Reference answer only in Judge; no category/evidence/answer labels enter Answer"},
        "cost_plan": {**estimate, "api_fee_usd": 0, "compute_cost_accounted": False},
        "planned_status_counts": dict(Counter(j["status"] for j in jobs)), "metrics": None,
        "usage": {"attempted_requests": 0}, "actual_paid_api_requests": 0,
        "limitations": ["Public validation is independent of previously used exact conversation identities under the supplied manifest; it is not a blind private benchmark.",
            "Local same-family Answer/Judge, independently worded prompt and complete-prefix packing are not confirmed production AML orchestration or scores.",
            "Only LoCoMo free-answer original validation is supported; LongMemEval has no independent validation component in this manifest and heldout is not supported.",
            "API fee is zero; GPU, electricity and other compute costs are not accounted. Failures stay in both fixed denominators."]}
    core.write_private_json(output / "source-checks.json", governance["source_checks"])
    core.write_private_json(output / "plan.json", [{k: j[k] for k in ("query_id", "side", "run_id", "status", "mode", "packing") if k in j} for j in jobs])
    start = time.monotonic()
    if args.execute:
        key_env = args.api_key_env or "AML_LOCAL_LLM_API_KEY"
        key = None if args.api_key_file is None and not os.environ.get(key_env, "").strip() else core.load_key(key_env, args.api_key_file)
        client = client_factory(settings, key)
        raw_path = output / "answers-judgments.jsonl"
        with raw_path.open("x", encoding="utf-8") as stream:
            os.chmod(raw_path, 0o600)
            def sink(row):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            results, ledger = calibration.execute_jobs(jobs, settings, encoding, client, sink)
        report["metrics"] = core.aggregate_results(results, len(queries))
        report["usage"] = {"attempted_requests": len(ledger.calls),
            "usage_reported_requests": sum(r["usage_status"] == "reported" for r in ledger.calls),
            "usage_unknown_requests": sum(r["usage_status"] == "unknown" for r in ledger.calls),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in ledger.calls),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in ledger.calls),
            "stopped_after_estimate_excess": ledger.stopped}
        report["private_raw_records_sha256"] = core.file_hash(raw_path)
        core.write_private_json(output / "ledger.json", ledger.calls)
    report["elapsed_seconds"] = time.monotonic() - start
    report["all_inputs_and_tools_unchanged"] = all(core.file_hash(Path(p)) == h for p, h in {**input_hashes, **tools}.items())
    if "files_sha256" in getattr(encoding, "metadata", {}):
        report["tokenizer_files_unchanged"] = core.local_tokenizer_files(args.tokenizer_path)[1] == encoding.metadata["files_sha256"]
    core.write_private_json(output / "summary.json", report)
    require(report["all_inputs_and_tools_unchanged"] and report.get("tokenizer_files_unchanged", True), "Validation inputs/tools/tokenizer changed during execution")
    return report


def parser():
    result = core.parser()
    result.description = __doc__
    result.add_argument("--validation-manifest", type=Path, required=True,
        help="Frozen public identity audit authorizing the complete original validation partition")
    result.set_defaults(max_questions=None)
    for action in result._actions:
        if action.dest == "max_questions":
            action.help = "Optional equality assertion of the full manifest question count; never a prefix selector"
    return result


def main(argv=None):
    try:
        report = run(parser().parse_args(argv))
        print(json.dumps({k: report[k] for k in ("status", "split", "matched_question_denominator", "cost_plan", "metrics", "all_inputs_and_tools_unchanged")}, indent=2))
        return 0
    except (core.EvaluationError, OSError, KeyError, TypeError, ValueError, OverflowError):
        print("Validation proxy stopped: invalid frozen provenance, configuration or execution limit.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
