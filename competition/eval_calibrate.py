#!/usr/bin/env python3
"""Local-only calibration against independently worded public free-answer rules.

Default is a no-network plan. Historical rejudging changes only the scorer and
does not establish any memory improvement. Fresh answers change the proxy
prompt as well. Neither is an official AML evaluation. Frozen v3 is imported
for immutable input checks, local transport, token counting and budget handling;
its source, historical outputs and metric definitions remain unchanged.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid

import eval_answer as core
import eval_contracts as contract

VERSION = "local-public-contract-calibration-v1"


def pack_memories(dataset, query, items, encoding, settings):
    if not isinstance(items, list) or len(items) > 100:
        raise core.EvaluationError("Invalid frozen retrieval list")
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str) or
                not item["id"] or not isinstance(item.get("content"), str) or not item["content"]):
            raise core.EvaluationError("Invalid frozen source item")
    parts = ["Memory " + str(i + 1) + ":\n" + item["content"] + "\n\n" for i, item in enumerate(items)]

    def build(n):
        body = "".join(parts[:n])
        return body, contract.build_answer_messages(dataset, query, body)

    def fits(n):
        body, messages = build(n)
        return (core.tokens(encoding, body) <= settings.memory_tokens and
                core.message_tokens(encoding, messages, settings.chat_overhead) <= settings.input_tokens)

    if not fits(0):
        raise core.EvaluationError("Question exceeds the input budget")
    low, high = 0, len(parts)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    body, messages = build(low)
    return messages, {"selected_items": low, "available_items": len(items),
        "memory_tokens": core.tokens(encoding, body),
        "estimated_input_tokens": core.message_tokens(encoding, messages, settings.chat_overhead),
        "packed_messages_sha256": core.canonical_hash(messages)}


def historical_answers(folder, runs, queries, spec):
    folder = folder.resolve()
    if not folder.is_relative_to(core.PRIVATE_ROOT.resolve()):
        raise core.EvaluationError("Historical raw outputs must remain private")
    summary_path = core.dev_input_file(folder / "summary.json")
    raw_path = core.dev_input_file(folder / "answers-judgments.jsonl")
    summary_hash = core.file_hash(summary_path)
    summary = core.read_json(summary_path)
    if (summary.get("version") != core.VERSION or
            summary.get("tool_sha256") != core.file_hash(Path(core.__file__)) or
            summary.get("status") != "executed_public_dev_proxy_not_official_aml"):
        raise core.EvaluationError("Require executed unchanged frozen-v3 historical outputs")
    if summary.get("source_runs") != [run[2] for run in runs]:
        raise core.EvaluationError("Historical retrieval lineage differs")
    if summary.get("prepared_file_sha256") != {k: v["sha256"] for k, v in spec["files"].items()}:
        raise core.EvaluationError("Historical prepared data differs")
    recorded_raw_hash = summary.get("private_raw_records_sha256")
    if (not isinstance(recorded_raw_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded_raw_hash) or
            core.file_hash(raw_path) != recorded_raw_hash):
        raise core.EvaluationError("Historical raw answers differ from their original summary")
    hashes = {str(summary_path): summary_hash, str(raw_path): recorded_raw_hash}
    rows = core.read_jsonl(raw_path)
    by_key = {(row.get("query_id"), row.get("side")): row for row in rows}
    count = summary.get("matched_question_denominator")
    if type(count) is not int or len(rows) != 2 * count or len(by_key) != len(rows):
        raise core.EvaluationError("Historical outputs have missing or duplicate rows")
    for side, run in zip(("baseline", "candidate"), runs):
        selected = [r for r in rows if r["side"] == side]
        if (len(selected) != count or any(r.get("run_id") != run[2]["run_id"] or
                r.get("query_id") not in run[1] for r in selected)):
            raise core.EvaluationError("Historical row scope differs")
    selected_ids = [r["query_id"] for r in rows if r["side"] == "baseline"]
    if core.canonical_hash(selected_ids) != summary.get("selected_query_ids_sha256"):
        raise core.EvaluationError("Historical question ordering differs")
    if any((q["query_id"], side) not in by_key for q in queries for side in ("baseline", "candidate")):
        raise core.EvaluationError("Historical answers omit a selected question")
    if any(core.file_hash(Path(path)) != value for path, value in hashes.items()):
        raise core.EvaluationError("Historical files changed during reading")
    return by_key, hashes, summary.get("packing")


def build_plan(runs, queries, gold, settings, encoding, previous=None):
    dataset = runs[0][0]["dataset"]
    contract.profile(dataset)
    jobs = []
    request_count = 0
    for query in queries:
        qid = query["query_id"]
        for side, (_, records, lineage) in zip(("baseline", "candidate"), runs):
            job = {"query_id": qid, "side": side, "run_id": lineage["run_id"],
                   "dataset": dataset, "query": query, "reference": gold[qid]["answer"],
                   "status": "planned", "mode": "rejudge" if previous is not None else "answer"}
            if records[qid].get("status") != "ok":
                job["status"] = "retrieval_error"
            elif previous is not None:
                old = previous[qid, side]
                job["packing"] = old.get("packing")
                if old.get("status") == "answer_error" or not isinstance(old.get("answer"), str) or not old["answer"].strip():
                    job["status"] = "historical_answer_error"
                else:
                    job["answer"] = old["answer"]
                    job["historical_answer_sha256"] = core.digest_bytes(old["answer"].encode())
            else:
                try:
                    job["messages"], job["packing"] = pack_memories(dataset, query, records[qid].get("retrieved"), encoding, settings)
                except (ValueError, core.EvaluationError):
                    job["status"] = "input_error"
            if job["status"] == "planned":
                messages = contract.build_judge_messages(dataset, query, job.get("answer", "placeholder"), job["reference"])
                bound = core.message_tokens(encoding, messages, settings.chat_overhead)
                if previous is None:
                    bound += 8 * settings.answer_output_tokens + settings.chat_overhead
                if bound > settings.input_tokens:
                    job["status"] = "input_error"
                else:
                    job["judge_input_bound"] = bound
                    request_count += 1 if previous is not None else 2
            jobs.append(job)
    return jobs, {"planned_requests_maximum": request_count,
                  "request_limit_sufficient": request_count <= settings.max_requests,
                  "estimated_uncached_api_cost_usd": 0.0, "compute_cost_accounted": False}


def execute_jobs(jobs, settings, encoding, client, sink):
    results, ledger = [], core.Ledger(settings)
    for job in jobs:
        result = {k: job[k] for k in ("query_id", "side", "run_id", "status", "mode")}
        result.update(correct=False, packing=job.get("packing"))
        stage = "answer" if job["mode"] == "answer" else "judge"
        if job["status"] == "planned":
            try:
                if job["mode"] == "answer":
                    row = ledger.reserve("answer", job["packing"]["estimated_input_tokens"], settings.answer_output_tokens)
                    response = client.complete(settings.answer_model, job["messages"], settings.answer_output_tokens)
                    ledger.settle(row, response)
                    answer = core.completion_text(response)
                    if core.tokens(encoding, answer) > settings.answer_output_tokens:
                        ledger.stopped = True
                        raise core.ProviderError("answer_exceeds_output_cap")
                else:
                    answer = job["answer"]
                    result["historical_answer_sha256"] = job["historical_answer_sha256"]
                result["answer"] = answer
                stage = "judge"
                messages = contract.build_judge_messages(job["dataset"], job["query"], answer, job["reference"])
                count = core.message_tokens(encoding, messages, settings.chat_overhead)
                if count > job["judge_input_bound"] or count > settings.input_tokens:
                    ledger.stopped = True
                    raise core.ProviderError("judge_exceeds_input_estimate")
                result["judge_messages_sha256"] = core.canonical_hash(messages)
                row = ledger.reserve("judge", job["judge_input_bound"], settings.judge_output_tokens)
                response = client.complete(settings.judge_model, messages, settings.judge_output_tokens, judge=True)
                ledger.settle(row, response)
                raw = core.completion_text(response)
                result["raw_judgment"] = raw
                try:
                    judgment = contract.parse_judgment(raw)
                except ValueError:
                    raise core.ProviderError("invalid_judgment") from None
                result.update(status="ok", correct=judgment["correct"], judgment=judgment)
            except core.ProviderError as error:
                result.update(status=stage + "_error", error_code=error.code)
            except Exception:
                result.update(status=stage + "_error", error_code="unexpected_client_error")
        sink(result)
        results.append(result)
    return results, ledger


def run(args, encoding=None, client_factory=core.ChatClient):
    if not args.local_loopback or args.tokenizer_path is None:
        raise core.EvaluationError("Calibration currently requires an explicit local loopback tokenizer profile")
    if args.input_usd_per_million not in (None, 0) or args.output_usd_per_million not in (None, 0):
        raise core.EvaluationError("Local API fee rates must be zero")
    if args.mode == "rejudge" and args.historical_output is None:
        raise core.EvaluationError("Rejudge mode requires historical frozen outputs")
    if args.mode == "answer" and args.historical_output is not None:
        raise core.EvaluationError("Fresh answer mode must not accept historical answers")
    kwargs = {"enable_thinking": False} if args.chat_template_kwargs is None else json.loads(args.chat_template_kwargs)
    settings = core.Settings(answer_model=args.answer_model, judge_model=args.judge_model,
        endpoint=args.endpoint, memory_tokens=args.memory_tokens, input_tokens=args.input_tokens,
        context_tokens=args.context_tokens, chat_overhead=args.chat_overhead,
        answer_output_tokens=args.answer_output_tokens, judge_output_tokens=args.judge_output_tokens,
        max_requests=args.max_requests, max_cost_usd=.10 if args.max_cost_usd is None else args.max_cost_usd,
        input_usd_per_million=0, output_usd_per_million=0, timeout=args.timeout,
        local_loopback=True, temperature=args.temperature, chat_template_kwargs=kwargs)
    settings.validate()
    encoding = encoding or core.offline_hf_encoding(args.tokenizer_path, kwargs)
    runs, queries, gold, spec = core.matched_inputs(args.baseline_run, args.candidate_run, args.data_root, args.max_questions)
    input_hashes = {}
    for folder_arg, run in zip((args.baseline_run, args.candidate_run), runs):
        folder = folder_arg.resolve()
        for name, field in (("summary.json", "summary_sha256"), ("retrieval.jsonl", "retrieval_sha256")):
            expected = run[2][field]
            if core.file_hash(folder / name) != expected:
                raise core.EvaluationError("Retrieval input changed after verification")
            input_hashes[str(folder / name)] = expected
        ingestion = folder.parent / run[2]["ingestion_run_id"]
        for name in ("summary", "retrieval"):
            path = core.dev_input_file(ingestion / (name + (".json" if name == "summary" else ".jsonl")))
            expected = run[2]["ingestion_" + name + "_sha256"]
            if core.file_hash(path) != expected:
                raise core.EvaluationError("Original ingestion input changed after verification")
            input_hashes[str(path)] = expected
    for name in ("histories", "queries", "gold"):
        path = args.data_root.resolve() / "prepared" / runs[0][0]["dataset"] / "dev" / (name + ".jsonl")
        input_hashes[str(path)] = spec["files"][name]["sha256"]
    previous = None
    historical_packing = None
    if args.mode == "rejudge":
        previous, old_hashes, historical_packing = historical_answers(args.historical_output, runs, queries, spec)
        input_hashes.update(old_hashes)
    tools = {str(Path(__file__)): core.file_hash(Path(__file__)),
             str(Path(core.__file__)): core.file_hash(Path(core.__file__)),
             str(Path(contract.__file__)): core.file_hash(Path(contract.__file__))}
    jobs, estimate = build_plan(runs, queries, gold, settings, encoding, previous)
    if args.execute and (args.max_cost_usd is None or not estimate["request_limit_sufficient"]):
        raise core.EvaluationError("Execution needs sufficient explicit request and cost limits")
    ident = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-calibration-" + uuid.uuid4().hex[:8]
    output = core.private_directory(args.output_dir or core.PRIVATE_ROOT / "calibration" / ident)
    report = {"version": VERSION, "evaluation_id": ident, "mode": args.mode,
        "status": "dry_run_no_network" if not args.execute else "executed_local_calibration_not_official_aml",
        "dataset": runs[0][0]["dataset"], "split": "dev", "matched_question_denominator": len(queries),
        "selected_query_ids_sha256": core.canonical_hash([q["query_id"] for q in queries]),
        "selection": "First matching questions in fixed prepared dev order; not selected by answers or scores",
        "source_runs": [run[2] for run in runs], "input_files_sha256": input_hashes,
        "tools_sha256": tools, "contract": contract.profile(runs[0][0]["dataset"]),
        "answer_rules_sha256": core.digest_bytes(contract.ANSWER_RULES.encode()),
        "judge_rules_sha256": core.digest_bytes(contract.JUDGE_RULES.encode()),
        "model": settings.answer_model, "temperature": settings.temperature,
        "request_limits": {"endpoint": settings.endpoint, "max_requests": settings.max_requests,
                           "max_cost_usd": settings.max_cost_usd, "timeout_seconds": settings.timeout,
                           "answer_output_tokens": settings.answer_output_tokens,
                           "judge_output_tokens": settings.judge_output_tokens,
                           "chat_template_kwargs": settings.chat_template_kwargs,
                           "attempts_per_call": 1, "api_fee_usd": 0},
        "new_answer_profile_packing_applied": args.mode == "answer",
        "historical_answer_packing": historical_packing,
        "new_profile_packing": {"memory_tokens": settings.memory_tokens, "input_tokens": settings.input_tokens,
                    "context_tokens": settings.context_tokens, "chat_reserve": settings.chat_overhead,
                    "tokenizer": getattr(encoding, "metadata", {}),
                    "rule": "Ordered complete item prefix; identical item bodies; independent single-user-message prompt",
                    "question_date_included": False},
        "cost_plan": estimate, "planned_status_counts": dict(Counter(j["status"] for j in jobs)),
        "metrics": None, "actual_paid_api_requests": 0,
        "limitations": ["Independent prompt wording, local Qwen and declared packing are not production AML replication.",
                        "Rejudging changes a measurement; it is not a memory-method gain.",
                        "Historical answers retain their original date-aware v3 Answer prompt in rejudge mode.",
                        "Same-family Answer/Judge bias remains; this reused development set is not a blind validation set.",
                        "MCQ, BEAM, CLBench, ScriptMem, PersonaMem and Streaming contracts are not implemented here."]}
    core.write_private_json(output / "plan.json", [{k: j[k] for k in ("query_id", "side", "run_id", "status", "mode", "packing") if k in j} for j in jobs])
    start = time.monotonic()
    if args.execute:
        key_env = args.api_key_env or "AML_LOCAL_LLM_API_KEY"
        key = None if args.api_key_file is None and not os.environ.get(key_env, "").strip() else core.load_key(key_env, args.api_key_file)
        client = client_factory(settings, key)
        with (output / "answers-judgments.jsonl").open("x", encoding="utf-8") as stream:
            os.chmod(stream.name, 0o600)
            def sink(row):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
            results, ledger = execute_jobs(jobs, settings, encoding, client, sink)
        report["metrics"] = core.aggregate_results(results, len(queries))
        if previous is not None:
            original = [previous[q["query_id"], side] for q in queries for side in ("baseline", "candidate")]
            report["historical_metrics_same_selected_questions"] = core.aggregate_results(original, len(queries))
            report["answers_reused_without_generation"] = sum("historical_answer_sha256" in row for row in results)
        report["usage"] = {"attempted_requests": len(ledger.calls),
            "usage_reported_requests": sum(r["usage_status"] == "reported" for r in ledger.calls),
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in ledger.calls),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in ledger.calls),
            "stopped_after_estimate_excess": ledger.stopped}
        core.write_private_json(output / "ledger.json", ledger.calls)
        report["private_raw_records_sha256"] = core.file_hash(output / "answers-judgments.jsonl")
    report["elapsed_seconds"] = time.monotonic() - start
    report["all_inputs_and_tools_unchanged"] = all(core.file_hash(Path(p)) == h for p, h in {**input_hashes, **tools}.items())
    if args.local_loopback and "files_sha256" in getattr(encoding, "metadata", {}):
        report["tokenizer_files_unchanged"] = core.local_tokenizer_files(args.tokenizer_path)[1] == encoding.metadata["files_sha256"]
    core.write_private_json(output / "summary.json", report)
    if not report["all_inputs_and_tools_unchanged"] or not report.get("tokenizer_files_unchanged", True):
        raise core.EvaluationError("Frozen calibration inputs or source changed during execution")
    return report


def parser():
    result = core.parser()
    result.description = __doc__
    result.add_argument("--mode", choices=("answer", "rejudge"), default="answer")
    result.add_argument("--historical-output", type=Path)
    return result


if __name__ == "__main__":
    try:
        report = run(parser().parse_args())
        print(json.dumps({k: report[k] for k in ("status", "mode", "matched_question_denominator", "cost_plan", "metrics", "all_inputs_and_tools_unchanged")}, indent=2))
    except (core.EvaluationError, ValueError) as error:
        print("Calibration failed: " + str(error))
        raise SystemExit(1)
