#!/usr/bin/env python3
"""Fixed two-complete-group PersonaMem-v1 dev Answer replay, default offline.

  python competition/eval_persona_pair.py plan
  python competition/eval_persona_pair.py replay-answer --baseline-run ... \
    --candidate-run ... --tokenizer-path ... --output-dir .local/... --execute

Only replays frozen retrieval.jsonl files: no Add, Search, database, embeddings,
downloads, remote purchasing, Judge or formal evaluation. Qwen on a literal
loopback H20 port is a local proxy, never official gpt-4o-mini accuracy. Plan
does not open gold/key files or create artifacts. Complete groups are selected
from frozen history order before looking at questions, retrievals or answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import eval_answer as qa
import eval_persona as persona
from eval_retrieval import source_segments, validate_search

HERE = Path(__file__).resolve().parent
PRIVATE_ROOT = HERE.parent / ".local"
VERSION = "personamem-v1-two-complete-dev-groups-local-qwen-replay-v1"
MODEL = "aml-qwen3-32b"
SHA = re.compile(r"[0-9a-f]{64}")
SOURCE_KEYS = {"role", "session_id", "request_id", "message_index", "message_id",
               "received_sequence", "timestamp_ms", "timestamp_utc", "native_message_index"}
Error = qa.EvaluationError


def checked_file(path):
    path = Path(path).absolute()
    qa.dev_input_file(path)  # Reject forbidden partitions and symlink containers before opening.
    if not path.is_file():
        raise Error("Require a regular frozen public-dev input")
    return path


def frozen_json(path, expected=None, lines=False):
    path = checked_file(path)
    before = persona.file_hash(path)
    if expected is not None and (not isinstance(expected, str) or not SHA.fullmatch(expected) or before != expected):
        raise Error("Frozen input SHA differs")
    value = qa.read_jsonl(path) if lines else qa.read_json(path)
    if persona.file_hash(path) != before:
        raise Error("Frozen input changed while reading")
    return value, before


def unique(rows, key):
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get(key), str)
                                       or not r[key] for r in rows):
        raise Error("Invalid prepared identities")
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise Error("Duplicate prepared identities")
    return result


def verify_prefix(history, lineage):
    cutoff = lineage.get("exclusive_end_index")
    if type(cutoff) is not int or cutoff < 1 or lineage.get("history_record_sha256") != persona.digest(history):
        raise Error("Historical prefix identity differs")
    sources = lineage.get("source_messages")
    if not isinstance(sources, list) or len(sources) != cutoff:
        raise Error("Historical prefix cutoff differs")
    messages, positions, sessions = [], {}, set()
    for session in history.get("sessions", []):
        sid = session.get("session_id")
        if not isinstance(sid, str) or not sid or sid in sessions:
            raise Error("Invalid historical session")
        sessions.add(sid)
        for i, message in enumerate(session.get("messages", [])):
            if message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str) or not message["content"]:
                raise Error("Invalid historical message")
            positions[(sid, i)] = message
            messages.append(message)
    if len(messages) != cutoff:
        raise Error("Historical messages exceed or omit the cutoff")
    seen = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or source.get("native_message_index") != index:
            raise Error("Historical native indices are not an exclusive prefix")
        position = (source.get("session_id"), source.get("message_position"))
        if position not in positions or position in seen:
            raise Error("Historical source position differs")
        seen.add(position)
        message = positions[position]
        content, role = message["content"], source.get("source_role")
        if role == "system":
            if message["role"] != "user" or not content.startswith("[historical system note]\n"):
                raise Error("Historical system note is not quoted data")
            content = content[len("[historical system note]\n"):]
        elif role != message["role"]:
            raise Error("Historical role differs")
        if qa.digest_bytes(content.encode("utf-8")) != source.get("source_content_sha256"):
            raise Error("Historical source text differs")
    return messages


def selection(data_root):
    root = Path(data_root).absolute()
    summary_path = root / "prepared" / "summary.json"
    summary, summary_sha = frozen_json(summary_path)
    if (summary.get("version") != persona.VERSION or summary.get("upstream") != persona.UPSTREAM
            or summary.get("revision") != persona.REVISION or summary.get("dataset_license") != "mit"
            or summary.get("status") != "prepared_public_native_proxy_not_official_aml"
            or summary.get("source_mapping_version") != "persona-native-exclusive-prefix-v1"):
        raise Error("Require the pinned public PersonaMem-v1 preparation")
    spec = summary["datasets"][persona.DATASET]["splits"]["dev"]
    folder = root / "prepared" / persona.DATASET / "dev"
    inputs, hashes = {}, {str(summary_path): summary_sha}
    for name in ("histories", "lineage", "queries"):
        path = folder / (name + ".jsonl")
        inputs[name], hashes[str(path)] = frozen_json(path, spec["files"][name]["sha256"], True)
    histories = unique(inputs["histories"], "sample_id")
    lineage = unique(inputs["lineage"], "sample_id")
    queries = unique(inputs["queries"], "query_id")
    if set(histories) != set(lineage) or len(histories) != spec["histories"] or len(queries) != spec["questions"]:
        raise Error("Prepared denominator differs")
    groups = []
    for sid in histories:  # Frozen file order; no question or label influences this choice.
        group = lineage[sid].get("group_id")
        if not isinstance(group, str) or not group:
            raise Error("A complete history group identity is required")
        if group not in groups:
            groups.append(group)
    if len(groups) < 2:
        raise Error("Two complete dev groups are required")
    chosen_groups = groups[:2]
    chosen = {sid: h for sid, h in histories.items() if lineage[sid]["group_id"] in chosen_groups}
    messages = {sid: verify_prefix(h, lineage[sid]) for sid, h in chosen.items()}
    selected_queries = []
    for row in queries.values():
        if row.get("sample_id") not in histories:
            raise Error("Question has no prepared historical scope")
        if row["sample_id"] in chosen:
            public = {key: row.get(key) for key in ("sample_id", "query_id", "query", "options", "all_options")}
            persona.build_answer_messages(public, "")  # Whitelist and validate all ordinary options.
            selected_queries.append(public)
    if not selected_queries or {q["sample_id"] for q in selected_queries} != set(chosen):
        raise Error("Selected complete histories have missing questions")
    return {"folder": folder, "spec": spec, "hashes": hashes, "histories": chosen,
            "lineage": {sid: lineage[sid] for sid in chosen}, "messages": messages,
            "queries": selected_queries, "all_query_ids": set(queries), "groups": chosen_groups,
            "selection_sha256": persona.digest([chosen_groups, list(chosen), [q["query_id"] for q in selected_queries]])}


def frozen_run(folder, selected):
    folder = Path(folder).absolute()
    summary, summary_sha = frozen_json(folder / "summary.json")
    config = summary.get("configuration", {})
    source = config.get("source_code_sha256", {})
    if (summary.get("dataset") != persona.DATASET or summary.get("split") != "dev"
            or summary.get("server_source_changed_during_run") or config.get("top_k") != 100
            or config.get("character_budget") != 0 or config.get("prepared_data") != selected["spec"]):
        raise Error("Require a frozen matching untrimmed Persona dev retrieval run")
    if (not isinstance(source, dict) or not source or any(not isinstance(v, str) or not SHA.fullmatch(v) for v in source.values())
            or source.get("server.py_at_start") != source.get("server.py_at_end")
            or "server.py_at_start" not in source):
        raise Error("Stable retrieval server source fingerprints are required")
    rows, retrieval_sha = frozen_json(folder / "retrieval.jsonl", lines=True)
    records = unique(rows, "query_id")
    if type(summary.get("question_count")) is not int or len(records) != summary["question_count"]:
        raise Error("Missing retrieval rows cannot disappear from the denominator")
    selected_ids = {q["query_id"] for q in selected["queries"]}
    if not selected_ids <= set(records) <= selected["all_query_ids"]:
        raise Error("Retrieval must include all questions in the fixed complete groups")
    for q in selected["queries"]:
        row = records[q["query_id"]]
        if row.get("sample_id") != q["sample_id"] or not isinstance(row.get("status"), str):
            raise Error("Retrieval historical scope or status differs")
    label = summary.get("model_label")
    if not isinstance(label, str) or not label.strip():
        raise Error("A declared retrieval arm label is required")
    return {"records": records, "label": label, "run_id": summary.get("run_id"),
            "source_code_sha256_declared": source,
            "hashes": {str(folder / "summary.json"): summary_sha, str(folder / "retrieval.jsonl"): retrieval_sha}}


def verified_items(record, history, lineage):
    items = validate_search({"data": record.get("retrieved")}, 100)
    originals = {m["content"] for s in history["sessions"] for m in s["messages"]}
    sessions = {s["session_id"]: s["messages"] for s in history["sessions"]}
    native = {(s["session_id"], s["message_position"]): s["native_message_index"] for s in lineage["source_messages"]}
    result = []
    for item in items:
        bodies = []
        for source, text in source_segments(item["content"], {}, original_texts=originals):
            if text not in originals:
                raise Error("Retrieved text is outside the native historical cutoff")
            if source is not None:
                if not isinstance(source, dict) or set(source) - SOURCE_KEYS or source.get("session_id") not in sessions:
                    raise Error("Invalid retrieved source metadata")
                matches = [i for i, m in enumerate(sessions[source["session_id"]])
                           if m["content"] == text and m["role"] == source.get("role")]
                if not matches:
                    raise Error("Retrieved source role or session is outside the cutoff")
                if "native_message_index" in source and (type(source["native_message_index"]) is not int
                        or source["native_message_index"] not in {native[(source["session_id"], i)] for i in matches}):
                    raise Error("Retrieved source native index is outside the cutoff")
            bodies.append(text)
        if not bodies:
            raise Error("Retrieved item has no verified historical text")
        # Verification alone splits bodies. Answer retains the returned role,
        # timestamps and source wrappers byte-for-byte inside whole items.
        result.append({"id": item["id"], "content": item["content"]})
    return result


class ByteEstimate:
    name = "utf8-byte-planning-estimate-not-a-model-tokenizer"
    def encode(self, text, disallowed_special=()):
        return text.encode("utf-8")


def pack(query, items, encoding, args):
    strings = ["Memory " + str(i + 1) + ":\n" + item["content"] + "\n\n" for i, item in enumerate(items)]
    if qa.message_tokens(encoding, persona.build_answer_messages(query, ""), args.chat_overhead) > args.input_tokens:
        return None, None
    count = 0
    for n in range(1, len(strings) + 1):
        body = "".join(strings[:n])
        if (qa.tokens(encoding, body) > args.memory_tokens or
                qa.message_tokens(encoding, persona.build_answer_messages(query, body), args.chat_overhead) > args.input_tokens):
            break
        count = n
    body = "".join(strings[:count])
    messages = persona.build_answer_messages(query, body)
    return messages, {"available_items": len(items), "selected_items": count,
        "memory_tokens_estimated": qa.tokens(encoding, body),
        "input_tokens_estimated": qa.message_tokens(encoding, messages, args.chat_overhead),
        "messages_sha256": persona.digest(messages)}


def validate_args(args):
    qa.validate_endpoint(args.endpoint, True)
    parsed = urlsplit(args.endpoint)
    if parsed.path != "/v1/chat/completions" or parsed.port is None or not 1024 <= parsed.port <= 65535:
        raise Error("Require an explicit local H20 chat-completions port")
    for key in ("memory_tokens", "input_tokens", "context_tokens", "output_tokens", "max_requests", "chat_overhead"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise Error("Require positive token and request bounds")
    if (args.chat_overhead < 64 or args.context_tokens > 65536 or args.input_tokens + args.output_tokens > args.context_tokens
            or not math.isfinite(args.timeout) or not 1 <= args.timeout <= 600
            or not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 86400):
        raise Error("Invalid local H20 context, timeout or duration bounds")
    if bool(args.baseline_run) != bool(args.candidate_run):
        raise Error("Supply both retrieval arms together")
    if args.execute and (args.command != "replay-answer" or not args.baseline_run or not args.output_dir):
        raise Error("Execution requires explicit replay-answer, both arms and a fresh private output")


def source_hashes():
    return {p.name: persona.file_hash(p) for p in
            (Path(__file__), HERE / "eval_persona.py", HERE / "eval_answer.py", HERE / "eval_retrieval.py")}


def unchanged(hashes):
    return all(persona.file_hash(checked_file(Path(p))) == h for p, h in hashes.items())


def private_output(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_relative_to(PRIVATE_ROOT.resolve()) or path == PRIVATE_ROOT.resolve():
        raise Error("Require a new non-symlink output beneath .local")
    missing, parent = [], path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    if not parent.is_dir():
        raise Error("Invalid private output parent")
    if not missing:
        raise Error("Never overwrite or resume an existing output")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    return path


def private_stream(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8")


def write_private(path, value):
    with private_stream(path) as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")


def scoring_references(selected):
    rows, sha = frozen_json(selected["folder"] / "gold.jsonl", selected["spec"]["files"]["gold"]["sha256"], True)
    labels = unique(rows, "query_id")
    if set(labels) != selected["all_query_ids"]:
        raise Error("Scoring reference denominator differs")
    gold = {}
    for query in selected["queries"]:
        row = labels[query["query_id"]]
        if row.get("sample_id") != query["sample_id"]:
            raise Error("Scoring historical scope differs")
        choice = persona.parse_gold_choice(row.get("answer"))
        if row.get("gold_option") != choice:
            raise Error("Scoring option differs")
        gold[query["query_id"]] = row["answer"]
    return gold, sha


def run(args, encoding=None, client_factory=qa.ChatClient):
    validate_args(args)
    sources = source_hashes()
    selected = selection(args.data_root)
    if encoding is None:
        if args.execute and args.tokenizer_path is None:
            raise Error("Explicit local execution requires an existing offline tokenizer")
        encoding = qa.offline_hf_encoding(args.tokenizer_path, {"enable_thinking": False}) if args.tokenizer_path else ByteEstimate()
    arms = [frozen_run(p, selected) for p in (args.baseline_run, args.candidate_run)] if args.baseline_run else []
    if arms and set(arms[0]["records"]) != set(arms[1]["records"]):
        raise Error("Both retrieval arms must contain identical dev question sets")
    jobs = []
    for query in selected["queries"]:
        for side, arm in zip(("baseline", "candidate"), arms):
            record = arm["records"][query["query_id"]]
            job = {"query_id": query["query_id"], "side": side, "status": "retrieval_error"}
            if record["status"] == "ok":
                items = verified_items(record, selected["histories"][query["sample_id"]], selected["lineage"][query["sample_id"]])
                messages, packing = pack(query, items, encoding, args)
                job.update(status="planned" if messages is not None else "input_error", messages=messages, packing=packing)
            jobs.append(job)
    planned = sum(j["status"] == "planned" for j in jobs)
    report = {"version": VERSION, "status": "public_dev_plan_no_model_calls", "dataset": persona.DATASET, "split": "dev",
        "proxy": "Local Qwen3-32B Answer replay; not official gpt-4o-mini accuracy or AML Overall",
        "selection": {"rule": "first two distinct complete group_id values in frozen dev histories order; all their questions",
            "group_ids": selected["groups"], "groups": 2, "histories": len(selected["histories"]),
            "questions_per_arm": len(selected["queries"]), "messages": sum(len(v) for v in selected["messages"].values()),
            "history_utf8_bytes": sum(len(m["content"].encode("utf-8")) for v in selected["messages"].values() for m in v),
            "question_options_utf8_bytes": sum(len((q["query"] + q["all_options"]).encode("utf-8")) for q in selected["queries"]),
            "selection_sha256": selected["selection_sha256"]},
        "budget": {"planned_answer_requests": planned, "maximum_paired_requests": 2 * len(selected["queries"]),
            "maximum_paired_input_tokens_by_declared_cap": 2 * len(selected["queries"]) * args.input_tokens,
            "request_limit": args.max_requests, "request_limit_sufficient": planned <= args.max_requests,
            "planned_input_tokens_estimated": sum(j["packing"]["input_tokens_estimated"] for j in jobs if j["status"] == "planned"),
            "maximum_output_tokens": planned * args.output_tokens, "memory_token_cap": args.memory_tokens,
            "input_token_cap": args.input_tokens, "output_token_cap": args.output_tokens, "context_token_cap": args.context_tokens,
            "max_seconds": args.max_seconds, "api_fee_usd": 0, "compute_cost_accounted": False},
        "tokenizer": getattr(encoding, "metadata", {"kind": encoding.name}),
        "model_requested": MODEL, "endpoint": args.endpoint, "temperature": 0, "enable_thinking": False,
        "arms": [{k: arm[k] for k in ("label", "run_id", "source_code_sha256_declared", "hashes")} for arm in arms],
        "input_sha256": selected["hashes"], "source_code_sha256": sources,
        "planned_status_counts": dict(Counter(j["status"] for j in jobs)), "metrics": None, "attempted_requests": 0,
        "limitations": ["Public dev, independently worded prompt and strict parser; not the official AML pipeline.",
            "Frozen retrievals are reused; this tool does not establish that v4 ingestion or Search was executed.",
            "Whole-item prefix packing retains returned role/time/source wrappers; official tokenizer/packing may differ.",
            "Two complete groups are correlated and small; no claim of representative Overall improvement.",
            "Byte-only dry planning is an estimate; execution requires an existing offline local tokenizer."]}
    if not args.execute:
        return report  # No gold/key/client/output access exists above this branch.
    if not report["budget"]["request_limit_sufficient"]:
        raise Error("The request ceiling cannot cover the fixed complete groups")
    hashes = dict(selected["hashes"])
    for arm in arms:
        hashes.update(arm["hashes"])
    gold, gold_sha = scoring_references(selected)  # Selection and prompts are already frozen.
    hashes[str(selected["folder"] / "gold.jsonl")] = gold_sha
    if not unchanged(hashes) or source_hashes() != sources:
        raise Error("Inputs changed before execution")
    output = private_output(args.output_dir)
    write_private(output / "plan.json", report)
    key = qa.load_key("AML_LOCAL_LLM_API_KEY", args.api_key_file) if args.api_key_file else None
    settings = qa.Settings(answer_model=MODEL, judge_model=MODEL, endpoint=args.endpoint,
        memory_tokens=args.memory_tokens, input_tokens=args.input_tokens, context_tokens=args.context_tokens,
        answer_output_tokens=args.output_tokens, judge_output_tokens=args.output_tokens, chat_overhead=args.chat_overhead,
        input_usd_per_million=0, output_usd_per_million=0, max_requests=args.max_requests,
        local_loopback=True, max_cost_usd=1, timeout=args.timeout, chat_template_kwargs={"enable_thinking": False})
    settings.validate()
    client = client_factory(settings, key)
    results, stopped, usage_rows = [], False, []
    begin = time.perf_counter()
    with private_stream(output / "answers.jsonl") as sink:
        for job in jobs:
            row = {"query_id": job["query_id"], "side": job["side"], "status": job["status"],
                   "correct": False, "packing": job.get("packing"), "elapsed_seconds": 0}
            if job["status"] == "planned":
                remaining = args.max_seconds - (time.perf_counter() - begin)
                if stopped or remaining <= 0 or report["attempted_requests"] >= args.max_requests:
                    row["status"] = "run_stopped"
                    stopped = True
                else:
                    report["attempted_requests"] += 1  # Reserve an attempt before transport; never retry.
                    attempt_begin = time.perf_counter()
                    try:
                        if isinstance(client, qa.ChatClient):
                            client.settings = replace(settings, timeout=min(settings.timeout, remaining))
                        response = client.complete(MODEL, job["messages"], args.output_tokens)
                        usage = response.get("usage") if isinstance(response, dict) else None
                        valid_usage = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                            for k in ("prompt_tokens", "completion_tokens"))
                        usage_rows.append({"status": "reported" if valid_usage else "unknown",
                            **({k: usage[k] for k in ("prompt_tokens", "completion_tokens")} if valid_usage else {})})
                        if not valid_usage or usage["prompt_tokens"] > job["packing"]["input_tokens_estimated"] or usage["completion_tokens"] > args.output_tokens:
                            stopped = True
                        if response.get("model") != MODEL:
                            raise qa.ProviderError("response_model_mismatch")
                        answer = qa.completion_text(response)
                        score = persona.score_choice(answer, gold[job["query_id"]])
                        row.update(status=score["status"], correct=score["correct"], selected_option=score["selected_option"], answer=answer)
                    except Exception as error:
                        row.update(status="answer_error", error_code=error.code if isinstance(error, qa.ProviderError) else "local_client_error")
                        stopped = True  # Includes 429 and unknown transport outcomes; no raw error text.
                        if len(usage_rows) < report["attempted_requests"]:
                            usage_rows.append({"status": "unknown"})
                    row["elapsed_seconds"] = time.perf_counter() - attempt_begin
            results.append(row)
            sink.write(persona.canonical_json(row) + "\n")
            sink.flush()
    denominator = len(selected["queries"])
    metrics = {}
    for side in ("baseline", "candidate"):
        rows = [r for r in results if r["side"] == side]
        metrics[side] = {"denominator": denominator, "result_rows": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / denominator,
            "parse_failures": sum(r["status"] == "invalid_answer" for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)), "elapsed_seconds": sum(r["elapsed_seconds"] for r in rows)}
    left = {r["query_id"]: r["correct"] for r in results if r["side"] == "baseline"}
    right = {r["query_id"]: r["correct"] for r in results if r["side"] == "candidate"}
    changed = not unchanged(hashes) or source_hashes() != sources
    report.update(status="invalid_inputs_changed" if changed else "completed_local_qwen_proxy",
        metrics=metrics, paired={"gained": sum(right[q] and not left[q] for q in left),
            "lost": sum(left[q] and not right[q] for q in left)}, elapsed_seconds=time.perf_counter() - begin,
        stopped=stopped, inputs_changed_during_run=changed, scoring_reference_sha256=gold_sha,
        usage={"reported_requests": sum(r["status"] == "reported" for r in usage_rows),
            "unknown_requests": sum(r["status"] == "unknown" for r in usage_rows),
            "reported_input_tokens": sum(r.get("prompt_tokens", 0) for r in usage_rows),
            "reported_output_tokens": sum(r.get("completion_tokens", 0) for r in usage_rows)},
        raw_answers_sha256=persona.file_hash(output / "answers.jsonl"))
    write_private(output / "summary.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", nargs="?", choices=("plan", "replay-answer"), default="plan")
    p.add_argument("--data-root", type=Path, default=persona.DEFAULT_OUTPUT)
    p.add_argument("--baseline-run", type=Path)
    p.add_argument("--candidate-run", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--endpoint", default="http://127.0.0.1:18080/v1/chat/completions")
    p.add_argument("--tokenizer-path", type=Path, help="Existing offline tokenizer; no download or model loading")
    p.add_argument("--api-key-file", type=Path, help="Optional explicit local auth file, read only with --execute")
    for name, value in (("memory-tokens", 32000), ("input-tokens", 64000), ("context-tokens", 65536),
                        ("output-tokens", 128), ("chat-overhead", 64), ("max-requests", 1000)):
        p.add_argument("--" + name, type=int, default=value)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-seconds", type=float, default=3600)
    return p


def main(argv=None):
    try:
        report = run(parser().parse_args(argv))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] != "invalid_inputs_changed" else 2
    except (Error, OSError, KeyError, TypeError, ValueError):
        print(json.dumps({"status": "rejected", "reason": "Invalid public dev provenance, local endpoint, complete groups, replay inputs or private output"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
