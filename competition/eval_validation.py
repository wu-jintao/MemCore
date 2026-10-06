#!/usr/bin/env python3
"""Freeze public-data identity splits and run synthetic memory governance checks.

No gold, model API or official evaluation is used. Publicly analyzed data is never
claimed to be blind. Existing output files cannot be overwritten by this CLI.
"""
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


VERSION = "memory-validation-governance-v1"
SPLIT_SEED = "independent-memory-validation-v1"
ROOT = Path(__file__).resolve().parent.parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:  # Unicode separators inside content are not file lines.
            if line.strip():
                yield json.loads(line)


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Identity must be a nonempty exact string")
    return value


def identity_values(record, singular, plural):
    values = record.get(plural, [])
    if not isinstance(values, list):
        raise ValueError("Identity collections require exact string lists")
    return [text(value) for value in ([record[singular]] if singular in record else []) + values]


def history_tokens(dataset, history, additional=None):
    """No answers/labels: exact user identities and conservative source overlap."""
    sid = text(history["sample_id"])
    tokens = {"sample:" + dataset + ":" + sid}
    for record in (history, additional or {}):
        users = identity_values(record, "user_id", "user_ids")
        conversations = identity_values(record, "conversation_id", "conversation_ids")
        tokens.update("user:" + text(identity) for identity in users)
        tokens.update("conversation:" + text(identity) for identity in conversations)
    sessions = history.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("History requires nonempty source sessions")
    for session in sessions:
        messages = session.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Session requires nonempty original messages")
        original = []
        for message in messages:
            if message.get("role") not in ("user", "assistant"):
                raise ValueError("Unsupported source role")
            text(message.get("content"))
            if "timestamp" in message and type(message["timestamp"]) is not int:
                raise ValueError("Source timestamp must be an integer")
            original.append({key: message[key] for key in ("role", "content", "timestamp") if key in message})
        # Prepared session IDs are sample-scoped hashes, so shared transcript
        # identity must ignore those IDs. Identical generic sessions over-group
        # conservatively; this never upgrades overlapping data into holdout.
        tokens.add("session-source:" + digest(original))
    return tokens


def build_validation_manifest(histories, queries, used=(), identities=(), seed=SPLIT_SEED,
                              validation_fraction=0.2, holdout_fraction=0.2, preserve_prepared_splits=False):
    if not isinstance(seed, str) or not seed:
        raise ValueError("A frozen split seed is required")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
           or x < 0 for x in (validation_fraction, holdout_fraction)) or validation_fraction + holdout_fraction > 1:
        raise ValueError("Invalid split fractions")
    identity_map = {}
    for row in identities:
        key = (text(row["dataset"]), text(row["sample_id"]))
        if key in identity_map:
            raise ValueError("Duplicate external identity mapping")
        identity_map[key] = row
    records, tokens, parents, original_splits = {}, {}, {}, defaultdict(set)
    for row in histories:
        key = (text(row["dataset"]), text(row["sample_id"]))
        original = row.get("original_split")
        if preserve_prepared_splits and original not in ("dev", "validation", "heldout"):
            raise ValueError("Preserving prepared splits requires declared original partitions")
        if original is not None:
            original_splits[key].add(original)
        current = history_tokens(key[0], row, identity_map.get(key))
        identity_record = {name: value for name, value in row.items() if name != "original_split"}
        if key in records and digest(records[key]) != digest(identity_record):
            raise ValueError("Conflicting duplicate history")
        records[key], tokens[key], parents[key] = identity_record, current, key
    if not records or set(identity_map) - set(records):
        raise ValueError("Empty histories or unmatched external identity mapping")

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owners = {}
    for key in sorted(records):
        for token in sorted(tokens[key]):
            if token in owners:
                left, right = find(key), find(owners[token])
                parents[max(left, right)] = min(left, right)
            else:
                owners[token] = key
    question_map = {}
    by_sample = defaultdict(list)
    for row in queries:
        if any(key in row for key in ("answer", "gold", "gold_units", "category")):
            raise ValueError("Labels must not enter the split selector")
        key = (text(row["dataset"]), text(row["sample_id"]))
        qkey = (key[0], text(row["query_id"]))
        if key not in records or (qkey in question_map and question_map[qkey] != key):
            raise ValueError("Unresolved or conflicting query/history scope")
        if qkey not in question_map:
            by_sample[key].append(qkey)
        question_map[qkey] = key
    if not question_map:
        raise ValueError("No validation questions")
    contaminated = set()
    used_entries = list(used)
    unmatched_samples, unmatched_queries = set(), set()
    for entry in used_entries:
        dataset = text(entry["dataset"])
        for field in ("sample_ids", "query_ids", "user_ids", "conversation_ids", "session_source_sha256"):
            if not isinstance(entry.get(field, []), list):
                raise ValueError("Usage identities require exact string lists")
            for identity in entry.get(field, []):
                text(identity)
        unmatched_samples.update((dataset, value) for value in entry.get("sample_ids", []) if (dataset, value) not in records)
        unmatched_queries.update((dataset, value) for value in entry.get("query_ids", []) if (dataset, value) not in question_map)
        contaminated.update((dataset, value) for value in entry.get("sample_ids", []) if (dataset, value) in records)
        for qid in entry.get("query_ids", []):
            if (dataset, qid) in question_map:
                contaminated.add(question_map[(dataset, qid)])
        used_tokens = {"user:" + text(value) for value in entry.get("user_ids", [])}
        used_tokens.update("conversation:" + text(value) for value in entry.get("conversation_ids", []))
        used_tokens.update("session-source:" + text(value) for value in entry.get("session_source_sha256", []))
        contaminated.update(key for key in records if tokens[key] & used_tokens)
    groups = defaultdict(list)
    for key in records:
        groups[find(key)].append(key)
    result = []
    for members in groups.values():
        members.sort()
        union = set().union(*(tokens[key] for key in members))
        stable_ids = sorted(token for token in union if token.startswith("user:"))
        if not stable_ids:
            stable_ids = sorted(token for token in union if token.startswith("conversation:"))
        group_id = digest(stable_ids or sorted(union))
        used_overlap = any(key in contaminated for key in members)
        origins = sorted(set().union(*(original_splits[key] for key in members)))
        if used_overlap:
            partition, reason = "development", "previously_used_identity_or_shared_source"
        elif preserve_prepared_splits:
            if len(origins) != 1:
                partition, reason = "development", "cross_prepared_split_identity_or_shared_source"
            else:
                partition, reason = {"dev": "development", "validation": "validation", "heldout": "holdout"}[origins[0]], "preserved_prepared_split"
        else:
            draw = int(digest([VERSION, seed, group_id]), 16) / (2 ** 256)
            partition = ("validation" if draw < validation_fraction else "holdout" if draw < validation_fraction + holdout_fraction else "development")
            reason = "frozen_identity_hash"
        result.append({"group_id": group_id, "partition": partition,
                       "reason": reason, "original_prepared_splits": origins,
                       "members": [{"dataset": d, "sample_id": s} for d, s in members],
                       "queries": [{"dataset": d, "query_id": q} for key in members for d, q in sorted(by_sample[key])],
                       "identity_tokens_sha256": digest(sorted(union))})
    result.sort(key=lambda group: group["group_id"])
    counts = {partition: sum(len(group["queries"]) for group in result if group["partition"] == partition)
              for partition in ("development", "validation", "holdout")}
    return {"schema": VERSION, "seed": None if preserve_prepared_splits else seed,
            "allocation": "preserved_prepared_splits" if preserve_prepared_splits else "stable_identity_hash",
            "fractions": None if preserve_prepared_splits else {"validation": validation_fraction, "holdout": holdout_fraction},
            "groups": result, "question_counts": counts, "independent_groups": len(result),
            "forced_development_groups": sum(group["partition"] == "development" and group["reason"] in
                                             ("previously_used_identity_or_shared_source", "cross_prepared_split_identity_or_shared_source") for group in result),
            "selection_reads_labels": False, "public_not_blind": True,
            "scope": "public_identity_overlap_audit_not_blind_evaluation",
            "unmatched_usage_identity_counts": {"samples": len(unmatched_samples), "queries": len(unmatched_queries)},
            "real_user_identity_available_for_all_histories": all(any(token.startswith("user:") for token in tokens[key]) for key in records),
            "policy": ["All questions from the same exact user/conversation/shared transcript component stay together.",
                       "Any previously analyzed/evaluated identity forces its entire overlap component into development.",
                       "Freeze input hashes, identities, seed and grouping before evaluation; do not seed-hop or pick by results.",
                       "Prepared sample IDs are constructed API scopes, not proof of independent real people.",
                       "Public data, including partition metadata inspected here, is not an untouched blind benchmark."] +
                       (["Preserve existing prepared assignments; quarantine components crossing original partitions into development."] if preserve_prepared_splits else []),
            "usage_ledger_sha256": digest(sorted(used_entries, key=digest))}


def load_prepared(folders):
    histories, queries, sources = [], [], []
    for folder in sorted(set(Path(value).resolve() for value in folders)):
        dataset = text(folder.parent.name)
        for name, target in (("histories", histories), ("queries", queries)):
            path = folder / (name + ".jsonl")
            before = file_hash(path)
            for row in jsonl(path):
                target.append({**row, "dataset": dataset, "original_split": folder.name})
            if file_hash(path) != before:
                raise ValueError("Prepared source changed while reading")
            sources.append({"path": str(path), "sha256": before})
    return histories, queries, sources


def native_identity_map(histories, raw_directory, manifest_path=ROOT / "competition" / "public_data_manifest.json"):
    """Select pinned native identity fields only; JSON decoding includes labels.

    Session reuse stays linked even if preparation attached different dates.
    Native source sessions/conversations are not independently verified people.
    """
    manifest_checksum = file_hash(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text())
    if file_hash(manifest_path) != manifest_checksum:
        raise ValueError("Identity pin manifest changed while reading")
    wanted = defaultdict(set)
    for row in histories:
        if row["dataset"] in ("longmemeval_s", "locomo_refined"):
            wanted[row["dataset"]].add(row["sample_id"])
    identities, sources = [], []
    for dataset in manifest["datasets"]:
        name = dataset["id"]
        if name not in wanted:
            continue
        specification = dataset["files"][0]
        path = Path(raw_directory) / specification["name"]
        before = file_hash(path)
        if before != specification["sha256"] or path.stat().st_size != specification["size"]:
            raise ValueError("Pinned native identity source mismatch")
        rows = json.loads(path.read_text(encoding="utf-8"))
        mapped = set()
        for row in rows:
            native = str(row["question_id"] if name == "longmemeval_s" else row["sample_id"])
            sample = hashlib.sha256((name + "\0" + native).encode()).hexdigest()[:32]
            if sample not in wanted[name]:
                continue
            if sample in mapped:
                raise ValueError("Duplicate pinned native sample identity")
            mapped.add(sample)
            if name == "longmemeval_s":
                ids = row["haystack_session_ids"]
                if not isinstance(ids, list) or not ids:
                    raise ValueError("Missing native source-session identities")
                conversations = sorted({name + "/native-session/" + text(str(identity)) for identity in ids})
            else:
                conversations = [name + "/native-conversation/" + text(native)]
            identities.append({"dataset": name, "sample_id": sample, "conversation_ids": conversations})
        if mapped != wanted[name] or file_hash(path) != before:
            raise ValueError("Unmapped prepared sample or changing native identity source")
        sources.append({"path": str(path.resolve()), "sha256": before, "pin_verified": True,
                        "pin_manifest_path": str(Path(manifest_path).resolve()), "pin_manifest_sha256": manifest_checksum,
                        "fields_selected": ["question_id", "haystack_session_ids"] if name == "longmemeval_s" else ["sample_id"],
                        "answer_or_evidence_labels_used": False, "real_user_identity": False})
    if set(wanted) != {row["dataset"] for row in identities}:
        raise ValueError("Missing recognized dataset identity pins")
    return sorted(identities, key=lambda row: (row["dataset"], row["sample_id"])), sources


def merge_identity_maps(*maps):
    merged = {}
    for rows in maps:
        for row in rows:
            key = (text(row["dataset"]), text(row["sample_id"]))
            target = merged.setdefault(key, {"dataset": key[0], "sample_id": key[1], "user_ids": set(), "conversation_ids": set()})
            for singular, plural in (("user_id", "user_ids"), ("conversation_id", "conversation_ids")):
                target[plural].update(identity_values(row, singular, plural))
    return [{**row, "user_ids": sorted(row["user_ids"]), "conversation_ids": sorted(row["conversation_ids"])}
            for _, row in sorted(merged.items())]


def load_identity_map(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return list(jsonl(path))
    if isinstance(value, dict) and "identities" in value:
        value = value["identities"]
    elif isinstance(value, dict) and "dataset" in value and "sample_id" in value:
        value = [value]
    if not isinstance(value, list):
        raise ValueError("Identity map requires a JSON identities array or JSONL rows")
    return value


def used_run(path):
    path = Path(path).resolve()
    before = {name: file_hash(path / name) for name in ("summary.json", "retrieval.jsonl")}
    summary = json.loads((path / "summary.json").read_text())
    if (summary.get("split") not in ("dev", "validation", "heldout")
            or not summary.get("dataset") or not str(summary.get("status", "")).startswith("public_")):
        raise ValueError("A declared public proxy run is required")
    samples, queries = set(), set()
    for row in jsonl(path / "retrieval.jsonl"):
        samples.add(text(row["sample_id"]))
        queries.add(text(row["query_id"]))
    if not samples or any(file_hash(path / name) != checksum for name, checksum in before.items()):
        raise ValueError("Empty or changing prior retrieval run")
    return {"dataset": summary["dataset"], "sample_ids": sorted(samples), "query_ids": sorted(queries),
            "usage": "previously_used", "run_directory": str(path),
            "source_summary_sha256": before["summary.json"], "source_retrieval_sha256": before["retrieval.jsonl"]}


def discover_used_runs(directory):
    """Completed public proxy dev retrievals; private/external uses need a ledger."""
    directory = Path(directory).resolve()
    found, skipped = [], defaultdict(int)
    for path in sorted(directory.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if not (path.parent / "retrieval.jsonl").is_file():
            skipped["no_retrieval_file"] += 1
        elif summary.get("split") != "dev":
            skipped["not_development"] += 1
        elif (not str(summary.get("status", "")).startswith("public_") or not summary.get("dataset")
              or type(summary.get("question_count")) is not int or summary["question_count"] <= 0
              or isinstance(summary.get("elapsed_seconds"), bool)
              or not isinstance(summary.get("elapsed_seconds"), (int, float))
              or not math.isfinite(summary["elapsed_seconds"]) or summary["elapsed_seconds"] < 0):
            skipped["not_completed_public_proxy"] += 1
        else:
            found.append(path.parent)
    return found, {"directory": str(directory), "depth": "one run directory", "discovered_runs": [str(path) for path in found],
                   "skipped_summary_counts": dict(sorted(skipped.items())),
                   "completeness": "Only completed public dev retrieval runs in this directory. Supply a ledger for private, unfinished, analyzed-only or workspace-external uses."}


def synthetic_fixtures():
    steps, sources, cases, requests = [], {}, [], {}
    scope = "synthetic-validation/person-A"

    def add(label, messages, session="timeline", user=scope):
        payload = {"user_id": user, "session_id": session, "request_id": "synthetic-" + label, "messages": messages}
        requests[label] = payload
        ids = []
        for index, message in enumerate(messages):
            identity = json.dumps([user, payload["request_id"], index], ensure_ascii=False, separators=(",", ":"))
            mid = "msg_" + hashlib.sha256(identity.encode()).hexdigest()
            sources[mid] = {**message, "user_id": user, "session_id": session,
                            "request_id": payload["request_id"], "message_index": index,
                            "message_id": mid, "received_sequence": len(sources) + 1}
            ids.append(mid)
        steps.append({"action": "add", "payload": payload, "expected_status": 200})
        return ids

    def search(name, query, required=(), reference=None, capability=None, forbidden=(), compare_to=None, answer_type="string"):
        step = {"action": "search", "name": name, "payload": {"user_id": scope, "query": query, "top_k": 100},
                "required_source_ids": list(required), "forbidden_source_ids": list(forbidden)}
        if compare_to:
            step["compare_to"] = compare_to
        steps.append(step)
        if capability:
            cases.append({"case_id": name, "capability": capability, "query": query,
                          "answer_type": answer_type, "reference_answer": reference,
                          "required_source_ids": list(required)})

    old = add("nickname-old", [{"role": "user", "content": "My project nickname is Maple.", "timestamp": 1767225600000}])
    search("streaming-before-update", "What project nickname have I given so far?", old, "Maple", "streaming")
    new = add("nickname-new", [{"role": "user", "content": "Correction: my project nickname is Juniper now. Maple is the old nickname.", "timestamp": 1767312000000}])
    query = "What is my project nickname now, after the correction?"
    search("fact-current", query, old + new, "Juniper", "fact_update")
    search("fact-previous", "What was my earlier project nickname?", old + new, "Maple", "fact_history")
    profile = add("profile", [{"role": "user", "content": "I prefer concise responses with metric units."},
                              {"role": "assistant", "content": "I prefer long responses with imperial units."}], "profile")
    search("user-preference", "What responses and units do I prefer?", profile, "concise and metric", "preference")
    search("assistant-role", "What responses and units did the assistant say it prefers?", profile, "long and imperial", "role")
    rule = add("rule", [{"role": "user", "content": "For itinerary suggestions, never propose overnight buses."}], "rules")
    search("user-rule", "What rule did I set about overnight buses for itinerary suggestions?", rule, "no overnight buses", "rule")
    negation = add("negation", [{"role": "user", "content": "I did not visit Lisbon. I visited Porto."}], "travel")
    search("negation", "Did I visit Lisbon or Porto?", negation, "Porto", "negation")
    late = add("event-late", [{"role": "user", "content": "The studio exhibit opening was on January 12, 2026.", "timestamp": 1768176000000}], "events")
    early = add("event-early", [{"role": "user", "content": "The studio exhibit setup was on January 10, 2026.", "timestamp": 1768003200000}], "events")
    search("event-order", "Which studio exhibit event happened first, setup or opening?", early + late, "setup", "time_order")
    kit = add("list", [{"role": "user", "content": "Supply kit list: notebook, ruler, pencil."}], "supplies")
    search("complete-list", "List all three items in my supply kit.", kit, ["notebook", "ruler", "pencil"], "list", answer_type="set")
    foreign = add("foreign", [{"role": "user", "content": "My foreign archive tag is CobaltArchive."}], "timeline", "synthetic-validation/person-B")
    search("cross-user-unknown", "What is my foreign archive tag?", (), "unknown", "isolation", foreign)
    search("retry-before", query)
    steps.append({"action": "add", "payload": copy.deepcopy(requests["nickname-new"]), "expected_status": 200})
    search("retry-after", query, compare_to="retry-before")
    conflict = copy.deepcopy(requests["nickname-new"])
    conflict["messages"][0]["content"] = "A conflicting replacement must never commit."
    steps.append({"action": "add", "payload": conflict, "expected_status": 409})
    search("conflict-after", query, compare_to="retry-after")
    steps.append({"action": "restart"})
    search("restart-after", query, compare_to="conflict-after")
    return {"schema": VERSION, "scope": "synthetic-only", "benchmark_specific_logic": False,
            "sources": sources, "steps": steps, "semantic_cases": cases,
            "semantic_metric": "Closed synthetic canonical string / unordered-set exact match, separate from retrieval coverage."}


def verify_grounding(response, sources, user_id, visible):
    seen, items = set(), response.get("data") if isinstance(response, dict) else None
    if not isinstance(items, list) or len(items) > 100:
        raise ValueError("Invalid Search response")
    anchors = set()
    for item in items:
        if (not isinstance(item, dict) or item.get("id") not in sources or item["id"] in anchors
                or not isinstance(item.get("content"), str) or not item["content"]):
            raise ValueError("Invalid or duplicate anchor")
        anchors.add(item["id"])
        content, first = item["content"], True
        while content:
            if not content.startswith("[source "):
                raise ValueError("Missing original source metadata")
            origin, end = json.JSONDecoder().raw_decode(content[len("[source "):])
            end += len("[source ")
            if content[end:end + 2] != "]\n" or not isinstance(origin, dict):
                raise ValueError("Invalid source boundary")
            mid = origin.get("message_id")
            if mid not in visible or mid not in sources or sources[mid]["user_id"] != user_id or mid in seen:
                raise ValueError("Unseen or cross-user source")
            expected = sources[mid]
            if first and item["id"] != mid:
                raise ValueError("Anchor/source mismatch")
            for key in ("role", "session_id", "request_id", "message_index", "message_id", "received_sequence"):
                if origin.get(key) != expected[key]:
                    raise ValueError("Altered source provenance")
            if "timestamp" in expected:
                if origin.get("timestamp_ms") != expected["timestamp"]:
                    raise ValueError("Altered source timestamp")
                try:
                    instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=expected["timestamp"])
                except OverflowError:
                    if "timestamp_utc" in origin:
                        raise ValueError("Invented out-of-range source date")
                else:
                    if origin.get("timestamp_utc") != instant.isoformat(timespec="milliseconds").replace("+00:00", "Z"):
                        raise ValueError("Altered source date")
            elif "timestamp_ms" in origin or "timestamp_utc" in origin:
                raise ValueError("Invented source timestamp")
            if set(origin) - {"role", "session_id", "request_id", "message_index", "message_id", "received_sequence", "timestamp_ms", "timestamp_utc"}:
                raise ValueError("Unattributed extra source metadata")
            body = content[end + 2:]
            if not body.startswith(expected["content"]):
                raise ValueError("Original source text changed")
            content = body[len(expected["content"]):]
            seen.add(mid)
            first = False
            if content:
                separator = "\n\n[adjacent context]\n"
                if not content.startswith(separator):
                    raise ValueError("Unattributed or truncated extra text")
                content = content[len(separator):]
    return seen


class MemoryStoreAdapter:
    """Local Add/Search API statuses plus explicit restart; no model or network."""
    def __init__(self, database, server_module=None, context_grouping="reserved"):
        if server_module is None:
            path = ROOT / "server.py"
            spec = importlib.util.spec_from_file_location("synthetic_validation_server", path)
            server_module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = server_module
            spec.loader.exec_module(server_module)
        self.module, self.database, self.grouping = server_module, Path(database), context_grouping
        self.restart()

    def restart(self):
        self.store = self.module.MemoryStore(self.database, semantic_backend=None, context_grouping=self.grouping)

    def add(self, payload):
        try:
            return 200, self.store.add(payload)
        except self.module.ConflictError:
            return 409, {"error": "conflict"}

    def search(self, payload):
        return 200, self.store.search(payload)


def run_synthetic(client, fixture=None, answers=None):
    fixture = fixture or synthetic_fixtures()
    if answers is not None and not isinstance(answers, dict):
        raise ValueError("Synthetic answers require a case-ID mapping")
    visible, responses, rows = set(), {}, []
    for index, step in enumerate(fixture["steps"]):
        passed, checks = True, {}
        try:
            if step["action"] == "restart":
                client.restart()
                checks["restart_invoked"] = True
            elif step["action"] == "add":
                status, response = client.add(step["payload"])
                checks["expected_status"] = status == step["expected_status"]
                if status == 200:
                    payload = step["payload"]
                    checks["exact_ack"] = response == {"success": True, **{key: payload[key] for key in ("user_id", "session_id", "request_id")}}
                    visible.update(mid for mid, source in fixture["sources"].items()
                                   if source["user_id"] == payload["user_id"] and source["request_id"] == payload["request_id"])
            else:
                status, response = client.search(step["payload"])
                checks["http_200"] = status == 200
                grounded = verify_grounding(response, fixture["sources"], step["payload"]["user_id"], visible)
                checks["exact_source_grounding"] = True
                checks["required_evidence_present"] = set(step["required_source_ids"]) <= grounded
                checks["no_forbidden_source"] = not (set(step["forbidden_source_ids"]) & grounded)
                if "compare_to" in step:
                    checks["unchanged_after_retry_conflict_or_restart"] = response == responses[step["compare_to"]]
                responses[step["name"]] = response
            passed = all(checks.values())
        except Exception as error:
            passed, checks = False, {"error_type": type(error).__name__}
        rows.append({"step": index, "action": step["action"], "name": step.get("name"), "passed": passed, "checks": checks})
    semantic = []
    for case in fixture["semantic_cases"]:
        correct = None
        if answers is not None:
            value = answers.get(case["case_id"])
            normalize = lambda item: " ".join(item.casefold().split()) if isinstance(item, str) else None
            if case["answer_type"] == "set":
                correct = (isinstance(value, list) and all(isinstance(item, str) for item in value)
                           and len(value) == len(set(normalize(item) for item in value))
                           and {normalize(item) for item in value} == {normalize(item) for item in case["reference_answer"]})
            else:
                correct = normalize(value) == normalize(case["reference_answer"])
        semantic.append({"case_id": case["case_id"], "capability": case["capability"], "correct": correct})
    return {"schema": VERSION, "synthetic_only": True, "model_requests": 0, "fixture_sha256": digest(fixture),
            "behavior": {"passed": sum(row["passed"] for row in rows), "total": len(rows), "all_passed": all(row["passed"] for row in rows), "steps": rows},
            "source_grounding": {"passed_searches": sum(row["checks"].get("exact_source_grounding") is True for row in rows), "searches": sum(row["action"] == "search" for row in rows)},
            "semantic_accuracy": None if answers is None else sum(row["correct"] is True for row in semantic) / len(semantic),
            "semantic_metric": fixture["semantic_metric"], "semantic_cases": semantic,
            "answer_inputs": [{"case_id": case["case_id"], "query": case["query"],
                               "search_response": responses.get(case["case_id"])} for case in fixture["semantic_cases"]],
            "limitations": ["Evidence presence/provenance does not prove answer correctness.", "Supplied synthetic answers use restricted exact match, not an LLM judge or official AML score."]}


def write_new(path, value):
    if path is None:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    path = Path(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--prepared-dir", type=Path, action="append", required=True)
    split.add_argument("--used-run", type=Path, action="append", default=[])
    split.add_argument("--usage-ledger", type=Path)
    split.add_argument("--identity-map", type=Path)
    split.add_argument("--raw-identity-dir", type=Path, default=ROOT / ".local" / "public-data" / "raw",
                       help="Pinned native source identities for known public datasets; no label selection")
    split.add_argument("--seed", default=SPLIT_SEED)
    split.add_argument("--preserve-prepared-splits", action="store_true",
                       help="Inherit original public partitions; quarantine used/cross-partition identity components")
    split.add_argument("--output", type=Path)
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--run-local", action="store_true")
    synthetic.add_argument("--context-grouping", choices=("reserved", "emitted"), default="reserved")
    synthetic.add_argument("--answers", type=Path, help="Local JSON mapping case IDs to canonical string/list answers")
    synthetic.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "split":
        if not args.used_run and args.usage_ledger is None:
            parser.error("split requires a usage ledger or used runs; prior LME60 identities must be included")
        histories, queries, sources = load_prepared(args.prepared_dir)
        discovered, discovery = discover_used_runs(ROOT / "competition" / "results")
        used = [used_run(path) for path in sorted(set(discovered + [path.resolve() for path in args.used_run]))]
        if args.usage_ledger:
            used.extend(json.loads(args.usage_ledger.read_text())["entries"])
            sources.append({"path": str(args.usage_ledger.resolve()), "sha256": file_hash(args.usage_ledger)})
        native, native_sources = native_identity_map(histories, args.raw_identity_dir)
        identities = merge_identity_maps(native, load_identity_map(args.identity_map) if args.identity_map else [])
        sources.extend(native_sources)
        result = build_validation_manifest(histories, queries, used, identities, seed=args.seed,
                                           preserve_prepared_splits=args.preserve_prepared_splits)
        result["sources"] = sources
        result["usage_discovery"] = discovery
        result["native_identity_map_sha256"] = digest(native)
        result["native_identity_policy"] = "All native source-session/conversation overlap is grouped; no real person IDs or answer/evidence labels are inferred."
        if args.identity_map:
            result["identity_map_sha256"] = file_hash(args.identity_map)
    elif args.run_local:
        fixture = synthetic_fixtures()
        answers = json.loads(args.answers.read_text()) if args.answers else None
        with tempfile.TemporaryDirectory(prefix="memory-validation-synthetic-") as directory:
            result = run_synthetic(MemoryStoreAdapter(Path(directory) / "memory.sqlite3", context_grouping=args.context_grouping), fixture, answers)
        result["context_grouping"] = args.context_grouping
        result["server_sha256"] = file_hash(ROOT / "server.py")
    else:
        if args.answers:
            parser.error("--answers requires --run-local")
        result = synthetic_fixtures()
    result["validation_source_sha256"] = file_hash(Path(__file__))
    write_new(args.output, result)
    return result


if __name__ == "__main__":
    main()
