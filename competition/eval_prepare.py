#!/usr/bin/env python3
"""Prepare pinned PUBLIC upstream data; keep history, queries and grader gold separate.

This is a newly authored proxy adapter, not AML orchestration or an official score.
Only the Python standard library is needed. See README.md in this directory for usage.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / ".local" / "public-data"
MANIFEST = HERE / "public_data_manifest.json"
SPLITS = ("dev", "validation", "heldout")
SOURCE_MAPPING_VERSION = "session-occurrence-v2"
LEGACY_SOURCE_MAPPING_VERSION = "session-id-v1"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def normalized(value):
    # Whitespace only: do not lower-case or erase punctuation/factual distinctions.
    return " ".join(value.split())


def timestamp(value):
    formats = ("%Y/%m/%d (%a) %H:%M", "%I:%M %p on %d %B, %Y",
               "%I:%M %p on %d %B %Y", "%I:%M %p on %d %b, %Y")
    for fmt in formats:
        try:
            # Upstream dates have no timezone. UTC is a documented encoding convention.
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    raise ValueError("Unrecognized public session date: " + repr(value))


def fetch_files(manifest, root, dataset_ids, download):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for dataset in manifest["datasets"]:
        if dataset["id"] not in dataset_ids:
            continue
        for item in dataset["files"]:
            path = raw / item["name"]
            if not path.exists():
                if not download:
                    raise FileNotFoundError(str(path) + "; rerun with --download")
                temp = path.with_suffix(path.suffix + ".part")
                req = urllib.request.Request(item["url"], headers={"User-Agent": "public-memory-eval/1.0"})
                try:
                    with urllib.request.urlopen(req, timeout=60) as response, temp.open("wb") as out:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            out.write(block)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
            if path.stat().st_size != item["size"] or file_hash(path) != item["sha256"]:
                raise ValueError("Pinned public file checksum mismatch: " + str(path))


class UnionFind:
    def __init__(self, keys):
        self.parent = {x: x for x in keys}

    def find(self, key):
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def allocate_splits(dataset, groups, seed):
    ordered = sorted(set(groups), key=lambda x: digest(seed + "\0" + dataset + "\0" + x))
    n = len(ordered)
    if n < 3:
        raise ValueError("Need at least three independent history groups")
    dev_end = max(1, int(n * .6))
    val_end = min(n - 1, dev_end + max(1, int(n * .2)))
    return {group: "dev" if i < dev_end else "validation" if i < val_end else "heldout"
            for i, group in enumerate(ordered)}


def new_history(dataset, native_id):
    return {"sample_id": digest(dataset + "\0" + native_id)[:32], "sessions": []}


def occurrence_id(base_id, occurrence, prefix=""):
    """Preserve first-occurrence IDs; distinguish later source occurrences."""
    if type(occurrence) is not int or occurrence < 0:
        raise ValueError("Session occurrence must be a nonnegative integer")
    if occurrence == 0:
        return base_id
    identity = digest(base_id + "\0occurrence\0" + str(occurrence))
    return prefix + (identity[:24] if prefix else identity)


def add_session(history, native_session, date, source_messages, occurrence=0):
    base_sid = "s_" + digest(history["sample_id"] + "\0" + native_session)[:24]
    sid = occurrence_id(base_sid, occurrence, "s_")
    if any(session["session_id"] == sid for session in history["sessions"]):
        raise ValueError("Repeated source session requires its distinct occurrence index")
    messages = []
    units = []
    seen = Counter(normalized(m["text"]) for m in source_messages)
    for i, m in enumerate(source_messages):
        if m["role"] not in ("user", "assistant") or not isinstance(m["text"], str) or not m["text"].strip():
            raise ValueError("Unsupported public history message")
        speaker = (m.get("speaker") + ": ") if m.get("speaker") else ""
        content = "[session_date: " + date + "]\n" + speaker + m["text"]
        messages.append({"role": m["role"], "content": content, "timestamp": timestamp(date)})
        base_unit = digest(history["sample_id"] + "\0" + native_session + "\0" + str(i))
        units.append({"unit_id": occurrence_id(base_unit, occurrence),
                      "session_id": sid, "message_position": i, "text": m["text"],
                      "source_message_content": content, "native_id": m["native_id"],
                      "unique_within_session": seen[normalized(m["text"])] == 1})
    history["sessions"].append({"session_id": sid, "messages": messages})
    return units


def build_longmemeval(rows, dataset, seed):
    keys = [str(x["question_id"]) for x in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate public question IDs")
    uf = UnionFind(keys)
    evidence_owner = {}
    history_owner = {}
    for row in rows:
        qid = str(row["question_id"])
        # Labels influence only the GRADER split, never ingestion or retrieval behavior.
        for sid in row["answer_session_ids"]:
            if sid in evidence_owner:
                uf.union(qid, evidence_owner[sid])
            evidence_owner[sid] = qid
        identity = digest(json.dumps(row["haystack_session_ids"], separators=(",", ":")))
        if identity in history_owner:
            uf.union(qid, history_owner[identity])
        history_owner[identity] = qid
    members = defaultdict(list)
    for key in keys:
        members[uf.find(key)].append(key)
    group_ids = {key: digest("\0".join(sorted(group))) for group in members.values() for key in group}
    assigned = allocate_splits(dataset, group_ids.values(), seed)
    output = {split: [] for split in SPLITS}
    counts = Counter()
    source_owners = defaultdict(set)
    for row in rows:
        qid = str(row["question_id"])
        split = assigned[group_ids[qid]]
        counts["source_questions"] += 1
        for sid in row["haystack_session_ids"]:
            source_owners[sid].add(split)
        if qid.endswith("_abs"):
            counts["excluded_abstention"] += 1
            continue
        history = new_history(dataset, qid)
        gold = []
        gold_sessions = []
        sizes = [len(row[k]) for k in ("haystack_sessions", "haystack_session_ids", "haystack_dates")]
        if len(set(sizes)) != 1:
            raise ValueError("Mismatched LongMemEval history arrays")
        session_occurrences = Counter()
        for sid, date, session in zip(row["haystack_session_ids"], row["haystack_dates"], row["haystack_sessions"]):
            # Upstream has a few empty filler turns. They carry no retrievable text;
            # omit them explicitly rather than fabricate an utterance or violate Add.
            kept = [(i, m) for i, m in enumerate(session) if isinstance(m["content"], str) and m["content"].strip()]
            dropped = len(session) - len(kept)
            if any(m.get("has_answer") is True and not m["content"].strip() for m in session):
                raise ValueError("An empty turn is labelled as evidence; cannot prepare turn recall")
            counts["omitted_empty_history_turns"] += dropped
            if not kept:
                counts["omitted_empty_sessions"] += 1
                continue
            occurrence = session_occurrences[sid]
            session_occurrences[sid] += 1
            safe = [{"role": m["role"], "text": m["content"], "native_id": str(i)} for i, m in kept]
            units = add_session(history, sid, date, safe, occurrence=occurrence)
            gold.extend(unit for unit, (_, m) in zip(units, kept) if m.get("has_answer") is True)
            if sid in row["answer_session_ids"]:
                gold_sessions.append(history["sessions"][-1]["session_id"])
        if not gold:
            counts["excluded_missing_turn_gold"] += 1
            continue
        # A qid is opaque in API payloads. Original IDs/types exist only in grader files.
        query = {"sample_id": history["sample_id"], "query_id": digest(dataset + "\0" + qid), "query": row["question"]}
        label = {"sample_id": history["sample_id"], "query_id": query["query_id"], "original_query_id": qid,
                 "category": row["question_type"], "answer": row["answer"], "question_date": row["question_date"],
                 "gold_units": gold, "gold_session_ids": sorted(set(gold_sessions))}
        output[split].append((history, [query], [label]))
        counts["eligible_questions"] += 1
    audit = {"independent_groups": len(members), "group_split_counts": dict(Counter(assigned.values())),
             "reused_history_session_ids_across_splits": sum(len(v) > 1 for v in source_owners.values()),
             "grouping_boundary": "Whole constructed history plus connected shared annotated evidence sessions. No stable native real-user IDs are provided.",
             "limitations": "Shared filler history sources can occur across splits. This is not a disjoint-corpus evaluation.",
             **dict(counts)}
    return output, audit


def evidence_ids(values):
    if not isinstance(values, list):
        return None
    result = []
    for value in values:
        if not isinstance(value, str):
            return None
        matches = list(re.finditer(r"D(\d+):(\d+)", value))
        remainder = re.sub(r"D\d+:\d+", "", value)
        if not matches or re.sub(r"[\s,;]+", "", remainder):
            return None
        result.extend("D" + str(int(m[1])) + ":" + str(int(m[2])) for m in matches)
    return sorted(set(result))


def build_locomo(rows, dataset, seed):
    native_ids = [str(x["sample_id"]) for x in rows]
    if len(set(native_ids)) != len(rows):
        raise ValueError("Duplicate public conversation IDs")
    assigned = allocate_splits(dataset, native_ids, seed)
    output = {split: [] for split in SPLITS}
    counts = Counter()
    for row in rows:
        conversation = row["conversation"]
        native_id = str(row["sample_id"])
        history = new_history(dataset, native_id)
        units_by_native = {}
        sessions = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        for key in sessions:
            safe = []
            for m in conversation[key]:
                if m["speaker"] not in (conversation["speaker_a"], conversation["speaker_b"]):
                    raise ValueError("Unrecognized conversation speaker")
                role = "user" if m["speaker"] == conversation["speaker_a"] else "assistant"
                safe.append({"role": role, "speaker": m["speaker"], "text": m["text"], "native_id": m["dia_id"]})
            units = add_session(history, key, conversation[key + "_date_time"], safe)
            for unit in units:
                ids = evidence_ids([unit["native_id"]])
                if not ids or len(ids) != 1 or ids[0] in units_by_native:
                    raise ValueError("Nonunique/invalid original dialogue ID")
                units_by_native[ids[0]] = unit
        queries, labels = [], []
        for i, qa in enumerate(row["qa"]):
            counts["source_questions"] += 1
            if qa.get("is_multi_modality"):
                counts["excluded_multimodal"] += 1
                continue
            ids = evidence_ids(qa.get("evidence"))
            if not ids:
                counts["excluded_unparseable_or_empty_evidence"] += 1
                continue
            if any(key not in units_by_native for key in ids):
                counts["excluded_unresolved_evidence"] += 1
                continue
            qid = native_id + "#q" + str(i).zfill(4)
            query = {"sample_id": history["sample_id"], "query_id": digest(dataset + "\0" + qid), "query": qa["question"]}
            gold = [units_by_native[key] for key in ids]
            queries.append(query)
            labels.append({"sample_id": history["sample_id"], "query_id": query["query_id"], "original_query_id": qid,
                           "category": str(qa["category"]), "answer": qa["answer"], "gold_units": gold,
                           "gold_session_ids": sorted(set(unit["session_id"] for unit in gold))})
            counts["eligible_questions"] += 1
        if queries:
            output[assigned[native_id]].append((history, queries, labels))
    return output, {"independent_groups": len(rows), "group_split_counts": dict(Counter(assigned.values())),
                    "grouping_boundary": "Every conversation and all its questions stay in one split.", **dict(counts)}


def prepare(root, manifest, dataset_ids):
    summary = {"status": "public_proxy_not_official_aml", "manifest_sha256": file_hash(MANIFEST),
               "source_mapping_version": SOURCE_MAPPING_VERSION,
               "split_seed": manifest["split"]["seed"], "timezone_convention": "UTC encoding of upstream timezone-unspecified session dates; original date text is also retained.",
               "datasets": {}}
    for dataset in manifest["datasets"]:
        name = dataset["id"]
        if name not in dataset_ids:
            continue
        raw = root / "raw" / dataset["files"][0]["name"]
        rows = json.loads(raw.read_text(encoding="utf-8"))
        builder = build_longmemeval if name == "longmemeval_s" else build_locomo
        output, audit = builder(rows, name, manifest["split"]["seed"])
        audit["splits"] = {}
        for split, entries in output.items():
            target = root / "prepared" / name / split
            if (target / "heldout-evaluated.lock").exists():
                raise ValueError("Cannot overwrite previously evaluated heldout preparation")
            target.mkdir(parents=True, exist_ok=True)
            paths = {key: target / (key + ".jsonl") for key in ("histories", "queries", "gold")}
            with paths["histories"].open("w", encoding="utf-8") as hf, paths["queries"].open("w", encoding="utf-8") as qf, paths["gold"].open("w", encoding="utf-8") as gf:
                for history, queries, labels in sorted(entries, key=lambda e: e[0]["sample_id"]):
                    hf.write(json_line(history))
                    for query in queries:
                        qf.write(json_line(query))
                    for label in labels:
                        gf.write(json_line(label))
            audit["splits"][split] = {
                "histories": len(entries), "questions": sum(len(q) for _, q, _ in entries),
                "messages": sum(len(s["messages"]) for h, _, _ in entries for s in h["sessions"]),
                "gold_turn_annotations": sum(len(g["gold_units"]) for _, _, labels in entries for g in labels),
                "files": {key: {"size": p.stat().st_size, "sha256": file_hash(p)} for key, p in paths.items()}}
        summary["datasets"][name] = audit
        del rows, output
    path = root / "prepared" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def migrate_history_sources(history, labels):
    """Upgrade one prepared history and grader labels without examining future QA.

    The old full source content disambiguates repeated IDs. Ambiguous locations
    cannot be recovered from legacy preparation and are explicitly rejected.
    """
    groups = defaultdict(list)
    occurrences = Counter()
    sessions = []
    for session in history["sessions"]:
        old_sid = session["session_id"]
        occurrence = occurrences[old_sid]
        occurrences[old_sid] += 1
        sid = occurrence_id(old_sid, occurrence, "s_")
        sessions.append({**session, "session_id": sid})
        groups[old_sid].append((sid, occurrence, session["messages"]))
    if len({session["session_id"] for session in sessions}) != len(sessions):
        raise ValueError("Session occurrence IDs are not unique")
    upgraded_labels = []
    for label in labels:
        units = []
        for unit in label["gold_units"]:
            position = unit["message_position"]
            content = unit.get("source_message_content")
            matches = [(sid, occurrence) for sid, occurrence, messages in groups[unit["session_id"]]
                       if type(position) is int and 0 <= position < len(messages)
                       and isinstance(content, str) and messages[position]["content"] == content]
            if len(matches) != 1:
                raise ValueError("Legacy evidence location is missing or ambiguous; cannot guess its occurrence")
            sid, occurrence = matches[0]
            units.append({**unit, "session_id": sid, "unit_id": occurrence_id(unit["unit_id"], occurrence)})
        if len({unit["unit_id"] for unit in units}) != len(units):
            raise ValueError("Evidence occurrence IDs are not unique")
        # Legacy preparation marked every matching native session as coarse gold.
        # Preserve that meaning, including occurrences with no annotated gold turn.
        gold_sessions = []
        for old_sid in label["gold_session_ids"]:
            if old_sid not in groups:
                raise ValueError("Legacy coarse gold session does not exist")
            gold_sessions.extend(sid for sid, _, _ in groups[old_sid])
        upgraded_labels.append({**label, "gold_units": units, "gold_session_ids": sorted(set(gold_sessions))})
    return {**history, "sessions": sessions}, upgraded_labels, sum(n - 1 for n in occurrences.values())


def migrate_dev_sources(source_root, target_root, dataset_ids):
    """Create a separate DEV-only mapping version; never overwrite old evidence."""
    if source_root.resolve() == target_root.resolve() or target_root.exists():
        raise ValueError("Dev mapping migration requires a new, absent data root")
    source_summary = source_root / "prepared" / "summary.json"
    original = json.loads(source_summary.read_text())
    if original.get("source_mapping_version", LEGACY_SOURCE_MAPPING_VERSION) != LEGACY_SOURCE_MAPPING_VERSION:
        raise ValueError("Dev mapping migration expects legacy session-id-v1 preparation")
    summary = {"status": "public_proxy_not_official_aml", "source_mapping_version": SOURCE_MAPPING_VERSION,
               "manifest_sha256": original["manifest_sha256"], "split_seed": original["split_seed"],
               "timezone_convention": original["timezone_convention"], "scope": "dev_only_source_mapping_migration",
               "derived_from_prepared_summary_sha256": file_hash(source_summary), "datasets": {}}
    for dataset in dataset_ids:
        specification = original["datasets"][dataset]["splits"]["dev"]
        folder = source_root / "prepared" / dataset / "dev"
        for name in ("histories", "queries", "gold"):
            if file_hash(folder / (name + ".jsonl")) != specification["files"][name]["sha256"]:
                raise ValueError("Legacy dev data checksum mismatch")
        with (folder / "gold.jsonl").open(encoding="utf-8") as source:
            labels = [json.loads(line) for line in source if line.strip()]
        by_sample = defaultdict(list)
        for index, label in enumerate(labels):
            by_sample[label["sample_id"]].append((index, label))
        target = target_root / "prepared" / dataset / "dev"
        target.mkdir(parents=True)
        seen_samples, repeated_histories, repeated_occurrences = set(), 0, 0
        with (folder / "histories.jsonl").open(encoding="utf-8") as source, (target / "histories.jsonl").open("w", encoding="utf-8") as out:
            for line in source:
                if not line.strip():
                    continue
                history = json.loads(line)
                sample = history["sample_id"]
                if sample in seen_samples:
                    raise ValueError("Duplicate prepared dev history ID")
                seen_samples.add(sample)
                indexes = by_sample[sample]
                upgraded, gold, repeats = migrate_history_sources(history, [label for _, label in indexes])
                out.write(json_line(upgraded))
                for (index, _), label in zip(indexes, gold):
                    labels[index] = label
                repeated_histories += bool(repeats)
                repeated_occurrences += repeats
        if seen_samples != set(by_sample):
            raise ValueError("Prepared dev histories and labels differ")
        with (target / "gold.jsonl").open("w", encoding="utf-8") as out:
            for label in labels:
                out.write(json_line(label))
        shutil.copyfile(folder / "queries.jsonl", target / "queries.jsonl")
        upgraded_specification = {**specification, "files": {name: {"size": (target / (name + ".jsonl")).stat().st_size,
                                  "sha256": file_hash(target / (name + ".jsonl"))} for name in ("histories", "queries", "gold")}}
        summary["datasets"][dataset] = {"splits": {"dev": upgraded_specification},
                                       "migration": {"repeated_session_histories": repeated_histories,
                                                     "additional_session_occurrences": repeated_occurrences,
                                                     "query_order_and_content_preserved": True,
                                                     "scope": "IDs only; original messages, questions, answers, annotations and fixed split preserved"}}
    path = target_root / "prepared" / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", default="longmemeval_s,locomo_refined")
    parser.add_argument("--download", action="store_true", help="Download pinned public files if absent; never a protected AML corpus")
    parser.add_argument("--migrate-dev-from", type=Path, help="Upgrade only legacy prepared dev into a NEW --data-root; no raw or validation/heldout access")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    names = args.datasets.split(",")
    allowed = {d["id"] for d in manifest["datasets"]}
    if not set(names) <= allowed:
        parser.error("Unknown dataset")
    if args.migrate_dev_from:
        if args.download:
            parser.error("Dev migration does not download or inspect raw datasets")
        print(json.dumps(migrate_dev_sources(args.migrate_dev_from, args.data_root, names), ensure_ascii=False, indent=2))
        return
    fetch_files(manifest, args.data_root, names, args.download)
    print(json.dumps(prepare(args.data_root, manifest, names), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
