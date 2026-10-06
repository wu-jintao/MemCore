#!/usr/bin/env python3
"""Research-only reranking of frozen PUBLIC dev candidates; no Add or paid API.

prepare exports question/candidates only. rank never reads gold or grader code.
score is a separate, local-only evidence-recall stage. Original runs stay intact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import time

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
VERSION = "frozen-dev-windowed-reranker-v1"
MODEL_FILES = ("README.md", "config.json", "model.safetensors", "sentencepiece.bpe.model",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    return path


def load_source(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("split") != "dev":
        raise ValueError("Only PUBLIC dev is authorized; reject before reading responses")
    if (summary.get("dataset"), summary.get("question_count")) not in {
            ("locomo_refined", 547), ("longmemeval_s", 60)}:
        raise ValueError("Only the frozen Lo547 / LME60 source sets are authorized")
    config = summary["configuration"]
    if config.get("top_k") != 100 or config.get("character_budget") != 0:
        raise ValueError("Require full existing top100 responses")
    if summary.get("server_source_changed_during_run"):
        raise ValueError("Source changed; cannot declare a frozen baseline")
    records = list(jsonl(folder / "retrieval.jsonl"))
    if len(records) != summary["question_count"] or len({r["query_id"] for r in records}) != len(records):
        raise ValueError("Incomplete/duplicate source responses")
    if any(r.get("status") != "ok" for r in records):
        raise ValueError("Source must be completely successful")
    for record in records:
        validate_items(record["retrieved"])
    return summary, records


def validate_items(items):
    if not isinstance(items, list) or not items or len(items) > 100:
        raise ValueError("Expected 1..100 complete candidate items")
    ids = []
    for item in items:
        if (set(item) != {"id", "content", "score"} or not isinstance(item["id"], str)
                or not isinstance(item["content"], str) or not item["content"]
                or type(item["score"]) not in (float, int) or not math.isfinite(item["score"])):
            raise ValueError("Invalid candidate schema/content/score")
        ids.append(item["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate IDs")


def assert_permutation(before, after):
    validate_items(before)
    validate_items(after)
    if {r["id"]: r["content"] for r in before} != {r["id"]: r["content"] for r in after}:
        raise ValueError("Candidate source set or exact original content changed")


def validate_bundle(manifest, records):
    if (manifest.get("split") != "dev" or manifest.get("exported_gold") is not False
            or (manifest.get("dataset"), manifest.get("source_question_count")) not in
            {("locomo_refined", 547), ("longmemeval_s", 60)}):
        raise ValueError("Only authorized PUBLIC dev question/candidate bundles")
    if len(records) != manifest["question_count"] or not 1 <= len(records) <= manifest["source_question_count"]:
        raise ValueError("Incomplete/oversized bundle")
    ids = [r.get("query_id") for r in records]
    if ids != manifest["query_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Query sequence/uniqueness changed")
    for record in records:
        if set(record) != {"query_id", "sample_id", "query", "retrieved"}:
            raise ValueError("Unexpected ranker input; no label/gold fields allowed")
        if not isinstance(record["query"], str) or not record["query"].strip():
            raise ValueError("Empty/invalid query")
        validate_items(record["retrieved"])


def intervals(length, width, overlap):
    if length < 0 or width < 1 or not 0 <= overlap < width:
        raise ValueError("Invalid token window configuration")
    if not length:
        return [(0, 0)]
    result = []
    start = 0
    while start < length:
        end = min(length, start + width)
        result.append((start, end))
        if end == length:
            break
        start = end - overlap
    return result


def item_pairs(tokenizer, query, content, maximum=512, query_width=128, overlap=64, max_pairs=4096):
    # All original token IDs are covered; neither query nor document is truncated.
    qids = tokenizer.encode(query, add_special_tokens=False, truncation=False)
    dids = tokenizer.encode(content, add_special_tokens=False, truncation=False)
    special = tokenizer.num_special_tokens_to_add(pair=True)
    if not qids or maximum <= special + query_width + overlap:
        raise ValueError("Empty query or insufficient pair budget")
    features = []
    qwindows = intervals(len(qids), query_width, min(overlap, query_width // 4))
    document_windows = 0
    for qstart, qend in qwindows:
        q = qids[qstart:qend]
        width = maximum - special - len(q)
        for start, end in intervals(len(dids), width, overlap):
            pair = tokenizer.prepare_for_model(q, pair_ids=dids[start:end], add_special_tokens=True,
                                               truncation=False, return_attention_mask=True,
                                               return_token_type_ids=False)
            if len(pair["input_ids"]) > maximum:
                raise ValueError("Tokenizer exceeded the declared pair budget")
            features.append(pair)
            document_windows += 1
            if len(features) > max_pairs:
                raise ValueError("Declared pair capacity exceeded; no silently dropped windows")
    return features, {"query_tokens": len(qids), "document_tokens": len(dids),
                      "query_windows": len(qwindows), "pairs": document_windows,
                      "original_content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def prepare(args):
    source = Path(args.source_run)
    summary, records = load_source(source)
    query_file = Path(args.data_root) / "prepared" / summary["dataset"] / "dev" / "queries.jsonl"
    expected = summary["configuration"]["prepared_data"]["files"]["queries"]["sha256"]
    if file_hash(query_file) != expected:
        raise ValueError("Prepared public dev questions changed")
    wanted = {r["query_id"] for r in records}
    if args.max_queries:
        chosen = sorted(wanted, key=lambda q: hashlib.sha256((VERSION + q).encode()).hexdigest())[:args.max_queries]
        wanted = set(chosen)
    queries = {r["query_id"]: r for r in jsonl(query_file) if r["query_id"] in wanted}
    if set(queries) != wanted:
        raise ValueError("Missing question mapping")
    selected = [r for r in records if r["query_id"] in wanted]
    folder = new_directory(args.output)
    with (folder / "input.jsonl").open("w", encoding="utf-8") as out:
        for record in selected:
            query = queries[record["query_id"]]
            if query["sample_id"] != record["sample_id"]:
                raise ValueError("Question/history mismatch")
            # Deliberately export no category, answer, gold, metric or original label.
            out.write(json.dumps({"query_id": record["query_id"], "sample_id": record["sample_id"],
                                  "query": query["query"], "retrieved": record["retrieved"]}, ensure_ascii=False) + "\n")
    manifest = {"version": VERSION, "dataset": summary["dataset"], "split": "dev",
                "source_run_id": summary["run_id"],
                "ingestion_run_id": summary.get("ingestion_run_id", summary["run_id"]),
                "source_question_count": len(records), "question_count": len(selected),
                "source_summary_sha256": file_hash(source / "summary.json"),
                "source_retrieval_sha256": file_hash(source / "retrieval.jsonl"),
                "input_sha256": file_hash(folder / "input.jsonl"), "query_file_sha256": expected,
                "query_ids": [r["query_id"] for r in selected], "source_configuration": summary["configuration"],
                "subset_policy": "sha256(version+query_id) ascending selection; retain original output order; no labels",
                "ranker_input": "original query plus exact returned item content including source/adjacency wrappers",
                "exported_gold": False, "helper_sha256": file_hash(__file__)}
    write_json(folder / "manifest.json", manifest)
    # Local-only baseline for matching subset analysis; do not transfer this directory to the ranker.
    baseline = new_directory(folder / "baseline")
    base_summary = dict(summary)
    base_summary["question_count"] = len(selected)
    base_summary["run_id"] = summary["run_id"] + "-subset-" + hashlib.sha256("\n".join(manifest["query_ids"]).encode()).hexdigest()[:10]
    base_summary["ingestion_run_id"] = manifest["ingestion_run_id"]
    base_summary["source_original_run_id"] = summary["run_id"]
    base_summary["model_label"] = summary["model_label"] + " (frozen matching subset)"
    write_json(baseline / "summary.json", base_summary)
    (baseline / "retrieval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    return {"question_count": len(selected), "dataset": summary["dataset"], "input_sha256": manifest["input_sha256"]}


def download_model(args):
    from huggingface_hub import HfApi, snapshot_download
    folder = Path(args.model_dir)
    folder.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise ValueError("Official repository revision mismatch")
    metadata = {item.rfilename: item for item in info.siblings}
    if not set(MODEL_FILES) <= set(metadata):
        raise ValueError("Missing official model files")
    snapshot_download(MODEL, revision=REVISION, local_dir=folder, allow_patterns=list(MODEL_FILES),
                      token=False, max_workers=2)
    files = {}
    for name in MODEL_FILES:
        path = folder / name
        item = metadata[name]
        actual = file_hash(path)
        lfs = getattr(item, "lfs", None)
        expected_lfs = getattr(lfs, "sha256", None)
        if expected_lfs:
            if actual != expected_lfs:
                raise ValueError("Official LFS SHA256 mismatch: " + name)
        else:
            blob = hashlib.sha1()
            blob.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    blob.update(block)
            if blob.hexdigest() != item.blob_id:
                raise ValueError("Official git blob mismatch: " + name)
        files[name] = {"bytes": path.stat().st_size, "sha256": actual,
                       "official_git_blob": item.blob_id, "official_lfs_sha256": expected_lfs}
    manifest = {"model": MODEL, "revision": REVISION, "official_metadata_verified": True,
                "origin": "https://huggingface.co/" + MODEL, "files": files}
    write_json(folder / "model-manifest.json", manifest)
    return {"revision": REVISION, "files": len(files), "manifest_sha256": file_hash(folder / "model-manifest.json")}


class Engine:
    def __init__(self, args):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                          CUBLAS_WORKSPACE_CONFIG=":4096:8")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.args = torch, args
        directory = Path(args.model_dir)
        self.manifest = json.loads((directory / "model-manifest.json").read_text())
        if (self.manifest.get("model"), self.manifest.get("revision")) != (MODEL, REVISION):
            raise ValueError("Unexpected model/revision")
        if self.manifest.get("official_metadata_verified") is not True or set(self.manifest["files"]) != set(MODEL_FILES):
            raise ValueError("Incomplete/unverified official model manifest")
        for name, spec in self.manifest["files"].items():
            if file_hash(directory / name) != spec["sha256"]:
                raise ValueError("Local model file changed: " + name)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation="eager").to(args.device).eval()
        self.versions = {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy")}

    def rerank(self, query, items):
        validate_items(items)
        flat, owners, traces = [], [], []
        for index, item in enumerate(items):
            pairs, trace = item_pairs(self.tokenizer, query, item["content"], self.args.max_pair_tokens,
                                     self.args.query_window, self.args.overlap, self.args.max_pairs_per_item)
            flat.extend(pairs)
            owners.extend([index] * len(pairs))
            traces.append({"id": item["id"], "original_rank": index + 1, "original_rrf_score": item["score"], **trace})
        scores = [-math.inf] * len(items)
        with self.torch.inference_mode():
            for start in range(0, len(flat), self.args.batch_size):
                batch = self.tokenizer.pad(flat[start:start + self.args.batch_size], padding="max_length",
                                           max_length=self.args.max_pair_tokens, return_tensors="pt")
                batch = {key: value.to(self.args.device) for key, value in batch.items()}
                values = self.model(**batch).logits.reshape(-1).float().cpu().tolist()
                for owner, value in zip(owners[start:start + self.args.batch_size], values):
                    if not math.isfinite(value):
                        raise ValueError("Non-finite cross-encoder score")
                    scores[owner] = max(scores[owner], value)
        for index, trace in enumerate(traces):
            trace["reranker_score"] = scores[index]
        ordered = sorted(range(len(items)), key=lambda index: (-scores[index], index))
        result = [{**items[index], "score": scores[index]} for index in ordered]
        assert_permutation(items, result)
        return result, traces


def rank(args):
    inputs = Path(args.input_dir)
    manifest = json.loads((inputs / "manifest.json").read_text())
    if manifest.get("split") != "dev" or manifest.get("exported_gold") is not False:
        raise ValueError("Only question/candidate-only PUBLIC dev bundles")
    if file_hash(inputs / "input.jsonl") != manifest["input_sha256"]:
        raise ValueError("Frozen input changed")
    input_records = list(jsonl(inputs / "input.jsonl"))
    validate_bundle(manifest, input_records)
    output = new_directory(args.output)
    write_json(output / "input-manifest.json", manifest)
    started = time.monotonic()
    engine = Engine(args)
    load_seconds = time.monotonic() - started
    count, pair_count, seconds = 0, 0, []
    with (output / "retrieval.jsonl").open("w", encoding="utf-8") as out:
        for record in input_records:
            t = time.monotonic()
            items, traces = engine.rerank(record["query"], record["retrieved"])
            elapsed = time.monotonic() - t
            seconds.append(elapsed)
            pair_count += sum(trace["pairs"] for trace in traces)
            count += 1
            row = {"query_id": record["query_id"], "sample_id": record["sample_id"], "status": "ok",
                   "search_seconds": elapsed, "retrieved": items, "reranker_trace": traces}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({"stage": "rerank", "completed": count, "total": manifest["question_count"],
                              "pairs": pair_count, "seconds": round(elapsed, 3)}), flush=True)
    if count != manifest["question_count"]:
        raise ValueError("Incomplete query set")
    config = dict(manifest["source_configuration"])
    config["reranker"] = {"model": MODEL, "revision": REVISION, "dtype": "float32", "device": args.device,
                          "batch_size": args.batch_size, "max_pair_tokens": args.max_pair_tokens,
                          "query_window": args.query_window, "overlap": args.overlap,
                          "max_pairs_per_item": args.max_pairs_per_item, "aggregate": "max over every query/document window pair",
                          "tie_break": "original candidate rank", "tf32": False, "attention": "eager",
                          "text": manifest["ranker_input"]}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-rerank-" + secrets.token_hex(4)
    summary = {"status": "public_dev_reranker_proxy_no_answer_judge", "run_id": run_id,
               "ingestion_run_id": manifest["ingestion_run_id"], "source_run_id": manifest["source_run_id"],
               "model_label": "BM25/E5-small + BGE-reranker-v2-m3 FP32 window-max",
               "dataset": manifest["dataset"], "split": "dev", "question_count": count, "configuration": config,
               "add_requests": 0, "new_search_requests": 0, "paid_requests": 0,
               "candidate_set_and_content_unchanged": True, "reranker_pairs": pair_count,
               "python_version": platform.python_version(), "package_versions": engine.versions,
               "model_manifest": engine.manifest, "model_manifest_sha256": file_hash(Path(args.model_dir) / "model-manifest.json"),
               "source_input_sha256": manifest["input_sha256"], "source_retrieval_sha256": manifest["source_retrieval_sha256"],
               "helper_sha256": file_hash(__file__), "load_seconds": load_seconds,
               "rerank_seconds": sum(seconds), "per_query_seconds": seconds,
               "device_name": engine.torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU",
               "peak_cuda_allocated_bytes": engine.torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else None,
               "limitations": ["Frozen-response reranking only; no Add/full retrieval or production deployment.",
                               "Evidence ordering/coverage is a proxy, not Answer/Judge accuracy or official AML score.",
                               "Internal GPU research runtime is not part of the public serving path or a CPU capacity claim."]}
    write_json(output / "summary.json", summary)
    return {"run_id": run_id, "questions": count, "pairs": pair_count, "rerank_seconds": sum(seconds)}


def score(args):
    # This stage reads gold only for evaluation AFTER model ranking has finished.
    from eval_retrieval import GRADER_VERSION, KS, aggregate, evidence_hits, requests_for_history
    from eval_prepare import LEGACY_SOURCE_MAPPING_VERSION
    from eval_token_budget import prepared_inputs
    original, source_rows = load_source(args.source_run)
    folder = Path(args.candidate)
    summary = json.loads((folder / "summary.json").read_text())
    if summary.get("split") != "dev" or summary["source_retrieval_sha256"] != file_hash(Path(args.source_run) / "retrieval.jsonl"):
        raise ValueError("Candidate provenance mismatch")
    records = list(jsonl(folder / "retrieval.jsonl"))
    baseline = {r["query_id"]: r for r in source_rows}
    if len(records) != summary["question_count"] or len({r["query_id"] for r in records}) != len(records):
        raise ValueError("Incomplete or duplicate candidate records")
    for row in records:
        before = baseline[row["query_id"]]
        if row["sample_id"] != before["sample_id"] or row.get("status") != "ok":
            raise ValueError("Candidate query/source/status mismatch")
        assert_permutation(before["retrieved"], row["retrieved"])
    labels, queries, histories, frequencies = prepared_inputs(Path(args.data_root), summary["dataset"], [(folder, summary, records)])
    mapping = summary["configuration"].get("source_mapping_version", LEGACY_SOURCE_MAPPING_VERSION)
    sources = {sid: requests_for_history(history, summary["ingestion_run_id"], mapping)[2:] for sid, history in histories.items()}
    output = new_directory(args.output)
    scored = []
    for row in records:
        sid, qid = row["sample_id"], row["query_id"]
        source_map, id_map = sources[sid]
        metrics = {str(k): {**evidence_hits(row["retrieved"][:k], labels[qid], source_map, id_map, frequencies[sid]),
                           "returned_items": len(row["retrieved"][:k]),
                           "returned_characters": sum(len(x["content"]) for x in row["retrieved"][:k])} for k in KS}
        scored.append({**row, "category": labels[qid]["category"], "original_query_id": labels[qid]["original_query_id"], "metrics": metrics})
    summary = {**summary, "grader_version": GRADER_VERSION,
               "metrics": {str(k): aggregate(scored, k, len(scored)) for k in KS},
               "unscored_candidate_retrieval_sha256": file_hash(folder / "retrieval.jsonl"),
               "unscored_candidate_summary_sha256": file_hash(folder / "summary.json"),
               "grading_code_sha256": {name: file_hash(Path(__file__).parent / name) for name in
                                        ("eval_rerank.py", "eval_retrieval.py", "eval_prepare.py", "eval_token_budget.py")}}
    write_json(output / "summary.json", summary)
    (output / "retrieval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored))
    return {"run_id": summary["run_id"], "questions": len(scored), "metrics": summary["metrics"]}


def functional(args):
    engine = Engine(args)
    query = "What is the capital of France?"
    items = [{"id": "tail", "content": "Leaves fall from trees. " * 800 + "Paris is the capital of France.", "score": 0.01},
             {"id": "other", "content": "Tokyo is the capital of Japan.", "score": 0.02}]
    a, traces = engine.rerank(query, items)
    b, _ = engine.rerank(query, items)
    assert_permutation(items, a)
    assert_permutation(items, b)
    if a != b or a[0]["id"] != "tail" or traces[0]["pairs"] <= 1:
        raise ValueError("Functional long-tail / deterministic rerank check failed")
    result = {"status": "synthetic_functional_only_not_benchmark", "model": MODEL, "revision": REVISION,
              "exact_repeated_output": True, "long_document_pairs": traces[0]["pairs"],
              "source_text_unchanged": True, "package_versions": engine.versions, "scores": {x["id"]: x["score"] for x in a}}
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-queries", type=int, default=0)
    d = commands.add_parser("download")
    d.add_argument("--model-dir", type=Path, required=True)
    for name in ("rank", "functional"):
        p = commands.add_parser(name)
        p.add_argument("--model-dir", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "rank":
            p.add_argument("--input-dir", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--max-pair-tokens", type=int, default=512)
        p.add_argument("--query-window", type=int, default=128)
        p.add_argument("--overlap", type=int, default=64)
        p.add_argument("--max-pairs-per-item", type=int, default=4096)
    p = commands.add_parser("score")
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "max_queries", 0) < 0:
        parser.error("max-queries cannot be negative")
    if args.command in ("rank", "functional"):
        if (args.batch_size < 1 or not 16 <= args.query_window <= 256 or args.max_pair_tokens != 512
                or not 0 <= args.overlap < args.query_window or args.max_pairs_per_item < 1):
            parser.error("Invalid declared reranker capacity/window parameters")
    print(json.dumps(globals()[{"download": "download_model"}.get(args.command, args.command)](args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
