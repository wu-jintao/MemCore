#!/usr/bin/env python3
"""Audit a completed PUBLIC dev semantic index without loading a model or calling APIs.

Checks fixed model files and the actual stored fingerprint/dimension. It does not
read questions, answers or message text, and never rewrites old run artifacts.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import struct
import sys

from eval_prepare import DEFAULT_ROOT, HERE, file_hash

sys.path.insert(0, str(HERE.parent))
from semantic import _fingerprint, _model_digest

REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def audit(model_dir, run_directories, data_root):
    manifest = json.loads((model_dir / "download-manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["model"] != "intfloat/multilingual-e5-small":
        raise ValueError("This audit expects the explicitly pinned public e5-small revision")
    for entry in manifest["files"]:
        path = model_dir / entry["path"]
        if path.stat().st_size != entry["size"] or file_hash(path) != entry["sha256"]:
            raise ValueError("Model file differs from the downloaded manifest: " + entry["path"])
    _, weights_hash = _model_digest(model_dir)
    settings = {"provider": "local-cpu", "model": manifest["model"], "weights": weights_hash,
                "dimension": 384, "normalization": "l2", "document_prefix": "passage: ", "query_prefix": "query: ",
                "segmentation": "source-offset-tokens-v1", "segment_tokens": 384, "overlap_tokens": 64,
                "max_seq_length": 512, "long_query": "normalized-segment-mean-v1"}
    expected = _fingerprint(settings)
    result = {"status": "completed_public_dev_index_audit_not_performance_or_aml_score",
              "model_revision": REVISION, "model_files_verified": len(manifest["files"]),
              "model_manifest_sha256": file_hash(model_dir / "download-manifest.json"),
              "semantic_source_sha256": file_hash(HERE.parent / "semantic.py"),
              "expected_index_settings": settings, "expected_fingerprint": expected, "runs": []}
    for folder in run_directories:
        summary = json.loads((folder / "summary.json").read_text())
        if summary["split"] != "dev" or summary["configuration"]["embedding_provider"] != "local":
            raise ValueError("Only completed local semantic public dev runs are audited")
        if summary["configuration"]["source_code_sha256"]["semantic.py"] != result["semantic_source_sha256"]:
            raise ValueError("Semantic source differs from the run; audit its saved version instead")
        db = data_root / "runs" / summary["run_id"] / "memory.sqlite3"
        wal = db.with_name(db.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Run database still has WAL data; wait for its service to close cleanly")
        # Immutable is safe only for the completed, WAL-free local snapshot above.
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
            groups = connection.execute("SELECT fingerprint,dimension,length(vector),COUNT(*) FROM vectors GROUP BY fingerprint,dimension,length(vector)").fetchall()
            if not groups or any(f != expected or dimension != 384 or size != 384 * 4 for f, dimension, size, _ in groups):
                raise ValueError("Stored index settings or float32 dimension differ from the declared e5 configuration")
            messages = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            sample = connection.execute("SELECT vector FROM vectors ORDER BY user_id,message_id,segment_index LIMIT 16").fetchall()
            deviations = [abs(sum(x * x for x in struct.unpack("<384f", row[0])) - 1) for row in sample]
            if max(deviations) > 1e-5:
                raise ValueError("Stored sample vector norms do not match L2 normalization")
            result["runs"].append({"run_id": summary["run_id"], "model_label": summary["model_label"],
                                   "fingerprint_matches": True, "message_count": messages,
                                   "vector_segment_count": sum(n for _, _, _, n in groups),
                                   "normalization_sample_count": len(sample),
                                   "max_sample_squared_norm_deviation": max(deviations)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model_dir, args.runs, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "verified_model_files": result["model_files_verified"],
                      "audited_dev_runs": len(result["runs"])}))


if __name__ == "__main__":
    main()
