#!/usr/bin/env python3
"""Offline synthetic checks only; never an actual public-dev Search experiment."""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval_prepare import SOURCE_MAPPING_VERSION, file_hash, json_line
from eval_retrieval import requests_for_history
from eval_research import (backup_source, database_identity, fuse_ast_sha256, inspect_source, safe_environment,
                           search_ast_sha256, search_records, validate_candidate)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import MemoryStore


class SearchResearchTests(unittest.TestCase):
    def fixture(self, base):
        root = base / "data"
        run = base / "run"
        snapshots = run / "source_snapshots"
        snapshots.mkdir(parents=True)
        starter = Path(__file__).resolve().parent.parent
        for name in ("server.py", "semantic.py", "competition/eval_prepare.py", "competition/eval_retrieval.py"):
            shutil.copyfile(starter / name, snapshots / Path(name).name)
        history = {"sample_id": "synthetic-history", "sessions": [{"session_id": "synthetic-session", "messages": [
            {"role": "user", "content": "Original history source alpha.", "timestamp": 0}]}]}
        user, payloads, sources, ids = requests_for_history(history, "original-ingestion")
        database = root / "runs" / "original-ingestion" / "memory.sqlite3"
        store = MemoryStore(database, context_radius=1)
        for payload in payloads:
            store.add(payload)  # Synthetic local fixture setup, never a dataset Add/HTTP/model request.
        folder = root / "prepared" / "synthetic-public-fixture" / "dev"
        folder.mkdir(parents=True)
        rows = {"histories": history,
                "queries": {"sample_id": history["sample_id"], "query_id": "synthetic-query",
                            "query": "FUTURE_QUESTION_SENTINEL"},
                "gold": {"sample_id": history["sample_id"], "query_id": "synthetic-query",
                         "original_query_id": "synthetic", "category": "synthetic",
                         "answer": "GOLD_ANSWER_SENTINEL",
                         "gold_units": [{"unit_id": "synthetic-unit", "session_id": "synthetic-session",
                                         "message_position": 0, "text": "Original history source alpha.",
                                         "source_message_content": "Original history source alpha."}],
                         "gold_session_ids": ["synthetic-session"]}}
        for name, row in rows.items():
            (folder / (name + ".jsonl")).write_text(json_line(row))
        specification = {"histories": 1, "questions": 1, "messages": 1, "gold_turn_annotations": 1,
                         "files": {name: {"size": (folder / (name + ".jsonl")).stat().st_size,
                                           "sha256": file_hash(folder / (name + ".jsonl"))} for name in rows}}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        code = {name: file_hash(snapshots / name) for name in ("semantic.py", "eval_prepare.py", "eval_retrieval.py")}
        code.update({"server.py_at_start": file_hash(snapshots / "server.py"),
                     "server.py_at_end": file_hash(snapshots / "server.py")})
        summary = {"status": "public_proxy_not_official_aml", "split": "dev", "dataset": "synthetic-public-fixture",
                   "run_id": "original-ingestion", "model_label": "synthetic-fixture",
                   "question_count": 1, "history_count": 1, "add_request_count": 1,
                   "ingestion_failed_histories": 0, "server_source_changed_during_run": False,
                   "configuration": {"base_url": "http://127.0.0.1:" + str(port), "launch_local": True,
                                     "embedding_provider": "disabled", "forced_local_backend": "lexical",
                                     "top_k": 100, "character_budget": 0, "search_concurrency": 1,
                                     "context_radius": 1, "semantic_weight": 1,
                                     "source_mapping_version": SOURCE_MAPPING_VERSION,
                                     "prepared_data": specification, "source_code_sha256": code}}
        (run / "summary.json").write_text(json.dumps(summary))
        old_response = {"query_id": "synthetic-query", "sample_id": "synthetic-history", "status": "ok",
                        "retrieved": [{"id": "old-response-id", "content": "OLD_RESPONSE_NEVER_REPLAYED_SENTINEL"}]}
        (run / "retrieval.jsonl").write_text(json_line(old_response))
        return run, root, database, history, user, payloads, ids

    def test_sqlite_backup_preserves_ids_corpus_and_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, root, database, _, _, _, _ = self.fixture(base)
            before = file_hash(database)
            source = inspect_source(run, root)
            target = base / "new-copy" / "memory.sqlite3"
            lineage = backup_source(source, target)
            self.assertTrue(lineage["source_database_unchanged"])
            self.assertEqual(file_hash(database), before)
            self.assertEqual(database_identity(target), source["metadata"]["actual_database_identity"])
            self.assertEqual(database_identity(target)["table_counts"]["requests"], 1)
            with self.assertRaisesRegex(ValueError, "independent"):
                backup_source(source, target)

    def test_fresh_search_uses_original_ingestion_namespace_without_add_or_response_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            run, root, _, _, user, payloads, ids = self.fixture(Path(directory))
            source = inspect_source(run, root)
            calls = []
            header = {"session_id": payloads[0]["session_id"], "request_id": payloads[0]["request_id"],
                      "message_index": 0}
            content = "[source " + json.dumps(header) + "]\n" + payloads[0]["messages"][0]["content"]
            class FreshClient:
                def request(self, route, payload):
                    calls.append((route, payload))
                    return 200, {"data": [{"id": next(iter(ids)), "content": content}]}, .01, 100
            rows = list(search_records(source, FreshClient()))
            self.assertEqual([route for route, _ in calls], ["/search"])
            self.assertEqual(calls[0][1]["user_id"], user)
            self.assertIn("original-ingestion", calls[0][1]["user_id"])
            self.assertNotIn("GOLD_ANSWER_SENTINEL", json.dumps(calls))
            self.assertEqual(rows[0]["retrieved"][0]["content"], content)
            self.assertNotIn("OLD_RESPONSE_NEVER_REPLAYED_SENTINEL", json.dumps(rows))
            self.assertEqual(rows[0]["metrics"]["100"]["source_verified_hit_turns"], 1)

    def test_validation_heldout_and_partial_sources_are_rejected_before_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for split in ("validation", "heldout"):
                run = base / split
                run.mkdir()
                (run / "summary.json").write_text(json.dumps({"configuration": {}, "split": split}))
                with self.assertRaisesRegex(ValueError, "PUBLIC dev"):
                    inspect_source(run, base / "missing-protected-data")
            run, root, _, _, _, _, _ = self.fixture(base / "partial")
            summary = json.loads((run / "summary.json").read_text())
            summary["ingestion_failed_histories"] = 1
            (run / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "complete original ingestion"):
                inspect_source(run, root)

    def test_candidate_only_character_cap_change_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run, root, _, _, _, _, _ = self.fixture(base)
            source = inspect_source(run, root)
            candidate = base / "candidate" / "server.py"
            candidate.parent.mkdir()
            text = (run / "source_snapshots/server.py").read_text()
            candidate.write_text(text.replace("MAX_EVIDENCE_CHARACTERS = 250_000", "MAX_EVIDENCE_CHARACTERS = 500_000"))
            shutil.copyfile(run / "source_snapshots/semantic.py", candidate.parent / "semantic.py")
            approved = validate_candidate(candidate, source)
            self.assertEqual(approved["service_whole_item_character_cap"], 500000)
            candidate.write_text(candidate.read_text().replace("MAX_QUERY_TERMS = 64", "MAX_QUERY_TERMS = 65"))
            with self.assertRaisesRegex(ValueError, "only MAX_EVIDENCE"):
                validate_candidate(candidate, source)

    def approved_method_fixture(self, base):
        """Small generic static review fixture, not a real RRF experiment."""
        run, root, _, _, _, _, _ = self.fixture(base)
        source = inspect_source(run, root)
        candidate = base / "approved-candidate" / "server.py"
        candidate.parent.mkdir()
        text = (run / "source_snapshots/server.py").read_text()
        # Change only this method's AST without depending on its real ranking
        # implementation; the marker is never executed by these static checks.
        marker = '    def _fuse(self, rankings):\n        reviewed_fixture_marker = "generic-static-review-fixture"\n'
        original = "    def _fuse(self, rankings):\n"
        self.assertEqual(text.count(original), 1)
        candidate.write_text(text.replace(original, marker).replace(
            "MAX_EVIDENCE_CHARACTERS = 250_000", "MAX_EVIDENCE_CHARACTERS = 500_000"))
        shutil.copyfile(run / "source_snapshots/semantic.py", candidate.parent / "semantic.py")
        return candidate, source, fuse_ast_sha256(candidate)

    def test_explicit_exact_method_hash_allows_one_reviewed_method_and_records_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, approved_hash = self.approved_method_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "only MAX_EVIDENCE"):
                validate_candidate(candidate, source)
            info = validate_candidate(candidate, source, approved_hash)
            self.assertEqual(info["server_ast_change_policy"], "cap_plus_explicitly_approved_fuse_v1")
            self.assertEqual(info["approved_fuse_ast_sha256"], approved_hash)
            self.assertEqual(info["candidate_fuse_ast_sha256"], approved_hash)
            self.assertNotEqual(info["original_fuse_ast_sha256"], approved_hash)
            self.assertEqual(info["service_whole_item_character_cap"], 500000)

    def test_unapproved_method_hash_and_malformed_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, approved_hash = self.approved_method_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "explicitly approved hash"):
                validate_candidate(candidate, source, "0" * 64)
            with self.assertRaisesRegex(ValueError, "exact lowercase SHA256"):
                validate_candidate(candidate, source, "not-a-reviewed-hash")
            candidate.write_text(candidate.read_text().replace(
                '"generic-static-review-fixture"', '"different-unreviewed-method"'))
            with self.assertRaisesRegex(ValueError, "explicitly approved hash"):
                validate_candidate(candidate, source, approved_hash)

    def test_reviewed_method_does_not_allow_add_sql_import_or_handler_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, approved_hash = self.approved_method_fixture(Path(directory))
            approved_text = candidate.read_text()
            changes = {
                "add": ("    def add(self, payload):", "    def add(self, payload, unapproved=False):"),
                "scoped_sql": ("WHERE user_id=? AND id IN (", "WHERE 1=? AND id IN ("),
                "import": ("import sqlite3\n", "import sqlite3\nimport subprocess\n"),
                "handler": ("        def do_POST(self):", "        def do_POST(self, unapproved=False):"),
                "query_limit": ("MAX_QUERY_TERMS = 64", "MAX_QUERY_TERMS = 65"),
            }
            for name, (before, after) in changes.items():
                with self.subTest(change=name):
                    self.assertIn(before, approved_text)
                    candidate.write_text(approved_text.replace(before, after))
                    with self.assertRaisesRegex(ValueError, "other server logic differs"):
                        validate_candidate(candidate, source, approved_hash)

    def test_reviewed_method_cannot_allow_changed_semantic_code(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, approved_hash = self.approved_method_fixture(Path(directory))
            semantic = candidate.parent / "semantic.py"
            semantic.write_text(semantic.read_text() + "\nUNAPPROVED_MODEL_SETTING = 1\n")
            with self.assertRaisesRegex(ValueError, "semantic code differs"):
                validate_candidate(candidate, source, approved_hash)

    def test_declared_disabled_backend_must_match_actual_stored_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            run, root, database, _, user, _, ids = self.fixture(Path(directory))
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("INSERT INTO vectors VALUES (?,?,?,?,?,?)",
                                   (user, next(iter(ids)), 0, "inconsistent-fingerprint", 384, b"\0" * 1536))
                connection.commit()
            with self.assertRaisesRegex(ValueError, "Disabled backend"):
                inspect_source(run, root)

    def search_method_fixture(self, base):
        run, root, _, _, _, _, _ = self.fixture(base)
        source = inspect_source(run, root)
        candidate = base / "reviewed-search" / "server.py"
        candidate.parent.mkdir()
        text = (run / "source_snapshots/server.py").read_text()
        marker = '    def search(self, payload):\n        reviewed_search_fixture = "synthetic-evidence-assembly"\n'
        candidate.write_text(text.replace("    def search(self, payload):\n", marker))
        shutil.copyfile(run / "source_snapshots/semantic.py", candidate.parent / "semantic.py")
        return candidate, source, search_ast_sha256(candidate)

    def test_search_change_requires_its_own_exact_hash_and_separate_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, approved = self.search_method_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "only MAX_EVIDENCE"):
                validate_candidate(candidate, source)
            with self.assertRaisesRegex(ValueError, "other server logic differs"):
                validate_candidate(candidate, source, fuse_ast_sha256(candidate))
            info = validate_candidate(candidate, source, approved_search_ast_sha256=approved)
            self.assertEqual(info["server_ast_change_policy"], "cap_plus_explicitly_reviewed_search_v1")
            self.assertEqual(info["approved_search_ast_sha256"], approved)
            self.assertNotEqual(info["original_search_ast_sha256"], approved)
            with self.assertRaisesRegex(ValueError, "explicitly approved hash"):
                validate_candidate(candidate, source, approved_search_ast_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "exact lowercase SHA256"):
                validate_candidate(candidate, source, approved_search_ast_sha256="not-a-hash")

    def test_reviewed_search_does_not_allow_other_methods_or_changed_approved_body(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, approved = self.search_method_fixture(Path(directory))
            text = candidate.read_text()
            for before, after in (
                ("    def add(self, payload):", "    def add(self, payload, changed=False):"),
                ("    def _neighbors(self, connection, document):", "    def _neighbors(self, connection, document, changed=False):"),
                ("import sqlite3\n", "import sqlite3\nimport subprocess\n"),
                ("        def do_POST(self):", "        def do_POST(self, changed=False):"),
            ):
                with self.subTest(before=before):
                    self.assertIn(before, text)
                    candidate.write_text(text.replace(before, after))
                    with self.assertRaisesRegex(ValueError, "other server logic differs"):
                        validate_candidate(candidate, source, approved_search_ast_sha256=approved)
            candidate.write_text(text.replace("synthetic-evidence-assembly", "unreviewed-body"))
            with self.assertRaisesRegex(ValueError, "explicitly approved hash"):
                validate_candidate(candidate, source, approved_search_ast_sha256=approved)
            candidate.write_text(text)
            semantic = candidate.parent / "semantic.py"
            semantic.write_text(semantic.read_text() + "\nUNREVIEWED_VECTOR_SETTING = 1\n")
            with self.assertRaisesRegex(ValueError, "semantic code differs"):
                validate_candidate(candidate, source, approved_search_ast_sha256=approved)

    def test_combined_search_and_fuse_policy_requires_both_exact_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, source, search_hash = self.search_method_fixture(Path(directory))
            candidate.write_text(candidate.read_text().replace(
                "    def _fuse(self, rankings):\n", "    def _fuse(self, rankings):\n        reviewed_fuse_fixture = True\n"))
            with self.assertRaisesRegex(ValueError, "other server logic differs"):
                validate_candidate(candidate, source, approved_search_ast_sha256=search_hash)
            info = validate_candidate(candidate, source, fuse_ast_sha256(candidate), search_hash)
            self.assertEqual(info["server_ast_change_policy"], "cap_plus_explicitly_reviewed_search_and_fuse_v1")
            self.assertEqual(info["candidate_search_ast_sha256"], search_hash)

    def test_inherited_paid_keys_proxy_and_embedding_settings_are_not_passed(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "SECRET_SENTINEL", "MEMORY_API_TOKEN": "SECRET_SENTINEL",
                                     "HTTP_PROXY": "http://wrong", "PYTHONPATH": "/untrusted",
                                     "MEMORY_EMBEDDING_PROVIDER": "http", "MEMORY_EMBEDDING_THREADS": "99"}):
            env = safe_environment("local", Path("/already-downloaded-model"))
        self.assertNotIn("SECRET_SENTINEL", env.values())
        self.assertNotIn("HTTP_PROXY", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["MEMORY_EMBEDDING_PROVIDER"], "local")
        self.assertEqual(env["MEMORY_EMBEDDING_THREADS"], "2")
        self.assertEqual(env["MEMORY_EMBEDDING_ALLOW_HTTP"], "0")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
