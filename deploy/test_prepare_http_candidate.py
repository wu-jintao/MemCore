"""Pure local construction checks: no real credentials, systemd or networking."""

from argparse import Namespace
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from deploy import prepare_http_candidate as candidate


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.layout = candidate.Layout(self.root / "opt", self.root / "data",
                                       self.root / "etc", self.root / "units")
        for path in (self.layout.base, self.layout.data, self.layout.config, self.layout.units):
            path.mkdir()
        old = self.layout.base / "releases" / "old"
        (old / "app").mkdir(parents=True)
        (old / "venv" / "bin").mkdir(parents=True)
        (old / "venv" / "bin" / "python").write_text("synthetic interpreter fixture")
        (old / "app" / "server.py").write_text("old production source fixture")
        (self.layout.base / "app").symlink_to(old / "app")
        (self.layout.base / "venv").symlink_to(old / "venv")
        (self.layout.config / "runtime.env").write_text(
            "MEMORY_API_TOKEN=synthetic-production-token\nHF_HOME=synthetic-production-cache\n")
        (self.layout.config / "runtime.env").chmod(0o600)
        (self.layout.units / "aml-memory.service").write_text("synthetic production unit")
        (self.layout.data / "db").mkdir()
        for name in ("memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm"):
            (self.layout.data / "db" / name).write_text("synthetic preserved " + name)
        self.stage = self.root / "stage"
        self.stage.mkdir()
        hashes = {}
        for name in candidate.SOURCE_NAMES:
            raw = ("# reviewed synthetic " + name + "\n").encode()
            (self.stage / name).write_bytes(raw)
            hashes[name] = candidate.digest_bytes(raw)
        manifest = self.stage / "manifest.json"
        manifest.write_text(json.dumps({"source_hashes": hashes}))
        model = self.stage / "model.env"
        values = dict(candidate.FIXED_MODEL_VALUES, MEMORY_EMBEDDING_API_KEY="synthetic-model-key",
                      MEMORY_EMBEDDING_URL="https://example.invalid/compatible-mode/v1/embeddings",
                      MEMORY_EMBEDDING_HTTP_TIMEOUT="30", MEMORY_EMBEDDING_OPERATION_TIMEOUT="600",
                      MEMORY_EMBEDDING_QUEUE_TIMEOUT="120", MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES="1536")
        self.model_raw = "".join(key + "=" + value + "\n" for key, value in values.items()).encode()
        model.write_bytes(self.model_raw)
        model.chmod(0o600)
        self.args = Namespace(run_id="v4-local-test", source_stage=self.stage,
                              model_env_stage=model, manifest=manifest,
                              manifest_sha256=candidate.digest_bytes(manifest.read_bytes()))
        self.commands = []
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(candidate.os.path, "ismount", return_value=True).start()
        mock.patch.object(candidate.os, "chown").start()
        mock.patch.object(candidate.pwd, "getpwnam", return_value=SimpleNamespace(
            pw_uid=os.getuid(), pw_gid=os.getgid())).start()
        # These temporary files belong to the test runner, not to production root.
        mock.patch.object(candidate, "root_private_file", side_effect=lambda path:
                          not path.is_symlink() and path.is_file() and
                          not path.stat().st_mode & 0o077).start()
        mock.patch.object(candidate, "command", side_effect=self.record_command).start()

    def record_command(self, arguments):
        self.commands.append(arguments)
        return "active" if arguments[:2] == ["systemctl", "show"] else ""

    def production_snapshot(self):
        paths = [self.layout.config / "runtime.env", self.layout.units / "aml-memory.service",
                 self.layout.base / "app" / "server.py", self.layout.base / "venv" / "bin/python"]
        paths.extend((self.layout.data / "db").iterdir())
        return ({str(path): (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
                 for path in paths},
                {name: os.readlink(self.layout.base / name) for name in ("app", "venv")})

    def test_prepare_preserves_production_and_private_input(self):
        before = self.production_snapshot()
        original_read_bytes = Path.read_bytes
        runtime = self.layout.config / "runtime.env"
        def guarded_read(path):
            if path == runtime:
                raise AssertionError("Preparation must not read production credentials")
            return original_read_bytes(path)
        with mock.patch.object(Path, "read_bytes", guarded_read):
            result = candidate.prepare(self.args, self.layout)
        self.assertEqual(before, self.production_snapshot())
        paths = self.layout.candidate(self.args.run_id)
        self.assertEqual((paths["config"] / "model.env").read_bytes(), self.model_raw)
        self.assertEqual((paths["config"] / "model.env").stat().st_mode & 0o777, 0o600)
        self.assertEqual(list((paths["data"] / "db").iterdir()), [])
        self.assertEqual(set(path.name for path in (paths["release"] / "app").iterdir()),
                         set(candidate.SOURCE_NAMES))
        self.assertFalse((paths["release"] / "venv").exists())
        self.assertFalse(result["production_modified"])
        self.assertFalse(result["started"])
        self.assertNotIn("synthetic-model-key", json.dumps(result))
        unit = paths["unit"].read_text()
        self.assertNotIn("synthetic-model-key", unit)
        self.assertNotIn("synthetic-production-token", unit)
        self.assertIn("--host 127.0.0.1 --port 18081", unit)
        self.assertIn("MemoryMax=12G", unit)
        self.assertIn("Restart=no", unit)
        self.assertIn("ReadOnlyPaths=", unit)
        self.assertLess(unit.index("runtime.env"), unit.index("model.env"))
        self.assertLess(unit.index("model.env"), unit.index("settings.env"))
        self.assertIn(str(paths["data"] / "cache" / "huggingface"),
                      (paths["config"] / "settings.env").read_text())
        self.assertEqual(self.commands, [["systemd-analyze", "verify", str(paths["unit"])]])

    def test_rejects_manifest_or_source_drift_before_writing(self):
        self.args.manifest_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            candidate.prepare(self.args, self.layout)
        self.args.manifest_sha256 = candidate.digest_bytes(self.args.manifest.read_bytes())
        (self.stage / "server.py").write_text("changed")
        with self.assertRaises(ValueError):
            candidate.prepare(self.args, self.layout)
        self.assertFalse(self.layout.candidate(self.args.run_id)["release"].exists())
        self.assertEqual(self.commands, [])

    def test_rejects_wrong_model_and_privilege_overrides(self):
        for invalid in (self.model_raw.replace(b"DIMENSION=2048", b"DIMENSION=1024"),
                        self.model_raw.replace(b"HTTP_RETRIES=0", b"HTTP_RETRIES=1"),
                        self.model_raw + b"MEMORY_API_TOKEN=attacker\n",
                        self.model_raw + b"PYTHONPATH=/tmp\n",
                        self.model_raw.replace(b"TIMEOUT=600", b"TIMEOUT=nan"),
                        self.model_raw.replace(b"https://example.invalid", b"http://127.0.0.1")):
            with self.subTest(invalid=invalid.splitlines()[-1].split(b"=", 1)[0]):
                with self.assertRaises(ValueError):
                    candidate.parse_model_environment(invalid)

    def test_http_queue_900_is_accepted_with_unchanged_operation_600(self):
        raw = self.model_raw.replace(b"MEMORY_EMBEDDING_QUEUE_TIMEOUT=120", b"MEMORY_EMBEDDING_QUEUE_TIMEOUT=900")
        values = candidate.parse_model_environment(raw)
        self.assertEqual(values["MEMORY_EMBEDDING_QUEUE_TIMEOUT"], "900")
        self.assertEqual(values["MEMORY_EMBEDDING_OPERATION_TIMEOUT"], "600")
        for timeout in (b"900.0001", b"901", b"inf", b"-inf", b"nan", b"0", b"-1"):
            invalid = raw.replace(b"MEMORY_EMBEDDING_QUEUE_TIMEOUT=900", b"MEMORY_EMBEDDING_QUEUE_TIMEOUT=" + timeout)
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                candidate.parse_model_environment(invalid)

    def test_run_ids_cannot_escape_candidate_paths(self):
        for run_id in ("../production", "a/b", "a.service\nExecStart=x", "", "a" * 65):
            with self.assertRaises(ValueError):
                self.layout.candidate(run_id)

    def test_empty_unquoted_prefixes_are_copied_but_other_empty_values_fail(self):
        raw = self.model_raw + b"MEMORY_EMBEDDING_QUERY_PREFIX=\nMEMORY_EMBEDDING_DOCUMENT_PREFIX=\n"
        values = candidate.parse_model_environment(raw)
        self.assertEqual(values["MEMORY_EMBEDDING_QUERY_PREFIX"], "")
        self.assertEqual(values["MEMORY_EMBEDDING_DOCUMENT_PREFIX"], "")
        self.args.model_env_stage.write_bytes(raw)
        candidate.prepare(self.args, self.layout)
        self.assertEqual((self.layout.candidate(self.args.run_id)["config"] / "model.env").read_bytes(), raw)
        for key, value in ((b"MEMORY_EMBEDDING_API_KEY", b"synthetic-model-key"),
                           (b"MEMORY_EMBEDDING_QUEUE_TIMEOUT", b"120")):
            for replacement in (b"", b'""'):
                with self.assertRaises(ValueError):
                    candidate.parse_model_environment(self.model_raw.replace(
                        key + b"=" + value, key + b"=" + replacement))

    def test_start_stop_status_only_target_candidate_and_refuse_changed_secret(self):
        candidate.prepare(self.args, self.layout)
        before = self.production_snapshot()
        self.commands.clear()
        paths = self.layout.candidate(self.args.run_id)
        unit = paths["unit"].name
        self.assertEqual(candidate.manage("start", self.args.run_id, self.layout)["active_state"], "active")
        candidate.manage("stop", self.args.run_id, self.layout)
        candidate.manage("status", self.args.run_id, self.layout)
        self.assertEqual(self.commands, [
            ["systemctl", "daemon-reload"], ["systemctl", "start", unit],
            ["systemctl", "show", unit, "--property=ActiveState", "--value"],
            ["systemctl", "stop", unit],
            ["systemctl", "show", unit, "--property=ActiveState", "--value"],
            ["systemctl", "show", unit, "--property=ActiveState", "--value"],
        ])
        self.assertEqual(before, self.production_snapshot())
        (paths["config"] / "model.env").write_bytes(self.model_raw.replace(
            b"synthetic-model-key", b"changed-synthetic-model-key"))
        self.commands.clear()
        with self.assertRaises(ValueError):
            candidate.manage("start", self.args.run_id, self.layout)
        self.assertEqual(self.commands, [])

    def test_failed_unit_verification_cleans_only_new_candidate(self):
        before = self.production_snapshot()
        with mock.patch.object(candidate, "command", side_effect=RuntimeError("synthetic verification failure")):
            with self.assertRaises(RuntimeError):
                candidate.prepare(self.args, self.layout)
        self.assertEqual(before, self.production_snapshot())
        for path in self.layout.candidate(self.args.run_id).values():
            self.assertFalse(path.exists())

    def test_private_umask_keeps_candidate_data_parent_traversable(self):
        previous = os.umask(0o077)
        try:
            candidate.prepare(self.args, self.layout)
        finally:
            os.umask(previous)
        paths = self.layout.candidate(self.args.run_id)
        self.assertEqual(paths["data"].parent.stat().st_mode & 0o777, 0o755)
        self.assertEqual(paths["data"].stat().st_mode & 0o777, 0o700)
        self.assertEqual(paths["config"].parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual((paths["release"] / "app").stat().st_mode & 0o777, 0o755)

    def test_systemd_single_paths_are_unquoted_and_exec_argv_stays_quoted(self):
        paths = self.layout.candidate(self.args.run_id)
        text = candidate.unit_text(paths, self.layout, (self.layout.base / "venv").resolve())
        lines = text.splitlines()
        self.assertIn("WorkingDirectory=" + str(paths["release"] / "app"), lines)
        for line in lines:
            if line.startswith(("WorkingDirectory=", "EnvironmentFile=")):
                self.assertNotIn('"', line)
                self.assertTrue(line.split("=", 1)[1].startswith("/"))
            if line.startswith("ExecStart="):
                self.assertTrue(line.startswith('ExecStart="/'))
                self.assertIn('" --host 127.0.0.1 --port 18081 --db "/', line)
        self.assertEqual(candidate.systemd_single_path(Path("/tmp/name%with-specifier")),
                         "/tmp/name%%with-specifier")
        with self.assertRaises(ValueError):
            candidate.systemd_single_path(Path("/tmp/path with space"))

    def test_refuses_candidate_parent_symlink_before_writing(self):
        (self.layout.data / "http-candidates").symlink_to(self.layout.data / "db")
        before = self.production_snapshot()
        with self.assertRaises(ValueError):
            candidate.prepare(self.args, self.layout)
        self.assertEqual(before, self.production_snapshot())
        self.assertFalse(self.layout.candidate(self.args.run_id)["release"].exists())

    def test_existing_private_data_namespace_is_preserved_and_refused(self):
        parent = self.layout.data / "http-candidates"
        parent.mkdir(mode=0o700)
        with self.assertRaises(ValueError):
            candidate.prepare(self.args, self.layout)
        self.assertEqual(parent.stat().st_mode & 0o777, 0o700)
        self.assertFalse(self.layout.candidate(self.args.run_id)["release"].exists())

    def test_cli_failure_does_not_print_private_details(self):
        output = io.StringIO()
        with mock.patch.object(candidate.os, "geteuid", return_value=0), \
                mock.patch.object(candidate, "manage", side_effect=ValueError("synthetic-secret-do-not-print")), \
                mock.patch.object(candidate.os, "umask"), contextlib.redirect_stdout(output):
            self.assertEqual(candidate.main(["status", "--run-id", "v4-local-test"]), 1)
        self.assertEqual(json.loads(output.getvalue()),
                         {"completed": False, "mode": "status", "error_type": "ValueError"})


if __name__ == "__main__":
    unittest.main()
