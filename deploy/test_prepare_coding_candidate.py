"""Pure local construction tests; no systemd, credentials, or network calls."""

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

from deploy import prepare_coding_candidate as candidate


class CodingCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.layout = candidate.Layout(self.root / "opt", self.root / "data",
                                       self.root / "etc", self.root / "units")
        for path in (self.layout.base, self.layout.data, self.layout.config, self.layout.units):
            path.mkdir()
        (self.layout.base / "venv/bin").mkdir(parents=True)
        (self.layout.base / "venv/bin/python").write_text("python fixture")
        (self.layout.config / "runtime.env").write_text("MEMORY_API_TOKEN=synthetic-private-token\n")
        (self.layout.config / "runtime.env").chmod(0o600)
        self.stage = self.root / "stage"; self.stage.mkdir()
        hashes = {}
        for name in candidate.SOURCE_NAMES:
            raw = ("# reviewed " + name + "\n").encode(); (self.stage / name).write_bytes(raw)
            hashes[name] = candidate.digest_bytes(raw)
        self.manifest = self.stage / "manifest.json"
        self.manifest.write_text(json.dumps({"files": hashes}))
        self.model = self.stage / "model.env"
        values = dict(candidate.FIXED_MODEL_VALUES,
                      MEMORY_EMBEDDING_URL="https://example.invalid/compatible-mode/v1/embeddings",
                      MEMORY_EMBEDDING_API_KEY="synthetic-model-key",
                      MEMORY_EMBEDDING_QUERY_PREFIX="", MEMORY_EMBEDDING_DOCUMENT_PREFIX="")
        self.model_raw = "".join(key + "=" + value + "\n" for key, value in values.items()).encode()
        self.model.write_bytes(self.model_raw); self.model.chmod(0o600)
        self.args = Namespace(run_id="coding-local-test", source_stage=self.stage,
                              model_env_stage=self.model, manifest=self.manifest,
                              manifest_sha256=candidate.digest_bytes(self.manifest.read_bytes()))
        self.commands = []
        mock.patch.object(candidate.os.path, "ismount", return_value=True).start()
        mock.patch.object(candidate.os, "chown").start()
        mock.patch.object(candidate.pwd, "getpwnam", return_value=SimpleNamespace(
            pw_uid=os.getuid(), pw_gid=os.getgid())).start()
        mock.patch.object(candidate, "root_private_file", side_effect=lambda path:
                          path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o077).start()
        mock.patch.object(candidate, "command", side_effect=self.record_command).start()
        self.addCleanup(mock.patch.stopall)

    def record_command(self, argv):
        self.commands.append(argv)
        return "active" if argv[:2] == ["systemctl", "show"] else ""

    def test_prepare_isolated_fixed_profile(self):
        result = candidate.prepare(self.args, self.layout)
        paths = self.layout.candidate(self.args.run_id)
        self.assertEqual(set(path.name for path in (paths["release"] / "app").iterdir()),
                         set(candidate.SOURCE_NAMES))
        self.assertEqual((paths["config"] / "model.env").read_bytes(), self.model_raw)
        self.assertEqual((paths["release"] / "app").stat().st_mode & 0o777, 0o755)
        self.assertEqual(list((paths["data"] / "db").iterdir()), [])
        unit = paths["unit"].read_text()
        self.assertIn('"--port" "18082"', unit)
        self.assertIn('"--profile" "coding"', unit)
        self.assertIn('"--semantic-weight" "3.0"', unit)
        self.assertIn('"--coding-weight" "2.0"', unit)
        self.assertIn('"--coding-result-limit" "12"', unit)
        self.assertIn('"--coding-character-limit" "180000"', unit)
        self.assertIn('"--max-body-bytes" "134217728"', unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("RestartSec=5s", unit)
        self.assertIn("StartLimitBurst=10", unit)
        self.assertNotIn("reverse_proxy", unit)
        self.assertFalse(result["production_modified"])
        self.assertFalse(result["public_route_modified"])
        self.assertEqual(self.commands, [["systemd-analyze", "verify", str(paths["unit"])]])

    def test_rejects_source_or_manifest_drift(self):
        self.args.manifest_sha256 = "0" * 64
        with self.assertRaises(ValueError): candidate.prepare(self.args, self.layout)
        self.args.manifest_sha256 = candidate.digest_bytes(self.manifest.read_bytes())
        (self.stage / "coding.py").write_text("changed")
        with self.assertRaises(ValueError): candidate.prepare(self.args, self.layout)
        self.assertEqual(self.commands, [])

    def test_model_profile_is_exact(self):
        candidate.parse_model_environment(self.model_raw)
        replacements = ((b"HTTP_RETRIES=2", b"HTTP_RETRIES=0"),
                        (b"SEGMENT_BYTES=8192", b"SEGMENT_BYTES=1536"),
                        (b"MAX_DOCUMENT_SEGMENTS=4", b"MAX_DOCUMENT_SEGMENTS=8"),
                        (b"DIMENSION=2048", b"DIMENSION=1024"))
        for old, new in replacements:
            with self.subTest(old=old), self.assertRaises(ValueError):
                candidate.parse_model_environment(self.model_raw.replace(old, new))
        with self.assertRaises(ValueError):
            candidate.parse_model_environment(self.model_raw + b"MEMORY_PROFILE=coding\n")

    def test_start_stop_status_only_target_candidate(self):
        candidate.prepare(self.args, self.layout); self.commands.clear()
        unit = self.layout.candidate(self.args.run_id)["unit"].name
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

    def test_changed_source_environment_unit_or_app_permissions_blocks_start(self):
        for kind in ("source", "environment", "unit", "permissions"):
            with self.subTest(kind=kind):
                run = self.args.run_id + "-" + kind
                args = Namespace(**{**vars(self.args), "run_id": run})
                candidate.prepare(args, self.layout)
                paths = self.layout.candidate(run)
                if kind == "source": (paths["release"] / "app/server.py").write_text("changed")
                elif kind == "environment": (paths["config"] / "model.env").write_bytes(self.model_raw.replace(b"synthetic-model-key", b"different-model-key"))
                else: paths["unit"].write_text(paths["unit"].read_text() + "\n# drift")
                self.commands.clear()
                with self.assertRaises(ValueError): candidate.manage("start", run, self.layout)
                self.assertEqual(self.commands, [])

    def test_failure_output_is_sanitized(self):
        output = io.StringIO()
        with mock.patch.object(candidate.os, "geteuid", return_value=0), \
                mock.patch.object(candidate, "manage", side_effect=ValueError("secret")), \
                mock.patch.object(candidate.os, "umask"), contextlib.redirect_stdout(output):
            self.assertEqual(candidate.main(["status", "--run-id", "coding-test"]), 1)
        self.assertEqual(json.loads(output.getvalue()),
                         {"completed": False, "mode": "status", "error_type": "ValueError"})

    def test_run_id_cannot_escape_layout(self):
        for value in ("../x", "a/b", "", "x" * 65):
            with self.assertRaises(ValueError): self.layout.candidate(value)


if __name__ == "__main__":
    unittest.main()
