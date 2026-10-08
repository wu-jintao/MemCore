#!/usr/bin/env python3
"""Prepare an isolated v4 candidate without changing the production deployment.

Preparation copies only reviewed source and a supplied private model environment.
It never contacts a model endpoint, downloads packages, or reads runtime.env.
Starting the candidate only starts its loopback systemd service; it sends no
synthetic requests. Production activation and public routing are separate work.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import subprocess
import sys
from urllib.parse import urlsplit


SOURCE_NAMES = ("server.py", "semantic.py")
FIXED_MODEL_VALUES = {
    "MEMORY_EMBEDDING_PROVIDER": "http",
    "MEMORY_EMBEDDING_ALLOW_HTTP": "1",
    "MEMORY_EMBEDDING_MODEL": "text-embedding-v4",
    "MEMORY_EMBEDDING_DIMENSION": "2048",
    "MEMORY_EMBEDDING_BATCH_SIZE": "10",
    "MEMORY_EMBEDDING_CONCURRENCY": "1",
    "MEMORY_EMBEDDING_HTTP_RETRIES": "0",
}
INTEGER_BOUNDS = {
    "MEMORY_EMBEDDING_THREADS": (1, 8),
    "MEMORY_EMBEDDING_SEGMENT_TOKENS": (16, 480),
    "MEMORY_EMBEDDING_OVERLAP_TOKENS": (0, 479),
    "MEMORY_EMBEDDING_MAX_DOCUMENT_SEGMENTS": (1, 4096),
    "MEMORY_EMBEDDING_MAX_TOTAL_SEGMENTS": (1, 16384),
    "MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS": (1, 64),
    "MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES": (32, 16384),
}
TIMEOUT_BOUNDS = {
    "MEMORY_EMBEDDING_HTTP_TIMEOUT": 120,
    "MEMORY_EMBEDDING_QUEUE_TIMEOUT": 900,
    "MEMORY_EMBEDDING_OPERATION_TIMEOUT": 1500,
}
REQUIRED_MODEL_KEYS = set(FIXED_MODEL_VALUES) | {
    "MEMORY_EMBEDDING_URL", "MEMORY_EMBEDDING_API_KEY",
    "MEMORY_EMBEDDING_HTTP_TIMEOUT", "MEMORY_EMBEDDING_OPERATION_TIMEOUT",
}
ALLOWED_MODEL_KEYS = (REQUIRED_MODEL_KEYS | set(INTEGER_BOUNDS) |
                      set(TIMEOUT_BOUNDS) | {"MEMORY_EMBEDDING_QUERY_PREFIX",
                                            "MEMORY_EMBEDDING_DOCUMENT_PREFIX"})
PREFIX_KEYS = {"MEMORY_EMBEDDING_QUERY_PREFIX", "MEMORY_EMBEDDING_DOCUMENT_PREFIX"}


@dataclass(frozen=True)
class Layout:
    base: Path = Path("/opt/aml-memory")
    data: Path = Path("/srv/aml-data")
    config: Path = Path("/etc/aml-memory")
    units: Path = Path("/etc/systemd/system")

    def candidate(self, run_id):
        validate_run_id(run_id)
        return {
            "release": self.base / "releases" / ("http-" + run_id),
            "data": self.data / "http-candidates" / run_id,
            "config": self.config / "http-candidates" / run_id,
            "unit": self.units / ("aml-memory-http-" + run_id + ".service"),
        }


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("Use a unique run id containing only letters, numbers, hyphens and underscores")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_bytes(path, maximum=16 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Input must be a bounded regular file, not a symlink")
    return path.read_bytes()


def load_manifest(path, expected_sha):
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("An explicit lowercase manifest SHA-256 is required")
    raw = file_bytes(path, 65536)
    if digest_bytes(raw) != expected_sha:
        raise ValueError("Source manifest SHA-256 differs from the supplied digest")
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError("Invalid source manifest") from None
    if not isinstance(manifest, dict) or set(manifest) not in ({"source_hashes"}, {"files"}):
        raise ValueError("Manifest must contain only a source_hashes or files mapping")
    hashes = next(iter(manifest.values()))
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_NAMES) or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in hashes.values())):
        raise ValueError("Manifest must pin exactly server.py and semantic.py")
    return dict(hashes)


def parse_model_environment(raw):
    """Validate only supplied candidate configuration; never report its values."""
    try:
        content = raw.decode("utf-8")
    except UnicodeError:
        raise ValueError("Candidate model environment must be UTF-8") from None
    if "\x00" in content or "\r" in content:
        raise ValueError("Candidate model environment contains unsupported control characters")
    values = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, text = line.partition("=")
        if not separator or key not in ALLOWED_MODEL_KEYS or key in values:
            raise ValueError("Candidate model environment has unknown or duplicate keys")
        try:
            parsed = [""] if not text and key in PREFIX_KEYS else shlex.split(text, comments=False, posix=True)
        except ValueError:
            raise ValueError("Candidate model environment has invalid quoting") from None
        if len(parsed) != 1:
            raise ValueError("Candidate model environment requires one quoted or literal value per key")
        if not parsed[0] and key not in PREFIX_KEYS:
            raise ValueError("Only candidate query and document prefixes may be empty")
        values[key] = parsed[0]
    if not REQUIRED_MODEL_KEYS <= set(values):
        raise ValueError("Candidate model environment is missing required explicit configuration")
    if any(values[key] != expected for key, expected in FIXED_MODEL_VALUES.items()):
        raise ValueError("Candidate must use HTTP v4, dimension 2048, batch 10, concurrency 1, retries 0")
    if not values["MEMORY_EMBEDDING_API_KEY"].strip():
        raise ValueError("Candidate model API key is empty")
    endpoint = urlsplit(values["MEMORY_EMBEDDING_URL"])
    try:
        endpoint.port
    except ValueError:
        raise ValueError("Candidate model endpoint has an invalid port") from None
    if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or
            endpoint.password or endpoint.query or endpoint.fragment or
            endpoint.path.rstrip("/") != "/compatible-mode/v1/embeddings"):
        raise ValueError("Candidate requires a credential-free HTTPS compatible embeddings endpoint")
    for key, (lower, upper) in INTEGER_BOUNDS.items():
        if key in values:
            try:
                number = int(values[key])
            except ValueError:
                raise ValueError("Invalid integer candidate configuration") from None
            if not lower <= number <= upper:
                raise ValueError("Integer candidate configuration is outside supported bounds")
    for key, maximum in TIMEOUT_BOUNDS.items():
        if key in values:
            try:
                number = float(values[key])
            except ValueError:
                raise ValueError("Invalid candidate timeout") from None
            if not math.isfinite(number) or not 0 < number <= maximum:
                raise ValueError("Candidate timeout is outside supported bounds")
    if int(values.get("MEMORY_EMBEDDING_OVERLAP_TOKENS", "64")) >= int(
            values.get("MEMORY_EMBEDDING_SEGMENT_TOKENS", "384")):
        raise ValueError("Candidate overlap must be smaller than its segment window")
    return values


def private_write(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chown(path, 0, 0)
        Path(path).chmod(0o600)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise


def root_private_file(path):
    path = Path(path)
    return (not path.is_symlink() and path.is_file() and path.stat().st_uid == 0 and
            not path.stat().st_mode & 0o077)


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30,
                            stdin=subprocess.DEVNULL)
    if result.returncode:
        # Captured output may include private paths/configuration; do not print it.
        raise RuntimeError("Candidate system command failed")
    return result.stdout.strip()


def systemd_quote(path):
    # Percent specifiers must stay literal, and argv/path entries must be quoted.
    return '"' + str(path).replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"') + '"'


def systemd_single_path(path):
    # WorkingDirectory parses one literal path rather than ExecStart argv. All
    # production layout components and validated run ids are whitespace-free.
    value = str(path)
    if not Path(path).is_absolute() or any(character.isspace() for character in value):
        raise ValueError("Candidate systemd single paths must be absolute and whitespace-free")
    return value.replace("%", "%%")


def unit_text(paths, layout, venv):
    app = paths["release"] / "app"
    cache = paths["data"] / "cache"
    database = paths["data"] / "db" / "memory.sqlite3"
    return "\n".join([
        "[Unit]", "Description=Isolated AML text-embedding-v4 candidate",
        "After=network-online.target", "Wants=network-online.target",
        "RequiresMountsFor=" + systemd_quote(paths["data"]), "", "[Service]",
        "Type=simple", "User=aml-memory", "Group=aml-memory",
        "WorkingDirectory=" + systemd_single_path(app),
        "EnvironmentFile=" + systemd_single_path(layout.config / "runtime.env"),
        "EnvironmentFile=" + systemd_single_path(paths["config"] / "model.env"),
        "EnvironmentFile=" + systemd_single_path(paths["config"] / "settings.env"),
        "Environment=PYTHONUNBUFFERED=1", "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=HF_HUB_OFFLINE=1", "Environment=TRANSFORMERS_OFFLINE=1",
        "Environment=HF_DATASETS_OFFLINE=1", "Environment=TOKENIZERS_PARALLELISM=false",
        "Environment=OMP_NUM_THREADS=2", "Environment=MKL_NUM_THREADS=2",
        "UnsetEnvironment=PYTHONPATH PYTHONHOME MEMORY_EMBEDDING_MODEL_DIR",
        "ExecStart=" + systemd_quote(venv / "bin/python") + " " +
        systemd_quote(app / "server.py") + " --host 127.0.0.1 --port 18081 --db " +
        systemd_quote(database),
        "Restart=no", "TimeoutStopSec=60", "UMask=0077",
        "NoNewPrivileges=true", "PrivateTmp=true", "ProtectSystem=strict",
        "ProtectHome=true", "ReadOnlyPaths=" + systemd_quote(venv),
        "ReadWritePaths=" + systemd_quote(paths["data"] / "db") + " " + systemd_quote(cache),
        "LimitNOFILE=65536", "TasksMax=256", "MemoryMax=12G", "",
    ])


def prepare(args, layout=Layout()):
    paths = layout.candidate(args.run_id)
    hashes = load_manifest(args.manifest, args.manifest_sha256)
    source = {name: file_bytes(Path(args.source_stage) / name) for name in SOURCE_NAMES}
    if {name: digest_bytes(raw) for name, raw in source.items()} != hashes:
        raise ValueError("Candidate source differs from the reviewed manifest")
    model_path = Path(args.model_env_stage)
    if model_path.stat().st_mode & 0o077:
        raise ValueError("Supplied model environment must have private file permissions")
    model_raw = file_bytes(model_path, 65536)
    parse_model_environment(model_raw)
    runtime = layout.config / "runtime.env"
    if not root_private_file(runtime):
        raise ValueError("Production runtime environment must remain a private root-owned regular file")
    if not os.path.ismount(layout.data):
        raise ValueError("Persistent production data disk is not mounted")
    venv = (layout.base / "venv").resolve(strict=True)
    if (not venv.is_dir() or not (venv / "bin/python").is_file() or
            not venv.is_relative_to(layout.base.resolve())):
        raise ValueError("Verified production virtual environment is unavailable")
    account = pwd.getpwnam("aml-memory")
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise ValueError("Candidate run id already exists; inspect it instead of overwriting")
    if any(paths[key].parent.is_symlink() for key in ("release", "data", "config", "unit")):
        raise ValueError("Candidate parent directories must not be symlinks")
    if paths["data"].parent.exists() and not paths["data"].parent.stat().st_mode & 0o001:
        raise ValueError("Existing candidate data parent is not traversable by the service account")
    created = []
    try:
        for key, mode in (("release", 0o755), ("data", 0o700), ("config", 0o700)):
            path = paths[key]
            parent_mode = 0o700 if key == "config" else 0o755
            parent_existed = path.parent.exists()
            path.parent.mkdir(parents=True, exist_ok=True, mode=parent_mode)
            if not parent_existed:
                # mkdir's mode is reduced by the private process umask. The new
                # data namespace must remain traversable by the service account.
                path.parent.chmod(parent_mode)
                os.chown(path.parent, 0, 0)
            path.mkdir(mode=mode)
            path.chmod(mode)
            created.append(path)
            os.chown(path, account.pw_uid if key == "data" else 0,
                     account.pw_gid if key == "data" else 0)
        app = paths["release"] / "app"
        app.mkdir(mode=0o755)
        app.chmod(0o755)
        for name, raw in source.items():
            target = app / name
            target.write_bytes(raw)
            target.chmod(0o644)
            os.chown(target, 0, 0)
        for relative in ("db", "cache", "cache/tmp", "cache/huggingface"):
            target = paths["data"] / relative
            target.mkdir(mode=0o700)
            target.chmod(0o700)
            os.chown(target, account.pw_uid, account.pw_gid)
        private_write(paths["config"] / "model.env", model_raw)
        cache = paths["data"] / "cache"
        settings = ("HF_HOME=" + str(cache / "huggingface") + "\nTMPDIR=" +
                    str(cache / "tmp") + "\nXDG_CACHE_HOME=" + str(cache) + "\n").encode()
        private_write(paths["config"] / "settings.env", settings)
        unit = unit_text(paths, layout, venv).encode()
        private_write(paths["unit"], unit)
        paths["unit"].chmod(0o644)
        created.append(paths["unit"])
        command(["systemd-analyze", "verify", str(paths["unit"])])
        state = {"prepared": True, "run_id": args.run_id, "source_hashes": hashes,
                 "manifest_sha256": args.manifest_sha256, "venv": str(venv),
                 "model_env_sha256": digest_bytes(model_raw),
                 "settings_sha256": digest_bytes(settings), "unit_sha256": digest_bytes(unit)}
        private_write(paths["config"] / "prepared.json", (json.dumps(state, indent=2) + "\n").encode())
    except BaseException:
        for path in reversed(created):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        raise
    return {"prepared": True, "run_id": args.run_id, "candidate_unit": paths["unit"].name,
            "candidate_release": str(paths["release"]), "candidate_data": str(paths["data"]),
            "production_modified": False, "started": False, "external_calls": False}


def verify_prepared(run_id, layout=Layout()):
    paths = layout.candidate(run_id)
    state = json.loads(file_bytes(paths["config"] / "prepared.json", 65536))
    if state.get("prepared") is not True or state.get("run_id") != run_id:
        raise ValueError("Candidate preparation state is invalid")
    if {name: digest_bytes(file_bytes(paths["release"] / "app" / name))
            for name in SOURCE_NAMES} != state["source_hashes"]:
        raise ValueError("Prepared candidate source changed")
    for name, expected in (("model.env", state["model_env_sha256"]),
                           ("settings.env", state["settings_sha256"])):
        target = paths["config"] / name
        if not root_private_file(target):
            raise ValueError("Prepared candidate environment permissions changed")
        raw = file_bytes(target, 65536)
        if digest_bytes(raw) != expected:
            raise ValueError("Prepared candidate environment changed")
        if name == "model.env":
            parse_model_environment(raw)
    if digest_bytes(file_bytes(paths["unit"], 65536)) != state["unit_sha256"]:
        raise ValueError("Prepared candidate service changed")
    venv = Path(state["venv"])
    if not venv.is_dir() or not (venv / "bin/python").is_file():
        raise ValueError("Shared candidate virtual environment is unavailable")
    return paths


def manage(mode, run_id, layout=Layout()):
    paths = verify_prepared(run_id, layout) if mode == "start" else layout.candidate(run_id)
    if not paths["unit"].is_file() or paths["unit"].is_symlink():
        raise ValueError("Candidate service is not prepared")
    unit = paths["unit"].name
    if mode == "start":
        command(["systemctl", "daemon-reload"])
        command(["systemctl", "start", unit])
    elif mode == "stop":
        command(["systemctl", "stop", unit])
    state = command(["systemctl", "show", unit, "--property=ActiveState", "--value"])
    if state not in {"active", "reloading", "inactive", "failed", "activating", "deactivating"}:
        raise RuntimeError("Candidate service returned an unexpected state")
    return {"run_id": run_id, "candidate_unit": unit, "active_state": state,
            "production_modified": False, "embedding_readiness_checked": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "start", "stop", "status"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-stage", type=Path)
    parser.add_argument("--model-env-stage", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        parser.error("Run this candidate preparation tool as root on the independent Ubuntu host")
    if args.mode == "prepare" and any(getattr(args, key) is None for key in (
            "source_stage", "model_env_stage", "manifest", "manifest_sha256")):
        parser.error("prepare requires source-stage, model-env-stage, manifest and manifest-sha256")
    os.umask(0o077)
    try:
        result = prepare(args) if args.mode == "prepare" else manage(args.mode, args.run_id)
        print(json.dumps(result))
    except Exception as failure:
        # No input/environment values, private reports, tracebacks, or subprocess
        # output are printed, including configuration-parser exceptions.
        print(json.dumps({"completed": False, "mode": args.mode,
                          "error_type": type(failure).__name__}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
