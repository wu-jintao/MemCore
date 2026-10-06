#!/usr/bin/env python3
"""Prepare, activate, or restore the reviewed offline E5-small deployment.

Run as root on the independent Ubuntu instance. Preparation does not stop the
service. Activation preserves the existing credentials byte-for-byte and moves
the complete stopped database directory, including SQLite sidecars, to private
recovery storage. No dependency/model downloads or public-network changes occur.
"""

import argparse
import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SOURCE_HASHES = {
    "server.py": "96124a789829afdfead08f74fb9c89ef7b41bbaf2523c90eb1746846e0d76ab9",
    "semantic.py": "8a0473f3ad91714ac5f88b4db177ef9f34e683dcfbcea1acbc3745c462b93145",
}
FINGERPRINT = "semantic-v1:3bd79d18f9fad555144ae06fe9f7a8bfbfefdc039f684259b789bda1e6d400f6"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MANIFEST_SHA = "5f03e6d0b3896e2f1cd830ba67c8d9f564da2cff93880f5433c60e6957e0d1a1"
SERVICE = Path("/etc/systemd/system/aml-memory.service")
ENV_FILE = Path("/etc/aml-memory/runtime.env")
BASE = Path("/opt/aml-memory")
DATABASE = Path("/srv/aml-data/db")
CACHE = Path("/srv/aml-data/cache")
CONFIG = {
    "MEMORY_EMBEDDING_PROVIDER": "local",
    "MEMORY_EMBEDDING_MODEL": "intfloat/multilingual-e5-small",
    "MEMORY_EMBEDDING_DIMENSION": "384",
    "MEMORY_EMBEDDING_BATCH_SIZE": "8",
    "MEMORY_EMBEDDING_THREADS": "2",
    "MEMORY_EMBEDDING_CONCURRENCY": "1",
    "MEMORY_EMBEDDING_SEGMENT_TOKENS": "384",
    "MEMORY_EMBEDDING_OVERLAP_TOKENS": "64",
    "MEMORY_EMBEDDING_MAX_DOCUMENT_SEGMENTS": "512",
    "MEMORY_EMBEDDING_MAX_TOTAL_SEGMENTS": "4096",
    "MEMORY_EMBEDDING_MAX_QUERY_SEGMENTS": "8",
    "MEMORY_EMBEDDING_QUEUE_TIMEOUT": "120",
    "MEMORY_EMBEDDING_OPERATION_TIMEOUT": "600",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def write_private(path, value):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".atomic-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o600)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run(arguments, *, environment=None, timeout=180):
    # Command/output never includes production credentials. The output of
    # package checks or private preflight is only stored in private recovery.
    result = subprocess.run(arguments, capture_output=True, text=True,
                            env=environment, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Subprocess failed: " + Path(arguments[0]).name)
    return result.stdout


def paths(run_id):
    return (BASE / "releases" / run_id,
            BASE / "recovery" / run_id,
            Path("/srv/aml-data/activation-recovery") / run_id)


def runtime_conflicts():
    # EnvironmentFile overrides service Environment assignments. Inspect keys
    # only; never log or return their values or the credential file contents.
    keys = {line.strip().split("=", 1)[0] for line in ENV_FILE.read_text().splitlines()
            if "=" in line and not line.lstrip().startswith("#")}
    return sorted(keys & (set(CONFIG) | {"MEMORY_EMBEDDING_MODEL_DIR", "PYTHONPATH", "PYTHONHOME"}))


def model_integrity(model):
    manifest = model / "download-manifest.json"
    if digest(manifest) != MANIFEST_SHA:
        raise RuntimeError("Model manifest differs from verified baseline")
    metadata = json.loads(manifest.read_text())
    if metadata["revision"] != MODEL_REVISION or len(metadata["files"]) != 10:
        raise RuntimeError("Unexpected model revision or file count")
    checked = []
    for item in metadata["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe model path")
        target = model / relative
        if target.stat().st_size != item["size"] or digest(target) != item["sha256"]:
            raise RuntimeError("Model integrity check failed")
        checked.append(item["path"])
    return {"model": metadata["model"], "revision": MODEL_REVISION,
            "manifest_sha256": MANIFEST_SHA, "verified_files": checked,
            "all_files_match": True}


PREFLIGHT = r'''
import sys,os,json,math,tempfile,sqlite3,importlib.metadata
from pathlib import Path
app,model,cache,fingerprint=sys.argv[1:]
sys.path.insert(0,app)
import server,semantic,torch,sentence_transformers
assert torch.version.cuda is None
backend=semantic.build_backend()
assert backend.fingerprint==fingerprint and backend.dimension==384
vectors=backend.embed_documents(['The deploycobalt archive is beside the window.'])[0]
query=backend.embed_query('Where is the deploycobalt archive?')
assert vectors and all(len(v)==384 and abs(sum(x*x for x in v)-1)<1e-5 for v in vectors)
assert len(query)==384 and all(math.isfinite(x) for x in query)
with tempfile.TemporaryDirectory(prefix='activation-preflight-',dir=cache) as directory:
 store=server.MemoryStore(Path(directory)/'preflight.sqlite3',semantic_backend=backend)
 payload={'request_id':'preflight-request','user_id':'preflight/complete-scope','session_id':'preflight-session','messages':[{'role':'user','content':'The deploycobalt archive is beside the window.'}]}
 assert store.add(payload)['success']
 assert store.add(payload)['success']
 found=store.search({'query':'deploycobalt','user_id':payload['user_id'],'top_k':1})
 assert 'deploycobalt' in found['data'][0]['content']
 assert store.search({'query':'deploycobalt','user_id':'another/complete-scope','top_k':1})=={'data':[]}
 with sqlite3.connect(str(store.db_path)) as c:
  assert c.execute('SELECT COUNT(*) FROM messages').fetchone()[0]==1
  assert c.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]==1
print(json.dumps({'passed':True,'python':sys.version.split()[0],'sys_prefix':sys.prefix,'cuda_version':torch.version.cuda,'fingerprint':backend.fingerprint,'dimension':backend.dimension,'vector_functionality':True,'store_scope_and_idempotency':True,'packages':{n:importlib.metadata.version(n) for n in ['torch','sentence-transformers','transformers','tokenizers','huggingface-hub','numpy']}}))
'''


def prepare(args):
    release, recovery, private = paths(args.run_id)
    if release.exists() or private.exists() or recovery.exists():
        raise RuntimeError("Run id already exists; use activate/rollback for that run")
    if not os.path.ismount("/srv/aml-data"):
        raise RuntimeError("Persistent data disk is not mounted")
    if shutil.disk_usage(BASE).free < 4 * 1024**3:
        raise RuntimeError("Insufficient standby space")
    if runtime_conflicts():
        raise RuntimeError("Runtime environment overrides the reviewed semantic configuration")
    source = Path(args.source_stage)
    environment_stage = Path(args.environment_stage)
    actual = {name: digest(source / name) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise RuntimeError("Source does not match reviewed candidate")
    original_model = environment_stage / "models/multilingual-e5-small"
    integrity = model_integrity(original_model)
    private.mkdir(parents=True, mode=0o700)
    recovery.mkdir(parents=True, mode=0o700)
    os.chmod(private.parent, 0o700)
    os.chmod(recovery.parent, 0o700)
    release.mkdir(parents=True, mode=0o755)
    release.parent.chmod(0o755)
    release.chmod(0o755)
    app = release / "app"
    app.mkdir(mode=0o755)
    app.chmod(0o755)
    for name in SOURCE_HASHES:
        shutil.copyfile(source / name, app / name)
        os.chmod(app / name, 0o644)
    venv = release / "venv"
    shutil.copytree(environment_stage / "venv", venv, symlinks=True,
                    ignore=shutil.ignore_patterns(".local"))
    # Installed packages are public artifacts; preserve executable bits while
    # making the copied interpreter and libraries traversable by the service.
    for directory, dirs, files in os.walk(venv):
        os.chmod(directory, 0o755)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, 0o755 if path.stat().st_mode & 0o111 else 0o644)
    model = release / "models/multilingual-e5-small"
    model.mkdir(parents=True, mode=0o755)
    for name in integrity["verified_files"] + ["download-manifest.json"]:
        target = model / name
        target.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
        shutil.copyfile(original_model / name, target)
        os.chmod(target, 0o644)
    model.parent.chmod(0o755)
    for directory, dirs, files in os.walk(model):
        os.chmod(directory, 0o755)
    model_integrity(model)
    account = pwd.getpwnam("aml-memory")
    cache = CACHE / ("activation-" + args.run_id)
    cache.mkdir(mode=0o700)
    os.chown(cache, account.pw_uid, account.pw_gid)
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("MEMORY_") and k not in ("PYTHONPATH", "PYTHONHOME")}
    environment.update(CONFIG)
    environment.update({"MEMORY_EMBEDDING_MODEL_DIR": str(model),
                        "HF_HOME": str(cache / "huggingface")})
    pip_check = run(["runuser", "-u", "aml-memory", "--", str(venv / "bin/python"),
                     "-m", "pip", "check"], environment=environment)
    preflight = json.loads(run(["runuser", "-u", "aml-memory", "--",
        str(venv / "bin/python"), "-c", PREFLIGHT, str(app), str(model), str(cache),
        FINGERPRINT], environment=environment))
    if Path(preflight["sys_prefix"]).resolve() != venv.resolve():
        raise RuntimeError("Copied interpreter uses the wrong environment")
    template = Path(__file__).with_name("aml-memory.service").read_text()
    if "MemoryMax=12G\n" not in template or "User=aml-memory\n" not in template:
        raise RuntimeError("Unexpected service template")
    service_environment = dict(CONFIG, MEMORY_EMBEDDING_MODEL_DIR=str(model))
    additions = "".join("Environment=" + key + "=" + value + "\n"
                        for key, value in service_environment.items())
    service = template.replace("ExecStart=", additions + "ExecStart=", 1)
    service_path = private / "candidate.service"
    service_path.write_text(service)
    service_path.chmod(0o600)
    run(["systemd-analyze", "verify", str(service_path)])
    result = {"prepared": True, "run_id": args.run_id, "release": str(release),
        "source_hashes": actual, "model": integrity, "fingerprint": FINGERPRINT,
        "pip_check_passed": True, "pip_check_output": pip_check.strip(),
        "unprivileged_preflight": preflight, "cache": str(cache),
        "runtime_environment_conflicts": [],
        "runtime_env_modified": False, "production_modified": False,
        "downloads_or_paid_api_calls": False, "memory_max": "12G",
        "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_private(private / "prepared.json", result)
    return {"prepared": True, "run_id": args.run_id, "release": str(release),
            "fingerprint": FINGERPRINT, "private_report": str(private / "prepared.json")}


def http(path, payload=None, token=None):
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    req = Request("http://127.0.0.1:8080" + path, data=body, headers=headers)
    try:
        with urlopen(req, timeout=180) as response:
            return response.status, json.loads(response.read())
    except HTTPError as response:
        return response.code, json.loads(response.read())


def healthy(timeout=150):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        active = subprocess.run(["systemctl", "is-active", "--quiet", "aml-memory"]).returncode == 0
        if active:
            try:
                if http("/health") == (200, {"status": "ok"}):
                    return True
            except Exception:
                pass
        time.sleep(.5)
    raise RuntimeError("Service did not become active and healthy")


def token_from(raw):
    values = [line.split("=", 1)[1] for line in raw.decode().splitlines()
              if line.startswith("MEMORY_API_TOKEN=")]
    if len(values) != 1:
        raise RuntimeError("Unexpected credential file format")
    tokens = shlex.split(values[0])
    if len(tokens) != 1 or not tokens[0]:
        raise RuntimeError("Invalid credential format")
    return tokens[0]


def restore(run_id):
    release, recovery, private = paths(run_id)
    run(["systemctl", "stop", "aml-memory"], timeout=90)
    for name in ("app", "venv"):
        saved = recovery / name
        if saved.exists() or saved.is_symlink():
            current = BASE / name
            if current.exists() or current.is_symlink():
                # Preserve failed new artifacts without overwriting old paths.
                os.replace(current, recovery / ("failed-" + name))
            os.replace(saved, current)
    saved_db = private / "db"
    if saved_db.exists():
        if DATABASE.exists():
            os.replace(DATABASE, private / "failed-semantic-db")
        os.replace(saved_db, DATABASE)
    for destination, name in ((ENV_FILE, "runtime.env"), (SERVICE, "old.service")):
        saved = private / name
        if saved.exists():
            shutil.copyfile(saved, destination)
            destination.chmod(0o600 if destination == ENV_FILE else 0o644)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "start", "aml-memory"])
    healthy()
    write_private(private / "rollback.json", {"restored": True, "service_healthy": True,
        "old_database_restored": DATABASE.exists(), "run_id": run_id,
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return {"restored": True, "run_id": run_id}


def activate(args):
    release, recovery, private = paths(args.run_id)
    prepared = json.loads((private / "prepared.json").read_text())
    if not prepared["prepared"] or prepared["fingerprint"] != FINGERPRINT:
        raise RuntimeError("Standby preflight was not verified")
    if {n: digest(release / "app" / n) for n in SOURCE_HASHES} != SOURCE_HASHES:
        raise RuntimeError("Standby source changed")
    if runtime_conflicts():
        raise RuntimeError("Runtime environment overrides the reviewed semantic configuration")
    model_integrity(release / "models/multilingual-e5-small")
    if (private / "activation-report.json").exists() or (private / "runtime.env").exists():
        raise RuntimeError("Activation already attempted; inspect report or use rollback")
    credential = ENV_FILE.read_bytes()  # In-memory only; never report its value.
    token = token_from(credential)
    shutil.copyfile(ENV_FILE, private / "runtime.env")
    (private / "runtime.env").chmod(0o600)
    shutil.copyfile(SERVICE, private / "old.service")
    (private / "old.service").chmod(0o600)
    old_sources = {n: digest(BASE / "app" / n) for n in SOURCE_HASHES}
    old_db_files = sorted(p.name for p in DATABASE.iterdir())
    result = {"run_id": args.run_id, "source_hashes": SOURCE_HASHES,
        "old_source_hashes": old_sources, "fingerprint": FINGERPRINT,
        "old_database_files_preserved": old_db_files, "old_system_paths": str(recovery),
        "old_data_and_config_paths": str(private), "model": prepared["model"],
        "unprivileged_preflight": prepared["unprivileged_preflight"],
        "production_network_modified": False, "official_evaluation": False}
    switched = False
    try:
        switched = True
        write_private(private / "activation-state.json", {"switch_started": True})
        run(["systemctl", "stop", "aml-memory"], timeout=90)
        if subprocess.run(["systemctl", "is-active", "--quiet", "aml-memory"]).returncode == 0:
            raise RuntimeError("Service remained active during database switch")
        for name in ("app", "venv"):
            os.replace(BASE / name, recovery / name)
            (BASE / name).symlink_to(release / name, target_is_directory=True)
        os.replace(DATABASE, private / "db")
        account = pwd.getpwnam("aml-memory")
        DATABASE.mkdir(mode=0o700)
        os.chown(DATABASE, account.pw_uid, account.pw_gid)
        shutil.copyfile(private / "candidate.service", SERVICE)
        SERVICE.chmod(0o644)
        if ENV_FILE.read_bytes() != credential:
            raise RuntimeError("Credential file changed before startup")
        run(["systemctl", "daemon-reload"])
        run(["systemctl", "start", "aml-memory"])
        healthy()
        with sqlite3.connect("file:" + str(DATABASE / "memory.sqlite3") + "?mode=ro", uri=True) as connection:
            result["new_database_started_empty"] = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        if not result["new_database_started_empty"]:
            raise RuntimeError("New database was not empty")
        user = "activation/" + args.run_id + "/complete-user"
        payload = {"request_id": "activation-request", "user_id": user,
            "session_id": "activation-session", "messages": [{"role": "user",
                "content": "The deploycobalt archive is beside the window.", "timestamp": 0}]}
        query = {"query": "deploycobalt", "user_id": user, "top_k": 1}
        status, ack = http("/add", payload, token)
        if status != 200 or ack.get("success") is not True:
            raise RuntimeError("Synthetic Add failed")
        before = http("/search", query, token)
        if before[0] != 200 or not before[1]["data"] or "deploycobalt" not in before[1]["data"][0]["content"]:
            raise RuntimeError("Synthetic immediate Search failed")
        if http("/add", payload, token) != (200, ack):
            raise RuntimeError("Synthetic identical retry failed")
        if http("/search", dict(query, user_id=user + "/other-run"), token) != (200, {"data": []}):
            raise RuntimeError("Complete user isolation failed")
        if http("/search", query)[0] != 401:
            raise RuntimeError("Authentication enforcement failed")
        with sqlite3.connect("file:" + str(DATABASE / "memory.sqlite3") + "?mode=ro", uri=True) as connection:
            records = connection.execute("SELECT fingerprint,dimension,COUNT(*) FROM vectors GROUP BY fingerprint,dimension").fetchall()
            count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if records != [(FINGERPRINT, 384, 1)] or count != 1:
            raise RuntimeError("Production vector index or idempotency count failed")
        run(["systemctl", "restart", "aml-memory"], timeout=90)
        healthy()
        if http("/search", query, token) != before or http("/add", payload, token) != (200, ack):
            raise RuntimeError("Restart evidence/idempotency changed")
        after_credential = ENV_FILE.read_bytes()
        if not hmac.compare_digest(credential, after_credential):
            raise RuntimeError("Credential file changed")
        result.update({"activated": True, "service_active_and_healthy": True,
            "add_search_retry_scope_auth_passed": True, "restart_evidence_preserved": True,
            "vector_fingerprint_dimension_count_verified": True, "stored_synthetic_messages": count,
            "runtime_env_bytes_unchanged": True, "runtime_env_sha_unchanged": digest(ENV_FILE) == digest(private / "runtime.env"),
            "token_unchanged": hmac.compare_digest(token, token_from(after_credential)),
            "credential_file_mode": oct(ENV_FILE.stat().st_mode & 0o777),
            "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        write_private(private / "activation-report.json", result)
        return {"activated": True, "run_id": args.run_id, "fingerprint": FINGERPRINT,
                "private_report": str(private / "activation-report.json"), "token_unchanged": True}
    except BaseException as error:
        # A failed start or assertion must restore a running lexical service,
        # not leave the prior deployment stopped. SIGKILL/power loss cannot be
        # caught; preserved paths support the explicit rollback command.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        result.update({"activated": False, "error_type": type(error).__name__})
        if switched:
            try:
                restore(args.run_id)
                result["rollback_restored_and_healthy"] = True
            except BaseException as rollback_error:
                result["rollback_restored_and_healthy"] = False
                result["rollback_error_type"] = type(rollback_error).__name__
        write_private(private / "activation-report.json", result)
        raise RuntimeError("Activation failed; inspect private recovery report") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "activate", "rollback"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-stage", default="/srv/aml-data/capacity-selected-96124a78")
    parser.add_argument("--environment-stage", default="/srv/aml-data/capacity-semantic")
    args = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r"e5small-96124a78-[A-Za-z0-9-]+", args.run_id):
        parser.error("Root and a unique reviewed e5small-96124a78 run id are required")
    os.umask(0o077)
    def interrupted(signum, frame):
        raise InterruptedError("Deployment interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        outcome = prepare(args) if args.mode == "prepare" else (
            activate(args) if args.mode == "activate" else restore(args.run_id))
        print(json.dumps(outcome))
    except BaseException as error:
        print(json.dumps({"completed": False, "mode": args.mode,
                          "run_id": args.run_id, "error_type": type(error).__name__}))
        sys.exit(1)


if __name__ == "__main__":
    main()
