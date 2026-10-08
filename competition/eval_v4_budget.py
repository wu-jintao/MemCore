#!/usr/bin/env python3
"""Budgeted text-embedding-v4 PUBLIC dev retrieval; dry-run unless --execute.

The model sees only prepared historical message bodies and original queries.
Gold stays in the existing local grader. No validation/heldout or AML calls.
Repeated --dataset flags share one cumulative token/request/deadline ledger.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shlex
import shutil
import stat
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIVATE_ROOT = ROOT / ".local"
sys.path.insert(0, str(ROOT))
from semantic import EmbeddingConfig, EmbeddingError, HttpEmbeddingBackend, normalize_vector, _utf8_segments
from server import MemoryStore, make_server
from eval_compare import compare
from eval_prepare import MANIFEST, SOURCE_MAPPING_VERSION, file_hash
from eval_retrieval import HTTPClient, KS, chunks, read_jsonl, run_evaluation

MODEL = "text-embedding-v4"
DIMENSION = 2048
BATCH_SIZE = 10
SEGMENT_BYTES = 1536
ALLOWED_DATA_ROOTS = (
    PRIVATE_ROOT / "public-data-occurrence-v2",
    PRIVATE_ROOT / "calibration-20261006" / "locomo-data-v2",
)


class BudgetStopped(EmbeddingError):
    """Terminal run stop; another upstream request is forbidden."""


def atomic_json(path, value):
    """Private, fsynced file replacement; never persist credentials or prompts."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(6))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/").split("/")[-1] != "embeddings"
            or "agentmemoryleaderboard.ai" in parsed.hostname.casefold()):
        raise ValueError("An explicit HTTPS embeddings endpoint without credentials is required")
    return endpoint


def read_model_key(path, endpoint, key_name):
    """Parse literal dotenv assignments without shell evaluation or env export."""
    path = Path(path)
    information = path.lstat()
    if not stat.S_ISREG(information.st_mode):
        raise ValueError("Model environment file must be a private regular file")
    # O_NOFOLLOW and fstat keep the permission check bound to the opened file,
    # including if a path changes between a caller's preflight and this read.
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        raise ValueError("Model environment file must be an accessible private regular file") from None
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        information = os.fstat(source.fileno())
        if (not stat.S_ISREG(information.st_mode) or information.st_mode & 0o077
                or information.st_uid != os.getuid()):
            raise ValueError("Model environment file must be a private regular file (0600)")
        raw = source.read(65537)
        if information.st_size > 65536 or len(raw.encode("utf-8")) > 65536:
            raise ValueError("Model environment file is too large")
    values = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, literal = line.partition("=")
        name = name.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in values:
            raise ValueError("Invalid or duplicate model environment assignment")
        try:
            words = shlex.split(literal, comments=True, posix=True)
        except ValueError:
            raise ValueError("Invalid literal model environment assignment") from None
        if len(words) > 1:
            raise ValueError("Model environment values must be literal strings")
        values[name] = words[0] if words else ""
    fixed = {"MEMORY_EMBEDDING_MODEL": MODEL, "MEMORY_EMBEDDING_DIMENSION": "2048",
             "MEMORY_EMBEDDING_BATCH_SIZE": "10", "MEMORY_EMBEDDING_CONCURRENCY": "1",
             "MEMORY_EMBEDDING_HTTP_RETRIES": "0", "MEMORY_EMBEDDING_URL": endpoint,
             "MEMORY_EMBEDDING_QUERY_PREFIX": "", "MEMORY_EMBEDDING_DOCUMENT_PREFIX": "",
             "MEMORY_EMBEDDING_HTTP_SEGMENT_BYTES": "1536"}
    if any(name in values and values[name] != expected for name, expected in fixed.items()):
        raise ValueError("Model environment conflicts with the fixed v4 experiment")
    key = values.get(key_name, "")
    if not key or "\n" in key or "\r" in key:
        raise ValueError("The selected API key assignment is missing or invalid")
    return key


class BudgetLedger:
    def __init__(self, path, max_input_tokens, max_requests, max_runtime_seconds, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        self.start = clock()
        self.deadline = self.start + max_runtime_seconds
        self.lock = threading.RLock()
        self.state = {"schema": "v4-public-dev-budget-ledger-v1", "model": MODEL,
                      "dimension": DIMENSION, "batch_size": BATCH_SIZE, "concurrency": 1,
                      "retries": 0, "reservation": "sum(utf8_bytes + 32 per input)",
                      "max_input_tokens": max_input_tokens, "max_requests": max_requests,
                      "max_runtime_seconds": max_runtime_seconds, "stopped": False,
                      "stop_reason": None, "calls": [], "stop_events": []}
        self._save()

    def _save(self):
        self.state["elapsed_seconds"] = max(0, self.clock() - self.start)
        calls = self.state["calls"]
        self.state["attempted_requests"] = len(calls)
        self.state["successful_requests"] = sum(c["status"] == "ok" for c in calls)
        self.state["failed_requests"] = sum(c["status"] not in ("ok", "pending") for c in calls)
        self.state["reported_input_tokens"] = sum(c.get("actual_input_tokens") or 0 for c in calls)
        self.state["accounted_input_tokens"] = sum(c.get("accounted_input_tokens", c["reserved_input_tokens"]) for c in calls)
        self.state["unknown_usage_requests"] = sum(c["status"] != "pending" and c.get("actual_input_tokens") is None for c in calls)
        atomic_json(self.path, self.state)

    @property
    def stopped(self):
        with self.lock:
            return self.state["stopped"]

    def remaining_seconds(self):
        return max(0.0, self.deadline - self.clock())

    def stop(self, reason):
        with self.lock:
            if not self.state["stopped"]:
                self.state["stopped"] = True
                self.state["stop_reason"] = reason
                self.state["stop_events"].append({"reason": reason, "elapsed_seconds": self.clock() - self.start})
            self._save()

    def begin(self, texts, body_hash):
        reservation = sum(len(text.encode("utf-8")) + 32 for text in texts)
        with self.lock:
            if self.stopped:
                raise BudgetStopped("Upstream calls are locked after a terminal stop")
            reason = None
            if self.remaining_seconds() <= 0:
                reason = "runtime_limit"
            elif len(self.state["calls"]) >= self.state["max_requests"]:
                reason = "request_limit"
            elif self.state["accounted_input_tokens"] + reservation > self.state["max_input_tokens"]:
                reason = "token_reservation_limit"
            if reason:
                self.stop(reason)
                raise BudgetStopped("The next batch exceeds the fixed run budget")
            call = {"attempt": len(self.state["calls"]) + 1, "input_count": len(texts),
                    "input_utf8_bytes": reservation - 32 * len(texts),
                    "reserved_input_tokens": reservation, "request_body_sha256": body_hash,
                    "status": "pending", "actual_input_tokens": None,
                    "transport_started": False,
                    "started_seconds": self.clock() - self.start}
            self.state["calls"].append(call)
            self._save()  # Reservation is durable before the network attempt.
            return len(self.state["calls"]) - 1

    def finish(self, index, status, usage=None, http_status=None):
        with self.lock:
            call = self.state["calls"][index]
            if call["status"] != "pending":
                raise RuntimeError("An upstream attempt may be finalized only once")
            call.update(status=status, http_status=http_status,
                        elapsed_seconds=self.clock() - self.start - call["started_seconds"])
            if usage is not None:
                call["actual_input_tokens"] = usage["input_tokens"]
                call["reported_total_tokens"] = usage["total_tokens"]
                call["accounted_input_tokens"] = max(usage.values())
            else:
                call["accounted_input_tokens"] = call["reserved_input_tokens"]
            if usage is not None and max(usage.values()) > call["reserved_input_tokens"]:
                call["status"] = status = "usage_exceeds_reservation"
            if self.remaining_seconds() <= 0 and status == "ok":
                call["status"] = status = "runtime_limit"
            if status != "ok":
                self.stop(status)
            else:
                self._save()
            return status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "Embedding redirects are forbidden", headers, response)


def https_transport(request, timeout, limit):
    # No environment proxy, redirect, implicit retry, or additional endpoint.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return response.status, response.read(limit + 1)


def parse_usage(payload):
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        raise ValueError("missing_usage")
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    total = usage.get("total_tokens", prompt)
    if type(prompt) is not int or type(total) is not int or prompt <= 0 or total < prompt:
        raise ValueError("invalid_usage")
    return {"input_tokens": prompt, "total_tokens": total}


class BudgetedV4Backend(HttpEmbeddingBackend):
    def __init__(self, config, ledger, transport=https_transport):
        if (config.model != MODEL or config.dimension != DIMENSION or config.batch_size != BATCH_SIZE
                or config.concurrency != 1 or config.http_retries != 0
                or config.http_segment_bytes != SEGMENT_BYTES or config.query_prefix or config.document_prefix):
            raise ValueError("This experiment requires the fixed v4 configuration")
        validate_endpoint(config.endpoint)
        super().__init__(config)
        self.ledger, self.transport = ledger, transport

    def _embed(self, texts, query):
        try:
            return super()._embed(texts, query)
        except Exception:
            # Capacity/operation failures before or after a HTTP batch must not
            # leave a partially failed Add free to spend on another history.
            if not self.ledger.stopped:
                self.ledger.stop("embedding_operation_failure")
            raise

    def _encode_batch(self, texts, deadline):
        payload = {"model": MODEL, "input": texts, "encoding_format": "float", "dimensions": DIMENSION}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        index = self.ledger.begin(texts, hashlib.sha256(body).hexdigest())
        request = Request(self.config.endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.config.api_key})
        limit = min(16 * 1024 * 1024, len(texts) * DIMENSION * 40 + 65536)
        remaining = min(self.ledger.remaining_seconds(), max(0.0, deadline - time.monotonic()), self.config.http_timeout)
        outcome = queue.Queue(maxsize=1)

        def send_once():
            try:
                if self.ledger.stopped or self.ledger.remaining_seconds() <= 0:
                    raise BudgetStopped("Deadline reached before transport")
                # Recheck the remaining budget at the actual transport boundary;
                # durable reservation/fsync and thread scheduling consume time.
                with self.ledger.lock:
                    if self.ledger.stopped or self.ledger.remaining_seconds() <= 0:
                        raise BudgetStopped("Deadline reached before transport")
                    self.ledger.state["calls"][index]["transport_started"] = True
                outcome.put((True, self.transport(request, min(self.config.http_timeout, self.ledger.remaining_seconds()), limit)))
            except BaseException as error:
                outcome.put((False, error))

        # A daemon permits a strict caller deadline even if a transport ignores its
        # socket timeout. There is at most one in-flight call and no successor.
        usage, http_status = None, None
        try:
            if remaining <= 0:
                raise ValueError("runtime_limit")
            thread = threading.Thread(target=send_once, daemon=True)
            thread.start()
            try:
                good, result = outcome.get(timeout=remaining)
            except queue.Empty:
                raise ValueError("runtime_limit" if self.ledger.remaining_seconds() <= 0 else "transport_timeout") from None
            if not good:
                if isinstance(result, BudgetStopped):
                    raise ValueError("runtime_limit")
                if isinstance(result, HTTPError):
                    http_status = result.code
                    result.close()
                    raise ValueError("http_" + str(http_status))
                raise ValueError("transport_failure")
            http_status, raw = result
            if http_status != 200:
                raise ValueError("http_" + str(http_status))
            if len(raw) > limit:
                raise ValueError("response_size_limit")
            try:
                response = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                raise ValueError("malformed_response") from None
            usage = parse_usage(response)
            if response.get("model") != MODEL:
                raise ValueError("response_model_mismatch")
            data = response.get("data")
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError("invalid_embedding_count")
            ordered = [None] * len(texts)
            for item in data:
                if (not isinstance(item, dict) or type(item.get("index")) is not int
                        or not 0 <= item["index"] < len(texts) or ordered[item["index"]] is not None):
                    raise ValueError("invalid_embedding_indexes")
                ordered[item["index"]] = normalize_vector(item.get("embedding"), DIMENSION)
            status = self.ledger.finish(index, "ok", usage, http_status)
            if status != "ok":
                raise BudgetStopped("The upstream attempt exceeded its reservation or deadline")
            return ordered
        except BudgetStopped:
            raise
        except Exception as error:
            reason = str(error) if isinstance(error, ValueError) and re.fullmatch(r"[a-z_]+[0-9]*", str(error)) else "malformed_embedding_response"
            self.ledger.finish(index, reason, usage, http_status)
            raise BudgetStopped("The upstream attempt failed; all further model calls are blocked") from None


def source_plan(args, dataset, permitted_roots=ALLOWED_DATA_ROOTS):
    root = Path(args.data_root).resolve()
    if root not in {Path(path).resolve() for path in permitted_roots}:
        raise ValueError("Only the pinned existing public dev roots are allowed")
    summary = json.loads((root / "prepared" / "summary.json").read_text())
    if summary.get("status") != "public_proxy_not_official_aml" or summary.get("source_mapping_version") != SOURCE_MAPPING_VERSION:
        raise ValueError("A public session-occurrence-v2 preparation is required")
    spec = summary["datasets"][dataset]["splits"]["dev"]
    folder = root / "prepared" / dataset / "dev"
    for name in ("histories", "queries", "gold"):
        source = folder / (name + ".jsonl")
        if source.resolve() != source or file_hash(source) != spec["files"][name]["sha256"]:
            raise ValueError("Prepared public dev data checksum mismatch")
    histories = read_jsonl(folder / "histories.jsonl")
    maximum = args.max_histories if args.max_histories is not None else (args.locomo_max_histories if dataset == "locomo_refined" else args.lme_max_histories)
    histories = histories[:maximum]
    selected = {history["sample_id"] for history in histories}
    queries = sorted(read_jsonl(folder / "queries.jsonl"), key=lambda q: (q["sample_id"], q["query_id"]))
    queries = [query for query in queries if query["sample_id"] in selected]
    if args.max_queries:
        queries = queries[:args.max_queries]
    selected = {query["sample_id"] for query in queries}
    histories = [history for history in histories if history["sample_id"] in selected]
    if not queries:
        raise ValueError("No public dev queries selected")
    for history in histories:
        sessions = [s["session_id"] for s in history["sessions"]]
        if len(sessions) != len(set(sessions)):
            raise ValueError("Duplicate source sessions require occurrence-v2 migration")
    texts = [message["content"] for history in histories for session in history["sessions"] for message in session["messages"]]
    def segments(text, query=False):
        return _utf8_segments(text, SEGMENT_BYTES, 8 if query else 512)
    planned_reservation, planned_requests = 0, 0
    for history in histories:
        for session in history["sessions"]:
            for _, messages in chunks(session):
                batch_segments = [part for message in messages for part in segments(message["content"])]
                if len(batch_segments) > 4096:
                    raise ValueError("A prepared Add exceeds the fixed segmentation capacity")
                planned_reservation += sum(len(part.encode("utf-8")) + 32 for part in batch_segments)
                planned_requests += math.ceil(len(batch_segments) / BATCH_SIZE)
    for query in queries:
        parts = segments(query["query"], query=True)
        planned_reservation += sum(len(part.encode("utf-8")) + 32 for part in parts)
        planned_requests += math.ceil(len(parts) / BATCH_SIZE)
    return {"dataset": dataset, "split": "dev", "max_histories": maximum,
            "max_queries": args.max_queries, "histories": len(histories), "questions": len(queries),
            "messages": len(texts), "source_utf8_bytes": sum(len(text.encode("utf-8")) for text in texts),
            "planning_reservation_upper_estimate": planned_reservation,
            "planned_embedding_requests_without_failures": planned_requests,
            "selection_sha256": hashlib.sha256(json.dumps([q["query_id"] for q in queries]).encode()).hexdigest(),
            "prepared_data": spec, "token_estimate_is_provider_usage": False}


@contextmanager
def loopback_store(database, backend=None):
    store = MemoryStore(database, semantic_backend=backend, context_radius=1,
                        semantic_weight=1, context_grouping="reserved")
    token = secrets.token_urlsafe(32)
    service = make_server(("127.0.0.1", 0), store, api_token=token)
    worker = threading.Thread(target=service.serve_forever, daemon=True)
    worker.start()
    try:
        yield HTTPClient("http://127.0.0.1:" + str(service.server_address[1]), token, 1500)
    finally:
        service.shutdown()
        service.server_close()
        worker.join(timeout=5)


def run_side(args, plan, output, run_id, backend=None):
    output.mkdir(mode=0o700)
    database = output / "memory.sqlite3"
    snapshots = output / "source_snapshots"
    snapshots.mkdir(mode=0o700)
    for path in (ROOT / "server.py", ROOT / "semantic.py", HERE / "eval_retrieval.py",
                 HERE / "eval_prepare.py", HERE / "eval_compare.py", Path(__file__), MANIFEST):
        shutil.copyfile(path, snapshots / path.name)
        os.chmod(snapshots / path.name, 0o600)
    start_hashes = {name: file_hash(ROOT / name) for name in ("server.py", "semantic.py")}
    run_args = SimpleNamespace(data_root=Path(args.data_root), dataset=plan["dataset"], split="dev",
                              max_histories=plan["max_histories"], max_queries=plan["max_queries"],
                              final_heldout=False, top_k=100, character_budget=0, search_concurrency=1,
                              model_label="v4-equal-rrf-context1-reserved-cap500k" if backend else "lexical-context1-reserved-cap500k")
    started = time.monotonic()
    with loopback_store(database, backend) as client:
        summary, specification = run_evaluation(run_args, client, run_id, output)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["scope"] = "public_dev_retrieval_v4_embedding_only_no_answer_judge_or_official_evaluation" if backend else "public_dev_lexical_retrieval_only"
    summary["configuration"] = {"prepared_data": specification, "top_k": 100, "character_budget": 0,
                                "source_mapping_version": SOURCE_MAPPING_VERSION, "launch_local": True,
                                "embedding_provider": "http" if backend else "disabled", "model": MODEL if backend else None,
                                "dimension": DIMENSION if backend else None, "context_radius": 1,
                                "context_grouping": "reserved", "semantic_weight": 1,
                                "search_concurrency": 1, "add_concurrency": 1,
                                "max_histories": plan["max_histories"], "max_queries": plan["max_queries"],
                                "source_code_sha256": {p.name: file_hash(p) for p in snapshots.iterdir()}}
    end_hashes = {name: file_hash(ROOT / name) for name in ("server.py", "semantic.py")}
    summary["configuration"]["core_source_sha256_at_start"] = start_hashes
    summary["configuration"]["core_source_sha256_at_end"] = end_hashes
    summary["server_source_changed_during_run"] = start_hashes != end_hashes
    atomic_json(output / "summary.json", summary)
    if summary["server_source_changed_during_run"]:
        raise ValueError("Core source changed during the evaluation")
    return summary


def execute(args, plans, output, transport=https_transport, key_reader=read_model_key):
    ledger = BudgetLedger(output / "ledger.json", args.max_input_tokens, args.max_requests, args.max_runtime_seconds)
    report = {"status": "running_public_dev_proxy", "official_score": None, "runs": [], "skipped": [],
              "no_answer_judge": True, "plans": plans}
    deadline_timer = threading.Timer(args.max_runtime_seconds, ledger.stop, args=("runtime_limit",))
    deadline_timer.daemon = True
    deadline_timer.start()
    try:
        key = key_reader(args.model_env_file, args.endpoint, args.api_key_name)
        config = EmbeddingConfig(provider="http", model=MODEL, dimension=DIMENSION, batch_size=BATCH_SIZE,
                                 concurrency=1, http_retries=0, endpoint=args.endpoint, api_key=key, allow_http=True,
                                 http_segment_bytes=SEGMENT_BYTES, operation_timeout=1500, http_timeout=120)
        backend = BudgetedV4Backend(config, ledger, transport)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
        for position, plan in enumerate(plans):
            if ledger.stopped or ledger.remaining_seconds() <= 0:
                if not ledger.stopped:
                    ledger.stop("runtime_limit")
                report["skipped"].extend({"dataset": item["dataset"], "questions": item["questions"], "reason": "terminal_budget_stop"} for item in plans[position:])
                break
            name = plan["dataset"]
            baseline = output / (name + "-lexical")
            candidate = output / (name + "-v4")
            report["active_stage"] = {"dataset": name, "side": "lexical", "directory": str(baseline)}
            atomic_json(output / "report.json", report)
            lexical = run_side(args, plan, baseline, timestamp + "-lexical-" + str(position))
            report["active_stage"] = {"dataset": name, "side": "v4", "directory": str(candidate)}
            atomic_json(output / "report.json", report)
            v4 = run_side(args, plan, candidate, timestamp + "-v4-" + str(position), backend)
            comparison = compare([baseline, candidate], Path(args.data_root), KS, [0])
            atomic_json(output / (name + "-comparison.json"), comparison)
            report["runs"].append({"dataset": name, "lexical_directory": str(baseline), "v4_directory": str(candidate),
                                   "questions": v4["question_count"], "v4_ingestion_failed_histories": v4["ingestion_failed_histories"],
                                   "lexical_successful_queries": lexical["metrics_by_k"]["100"]["successful_questions"],
                                   "v4_successful_queries": v4["metrics_by_k"]["100"]["successful_questions"]})
            atomic_json(output / "report.json", report)
        report["status"] = "stopped_public_dev_proxy" if ledger.stopped else "completed_public_dev_proxy"
        report.pop("active_stage", None)
    except Exception:
        ledger.stop("runner_failure")
        report["status"] = "failed_public_dev_proxy"
        report["error"] = "Evaluation setup or local execution failed; inspect private stage files"
    finally:
        deadline_timer.cancel()
        report["budget"] = {key: value for key, value in ledger.state.items() if key != "calls"}
        atomic_json(output / "report.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", action="append", choices=("locomo_refined", "longmemeval_s"), required=True,
                        help="Repeat LoCoMo then LME to share one cumulative budget")
    result.add_argument("--data-root", type=Path, default=PRIVATE_ROOT / "public-data-occurrence-v2")
    result.add_argument("--model-env-file", type=Path, required=True, help="Read only with --execute; never copied or printed")
    result.add_argument("--api-key-name", choices=("MEMORY_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY"), default="MEMORY_EMBEDDING_API_KEY")
    result.add_argument("--endpoint", required=True)
    result.add_argument("--max-input-tokens", type=int, required=True)
    result.add_argument("--max-requests", type=int, required=True)
    result.add_argument("--max-runtime-seconds", type=float, default=1800)
    result.add_argument("--max-histories", type=int, help="Single-dataset runs only; fixed prepared prefix")
    result.add_argument("--max-queries", type=int, default=0, help="Single-dataset runs only; 0 means all queries in selected histories")
    result.add_argument("--locomo-max-histories", type=int, default=6)
    result.add_argument("--lme-max-histories", type=int, default=5)
    result.add_argument("--output-dir", type=Path, required=True, help="New private directory under starter/.local")
    result.add_argument("--execute", action="store_true", help="Explicitly enable v4 HTTPS calls; otherwise plan without reading credentials")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    validate_endpoint(args.endpoint)
    if (args.max_input_tokens <= 0 or args.max_requests <= 0 or not math.isfinite(args.max_runtime_seconds)
            or args.max_runtime_seconds <= 0 or args.max_queries < 0
            or min(args.locomo_max_histories, args.lme_max_histories) <= 0
            or (args.max_histories is not None and args.max_histories <= 0)):
        raise ValueError("Budgets, deadline and history limits must be positive")
    if len(set(args.dataset)) != len(args.dataset) or (len(args.dataset) > 1 and (args.max_histories is not None or args.max_queries)):
        raise ValueError("Repeated datasets must be unique and use per-dataset history limits")
    if args.dataset != sorted(args.dataset, key=lambda name: name != "locomo_refined"):
        raise ValueError("Combined runs must evaluate LoCoMo before LME")
    plans = [source_plan(args, dataset) for dataset in args.dataset]
    output = args.output_dir.resolve()
    if not output.is_relative_to(PRIVATE_ROOT.resolve()) or output == PRIVATE_ROOT.resolve():
        raise ValueError("Outputs must be under the private starter/.local directory")
    previous_mask = os.umask(0o077)
    try:
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        plan = {"status": "public_dev_plan_no_model_calls", "model": MODEL, "dimension": DIMENSION,
                "batch_size": 10, "concurrency": 1, "retries": 0, "segment_bytes": SEGMENT_BYTES,
                "endpoint": args.endpoint, "max_input_tokens": args.max_input_tokens,
                "max_requests": args.max_requests, "max_runtime_seconds": args.max_runtime_seconds,
                "datasets": plans, "credential_file_read": False,
                "limitations": ["UTF-8 reservation is conservative planning, not the provider tokenizer.",
                                "Actual per-call usage is required; unknown usage or any failed request stops upstream calls.",
                                "Evidence recall is not answer correctness or an official AML score."]}
        atomic_json(output / "plan.json", plan)
        report = execute(args, plans, output) if args.execute else plan
        print(json.dumps({"status": report["status"], "output_directory": str(output),
                          "model_requests": report.get("budget", {}).get("attempted_requests", 0)}, ensure_ascii=False))
        return 0 if report["status"] in ("public_dev_plan_no_model_calls", "completed_public_dev_proxy") else 2
    finally:
        os.umask(previous_mask)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print(json.dumps({"status": "rejected", "error": "Invalid public-dev source, private destination, endpoint or fixed experiment configuration"}))
        sys.exit(2)
