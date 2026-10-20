#!/usr/bin/env python3
"""HLE inference and official-prompt judging against OpenAI-compatible APIs."""
import argparse
import ast
import base64
from contextlib import contextmanager
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
import warnings

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen3.6-35B-A3B"
DEFAULT_BASE = "http://10.119.27.14:8200/v1"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "integer"},
        "strict": {"type": "boolean", "enum": [True]},
    },
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
    "additionalProperties": False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_entries(path):
    if path.exists():
        with path.open() as handle:
            for number, line in enumerate(handle, 1):
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{number}; preserve and repair it before resuming") from exc
                yield item


def read_records(path):
    return {item["id"]: item for item in read_entries(path)}


def append_record(path, item):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def model_directory(model):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "__", model).strip(".")
    if not value:
        raise ValueError("Model name cannot be used as a directory name")
    return value


def run_directory(args):
    name = "hle" + (f"-{args.subset}" if args.subset else "")
    return args.results_root.resolve() / model_directory(args.model) / name


def load_constant(path, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        nodes = ast.parse(path.read_text()).body
    for node in nodes:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"Missing {name} in {path}")


def official_protocol():
    checkout = ROOT / "third_party/hle"
    prediction = checkout / "hle_eval/run_model_predictions.py"
    judge = checkout / "hle_eval/run_judge_results.py"
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    lock = json.loads((ROOT / "archive/benchmarks.lock.json").read_text())
    if commit != lock["datasets"]["hle"]["code_commit"]:
        raise ValueError("Official HLE checkout differs from the pinned commit")
    return {
        "commit": commit,
        "prediction_source_sha256": digest(prediction),
        "judge_source_sha256": digest(judge),
        "system_prompt": load_constant(prediction, "SYSTEM_PROMPT"),
        "judge_prompt": load_constant(judge, "JUDGE_PROMPT"),
        "answer_extraction": "Official JUDGE_PROMPT extracts final answer and confidence; no regex or exact-match replacement",
    }


def load_dataset(scope, limit, subset=None):
    path = ROOT / "datasets/hle" / ("text_only.jsonl" if scope == "text-only" else "test.jsonl")
    manifest = json.loads((ROOT / "archive/manifests/core-files.sha256.json").read_text())
    relative = str(path.relative_to(ROOT / "datasets"))
    expected = next(row for row in manifest["files"] if row["path"] == relative)
    sha = digest(path)
    if sha != expected["sha256"]:
        raise ValueError("HLE JSONL differs from the pinned dataset manifest")
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate HLE task IDs")
    expected_count = 2158 if scope == "text-only" else 2500
    if len(rows) != expected_count or (scope == "text-only" and any(row.get("image") for row in rows)):
        raise ValueError("HLE dataset scope or task count is invalid")
    parent_count = len(rows)
    subset_info = {}
    if subset:
        if scope != "text-only":
            raise ValueError("The fixed diagnostic subset requires --scope text-only")
        subset_manifest = ROOT / "archive/manifests/hle-text-only-10pct-seed42.json"
        fixed = json.loads(subset_manifest.read_text())
        if fixed["name"] != subset or fixed["parent_sha256"] != sha or fixed["parent_size"] != parent_count:
            raise ValueError("The fixed subset does not match its pinned parent dataset")
        path = ROOT / "datasets/hle/text_only_10pct_seed42.jsonl"
        if fixed["path"] != str(path.relative_to(ROOT)) or digest(path) != fixed["sha256"]:
            raise ValueError("HLE subset differs from its pinned manifest")
        with path.open() as handle:
            subset_rows = [json.loads(line) for line in handle]
        parent = {row["id"]: row for row in rows}
        if (len(subset_rows) != 216 or fixed["size"] != 216
                or len(set(fixed["task_ids"])) != 216
                or [r["id"] for r in subset_rows] != fixed["task_ids"]
                or any(parent.get(row["id"]) != row for row in subset_rows)):
            raise ValueError("HLE subset IDs or content differ from the fixed parent rows")
        rows, sha = subset_rows, fixed["sha256"]
        subset_info = {"subset": subset, "subset_manifest": str(subset_manifest.relative_to(ROOT)),
            "subset_manifest_sha256": digest(subset_manifest), "selection_seed": fixed["seed"],
            "selection_method": fixed["method"], "subset_available_questions": len(rows)}
    selected = rows[:limit] if limit else rows
    lock = json.loads((ROOT / "archive/benchmarks.lock.json").read_text())["datasets"]["hle"]
    info = {
        "repo": lock["hf_repo"], "revision": lock["hf_revision"], "split": "test",
        "scope": scope, "includes_images": scope == "full", "path": str(path.relative_to(ROOT)),
        "sha256": sha, "available_questions": parent_count, "selected_questions": len(selected),
        "complete_scope": not subset and len(selected) == parent_count,
        "task_ids": [row["id"] for row in selected], **subset_info,
    }
    return selected, info


def question_messages(row, system_prompt):
    content = [{"type": "text", "text": row["question"]}]
    if row.get("image"):
        path = (ROOT / "datasets/hle" / row["image"]).resolve()
        if not path.is_relative_to((ROOT / "datasets/hle/images").resolve()):
            raise ValueError("Image path escapes the HLE image directory")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    # Answers, rationales, image previews and judge instructions never enter generation requests.
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


class APIError(Exception):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


class Client:
    def __init__(self, base_url, key_env, timeout, retries):
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment or parts.username:
            raise ValueError("Use an HTTP(S) base URL without credentials or query parameters")
        self.base = base_url.rstrip("/")
        if not parts.path.rstrip("/"):
            self.base += "/v1"
        self.key = os.environ.get(key_env, "")
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    def post(self, path, body, tokenize=False):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.trust_env = False
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        url = (self.base[:-3] if self.base.endswith("/v1") else self.base) + path if tokenize else self.base + path
        attempts = []
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            retryable = True
            try:
                response = self.local.session.post(url, json=body, headers=headers,
                    timeout=(10, min(self.timeout, 60) if tokenize else self.timeout), allow_redirects=False)
                if response.status_code != 200:
                    retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                    raise ValueError(f"HTTP {response.status_code}: {response.text[:2000]}")
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("API returned a non-object JSON response")
                attempts.append({"attempt": attempt + 1, "status": "ok", "elapsed_sec": time.monotonic() - started})
                return data, attempts
            except (requests.RequestException, ValueError) as exc:
                error = str(exc)
                if self.key:
                    error = error.replace(self.key, "[REDACTED]")
                attempts.append({"attempt": attempt + 1, "status": "error", "elapsed_sec": time.monotonic() - started, "error": error})
                if not retryable or attempt == self.retries:
                    raise APIError(error, attempts) from exc
                time.sleep(min(2 ** attempt, 10))


class EndpointPool:
    """Lease one endpoint for both tokenization and generation of a task."""

    def __init__(self, bases, key_env, timeout, retries, workers_per_endpoint, registry=None, model=None):
        self.clients = [Client(base, key_env, timeout, retries) for base in bases]
        self.bases = [client.base for client in self.clients]
        if len(set(self.bases)) != len(self.bases):
            raise ValueError("Generation endpoints must be distinct")
        self.registry = Path(registry).resolve() if registry else None
        self.registry_configuration = None
        self.registry_last_checked = 0
        self.registry_lock = threading.Lock()
        self.registered_jobs = {}
        self.client_options = (key_env, timeout, retries)
        self.workers_per_endpoint = workers_per_endpoint
        replica_count = 0
        if self.registry:
            manifest = json.loads(self.registry.read_text())
            self.registry_configuration = self.registry_identity(manifest)
            if manifest["model"] != model:
                raise ValueError("Replica registry model differs from the tested model")
            replica_count = len(manifest["replicas"])
        self.workers = workers_per_endpoint * (len(self.clients) + replica_count)
        if not self.workers:
            raise ValueError("No generation endpoints configured")
        self.slots = Queue()
        for client in self.clients:
            for _ in range(workers_per_endpoint):
                self.slots.put(client)

    def registry_identity(self, manifest):
        jobs = [row["name"] for row in manifest["replicas"]]
        if len(set(jobs)) != len(jobs) or len(jobs) != manifest["new_replica_count"]:
            raise ValueError("Invalid or duplicate replica jobs in registry")
        return {"path": str(self.registry), "created_at": manifest["created_at"],
            "model": manifest["model"], "jobs": jobs}

    def refresh(self):
        if not self.registry:
            return
        with self.registry_lock:
            checked = time.monotonic()
            if checked - self.registry_last_checked < 1:
                return
            manifest = json.loads(self.registry.read_text())
            if self.registry_identity(manifest) != self.registry_configuration:
                raise ValueError("Replica registry identity changed during evaluation")
            for row in manifest["replicas"]:
                if not row.get("ready"):
                    continue
                client = Client(row["base_url"], *self.client_options)
                previous = self.registered_jobs.get(row["name"])
                if previous:
                    if previous != client.base:
                        raise ValueError("A registered replica changed its endpoint")
                    continue
                if client.base in self.bases:
                    raise ValueError("Replica registry contains a duplicate endpoint")
                self.clients.append(client)
                self.bases.append(client.base)
                self.registered_jobs[row["name"]] = client.base
                for _ in range(self.workers_per_endpoint):
                    self.slots.put(client)
            self.registry_last_checked = checked

    @contextmanager
    def lease(self):
        while True:
            self.refresh()
            try:
                client = self.slots.get(timeout=1)
                break
            except Empty:
                pass
        try:
            yield client
        finally:
            self.slots.put(client)


def raw_path(folder, stage, task_id):
    name = hashlib.sha256(task_id.encode()).hexdigest()[:24] + f"-{time.time_ns()}.json"
    return folder / "raw" / stage / name


def generate_one(row, args, client, protocol, folder):
    started = time.monotonic()
    result = {"id": row["id"], "status": "error", "started_at": now()}
    try:
        messages = question_messages(row, protocol["system_prompt"])
        template = {} if args.thinking == "default" else {"enable_thinking": args.thinking == "on"}
        counted, attempts = client.post("/tokenize", {
            "model": args.model, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": template,
        }, tokenize=True)
        prompt_tokens = counted["count"]
        server_limit = counted.get("max_model_len") or args.context_limit
        budget = min(args.max_output_tokens, min(args.context_limit, server_limit) - prompt_tokens)
        result.update({"prompt_tokens": prompt_tokens, "max_output_tokens": budget, "tokenize_attempts": attempts})
        if budget <= 0:
            raise ValueError("Prompt exhausts the configured or server context window")
        request = {"model": args.model, "messages": messages, "max_tokens": budget,
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "seed": args.seed, "stream": False, "chat_template_kwargs": template}
        response, attempts = client.post("/chat/completions", request)
        path = raw_path(folder, "generate", row["id"])
        write_json(path, response)
        result.update({"raw_response": str(path.relative_to(folder)), "attempts": attempts})
        choice = response["choices"][0]
        message = choice["message"]
        text = message.get("content")
        result.update({"response": text or "", "finish_reason": choice.get("finish_reason"),
            "truncated": choice.get("finish_reason") == "length", "usage": response.get("usage", {})})
        if message.get("tool_calls"):
            raise ValueError("Unexpected tool calls in a no-tools HLE run")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("No final response content (reasoning-only outputs are not final answers)")
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, APIError):
            result["attempts"] = exc.attempts
    result["elapsed_sec"] = time.monotonic() - started
    return result


def judge_one(row, prediction, args, client, protocol, folder):
    started = time.monotonic()
    result = {"id": row["id"], "status": "error", "prediction_sha256": hashlib.sha256(prediction["response"].encode()).hexdigest()}
    try:
        prompt = protocol["judge_prompt"].format(question=row["question"], correct_answer=row["answer"], response=prediction["response"])
        request = {
            "model": args.judge_model, "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": args.judge_max_tokens, "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "ExtractedAnswer", "strict": True, "schema": JUDGE_SCHEMA}},
        }
        if args.judge_temperature is not None:
            request["temperature"] = args.judge_temperature
        response, attempts = client.post("/chat/completions", request)
        path = raw_path(folder, "judge", row["id"])
        write_json(path, response)
        result.update({"raw_response": str(path.relative_to(folder)), "attempts": attempts, "usage": response.get("usage", {})})
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("Judge response was truncated")
        answer = json.loads(choice["message"]["content"])
        if set(answer) != set(JUDGE_SCHEMA["required"]):
            raise ValueError("Judge output fields do not match the official schema")
        if answer["correct"] not in {"yes", "no"} or answer["strict"] is not True:
            raise ValueError("Invalid judge correctness or strict field")
        if type(answer["confidence"]) is not int or not 0 <= answer["confidence"] <= 100:
            raise ValueError("Judge confidence must be an integer in [0, 100]")
        if not isinstance(answer["extracted_final_answer"], str) or not isinstance(answer["reasoning"], str):
            raise ValueError("Judge answer and reasoning must be strings")
        result.update({"status": "ok", "judge_response": answer})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, APIError):
            result["attempts"] = exc.attempts
    result["elapsed_sec"] = time.monotonic() - started
    return result


def save_or_match(path, configuration):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["configuration"] != configuration:
            raise ValueError(f"Configuration differs from {path}; do not mix runs. Move the old result directory before changing settings")
    else:
        write_json(path, {"created_at": now(), "configuration": configuration})


def summarize(folder):
    meta = json.loads((folder / "run.json").read_text())["configuration"]
    ids = meta["dataset"]["task_ids"]
    predictions = read_records(folder / "predictions.jsonl")
    grades = read_records(folder / "grades.jsonl")
    successful, judged, correct = [], [], 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    judge_usage = dict(usage)
    for prediction in read_entries(folder / "predictions.jsonl"):
        if prediction["id"] in ids:
            for key in usage:
                usage[key] += prediction.get("usage", {}).get(key, 0) or 0
    for grade in read_entries(folder / "grades.jsonl"):
        if grade["id"] in ids:
            for key in judge_usage:
                judge_usage[key] += grade.get("usage", {}).get(key, 0) or 0
    for task_id in ids:
        prediction = predictions.get(task_id, {})
        if prediction.get("status") != "ok":
            continue
        successful.append(prediction)
        grade = grades.get(task_id, {})
        if grade.get("status") == "ok" and grade.get("prediction_sha256") == hashlib.sha256(prediction["response"].encode()).hexdigest():
            judged.append(grade)
            correct += grade["judge_response"]["correct"] == "yes"
    n = len(ids)
    pending = len(successful) - len(judged)
    judge_config = json.loads((folder / "judge.json").read_text())["configuration"] if (folder / "judge.json").exists() else None
    complete = judge_config is not None and len(predictions.keys() & set(ids)) == n and pending == 0
    accuracy = 100 * correct / n if complete else None
    summary = {"updated_at": now(), "benchmark": "HLE", "scope": meta["dataset"]["scope"],
        "model": meta["model"], "tools": False, "expected": n,
        "complete_dataset_scope": meta["dataset"]["complete_scope"],
        "generation_attempted": len(predictions.keys() & set(ids)), "generation_succeeded": len(successful),
        "generation_failed": sum(predictions.get(i, {}).get("status") == "error" for i in ids),
        "truncated": sum(bool(predictions.get(i, {}).get("truncated")) for i in ids),
        "judged": len(judged), "judge_failed": sum(grades.get(i, {}).get("status") == "error" for i in ids),
        "pending_judgment": pending, "correct": correct, "accuracy_percent": accuracy,
        "provisional_correct_over_all_percent": 100 * correct / n,
        "scoring_complete": complete, "accuracy_denominator": n,
        "judge": judge_config,
        "generation_usage": usage, "judge_usage": judge_usage,
        "uncertainty": {"method": "official Wald 95% half-width", "half_width_percentage_points":
            1.96 * math.sqrt(accuracy * (100 - accuracy) / n) if accuracy is not None else None},
        "calibration_error": None,
        "note": "All selected tasks remain in the denominator. Accuracy is withheld while generation or judging is pending. GPU cost and calibration error are not estimated."}
    if meta["dataset"].get("subset"):
        summary["subset"] = meta["dataset"]["subset"]
    write_json(folder / "summary.json", summary)
    return summary


def process_jobs(args, rows, operation, output, folder, valid_record=None, workers=None):
    previous = read_records(output)
    pending = [row for row in rows if row["id"] not in previous
        or (args.retry_failed and previous[row["id"]]["status"] != "ok")
        or (valid_record is not None and previous[row["id"]]["status"] == "ok" and not valid_record(previous[row["id"]], row))]
    with ThreadPoolExecutor(max_workers=workers or args.workers) as executor:
        futures = {executor.submit(operation, row): row["id"] for row in pending}
        last_summary = time.monotonic()
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append_record(output, result)
            previous[result["id"]] = result
            progress = {"updated_at": now(), "phase": output.stem, "id": result["id"],
                "status": result["status"], "attempted": len(previous), "expected": len(rows),
                "succeeded": sum(item["status"] == "ok" for item in previous.values()),
                "failed": sum(item["status"] != "ok" for item in previous.values()), "error": result.get("error")}
            write_json(folder / "progress.json", progress)
            if index % 32 == 0 or time.monotonic() - last_summary >= 60:
                summarize(folder)
                last_summary = time.monotonic()
            print(json.dumps(progress, ensure_ascii=False), flush=True)


def generation_configuration(args, dataset, protocol, pool):
    configuration = {"model": args.model, "base_url": None if args.replicas_only else args.base_url[0].rstrip("/"), "dataset": dataset,
        "context_limit": args.context_limit, "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "seed": args.seed, "thinking": args.thinking, "preserve_thinking": "not_applicable_single_turn",
        "tools": False, "workers": args.workers, "timeout_sec": args.timeout,
        "retries": args.retries, "api_key_env": args.api_key_env, "official_protocol": protocol, "runner_sha256": digest(__file__),
        "checkpoint_revision": None, "quantization": "not_reported_by_endpoint"}
    if len(pool.clients) > 1 or pool.registry:
        configuration.update({"base_urls": list(pool.bases), "workers_per_endpoint": args.workers,
            "total_workers": pool.workers, "routing": "shared task queue; endpoint leased for tokenize and completion"})
    if pool.registry:
        configuration["replica_registry"] = pool.registry_configuration
    if args.replicas_only:
        configuration["replicas_only"] = True
    return configuration


def generate(args):
    rows, dataset = load_dataset(args.scope, args.limit, args.subset)
    protocol = official_protocol()
    bases = [] if args.replicas_only else [*args.base_url, *args.replica_base_url]
    pool = EndpointPool(bases, args.api_key_env, args.timeout, args.retries,
        args.workers, args.replica_manifest, args.model)
    folder = run_directory(args)
    configuration = generation_configuration(args, dataset, protocol, pool)
    folder.mkdir(parents=True, exist_ok=True)
    save_or_match(folder / "run.json", configuration)
    summarize(folder)
    write_json(folder / "generation-state.json", {"state": "running", "started_at": now(), "pid": os.getpid()})
    print(json.dumps({"state": "started", "result_directory": str(folder), "questions": len(rows),
        "scope": args.scope, "context_limit": args.context_limit, "max_output_tokens": args.max_output_tokens,
        "workers_per_endpoint": args.workers, "total_workers": pool.workers}, ensure_ascii=False), flush=True)
    def operation(row):
        with pool.lease() as client:
            result = generate_one(row, args, client, protocol, folder)
            result.update({"base_url": client.base, "runner_sha256": configuration["runner_sha256"]})
            return result

    process_jobs(args, rows, operation, folder / "predictions.jsonl", folder, workers=pool.workers)
    summary = summarize(folder)
    write_json(folder / "generation-state.json", {"state": "finished", "finished_at": now(), "pid": os.getpid()})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["generation_failed"] == 0 else 2


def judge_configuration(args, protocol):
    return {"model": args.judge_model, "base_url": args.judge_base_url.rstrip("/"),
        "max_completion_tokens": args.judge_max_tokens, "temperature": args.judge_temperature, "official_commit": protocol["commit"],
        "prompt_sha256": hashlib.sha256(protocol["judge_prompt"].encode()).hexdigest(),
        "schema": JUDGE_SCHEMA, "self_judge_diagnostic": args.judge_model == args.model, "workers": args.workers,
        "timeout_sec": args.timeout, "retries": args.retries, "api_key_env": args.judge_api_key_env,
        "model_revision": args.judge_model_revision}


def judge(args):
    folder = run_directory(args)
    config = json.loads((folder / "run.json").read_text())["configuration"]
    if args.model != config["model"]:
        raise ValueError("Generation model does not match the saved run")
    dataset = config["dataset"]
    rows, loaded = load_dataset(dataset["scope"], dataset["selected_questions"], dataset.get("subset"))
    if loaded != dataset:
        raise ValueError("Dataset changed since generation")
    predictions = read_records(folder / "predictions.jsonl")
    if len(predictions.keys() & set(dataset["task_ids"])) != len(rows):
        raise ValueError("Wait for all generation tasks before starting the judge")
    protocol = official_protocol()
    if protocol != config["official_protocol"]:
        raise ValueError("Official grading protocol changed since generation")
    self_judge = args.judge_model == args.model
    if self_judge and not args.allow_self_judge:
        raise ValueError("Using the tested model as judge requires --allow-self-judge and is a diagnostic only")
    client = Client(args.judge_base_url, args.judge_api_key_env, args.timeout, args.retries)
    save_or_match(folder / "judge.json", judge_configuration(args, protocol))
    rows = [row for row in rows if predictions.get(row["id"], {}).get("status") == "ok"]
    # Failed generation tasks are retained in the accuracy denominator, without judge requests.
    process_jobs(args, rows, lambda row: judge_one(row, predictions[row["id"]], args, client, protocol, folder),
        folder / "grades.jsonl", folder, valid_record=lambda record, row:
        record["prediction_sha256"] == hashlib.sha256(predictions[row["id"]]["response"].encode()).hexdigest())
    summary = summarize(folder)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["scoring_complete"] else 2


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ["generate", "judge", "run", "summarize"]:
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL, help="Tested model; also the result directory name")
        p.add_argument("--results-root", type=Path, default=ROOT / "results")
        p.add_argument("--subset", choices=["text-only-10pct"],
            help="Fixed stratified 216-question diagnostic set; results use a separate benchmark directory")
        p.add_argument("--full", action="store_const", dest="subset", const=None,
            help="Evaluate the full selected scope, overriding --subset; 2158 questions for text-only")
        if name == "summarize":
            continue
        p.add_argument("--workers", type=positive, default=32)
        p.add_argument("--timeout", type=positive, default=7200)
        p.add_argument("--retries", type=int, choices=range(4), default=1)
        p.add_argument("--retry-failed", action="store_true", help="Explicitly append new attempts for failed tasks")
        if name in {"generate", "run"}:
            p.add_argument("--base-url", nargs="+", default=[DEFAULT_BASE],
                help="One or more identical-model endpoints, separated by spaces; --workers applies to each endpoint")
            p.add_argument("--replica-base-url", action="append", default=[],
                help="Additional identical-model endpoint; repeat per replica, with --workers per endpoint")
            p.add_argument("--replica-manifest", type=Path,
                help="Replica registry; verified ready endpoints join automatically, with --workers per endpoint")
            p.add_argument("--replicas-only", action="store_true",
                help="Use only replicas in --replica-manifest; do not request --base-url")
            p.add_argument("--api-key-env", default="VLLM_API_KEY")
            p.add_argument("--scope", choices=["text-only", "full"], default="text-only")
            p.add_argument("--limit", type=positive, help="Debug subset; never reported as the complete scope")
            p.add_argument("--context-limit", type=positive, default=262144)
            p.add_argument("--max-output-tokens", type=positive, default=131072)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--top-p", type=float, default=1)
            p.add_argument("--top-k", type=int, default=20)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--thinking", choices=["on", "off", "default"], default="on")
        if name in {"judge", "run"}:
            p.add_argument("--judge-base-url", required=True)
            p.add_argument("--judge-model", required=True)
            p.add_argument("--judge-model-revision", help="Provider snapshot/revision, when available; never guessed")
            p.add_argument("--judge-api-key-env", default="HLE_JUDGE_API_KEY")
            p.add_argument("--judge-max-tokens", type=positive, default=4096)
            p.add_argument("--judge-temperature", type=float, help="Omit to follow the official judge's server default")
            p.add_argument("--allow-self-judge", action="store_true")
    return root


def main(argv=None):
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)
    args = parser().parse_args(argv)
    if args.command in {"generate", "run"}:
        if args.replicas_only and (not args.replica_manifest or args.replica_base_url):
            raise ValueError("--replicas-only requires --replica-manifest and excludes --replica-base-url")
        if args.context_limit > 262144 or args.max_output_tokens > args.context_limit:
            raise ValueError("Context must not exceed 256K and the output budget must fit within it")
        if args.temperature < 0 or not 0 < args.top_p <= 1:
            raise ValueError("Invalid sampling parameters")
    if args.command == "summarize":
        print(json.dumps(summarize(run_directory(args)), ensure_ascii=False, indent=2))
        return 0
    folder = run_directory(args)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another inference or judge process is already using this model result directory") from exc
        if args.command == "run":
            if args.judge_model == args.model and not args.allow_self_judge:
                raise ValueError("The combined run requires an independent judge or explicit --allow-self-judge")
            Client(args.judge_base_url, args.judge_api_key_env, args.timeout, args.retries)
            if not os.environ.get(args.judge_api_key_env):
                raise ValueError(f"Missing judge credential environment variable: {args.judge_api_key_env}")
            save_or_match(folder / "judge.json", judge_configuration(args, official_protocol()))
            generate(args)  # Generation failures remain in the denominator; still judge successful answers.
            return judge(args)
        return generate(args) if args.command == "generate" else judge(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
