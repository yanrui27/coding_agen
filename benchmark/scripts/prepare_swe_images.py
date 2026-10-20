#!/usr/bin/env python3
"""Pin, pull and check all official SWE-bench Pro V2 HARD-51 task images."""
from __future__ import annotations

import concurrent.futures
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib

import requests

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".cache/swe-build"
STORE = ROOT / ".cache/docker-images/swe-bench-pro-v2-hard"
DISK = STORE.with_suffix(".ext4")
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
for key in PROXIES:
    os.environ.pop(key, None)
os.environ.update(NO_PROXY="*", no_proxy="*", DOCKER_HOST="unix:///run/coding-agent-swe-docker/docker.sock")
REPO = "ghcr.io/scaleapi/swe-bench_pro-v2"
DOWNLOAD_REPO = "ghcr.nju.edu.cn/scaleapi/swe-bench_pro-v2"
GIB = 1024 ** 3
STORAGE_LOCK = threading.Lock()
PARALLEL = 2


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def capture(args, timeout=60):
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def ensure_space(required=8 * GIB):
    with STORAGE_LOCK:
        _ensure_space(required)


def _ensure_space(required):
    if shutil.disk_usage(STORE).free >= required:
        return
    device = capture(["findmnt", "-n", "-o", "SOURCE", "--target", str(STORE)])
    backing = capture(["losetup", "-n", "-O", "BACK-FILE", device])
    if not device.startswith("/dev/loop") or Path(backing).resolve() != DISK.resolve():
        raise RuntimeError("Refusing to resize an unrelated image filesystem")
    while shutil.disk_usage(STORE).free < required:
        previous = DISK.stat().st_size
        size = min(512 * GIB, previous + 32 * GIB)
        if size == previous:
            raise RuntimeError("SWE Docker storage reached its 512 GiB disk capacity limit")
        with DISK.open("r+b") as f:
            f.truncate(size)
        subprocess.run(["losetup", "--set-capacity", device], check=True)
        subprocess.run(["resize2fs", device], check=True, timeout=120)
        print(now(), "Expanded SWE Docker disk capacity to", size // GIB, "GiB", flush=True)


class Registry:
    def __init__(self):
        self.local = threading.local()
        session = self.session()
        response = session.get("https://ghcr.io/token", params={
            "service": "ghcr.io", "scope": "repository:scaleapi/swe-bench_pro-v2:pull"}, timeout=(15, 60))
        if response.status_code != 200:
            raise RuntimeError(f"Anonymous registry authentication returned HTTP {response.status_code}")
        self.token = response.json()["token"]

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.trust_env = False
        return self.local.session

    def manifest(self, reference):
        accept = ", ".join(("application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json"))
        for attempt in range(3):
            try:
                response = self.session().get(
                    "https://ghcr.io/v2/scaleapi/swe-bench_pro-v2/manifests/" + reference,
                    headers={"Authorization": "Bearer " + self.token, "Accept": accept}, timeout=(15, 60))
                if response.status_code != 200:
                    raise RuntimeError(f"Registry manifest {reference}: HTTP {response.status_code}")
                digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
                if response.headers.get("Docker-Content-Digest", digest) != digest:
                    raise RuntimeError("Registry manifest digest mismatch")
                (WORK / "metadata/registry" / (digest.split(":")[1] + ".json")).write_bytes(response.content)
                return response.json(), digest
            except (requests.RequestException, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    def resolve(self, task_id):
        manifest, tag_digest = self.manifest(task_id)
        digest = tag_digest
        if "manifests" in manifest:
            choices = [v for v in manifest["manifests"]
                       if v.get("platform", {}).get("os") == "linux"
                       and v.get("platform", {}).get("architecture") == "amd64"]
            if len(choices) != 1:
                raise RuntimeError(f"Expected exactly one linux/amd64 manifest for {task_id}")
            manifest, digest = self.manifest(choices[0]["digest"])
            if digest != choices[0]["digest"]:
                raise RuntimeError("Selected platform digest mismatch")
        return {"tag_digest": tag_digest, "manifest_digest": digest,
                "config_digest": manifest["config"]["digest"], "platform": "linux/amd64",
                "compressed_bytes": sum(v["size"] for v in manifest["layers"]),
                "layers": [{"digest": v["digest"], "size": v["size"]} for v in manifest["layers"]],
                "resolved_at": now()}


def image_info(reference):
    result = subprocess.run(["docker", "image", "inspect", reference], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def pull(task_id, record):
    expected = record["registry"]
    pinned = DOWNLOAD_REPO + "@" + expected["manifest_digest"]
    info = image_info(expected["config_digest"])
    if info is not None and not any(ref.endswith("@" + expected["manifest_digest"])
                                    for ref in info.get("RepoDigests", [])):
        info = None
    if info is None:
        ensure_space(max(8 * GIB, 3 * expected["compressed_bytes"] + GIB))
        for attempt in range(1, 4):
            repository = DOWNLOAD_REPO if attempt < 3 else REPO
            pinned = repository + "@" + expected["manifest_digest"]
            command = ["docker", "pull", "--platform", "linux/amd64", pinned]
            print(now(), "PULL", task_id, "attempt", attempt, flush=True)
            with (WORK / "logs" / (task_id + ".log")).open("a") as log:
                log.write("\n" + now() + " " + shlex.join(command) + "\n"); log.flush()
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 7200
                    while process.poll() is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Docker pull exceeded 7200 seconds")
                        ensure_space()
                        time.sleep(5)
                    if process.returncode == 0:
                        break
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
            if attempt == 3:
                raise RuntimeError("Docker pull failed after three attempts; see task log")
            time.sleep(5 * attempt)
        info = image_info(pinned)
    if info is None or info["Id"] != expected["config_digest"]:
        raise RuntimeError("Pulled image config ID differs from the pinned registry manifest")
    if (info["Os"], info["Architecture"]) != ("linux", "amd64"):
        raise RuntimeError("Pulled image has the wrong OS or architecture")
    record["download_ref"] = next((ref for ref in info.get("RepoDigests", []) if ref == pinned),
        next(ref for ref in info["RepoDigests"] if ref.endswith("@" + expected["manifest_digest"])))
    subprocess.run(["docker", "tag", info["Id"], record["official_ref"]], check=True, timeout=60)
    return info


def probe(reference):
    script = '''set -eu
repo=''
for candidate in /app /testbed; do
  if git -c safe.directory="$candidate" -C "$candidate" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    repo="$candidate"; break
  fi
done
test -n "$repo"
printf 'repository=%s\n' "$repo"
printf 'base_commit=%s\n' "$(git -c safe.directory="$repo" -C "$repo" rev-parse HEAD)"
printf 'cpu_max=%s\n' "$(cat /sys/fs/cgroup/cpu.max)"
printf 'memory_max=%s\n' "$(cat /sys/fs/cgroup/memory.max)"
printf 'memory_swap_max=%s\n' "$(cat /sys/fs/cgroup/memory.swap.max)"
printf 'uid=%s\n' "$(id -u)"
'''
    output = capture(["docker", "run", "--rm", "--pull", "never", "--network", "none",
        "--read-only", "--cpus", "4", "--memory", "32g", "--memory-swap", "32g",
        "--pids-limit", "512", "--entrypoint", "/bin/sh", reference, "-c", script], timeout=180)
    result = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if (result.get("cpu_max"), result.get("memory_max"), result.get("memory_swap_max")) != (
            "400000 100000", str(32 * GIB), "0"):
        raise RuntimeError(f"Container resource limits differ from requested defaults: {result}")
    return result


def prepare_one(task_id, previous):
    record = copy.deepcopy(previous)
    try:
        info = pull(task_id, record)
        checks = probe(record["official_ref"])
        record.pop("error", None)
        record.update(status="ready", image_id=info["Id"], repo_digests=info.get("RepoDigests", []),
                      image_size_bytes=info["Size"], checks=checks, finished_at=now())
        save(WORK / "metadata" / (task_id + ".json"), record)
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
        print(now(), "FAILED", task_id, record["error"], flush=True)
    return record


def verify_completed(m, ids, dataset):
    if len(ids) != 51 or set(m["tasks"]) != set(ids):
        raise RuntimeError("Final verification requires the exact official HARD-51 task set")
    if any(m["tasks"][task]["status"] != "ready" for task in ids):
        raise RuntimeError("Final verification found a task image that is not ready")
    if capture(["git", "-C", str(ROOT / dataset["code_directory"]), "rev-parse", "HEAD"]) != m["code_commit"]:
        raise RuntimeError("SWE source commit changed during image preparation")
    official_ids = (ROOT / dataset["code_directory"] / "v2/hard51_ids.txt").read_text().splitlines()
    if set(ids) != set(official_ids):
        raise RuntimeError("Final task set differs from official HARD-51 IDs")
    references = [m["tasks"][task]["official_ref"] for task in ids]
    images = json.loads(capture(["docker", "image", "inspect", *references], timeout=180))
    if len(images) != 51:
        raise RuntimeError("Final Docker inspection did not return all 51 task images")
    for task_id, info in zip(ids, images):
        record = m["tasks"][task_id]
        expected = record["registry"]
        if (info["Id"] != expected["config_digest"] or record["image_id"] != info["Id"]
                or (info["Os"], info["Architecture"]) != ("linux", "amd64")
                or record["official_ref"] not in info.get("RepoTags", [])
                or not any(ref.endswith("@" + expected["manifest_digest"])
                           for ref in info.get("RepoDigests", []))):
            raise RuntimeError(f"Final image identity or platform mismatch: {task_id}")
        checks = record["checks"]
        if (checks.get("cpu_max"), checks.get("memory_max"), checks.get("memory_swap_max")) != (
                "400000 100000", str(32 * GIB), "0"):
            raise RuntimeError(f"Missing successful 4 CPU / 32 GiB runtime check: {task_id}")
        for digest in {expected["tag_digest"], expected["manifest_digest"]}:
            raw = (WORK / "metadata/registry" / (digest.split(":")[1] + ".json")).read_bytes()
            if "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
                raise RuntimeError(f"Pinned registry evidence checksum mismatch: {task_id}")
        task = ROOT / record["task_directory"]
        for relative, key in (("task.toml", "task_config_sha256"), ("environment/Dockerfile", "dockerfile_sha256")):
            if hashlib.sha256((task / relative).read_bytes()).hexdigest() != record[key]:
                raise RuntimeError(f"Official task file changed during preparation: {task_id}/{relative}")
    docker = json.loads(capture(["docker", "info", "--format", "{{json .}}"] ))
    if docker["DockerRootDir"] != str(STORE) or docker["Driver"] != "overlay2":
        raise RuntimeError("Final Docker store or storage driver differs from the expected store")
    verification = {"status": "complete", "verified_at": now(), "task_count": 51,
        "task_images": 51, "failed": 0, "all_ids_and_digests_match": True,
        "runtime_checks": 51, "cpus": 4, "memory_gib": 32, "image_data_root": str(STORE),
        "export_archives": False, "benchmark_execution": "not_started"}
    job_path = WORK / "metadata/job.json"
    job = json.loads(job_path.read_text())
    job["completion_verification"] = verification
    save(job_path, job)
    m["completion_verification"] = verification
    print(now(), "VERIFIED", len(ids), "official task images", flush=True)


def run():
    for folder in (WORK, WORK / "logs", WORK / "metadata/registry"):
        folder.mkdir(parents=True, exist_ok=True)
    docker = json.loads(capture(["docker", "info", "--format", "{{json .}}"] ))
    if docker["DockerRootDir"] != str(STORE) or docker["Driver"] != "overlay2":
        raise RuntimeError("Refusing to prepare images in a different Docker store or storage driver")
    dataset = json.loads((ROOT / "archive/benchmarks.lock.json").read_text())["datasets"]["swe_bench_pro_v2_hard"]
    commit = capture(["git", "-C", str(ROOT / dataset["code_directory"]), "rev-parse", "HEAD"])
    if commit != dataset["code_commit"]:
        raise RuntimeError("SWE framework differs from the pinned source commit")
    ids = (ROOT / dataset["task_ids"]).read_text().splitlines()
    if len(ids) != 51 or len(set(ids)) != 51:
        raise RuntimeError("Expected the complete pinned HARD-51 task list")
    path = WORK / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "benchmark": "SWE-bench Pro V2 HARD-51", "task_count": 51,
        "code_commit": commit, "hf_revision": dataset["hf_revision"], "created_at": now(),
        "hard_basis": "Official v2/hard51_ids.txt at the pinned framework commit",
        "runtime_store": str(STORE), "docker_host": os.environ["DOCKER_HOST"],
        "export_archives": False, "resources": {"cpus": 4, "memory_gib": 32, "parallel": PARALLEL},
        "storage": {"initial_capacity_gib": 32, "growth_gib": 32, "max_capacity_gib": 512},
        "download_policy": "Direct; all six proxy variables cleared; requests trust_env=False",
        "tasks": {}}
    if (m["code_commit"], m["hf_revision"]) != (commit, dataset["hf_revision"]):
        raise RuntimeError("Refusing to mix pinned benchmark versions")
    if m["tasks"] and set(m["tasks"]) != set(ids):
        raise RuntimeError("Existing manifest has a different task set")
    official_ids = (ROOT / dataset["code_directory"] / "v2/hard51_ids.txt").read_text().splitlines()
    if set(ids) != set(official_ids):
        raise RuntimeError("Local task IDs differ from the official HARD-51 split")
    resources = {"cpus": 4, "memory_gib": 32, "parallel": PARALLEL}
    if m.get("resources") != resources:
        m.setdefault("resource_history", []).append({"previous": m.get("resources"),
            "current": resources, "changed_at": now(),
            "reason": "Overlap downloads within the unchanged aggregate 4 CPU / 32 GiB cgroup limit"})
    m["resources"] = resources
    for task_id in ids:
        task = ROOT / dataset["hard_directory"] / task_id
        config = tomllib.loads((task / "task.toml").read_text())
        reference = config["environment"]["docker_image"]
        if reference != REPO + ":" + task_id or (task / "environment/Dockerfile").read_text().strip() != "FROM " + reference:
            raise RuntimeError(f"Unexpected official image reference: {task_id}")
        m["tasks"].setdefault(task_id, {"official_ref": reference, "status": "pending",
            "task_directory": str(task.relative_to(ROOT)),
            "dockerfile_sha256": hashlib.sha256((task / "environment/Dockerfile").read_bytes()).hexdigest(),
            "task_config_sha256": hashlib.sha256((task / "task.toml").read_bytes()).hexdigest()})
    if m.get("download_repository") and m["download_repository"] != DOWNLOAD_REPO:
        m.setdefault("download_source_history", []).append({"previous": m["download_repository"],
            "current": DOWNLOAD_REPO, "changed_at": now(),
            "reason": "Direct download source changed; original official manifest digests remain pinned"})
    m.update(status="resolving", updated_at=now(), ready_task_images=sum(r["status"] == "ready" for r in m["tasks"].values()), failed_task_images=0,
             download_repository=DOWNLOAD_REPO, manifest_source=REPO)
    save(path, m)
    pending = [task_id for task_id in ids if "registry" not in m["tasks"][task_id]]
    if pending:
        registry = Registry()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(registry.resolve, task_id): task_id for task_id in pending}
            for future in concurrent.futures.as_completed(jobs):
                task_id = jobs[future]
                m["tasks"][task_id]["registry"] = future.result()
                m["updated_at"] = now(); save(path, m)
                print(now(), "PINNED", task_id, flush=True)
    blobs = {v["digest"]: v["size"] for r in m["tasks"].values() for v in r["registry"]["layers"]}
    m.update(status="running", unique_compressed_layer_bytes=sum(blobs.values()), unique_layers=len(blobs), updated_at=now())
    save(path, m)
    print(now(), "PLAN", len(ids), "images; unique compressed layer GiB", round(sum(blobs.values()) / GIB, 2), flush=True)
    save(WORK / "metadata/job.json", {"unit": "coding-agent-swe-images.service", "started_at": now(),
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256((ROOT / "run_swe.sh").read_bytes()).hexdigest(),
        "resource_limits": m["resources"], "docker_host": m["docker_host"], "image_data_root": str(STORE),
        "manifest_source": REPO, "download_repository": DOWNLOAD_REPO, "fallback_repository": REPO,
        "operation": "Official image preparation only; no model evaluation or submission", "export_archives": False})
    cached = {item["Repository"] + ":" + item["Tag"]: item["ID"] for item in
              (json.loads(line) for line in capture(["docker", "image", "ls", "--no-trunc", "--format", "json"]).splitlines())}
    todo = []
    for task_id in ids:
        record = m["tasks"][task_id]
        if record["status"] != "ready" or cached.get(record["official_ref"]) != record["registry"]["config_digest"]:
            record["status"] = "pending"
            todo.append(task_id)
    iterator = iter(todo)
    active = {}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        while True:
            while len(active) < PARALLEL:
                task_id = next(iterator, None)
                if task_id is None:
                    break
                record = m["tasks"][task_id]
                record.update(status="pulling", started_at=now())
                active[pool.submit(prepare_one, task_id, record)] = task_id
                print(now(), "TASK IMAGE", task_id, flush=True)
            m.update(active_tasks=list(active.values()), current_task=next(iter(active.values()), None),
                     ready_task_images=sum(v["status"] == "ready" for v in m["tasks"].values()),
                     failed_task_images=len(failures), updated_at=now())
            save(path, m)
            if not active:
                break
            done, _ = concurrent.futures.wait(active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task_id = active.pop(future)
                m["tasks"][task_id] = future.result()
                if m["tasks"][task_id]["status"] == "failed":
                    failures.append(task_id)
    if not failures:
        m.update(status="verifying", updated_at=now())
        save(path, m)
        verify_completed(m, ids, dataset)
    m.update(status="failed" if failures else "complete", finished_at=now(), updated_at=now())
    save(path, m)
    print(now(), "DONE", m["status"], m["ready_task_images"], flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except Exception as exc:
            path = WORK / "manifest.json"
            if path.exists():
                m = json.loads(path.read_text())
                m.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
                save(path, m)
            raise
