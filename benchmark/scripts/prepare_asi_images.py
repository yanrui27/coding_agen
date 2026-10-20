#!/usr/bin/env python3
"""Build official ASI seed42 task images in the project-local Docker store."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/asi-bench"
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
for _key in PROXIES:
    os.environ.pop(_key, None)
os.environ.update(NO_PROXY="*", no_proxy="*", DOCKER_BUILDKIT="0", PYTHONDONTWRITEBYTECODE="1")
sys.path.insert(0, str(SOURCE))
from ai4sci_bench.core.task import TaskLoader
from ai4sci_bench.runner.task_image import TaskImageBuilder


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def capture(args, timeout=30):
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def ensure_storage_space(required_gib=16):
    store = ROOT / ".cache/docker-images/asi-bench"
    disk = ROOT / ".cache/docker-images/asi-bench.ext4"
    if shutil.disk_usage(store).free >= required_gib * 1024 ** 3:
        return
    device = capture(["findmnt", "-n", "-o", "SOURCE", "--target", str(store)])
    if not device.startswith("/dev/loop") or not disk.exists():
        raise RuntimeError(f"ASI image storage has less than {required_gib} GiB free")
    backing = capture(["losetup", "-n", "-O", "BACK-FILE", device])
    if Path(backing).resolve() != disk.resolve():
        raise RuntimeError("Refusing to resize an unrelated filesystem")
    while shutil.disk_usage(store).free < required_gib * 1024 ** 3:
        previous = disk.stat().st_size
        size = min(64 * 1024 ** 3, previous + 16 * 1024 ** 3)
        if size == previous:
            raise RuntimeError("ASI image store reached its 64 GiB capacity limit")
        with disk.open("r+b") as f:
            f.truncate(size)
        subprocess.run(["losetup", "--set-capacity", device], check=True)
        subprocess.run(["resize2fs", device], check=True, timeout=120)
        print(now(), "Expanded ASI Docker storage to", size // 1024 ** 3, "GiB", flush=True)


class LocalBuilder(TaskImageBuilder):
    def __init__(self, cache, python_id):
        super().__init__(SOURCE)
        self.cache = cache
        self.python_id = python_id
        self.built_tags = []

    def _prepare(self, text):
        # Preserve task package specifications and the upstream image naming.
        text = text.replace("FROM python:3.12-slim", "FROM " + self.python_id)
        text = re.sub(
            r"RUN curl -LsSf https://astral\.sh/uv/install\.sh \| sh \\\n"
            r"    && mv /root/\.local/bin/uv /usr/local/bin/uv \\\n"
            r"    && mv /root/\.local/bin/uvx /usr/local/bin/uvx\n",
            "RUN python3 -m pip --isolated install --no-cache-dir --proxy '' --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 60 --retries 2 uv==0.8.22\n",
            text,
        )
        if "astral.sh/uv/install.sh" in text:
            raise RuntimeError("Unrecognized upstream uv installer; refusing an unrecorded recipe")
        text = text.replace("curl -", "curl -4 --noproxy '*' --connect-timeout 15 --max-time 180 -")
        text = text.replace("RUN uv pip install", "RUN UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv pip install")
        direct = "ENV " + " ".join(k + "=''" for k in PROXIES) + " NO_PROXY='*' no_proxy='*' UV_NO_CACHE=1 UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=2\n"
        if "FROM " + self.python_id in text:
            direct += (
                "RUN printf '%s\\n' 'Acquire::ForceIPv4 \"true\";' "
                "'Acquire::http::Proxy \"DIRECT\";' 'Acquire::https::Proxy \"DIRECT\";' "
                "'Acquire::Retries \"2\";' 'Acquire::http::Timeout \"30\";' "
                "'Acquire::https::Timeout \"30\";' > /etc/apt/apt.conf.d/99-direct "
                "&& sed -i 's|http://deb.debian.org/|https://mirrors.aliyun.com/|g' "
                "/etc/apt/sources.list.d/debian.sources\n"
            )
        # curl/npm inside install scripts also inherit explicit direct settings.
        text = re.sub(r"(^FROM[^\n]*\n)", lambda m: m[1] + direct, text, count=1, flags=re.M)
        if "FROM " + self.python_id in text:
            text += "\nENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/\n"
        return text

    def _build(self, tag, original, context):
        slug = tag.replace(":", "__").replace("/", "_")
        folder = self.cache / "dockerfiles"
        (folder / (slug + ".upstream.Dockerfile")).write_text(original)
        prepared = folder / (slug + ".Dockerfile")
        prepared.write_text(self._prepare(original))
        ensure_storage_space(32 if "torch==" in original else 16)
        args = ["docker", "build", "--cpu-period", "100000", "--cpu-quota", "400000",
                "--memory", "32g", "--memory-swap", "32g", "--force-rm", "--network", "host"]
        for key in PROXIES:
            args += ["--build-arg", key + "="]
        args += ["--build-arg", "NO_PROXY=*", "--build-arg", "no_proxy=*",
                 "-f", str(prepared), "-t", tag, str(context)]
        print(now(), "BUILD", tag, flush=True)
        with (self.cache / "logs" / (slug + ".log")).open("a") as log:
            log.write("\n" + now() + " " + shlex.join(args) + "\n")
            log.flush()
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
        self.built_tags.append(tag)

    def _build_image(self, tag, dockerfile):
        context = self.cache / "empty-context"
        context.mkdir(exist_ok=True)
        self._build(tag, dockerfile, context)

    def _build_image_from_file(self, tag, dockerfile_path, context_dir):
        self._build(tag, dockerfile_path.read_text(), context_dir)


def record_image(cache, manifest, tag):
    info = json.loads(capture(["docker", "image", "inspect", tag]))[0]
    meta = {"image_id": info["Id"], "repo_digests": info.get("RepoDigests", []),
            "image_created": info["Created"], "image_size_bytes": info["Size"],
            "recorded_at": now()}
    manifest["images"][tag] = meta
    write_json(cache / "manifest.json", manifest)
    return meta


def inspect_runtime(tag):
    probe = """import json, shutil, subprocess
from pathlib import Path
tools = {t: shutil.which(t) for t in ('python3','uv','git','curl','node','opencode')}
assert all(tools.values()), tools
def run(args): return subprocess.check_output(args, text=True, timeout=45).strip()
packages = run(['uv','pip','list','--python','/opt/venv/bin/python','--format','json'])
run(['uv','pip','check','--python','/opt/venv/bin/python'])
print(json.dumps({'tools':tools,'python':run(['python','--version']),
 'node':run(['node','--version']),'uv':run(['uv','--version']),
 'opencode':run(['opencode','--version']), 'packages':json.loads(packages),
 'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
 'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip()}))
"""
    out = capture(["docker", "run", "--rm", "--pull", "never", "--network", "none",
                   "--cpus", "4", "--memory", "32g", "--pids-limit", "512",
                   "--entrypoint", "python", tag, "-c", probe], timeout=120)
    return json.loads(out.splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/asi-build")
    parser.add_argument("--docker-host", default="unix:///run/coding-agent-asi-docker/docker.sock")
    parser.add_argument("--limit", type=int, help="Prepare the first N unique task images for debugging")
    args = parser.parse_args()
    os.environ["DOCKER_HOST"] = args.docker_host
    info = json.loads(capture(["docker", "info", "--format", "{{json .}}"] ))
    if Path(info["DockerRootDir"]).resolve() != ROOT / ".cache/docker-images/asi-bench":
        raise RuntimeError("Refusing to build into a Docker store outside the requested directory")
    cache = args.cache_dir.resolve()
    for sub in ("dockerfiles", "logs", "metadata"):
        (cache / sub).mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(cache, args.limit)
        except Exception as exc:
            path = cache / "manifest.json"
            if path.exists():
                state = json.loads(path.read_text())
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
                write_json(path, state)
            raise


def run(cache, limit):
    lock = json.loads((ROOT / "archive/benchmarks.lock.json").read_text())["datasets"]["asi_bench"]
    commit = capture(["git", "-C", str(SOURCE), "rev-parse", "HEAD"])
    if commit != lock["code_commit"]:
        raise RuntimeError("ASI source differs from the pinned code commit")
    # Reuse the already-local official Python image, without refreshing its mutable tag.
    base_info = json.loads(capture(["docker", "image", "inspect", "python:3.12-slim"]))[0]
    builder = LocalBuilder(cache, base_info["Id"])
    loader = TaskLoader(SOURCE / "tasks")
    groups = {}
    ids = (ROOT / lock["task_ids"]).read_text().splitlines()
    for instance in ids:
        task_id = instance.rsplit("__seed", 1)[0]
        metadata = loader.load_task_by_id(task_id)
        custom = metadata.get("runtime", {}).get("dockerfile")
        base_tag = builder._custom_dockerfile_tag(metadata["_task_dir"] / custom) if custom else builder._base_image_tag()
        tag = builder._task_agent_image_tag(metadata, "opencode", base_image=base_tag)
        group = groups.setdefault(tag, {"tag": tag, "tasks": [], "metadata": metadata,
                                       "runtime_base": base_tag})
        group["tasks"].append(task_id)
    state_path = cache / "manifest.json"
    manifest = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": 1, "benchmark": "ASI-Bench", "seed": 42,
        "code_commit": commit, "hf_revision": lock["hf_revision"],
        "agent": "opencode_cli", "agent_version": "1.17.15", "task_count": len(ids),
        "unique_task_images": len(groups), "images": {}, "tasks": {}, "created_at": now(),
        "build_resources": {"cpus": 4, "memory_bytes": 32 * 1024 ** 3, "parallel": 1},
        "recipe_adjustments": ["Pin FROM to the existing official Python image ID",
          "Install uv==0.8.22 from PyPI because astral.sh returns HTTP 403 directly",
          "Explicitly clear all six proxy variables and use curl --noproxy '*'",
          "Use the direct Aliyun Debian mirror with the original Debian signing keys",
          "Disable uv wheel caches in the final images; bound network retries/timeouts",
          "Use the direct Aliyun PyPI mirror without changing task package requirements"],
        "runtime_store": str(ROOT / ".cache/docker-images/asi-bench"), "archive_format": None,
        "python_image_id": base_info["Id"], "scope": "full" if limit is None else "debug_subset"}
    if (manifest["code_commit"], manifest["hf_revision"], manifest["python_image_id"]) != (
            commit, lock["hf_revision"], base_info["Id"]):
        raise RuntimeError("Image preparation inputs changed; refusing to mix cache manifests")
    resources = {"cpus": 4, "memory_bytes": 32 * 1024 ** 3, "parallel": 1}
    if manifest.get("build_resources") != resources:
        manifest.setdefault("build_resource_history", []).append({
            "previous": manifest.get("build_resources"), "current": resources,
            "changed_at": now(), "reason": "User requested 4 CPU / 32 GiB for future starts"})
    manifest["build_resources"] = resources
    manifest.update(status="running", updated_at=now(), archive_format=None,
                    docker_host=os.environ["DOCKER_HOST"],
                    runtime_store=str(ROOT / ".cache/docker-images/asi-bench"))
    write_json(state_path, manifest)
    record_image(cache, manifest, "python:3.12-slim")
    builder.ensure_base_image()
    record_image(cache, manifest, builder._base_image_tag())
    selected = list(groups.values())[:limit]
    failed = []
    for i, group in enumerate(selected, 1):
        tag = group["tag"]
        record = {"tasks": group["tasks"], "runtime_base": group["runtime_base"],
                  "packages_requested": group["metadata"].get("_runtime_packages", []),
                  "status": "building", "started_at": now()}
        manifest["tasks"][tag] = record
        manifest.update(current_image=tag, updated_at=now())
        write_json(state_path, manifest)
        print(now(), f"TASK IMAGE {i}/{len(selected)}", tag, flush=True)
        try:
            actual = builder.ensure_image(group["metadata"], agent_type="opencode")
            if actual != tag:
                raise RuntimeError(f"Image plan mismatch: {actual} vs {tag}")
            versions = inspect_runtime(tag)
            write_json(cache / "metadata" / (tag.replace(":", "__") + ".json"), versions)
            if group["runtime_base"] != builder._base_image_tag():
                record_image(cache, manifest, group["runtime_base"])
            record_image(cache, manifest, tag)
            record.update(status="ready", finished_at=now(), image_id=manifest["images"][tag]["image_id"])
        except Exception as exc:
            record.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            failed.append(tag)
            print(now(), "FAILED", tag, record["error"], flush=True)
        manifest.update(updated_at=now(), ready_task_images=sum(v["status"] == "ready" for v in manifest["tasks"].values()),
                        failed_task_images=len(failed))
        write_json(state_path, manifest)
    manifest.update(status="failed" if failed else "complete", finished_at=now(), updated_at=now())
    write_json(state_path, manifest)
    print(now(), "DONE", manifest["status"], manifest.get("ready_task_images"), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
