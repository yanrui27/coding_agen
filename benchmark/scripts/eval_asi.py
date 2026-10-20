#!/usr/bin/env python3
"""Launch the pinned ASI runner; leave task execution and collection upstream."""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/asi-bench"
for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(key, None)
os.environ.update(NO_PROXY="*", no_proxy="*", LITELLM_LOCAL_MODEL_COST_MAP="True",
                 DOCKER_HOST="unix:///run/coding-agent-asi-docker/docker.sock",
                 AI4SCI_DOCKER_NETWORK="host")
os.environ.setdefault("ASI_MODEL_API_KEY", "EMPTY")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / "scripts"))

from asi_agent import ASSIGNMENT, server_urls, use_server_pool


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def check_environment(options, resources):
    import requests
    from ai4sci_bench.core.task import TaskLoader

    lock = json.loads((ROOT / "archive/benchmarks.lock.json").read_text())["datasets"]["asi_bench"]
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["code_commit"]:
        raise ValueError("ASI source differs from archive/benchmarks.lock.json")
    if options["seed"] != 42 or options["agent"] != "opencode_cli" or options["sandbox"] != "os":
        raise ValueError("Prepared ASI images require seed42, --agent opencode_cli and --sandbox os")
    if options["instances_per_task"] != 1:
        raise ValueError("The fixed seed42 dataset has one instance per task")
    if options["agent_cmd"] or options["params"] or any(options[name] for name in (
            "include_test", "include_sample", "include_dev", "include_abandoned")):
        raise ValueError("Use the built-in adapter and the fixed final-task seed42 instances")
    if Path(options["tasks_dir"]).resolve() != SOURCE / "tasks":
        raise ValueError("Use the pinned official --tasks-dir third_party/asi-bench/tasks")
    ids = (ROOT / "archive/manifests/asi-seed42-task-ids.txt").read_text().splitlines()
    canonical = {task.removesuffix("__seed42") for task in ids}
    selected = canonical if options["tasks"] in (None, "all") else {task.strip() for task in options["tasks"].split(",")}
    if len(ids) != 60 or len(canonical) != 60 or not selected or not selected <= canonical:
        raise ValueError("--tasks must select task IDs from the pinned 60-task seed42 set")
    levels = [level.strip() for level in options["prompt_levels"].split(",")]
    if len(set(levels)) != len(levels) or not set(levels) <= {"b1", "b2", "b3", "b4"}:
        raise ValueError("--prompt-levels must select distinct b1,b2,b3,b4 levels")
    instances = Path(options["instances_dir"])
    loader = TaskLoader(SOURCE / "tasks")
    for task in selected:
        directory = instances / (task + "__seed42")
        if {p.name for p in instances.glob(task + "__*") if p.is_dir()} != {directory.name}:
            raise ValueError(f"Missing or mixed-seed instances: {task}")
        metadata = loader.load_task_by_id(task)
        needs_data = bool(metadata.get("input", {}).get("files"))
        if (needs_data and not (directory / "data").is_dir()) or any(not (directory / f"prompt_{level}.md").is_file() for level in levels):
            raise ValueError(f"Missing fixed seed42 inputs: {task}")
    cache = json.loads((ROOT / ".cache/asi-build/manifest.json").read_text())
    if cache["status"] != "complete" or (cache["code_commit"], cache["hf_revision"]) != (commit, lock["hf_revision"]):
        raise ValueError("ASI image preparation is incomplete or differs from the pinned release")
    subprocess.run(["bash", str(ROOT / "scripts/manage_asi.sh"), "docker", "info", "--format", "{{.DockerRootDir}}"],
                   check=True, stdout=subprocess.DEVNULL)
    records = [record for record in cache["tasks"].values() if set(record["tasks"]) & selected]
    tags = [tag for tag, record in cache["tasks"].items() if set(record["tasks"]) & selected]
    if not selected <= {task for record in records for task in record["tasks"]}:
        raise ValueError("A selected task has no prepared image")
    images = json.loads(subprocess.check_output(["docker", "image", "inspect", *tags], text=True))
    if len(images) != len(records) or any(image["Id"] != record["image_id"] for image, record in zip(images, records)):
        raise ValueError("A prepared ASI image no longer matches its recorded image ID")
    agent = json.loads(options["agent_config"])
    key = agent.get("api_key") or os.environ.get(agent.get("api_key_env", "ASI_MODEL_API_KEY"))
    if not key:
        raise ValueError("Set the model credential in the configured api_key_env")
    if agent.get("api_protocol") != "openai" or not agent.get("api_base") or not agent.get("model"):
        raise ValueError("Set model, api_base and api_protocol=openai in --agent-config")
    urls = server_urls(agent["api_base"])

    def check_server(url):
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(url + "/models", headers={"Authorization": "Bearer " + key}, timeout=15)
            response.raise_for_status()
            if agent["model"] not in {item["id"] for item in response.json()["data"]}:
                raise ValueError(f"Configured model is not advertised by {url}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        list(pool.map(check_server, urls))
    return {"framework_commit": commit, "hf_revision": lock["hf_revision"],
            "agent_version": cache["agent_version"], "tasks": sorted(selected),
            "conditions": len(selected) * len(levels), "task_images": len(images),
            "model_servers": urls, "parallel_sandboxes": options["parallel"],
            "maximum_sandbox_resources": {"cpus": resources.cpus * options["parallel"],
                                          "memory_mib": resources.memory_mib * options["parallel"]},
            "resources_per_task": {"cpus": resources.cpus, "memory_mib": resources.memory_mib},
            "grading": "seed42 private references; official Portal score pending"}


def set_resource_limits(options, resources):
    subprocess.run(["systemctl", "set-property", "--runtime", "coding-agent-asi.slice",
                    f"CPUQuota={resources.cpus * options['parallel'] * 100:g}%",
                    f"MemoryMax={resources.memory_mib * options['parallel']}M"], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpus", type=float, default=4)
    parser.add_argument("--memory-mib", type=int, default=32768)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--check", action="store_true")
    resources, native = parser.parse_known_args(argv)
    if resources.cpus <= 0 or resources.memory_mib <= 0:
        parser.error("CPU and memory limits must be positive")
    os.chdir(ROOT)
    from ai4sci_bench.cli import cli, run, _redact_agent_config
    from ai4sci_bench.runner.os_sandbox import OSSandbox

    OSSandbox.DEFAULT_CPU_LIMIT = str(resources.cpus)
    OSSandbox.DEFAULT_MEMORY_LIMIT = f"{resources.memory_mib}m"
    # Calling cli directly keeps .env files from restoring cleared proxy settings.
    if not native or native[0] != "run" or "--help" in native:
        cli.main(args=native, prog_name="asibench", standalone_mode=False)
        if "--help" in native:
            print("\nLocal launcher options: --cpus N, --memory-mib N, --print-config, --check")
        return 0
    with run.make_context("run", native[1:]) as context:
        options = context.params.copy()
    if not 1 <= options["parallel"] <= 8:
        raise ValueError("--parallel must be between 1 and 8")
    if options["timeout"] <= 0:
        raise ValueError("--timeout must be positive")
    config = {"runner": "ASI-Bench official CLI", "run_options": options.copy(),
              "endpoint_assignment": ASSIGNMENT,
              "resources_per_task": {"cpus": resources.cpus, "memory_mib": resources.memory_mib}}
    config["run_options"]["agent_config"] = _redact_agent_config(json.loads(options["agent_config"]))
    server_urls(config["run_options"]["agent_config"].get("api_base"))
    if resources.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    checked = check_environment(options, resources)
    if resources.check:
        summary = {key: value for key, value in checked.items() if key != "tasks"}
        print(json.dumps({"status": "ready", "task_count": len(checked["tasks"]), **summary},
                         indent=2, ensure_ascii=False))
        return 0
    output = Path(options["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launcher.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "launcher.json"
        resume = config["run_options"].pop("resume", None)
        if resume and Path(resume).resolve() != output:
            raise ValueError("--resume must refer to the same --output-dir")
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["configuration"] != config:
                raise ValueError("Run settings changed; choose a separate --output-dir")
            if not resume:
                raise ValueError("Results already exist; pass --resume with the output directory")
        elif any(p.name != ".launcher.lock" for p in output.iterdir()):
            raise ValueError("Output directory contains untracked results; choose a separate --output-dir")
        record = {"configuration": config, "preflight": checked, "status": "running",
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "agent_router_sha256": hashlib.sha256((ROOT / "scripts/asi_agent.py").read_bytes()).hexdigest()}
        save(path, record)
        try:
            set_resource_limits(options, resources)
            with use_server_pool(output):
                cli.main(args=native, prog_name="asibench", standalone_mode=False)
            record["status"] = "outputs_collected"
        except BaseException:
            record["status"] = "interrupted_or_failed"
            raise
        finally:
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            save(path, record)
    print("ASI outputs collected. seed42 official scores require the ASI-Bench Portal.")
    return 0


if __name__ == "__main__":
    import click

    try:
        raise SystemExit(main())
    except (click.Abort, KeyboardInterrupt):
        print("ASI: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ASI: {exc}", file=sys.stderr)
        raise SystemExit(1)
