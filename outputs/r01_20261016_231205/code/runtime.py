"""Megatron disk guard and local checkpoint/HF artifact checks."""
import json
import hashlib
import subprocess
import time
from datetime import datetime
import shutil
import struct
import sys
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def latest(root):
    candidates = [p for p in Path(root).glob("checkpoint-*")
                  if p.name.removeprefix("checkpoint-").isdigit()
                  and (p / "complete.json").is_file()
                  and (p / "latest_checkpointed_iteration.txt").is_file()]
    if not candidates:
        raise RuntimeError(f"No complete checkpoint under {root}")
    return max(candidates, key=lambda p: int(p.name.removeprefix("checkpoint-")))


def headers(root):
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    result = {}
    for name in sorted(set(index["weight_map"].values())):
        path = root / name
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < length < 128 * 1024**2, f"Invalid header: {path}"
            header = json.loads(stream.read(length))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        assert path.stat().st_size == 8 + length + max(v["data_offsets"][1] for v in tensors.values())
        for key, value in tensors.items():
            assert key not in result and index["weight_map"][key] == name, key
            result[key] = {"shape": value["shape"], "dtype": value["dtype"]}
    assert set(result) == set(index["weight_map"])
    return result


def verify_hf(base, exported):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    base, exported = Path(base), Path(exported)
    expected, actual = headers(base), headers(exported)
    assert set(actual) == set(expected), {
        "missing": sorted(set(expected) - set(actual))[:20],
        "extra": sorted(set(actual) - set(expected))[:20],
    }
    for key in actual:
        assert actual[key]["shape"] == expected[key]["shape"], key
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        assert (exported / name).is_file(), name
    assert json.loads((exported / "config.json").read_text())["model_type"] == \
        json.loads((base / "config.json").read_text())["model_type"]
    index = json.loads((exported / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(exported / shard), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                tensor = stream.get_tensor(key).reshape(-1)
                for chunk in tensor.split(4 * 1024**2):
                    assert bool(torch.isfinite(chunk).all()), f"Non-finite weights: {key}"
        print(f"Verified {shard}", flush=True)
    write_json(exported / "verification.json", {
        "status": "passed", "base": str(base), "parameter_tensors": len(actual),
        "checks": ["complete shard storage", "base key and shape match", "all weights finite", "config and tokenizer files"],
    })

PROJECT = Path("/mnt/data/rui.yan/projects/coding_agent")
MODEL = Path("/mnt/model/ckpts/Qwen3.6-35B-A3B")
DATASETS = ("science", "math")
ACTIVATE = Path("/mnt/code/yuhao.feng/qwen36_omni/training/qwen36_audio_swift/activate_megatron_swift.sh")
ARCHIVE_SHA = ACTIVATE.parents[2] / "qwen36-omni-megatron-swift-cu128-20260614.tar.gz.sha256"
COMMON = {"model": str(MODEL), "model_type": "qwen3_5_moe", "template": "qwen3_5",
          "loss_scale": "default", "max_length": 65536, "truncation_strategy": "delete"}


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()

def command(program, parameters):
    result = program.split()
    for key, value in parameters.items():
        if value is None:
            continue
        values = value if isinstance(value, (tuple, list)) else [value]
        result += ["--" + key] + [str(v).lower() if isinstance(v, bool) else str(v) for v in values]
    return result

def caches():
    manifest = json.loads((PROJECT / "logs/sft/manifest.json").read_text())
    records = {r["dataset"]: r for r in manifest["files"]}
    tokenizer = {name: checksum(MODEL / name) for name in
                 ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]}
    runtime = {str(p): checksum(p) for p in [ACTIVATE, ARCHIVE_SHA]}
    result = []
    for name in DATASETS:
        source = records[name]
        path = PROJECT / f"sft/data/{name}.train.jsonl"
        assert path.resolve() == Path(source["path"]).resolve() and path.stat().st_size == source["bytes"]
        config = {"source": source, "encoding": COMMON, "tokenizer": tokenizer,
                  "runtime": runtime, "packing": False, "split_dataset_ratio": 0}
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
        result.append((PROJECT / "cache/sft" / key, config))
    return result

def run_command(argv, log, env):
    print("Running: " + " ".join(argv), flush=True)
    with log.open("ab", buffering=0) as target:
        process = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            target.write(line)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, argv)

def wait_for(paths, run_dir):
    deadline = time.monotonic() + 86400
    while not all(p.exists() for p in paths):
        failures = list(run_dir.glob("node*.failed.json"))
        if failures:
            raise RuntimeError(f"Peer failed: {failures[0].read_text()}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {paths}")
        time.sleep(5)


def prepare_cache(run_dir, rank):
    run_dir = Path(run_dir)
    (run_dir / "logs").mkdir(exist_ok=True)
    plans = caches()
    cache, config = plans[rank]
    if not (cache / "complete.json").exists():
        if cache.exists():
            raise RuntimeError(f"Incomplete cache exists: {cache}")
        assert checksum(Path(config["source"]["path"])) == config["source"]["sha256"], "Source checksum mismatch"
        env = os.environ.copy()
        for name in ["NNODES", "NODE_RANK", "NPROC_PER_NODE", "MASTER_ADDR", "MASTER_PORT"]:
            env.pop(name, None)
        args = {**COMMON, "dataset": config["source"]["path"], "split_dataset_ratio": 0,
                "to_cached_dataset": True, "output_dir": str(cache), "dataset_num_proc": 16,
                "load_from_cache_file": True, "remove_unused_columns": False, "packing": False, "streaming": False}
        run_command(command("swift export", args), run_dir / f"logs/encode_node{rank}.log", env)
        from datasets import load_from_disk
        data = load_from_disk(str(cache / "train"))
        lengths = [max(v) if isinstance(v, list) else v for v in data["lengths"]]
        assert lengths and 0 < min(lengths) <= max(lengths) <= COMMON["max_length"]
        assert len(data) <= config["source"]["rows"]
        write_json(cache / "config.json", config)
        write_json(cache / "complete.json", {"input_rows": config["source"]["rows"], "retained_rows": len(data),
                   "deleted_rows": config["source"]["rows"] - len(data), "tokens": sum(lengths), "max_length": max(lengths)})
    wait_for([p / "complete.json" for p, _ in plans], run_dir)
    for path, expected in plans:
        assert json.loads((path / "config.json").read_text()) == expected, f"Cache config mismatch: {path}"
    if rank == 0:
        target = run_dir / "cache_paths.txt"
        temporary = target.with_suffix(".tmp")
        temporary.write_text("\n".join(str(p / "train") for p, _ in plans) + "\n")
        temporary.replace(target)
        write_json(run_dir / "cache_stats.json", {config["source"]["dataset"]:
                   json.loads((p / "complete.json").read_text()) for p, config in plans})
    wait_for([run_dir / "cache_paths.txt"], run_dir)


def scratch(run_dir, rank):
    needed = 3 * caches()[rank][1]["source"]["bytes"] + 50 * 1024**3
    local = Path("/tmp")
    if shutil.disk_usage(local).free < needed:
        local = PROJECT / "cache/scratch"
    path = local / Path(run_dir).name / f"node{rank}"
    path.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(path).free >= needed, "Insufficient preprocessing scratch space"
    (path / "tmp").mkdir(exist_ok=True)
    return path


def publish(run_dir):
    run_dir = Path(run_dir)
    staging = run_dir / "hf_export"
    destination = PROJECT / "ckpts" / f"{run_dir.name}_hf"
    assert not destination.exists(), destination
    for name in ["chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
                 "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"]:
        if (MODEL / name).exists() and not (staging / name).exists():
            shutil.copy2(MODEL / name, staging / name)
    verify_hf(MODEL, staging)
    destination.parent.mkdir(exist_ok=True)
    staging.rename(destination)
    write_json(run_dir / "complete.json", {"hf_model": str(destination), "checkpoint": str(latest(run_dir / "checkpoints"))})

if __name__ == "__main__":
    import os
    if sys.argv[1] == "latest":
        print(latest(sys.argv[2]))
    elif sys.argv[1] == "verify-hf":
        verify_hf(*sys.argv[2:4])
    elif sys.argv[1] == "scratch":
        print(scratch(sys.argv[2], int(sys.argv[3])))
    elif sys.argv[1] == "cache":
        COMMON.update(model=sys.argv[4], max_length=int(sys.argv[5]))
        MODEL = Path(sys.argv[4])
        prepare_cache(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "trained":
        root, rank = Path(sys.argv[2]), int(sys.argv[3])
        (root / f"node{rank}.trained").touch()
        wait_for([root / f"node{i}.trained" for i in range(2)], root)
    elif sys.argv[1] == "publish":
        MODEL = Path(sys.argv[3])
        publish(sys.argv[2])
    elif sys.argv[1] == "wait-hf":
        wait_for([Path(sys.argv[2]) / "complete.json"], Path(sys.argv[2]))
    elif sys.argv[1] == "failed":
        write_json(Path(sys.argv[2]) / f"node{sys.argv[3]}.failed.json", {"exit_code": int(sys.argv[4])})
    elif sys.argv[1] == "status":
        write_json(Path(sys.argv[2]) / f"node{sys.argv[3]}.status.json", {"stage": sys.argv[4], "time": datetime.now().isoformat()})
    else:
        raise SystemExit("Unknown runtime operation")
else:
    import torch
    import torch.distributed as dist
    from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

    class DiskGuard(MegatronCallback):
        def __init__(self, trainer):
            super().__init__(trainer)
            self.planned_iters = self.args.train_iters
            self.last_bytes = 0
            self.minimum = 1024 * 1024**3
            base = Path(self.args.model_info.model_dir)
            self.model_bytes = sum(p.stat().st_size for p in base.glob("*.safetensors"))
            self.stopped_for_space = False

        def space(self):
            values = [0, 0]
            if dist.get_rank() == 0:
                values = [shutil.disk_usage(self.args.output_dir).free,
                          max(self.minimum, 10 * self.model_bytes,
                              2 * self.last_bytes + 2 * self.model_bytes)]
            tensor = torch.tensor(values, dtype=torch.int64, device=torch.cuda.current_device())
            dist.broadcast(tensor, src=0)
            return tensor.tolist()

        def on_train_begin(self):
            free, reserve = self.space()
            if free < reserve:
                raise RuntimeError(f"Insufficient disk before training: free={free}, reserve={reserve}")
            if dist.get_rank() == 0:
                print(f"Disk guard: free={free}, reserve={reserve}; retaining every checkpoint", flush=True)

        def on_step_end(self):
            free, reserve = self.space()
            if free >= reserve:
                return
            # This Megatron-SWIFT version terminates its loop using train_iters.
            # All ranks take the same decision; DefaultFlow then saves the final step.
            self.args.train_iters = self.state.iteration
            self.state.should_save = True
            self.stopped_for_space = True
            if dist.get_rank() == 0:
                write_json(Path(self.args.output_dir) / "space_stop.json", {
                    "step": self.state.iteration, "planned_iters": self.planned_iters,
                    "free_bytes": free, "reserved_bytes": reserve,
                })
                print(f"Disk guard requested clean stop at step {self.state.iteration}", flush=True)

        def on_save(self, output_dir):
            dist.barrier()
            if dist.get_rank() == 0:
                root = Path(output_dir)
                self.last_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
                write_json(root / "complete.json", {
                    "step": self.state.iteration, "bytes": self.last_bytes,
                    "optimizer_and_rng": True, "stopped_for_space": self.stopped_for_space,
                })
            dist.barrier()

        def on_train_end(self):
            if dist.get_rank() == 0:
                write_json(Path(self.args.output_dir) / "training_complete.json", {
                    "step": self.state.iteration, "planned_iters": self.planned_iters,
                    "stopped_for_space": self.stopped_for_space,
                    "last_checkpoint": self.state.last_model_checkpoint,
                })

    megatron_callbacks_map["disk_guard"] = DiskGuard
