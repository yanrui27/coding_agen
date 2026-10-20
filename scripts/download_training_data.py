#!/usr/bin/env python3
"""Download pinned HF dataset snapshots directly, without proxy or HF caches."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import struct
import threading
import time
from urllib.parse import quote

for _key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_key, None)
os.environ.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
                  GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")

import requests

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/training"
WORK = PROJECT / "logs/download_training_data"
ENDPOINT = "https://hf-mirror.com"
REPOS = [
    ("nvidia/Nemotron-SFT-Math-v4", "main", "HLE mathematics"),
    ("nvidia/Nemotron-SFT-Science-v2", "main", "HLE science; teacher filtering required"),
    ("centre-for-swe/MindForge-27B-Training-Trajectories", "main", "SWE"),
    ("open-athena/recursive-task-synthesis-glm-5.3-rollouts", "main", "Terminal; success filtering required"),
    ("amphora/rlmath-kimi-k3-trajectories", "main", "Mathematics with tools"),
    ("FrontisAI/NatureBench-traces", "main", "ASI; teacher/success filtering required"),
    ("open-athena/pdbthink-glm53-teacher-traces", "supplement", "Protein structure reasoning"),
    ("nvidia/Nemotron-Math-Proofs-v3-SFT", "supplement", "Mathematical proofs; role/quality filtering required"),
    ("XiaomiMiMo/MiMo-V2.6-RL-oss", "rl", "RL environments, not ready-made SFT responses"),
]
LOCK = threading.RLock()
LOCAL = threading.local()
STATE = {}
STOP = threading.Event()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    return LOCAL.session


def log(message):
    with LOCK:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def manifest():
    with LOCK:
        atomic_json(ROOT / "manifest.json", {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "transport": "direct; all inherited HTTP/HTTPS/ALL proxy variables removed",
            "download_endpoint": ENDPOINT,
            "scope": "Complete original repository snapshots, including unsuccessful attempts and other teachers; no training subsets created.",
            "datasets": list(STATE.values()),
        })


def safe_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or ".git" in p.parts:
        raise ValueError("Unsafe repository path")
    return Path(*p.parts)


def metadata(repo):
    path = WORK / "metadata" / (repo.replace("/", "__") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    # Reuse this session's already inspected, pinned inventory when available.
    inspected = Path("/tmp/coding_agent_dataset_inventory") / path.name
    if inspected.exists():
        data = json.loads(inspected.read_text())
    else:
        response = session().get(f"{ENDPOINT}/api/datasets/{repo}",
                                 params={"blobs": "true"}, timeout=(20, 120))
        response.raise_for_status()
        data = response.json()
    atomic_json(path, data)
    return data


def git_command(gitdir, *args):
    return ["git", "-c", "http.proxy=", "-c", "https.proxy=", f"--git-dir={gitdir}", *args]


def export_git_files(repo, data, target):
    """Use a Git pack for small files, avoiding tens of thousands of HTTP requests."""
    if repo == "FrontisAI/NatureBench-traces":
        return export_http_git_files(repo, data, target)
    ordinary = [f for f in data["siblings"] if not f.get("lfs")]
    local_complete = True
    for entry in ordinary:
        path = target / safe_path(entry["rfilename"])
        if not path.is_file() or path.stat().st_size != entry["size"]:
            local_complete = False
            break
        h = hashlib.sha1(b"blob " + str(entry["size"]).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != entry["blobId"]:
            local_complete = False
            break
    if local_complete:
        return len(ordinary), sum(f["size"] for f in ordinary)
    gitdir = WORK / (repo.split("/")[1] + ".git")
    clone_log = WORK / (repo.split("/")[1] + "-git.log")
    sha = data["sha"]
    with clone_log.open("a") as output:
        # An earlier supervisor may have exited while Git was still receiving a pack.
        while gitdir.exists() and any((gitdir / "objects/pack").glob("tmp_pack*")):
            check = subprocess.run(git_command(gitdir, "rev-parse", "--verify", "HEAD"),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if check.returncode == 0:
                break
            time.sleep(5)
        if not gitdir.exists():
            subprocess.run(["git", "-c", "http.proxy=", "-c", "https.proxy=", "clone",
                            "--bare", "--depth", "1", f"{ENDPOINT}/datasets/{repo}", str(gitdir)],
                           stdout=output, stderr=output, check=True, timeout=7200)
        check = subprocess.run(git_command(gitdir, "cat-file", "-e", sha + "^{commit}"),
                               stdout=output, stderr=output)
        if check.returncode:
            subprocess.run(git_command(gitdir, "fetch", "--depth", "1", "origin", sha),
                           stdout=output, stderr=output, check=True, timeout=7200)
    # Match the actual pinned tree to the API inventory before exporting anything.
    tree = subprocess.check_output(git_command(gitdir, "ls-tree", "-r", "-z", sha))
    entries = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        header, name = record.split(b"\t", 1)
        mode, typ, oid = header.decode().split()
        if typ != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"Unsupported tree entry: {repo}: {name!r}")
        entries[name.decode()] = (mode, oid)
    expected = {f["rfilename"]: f for f in data["siblings"]}
    if entries.keys() != expected.keys():
        raise ValueError(f"Pinned tree/inventory mismatch: {repo}")
    proc = subprocess.Popen(git_command(gitdir, "cat-file", "--batch"),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    count = total = 0
    try:
        for name, entry in expected.items():
            mode, oid = entries[name]
            if oid != entry["blobId"]:
                raise ValueError(f"Git object mismatch: {repo}: {name}")
            if entry.get("lfs"):
                continue
            proc.stdin.write((oid + "\n").encode())
            proc.stdin.flush()
            obj, typ, size = proc.stdout.readline().split()
            size = int(size)
            content = proc.stdout.read(size)
            ending = proc.stdout.read(1)
            if obj.decode() != oid or typ != b"blob" or len(content) != size or ending != b"\n":
                raise ValueError("Invalid cat-file response")
            actual = hashlib.sha1(b"blob " + str(size).encode() + b"\0" + content).hexdigest()
            if actual != entry["blobId"] or size != entry["size"]:
                raise ValueError(f"Git content verification failed: {repo}: {name}")
            dst = target / safe_path(name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.read_bytes() != content:
                dst.write_bytes(content)
            dst.chmod(0o750 if mode == "100755" else 0o640)
            count += 1
            total += size
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"git cat-file failed: {repo}")
    return count, total


def export_http_git_files(repo, data, target):
    """The NatureBench Git pack is slow; verify raw pinned Git blobs instead."""
    entries = [f for f in data["siblings"] if not f.get("lfs")]

    def fetch(entry):
        dst = target / safe_path(entry["rfilename"])
        size = entry["size"]
        prefix = b"blob " + str(size).encode() + b"\0"
        if dst.exists() and dst.stat().st_size == size:
            if hashlib.sha1(prefix + dst.read_bytes()).hexdigest() == entry["blobId"]:
                return
        url = f"{ENDPOINT}/datasets/{repo}/raw/{data['sha']}/{quote(entry['rfilename'])}"
        for attempt in range(12):
            try:
                if dst.exists() and dst.stat().st_size == size:
                    if hashlib.sha1(prefix + dst.read_bytes()).hexdigest() == entry["blobId"]:
                        return
                response = session().get(url, timeout=(20, 120))
                response.raise_for_status()
                content = response.content
                if len(content) != size or hashlib.sha1(prefix + content).hexdigest() != entry["blobId"]:
                    raise ValueError("Git blob verification failed")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(content)
                dst.chmod(0o640)
                return
            except Exception as exc:
                if attempt == 11:
                    raise RuntimeError(f"HTTP Git blob download failed: {entry['rfilename']}: {type(exc).__name__}") from None
                time.sleep(min(60, 2 ** attempt))

    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch, entries))
    return len(entries), sum(f["size"] for f in entries)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def batch_actions(repo, objects):
    for attempt in range(12):
        try:
            response = session().post(f"{ENDPOINT}/datasets/{repo}.git/info/lfs/objects/batch",
                                      json={"operation": "download", "transfers": ["basic"],
                                            "objects": objects}, timeout=(20, 120))
            response.raise_for_status()
            result = {}
            for obj in response.json()["objects"]:
                if "error" in obj:
                    raise RuntimeError("LFS object unavailable")
                action = obj["actions"]["download"]
                # Mirror rewrites this hostname to an unavailable alias. Its
                # signed URL works on the original upstream CDN, directly.
                action["href"] = action["href"].replace(".hf-mirror.org/", ".hf.co/")
                result[obj["oid"]] = action
            return result
        except Exception as exc:
            if attempt == 11:
                raise RuntimeError(f"LFS batch failed: {repo}: {type(exc).__name__}") from None
            log(f"LFS batch retry: {repo}: {type(exc).__name__}")
            time.sleep(min(60, 2 ** attempt))


def download_object(repo, target, oid, files, action):
    size = files[0]["size"]
    dst = target / safe_path(files[0]["rfilename"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = WORK / "partials" / (repo.split("/")[1] + "--" + oid)
    verified = dst.exists() and dst.stat().st_size == size and sha256_file(dst) == oid
    if not verified and size >= 128 * 1024 * 1024:
        # Segment large files; HTTPS transfers never inherit aria2 proxy settings.
        aria_log = WORK / "partials" / (part.name + ".log")
        for attempt in range(8):
            if attempt:
                action = batch_actions(repo, [{"oid": oid, "size": size}])[oid]
            with aria_log.open("a") as output:
                result = subprocess.run([
                "aria2c", "--no-conf=true", "--all-proxy=", "--http-proxy=", "--https-proxy=",
                "--ftp-proxy=", "--max-connection-per-server=8", "--split=8", "--min-split-size=32M",
                "--file-allocation=none", "--continue=true", "--auto-file-renaming=false",
                "--allow-overwrite=true", "--check-integrity=true", f"--checksum=sha-256={oid}",
                "--connect-timeout=30", "--timeout=90", "--max-tries=12", "--retry-wait=5",
                "--summary-interval=0", "--console-log-level=warn", "--download-result=hide",
                f"--dir={part.parent}", f"--out={part.name}", action["href"],
                ], stdout=output, stderr=output)
            if result.returncode == 0:
                break
            if attempt == 7:
                raise RuntimeError(f"aria2 failed: {repo}/{files[0]['rfilename']}: exit {result.returncode}")
            log(f"Segmented transfer retry: {repo}/{files[0]['rfilename']}: exit {result.returncode}")
            time.sleep(min(60, 2 ** attempt))
        if part.stat().st_size != size:
            raise ValueError("aria2 size mismatch")
        part.replace(dst)
        dst.chmod(0o640)
        aria_log.unlink(missing_ok=True)
        verified = True
    if not verified:
        for attempt in range(12):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > size:
                    part.unlink()
                    offset = 0
                h = hashlib.sha256()
                if offset:
                    with part.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            h.update(chunk)
                if offset < size:
                    headers = dict(action.get("header", {}))
                    if offset:
                        headers["Range"] = f"bytes={offset}-"
                    with session().get(action["href"], headers=headers, stream=True,
                                       timeout=(30, 120)) as response:
                        if response.status_code in (401, 403, 404):
                            action = batch_actions(repo, [{"oid": oid, "size": size}])[oid]
                            raise RuntimeError("Refreshing expired LFS URL")
                        response.raise_for_status()
                        if offset and response.status_code == 200:
                            offset = 0
                            h = hashlib.sha256()
                        if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Incorrect resume range")
                        with part.open("ab" if offset else "wb") as stream:
                            for chunk in response.iter_content(4 * 1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                                    h.update(chunk)
                if part.stat().st_size != size or h.hexdigest() != oid:
                    if part.stat().st_size == size:
                        part.unlink()
                    raise ValueError("LFS size/SHA256 mismatch")
                part.replace(dst)
                dst.chmod(0o640)
                break
            except Exception as exc:
                if attempt == 11:
                    raise RuntimeError(f"Download failed: {repo}: {files[0]['rfilename']}: {type(exc).__name__}") from None
                log(f"Retry {repo}/{files[0]['rfilename']}: {type(exc).__name__}")
                time.sleep(min(60, 2 ** attempt))
    for entry in files[1:]:
        other = target / safe_path(entry["rfilename"])
        other.parent.mkdir(parents=True, exist_ok=True)
        if not other.exists() or other.stat().st_size != size or sha256_file(other) != oid:
            shutil.copyfile(dst, other)
            other.chmod(0o640)
    with LOCK:
        state = STATE[repo]
        state["verified_files"] += len(files)
        state["verified_bytes"] += size * len(files)
        with (WORK / "verified-lfs.jsonl").open("a") as output:
            output.write(json.dumps({"repo": repo, "sha256": oid, "size": size,
                                     "paths": [f["rfilename"] for f in files]}) + "\n")


def setup(repo, category, purpose, pool, futures):
    data = metadata(repo)
    target = ROOT / repo.split("/")[1]
    target.mkdir(exist_ok=True)
    with LOCK:
        STATE[repo] = {"repo": repo, "source": f"https://huggingface.co/datasets/{repo}",
                       "revision": data["sha"], "directory": target.name, "category": category,
                       "purpose": purpose, "status": "downloading",
                       "expected_files": len(data["siblings"]),
                       "expected_bytes": sum(f["size"] for f in data["siblings"]),
                       "verified_files": 0, "verified_bytes": 0}
    manifest()
    log(f"Exporting Git files: {repo}")
    count, total = export_git_files(repo, data, target)
    with LOCK:
        STATE[repo]["verified_files"] += count
        STATE[repo]["verified_bytes"] += total
    log(f"Exported and verified {count} Git files: {repo}")
    groups = {}
    for entry in data["siblings"]:
        if entry.get("lfs"):
            groups.setdefault(entry["lfs"]["sha256"], []).append(entry)
    items = sorted(groups.items(), key=lambda pair: pair[1][0]["size"])
    for start in range(0, len(items), 128):
        batch = items[start:start + 128]
        actions = batch_actions(repo, [{"oid": oid, "size": fs[0]["size"]} for oid, fs in batch])
        for oid, fs in batch:
            future = pool.submit(download_object, repo, target, oid, fs, actions[oid])
            with LOCK:
                futures[future] = repo
    log(f"Queued {len(groups)} unique LFS objects: {repo}")


def reporter():
    while not STOP.wait(30):
        manifest()
        with LOCK:
            files = sum(s["verified_files"] for s in STATE.values())
            size = sum(s["verified_bytes"] for s in STATE.values())
        partial = 0
        for path in (WORK / "partials").glob("*"):
            if path.name.endswith((".aria2", ".log")):
                continue
            try:
                control = path.with_name(path.name + ".aria2")
                if control.exists():
                    data = control.read_bytes()
                    offset = 10 + struct.unpack_from(">I", data, 6)[0]
                    piece, total, _, length = struct.unpack_from(">IQQI", data, offset)
                    bits = data[offset + 24:offset + 24 + length]
                    partial += min(total, sum(x.bit_count() for x in bits) * piece)
                else:
                    partial += path.stat().st_size
            except (FileNotFoundError, struct.error):
                pass
        log(f"Verified {files} files, {size / 2**30:.2f} GiB; in progress {partial / 2**30:.2f} GiB")


def main():
    ROOT.mkdir(exist_ok=True)
    for directory in (WORK, WORK / "metadata", WORK / "partials"):
        directory.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()
    futures = {}
    errors = []
    with cf.ThreadPoolExecutor(max_workers=24) as downloads:
        with cf.ThreadPoolExecutor(max_workers=3) as exports:
            # Start the longest file transfers early, reducing the final long-file tail.
            order = sorted(REPOS, key=lambda item: {"nvidia/Nemotron-SFT-Science-v2": 0,
                           "nvidia/Nemotron-Math-Proofs-v3-SFT": 1,
                           "open-athena/recursive-task-synthesis-glm-5.3-rollouts": 2}.get(item[0], 3))
            jobs = {exports.submit(setup, *item, downloads, futures): item[0] for item in order}
            for future in cf.as_completed(jobs):
                try:
                    future.result()
                except Exception as exc:
                    repo = jobs[future]
                    log(f"SETUP ERROR {repo}: {type(exc).__name__}: {exc}")
                    errors.append(repo)
                    with LOCK:
                        if repo in STATE:
                            STATE[repo]["status"] = "failed"
        for future in cf.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                repo = futures[future]
                log(f"DOWNLOAD ERROR {repo}: {exc}")
                errors.append(repo)
                with LOCK:
                    STATE[repo]["status"] = "failed"
    for repo, _, _ in REPOS:
        if repo not in STATE or repo in errors:
            continue
        state = STATE[repo]
        data = metadata(repo)
        target = ROOT / state["directory"]
        expected = {f["rfilename"]: f["size"] for f in data["siblings"]}
        actual = {str(p.relative_to(target)): p.stat().st_size for p in target.rglob("*") if p.is_file()}
        if actual != expected or state["verified_files"] != state["expected_files"]:
            state["status"] = "failed"
            errors.append(repo)
        else:
            state["status"] = "complete"
            state["integrity"] = "All Git blobs matched upstream Git SHA1; all LFS objects matched upstream SHA256; all paths and sizes matched pinned repository inventory."
            log(f"COMPLETE {repo}: {len(actual)} files")
    STOP.set()
    manifest()
    log("FINISHED " + ("with failures: " + ", ".join(sorted(set(errors))) if errors else "all 9 datasets verified"))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
