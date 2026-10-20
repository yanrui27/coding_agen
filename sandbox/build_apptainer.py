#!/usr/bin/env python3
"""Build the Apptainer 1.5.4 toolkit that runs on ACP Workers (Ubuntu 22.04, GLIBC 2.35).

The benchmark toolkit needs GLIBC 2.38, so this extracts the official Jammy packages into
runtime/apptainer-1.5.4-jammy with relative rpaths; no system package is installed. Run it on a host
with GLIBC >= 2.38: the benchmark patchelf needs it, and the final version check runs the toolkit.
"""

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.request import ProxyHandler, Request, build_opener


SANDBOX_DIR = Path(__file__).resolve().parent
CACHE = SANDBOX_DIR / "runtime/packages"
PREFIX = SANDBOX_DIR / "runtime/apptainer-1.5.4-jammy"
PATCHELF = SANDBOX_DIR.parent / "benchmark/.sandbox-tools/apptainer/usr/bin/patchelf"
PPA = "https://ppa.launchpadcontent.net/apptainer/ppa/ubuntu/"
UBUNTU = "https://archive.ubuntu.com/ubuntu/"


def fetch(url):
    request = Request(url, headers={"User-Agent": "coding-agent-sandbox/1.0"})
    with build_opener(ProxyHandler({})).open(request, timeout=45) as response:
        return response.read()


def fields(block):
    return {
        line.split(": ", 1)[0]: line.split(": ", 1)[1]
        for line in block.splitlines()
        if ": " in line and not line.startswith(" ")
    }


def main():
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    # Without a working patchelf every ELF would be skipped silently and lose its rpath.
    subprocess.run([str(PATCHELF), "--version"], check=True, capture_output=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    PREFIX.mkdir(parents=True, exist_ok=True)
    index = gzip.decompress(fetch(PPA + "dists/jammy/main/binary-amd64/Packages.gz")).decode()
    app = next(fields(block) for block in index.split("\n\n") if block.startswith("Package: apptainer\n"))
    assert app["Version"] == "1.5.4-stable-1~jammy"
    assert app["SHA256"] == "0d052ff08d7d50b865ed64ca8d88930dd7050e39182d24891c846c1ccea0a1d7"
    wanted = {
        "libfuse3-3", "liblz4-1", "liblzma5", "liblzo2-2", "libprotobuf-c1",
        "libseccomp2", "libtalloc2", "libuuid1", "libzstd1", "zlib1g", "uidmap",
        "fakeroot", "libfakeroot", "libaudit1", "libcap-ng0", "libselinux1",
        "libpcre2-8-0", "libacl1", "libattr1", "libaudit-common",
    }
    selected = [(PPA, app)]
    for component in ("main", "universe"):
        path = CACHE / f"jammy-{component}-Packages.gz"
        if not path.exists():
            path.write_bytes(fetch(UBUNTU + f"dists/jammy/{component}/binary-amd64/Packages.gz"))
        for block in gzip.decompress(path.read_bytes()).decode().split("\n\n"):
            if block.partition("\n")[0].removeprefix("Package: ") in wanted:
                package = fields(block)
                selected.append((UBUNTU, package))
                wanted.remove(package["Package"])
        if not wanted:
            break
    assert not wanted, sorted(wanted)

    def download(item):
        base, package = item
        path = CACHE / Path(package["Filename"]).name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != package["SHA256"]:
            data = fetch(base + package["Filename"])
            assert hashlib.sha256(data).hexdigest() == package["SHA256"]
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        return {
            "package": package["Package"], "version": package["Version"],
            "filename": path.name, "url": base + package["Filename"],
            "sha256": package["SHA256"], "bytes": path.stat().st_size,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, selected))
    for record in records:
        subprocess.run(["dpkg-deb", "--extract", str(CACHE / record["filename"]), str(PREFIX)], check=True)
    for name in ("etc", "var"):
        (PREFIX / name).mkdir(exist_ok=True)
        alias = PREFIX / "usr" / name
        if not alias.exists():
            alias.symlink_to("../" + name, target_is_directory=True)
    library_dirs = [PREFIX / "usr/lib/x86_64-linux-gnu", PREFIX / "lib/x86_64-linux-gnu"]
    modified = []
    for path in PREFIX.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        probe = subprocess.run([str(PATCHELF), "--print-needed", str(path)], capture_output=True, text=True)
        if probe.returncode or not probe.stdout.strip():
            continue
        origin = ":".join("$ORIGIN/" + os.path.relpath(directory, path.parent) for directory in library_dirs)
        subprocess.run([str(PATCHELF), "--set-rpath", origin, str(path)], check=True)
        modified.append(str(path.relative_to(PREFIX)))
    for name in ("newuidmap", "newgidmap"):
        (PREFIX / "usr/bin" / name).chmod(0o755)
    config = PREFIX / "etc/apptainer/apptainer.conf"
    content = config.read_text()
    binary_path = f"{PREFIX}/usr/bin:{PREFIX}/usr/libexec/apptainer/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    content = re.sub(r"^#?\s*binary path\s*=.*$", "binary path = " + binary_path, content, flags=re.M)
    content = re.sub(r"^mksquashfs procs\s*=.*$", "mksquashfs procs = 2", content, flags=re.M)
    config.write_text(content + "\nmksquashfs mem = 256M\n")
    version = subprocess.check_output([str(PREFIX / "usr/bin/apptainer"), "--version"], text=True).strip()
    report = {
        "prefix": str(PREFIX), "version": version, "target_distribution": "Ubuntu 22.04 Jammy",
        "source": "Official Apptainer PPA and Ubuntu Jammy package indexes",
        "system_packages_installed": False, "benchmark_environment_modified": False,
        "packages": records, "relocated_elfs": modified,
    }
    (SANDBOX_DIR / "runtime/apptainer-1.5.4-jammy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"prefix": str(PREFIX), "version": version, "packages": len(records)}, indent=2))


if __name__ == "__main__":
    main()
