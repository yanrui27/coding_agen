#!/usr/bin/env python3
"""Install the checked official Apptainer binary under this project, without sudo."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/downloads'
PACKAGE = CACHE / 'apptainer_1.5.4_amd64.deb'
PREFIX = ROOT / '.sandbox-tools/apptainer'
ASSET = 'https://api.github.com/repos/apptainer/apptainer/releases/assets/581679855'
SHA256 = '281cf9bd68f16b6f7c34a3a94d77668bc327b9c09db0412ce7890162734a97f7'
SIZE = 32859864
PART_SIZE = 4 * 1024 * 1024
SLICE_SIZE = 512 * 1024


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download():
    if PACKAGE.exists() and PACKAGE.stat().st_size == SIZE and digest(PACKAGE) == SHA256:
        return
    request = urllib.request.Request(ASSET, headers={'Accept': 'application/octet-stream', 'Range': 'bytes=0-0'})
    with opener().open(request, timeout=20) as response:
        url = response.geturl()
        assert 'release-assets.githubusercontent.com/' in url, 'Expected the official release asset redirect'
    # Reuse prefixes left by an interrupted download; split remaining ranges
    # into small independent slices so stalled connections can be resumed.
    jobs, parts = [], []
    for index, start in enumerate(range(0, SIZE, PART_SIZE)):
        length = min(PART_SIZE, SIZE - start)
        prefix = CACHE / f'apptainer.part{index:02d}'
        have = prefix.stat().st_size if prefix.exists() else 0
        assert have <= length, prefix
        slices = []
        for offset in range(have, length, SLICE_SIZE):
            size = min(SLICE_SIZE, length - offset)
            piece = CACHE / f'apptainer.part{index:02d}.offset{offset}'
            slices.append(piece)
            jobs.append((start + offset, size, piece))
        parts.append((prefix, slices))

    def fetch(job):
        start, size, piece = job
        for attempt in range(6):
            have = piece.stat().st_size if piece.exists() else 0
            assert have <= size, piece
            if have == size:
                return
            try:
                first, last = start + have, start + size - 1
                request = urllib.request.Request(url, headers={'Range': f'bytes={first}-{last}'})
                with opener().open(request, timeout=20) as response:
                    assert response.status == 206, response.status
                    assert response.headers['Content-Range'].startswith(f'bytes {first}-{last}/')
                    with piece.open('ab') as handle:
                        while block := response.read(64 * 1024):
                            handle.write(block)
                assert piece.stat().st_size == size, piece
                print(f'Downloaded {piece.name}: {size} bytes', flush=True)
                return
            except (OSError, TimeoutError):
                if attempt == 5:
                    raise
                time.sleep(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, jobs))
    with PACKAGE.open('wb') as handle:
        for prefix, slices in parts:
            if prefix.exists():
                handle.write(prefix.read_bytes())
            for piece in slices:
                handle.write(piece.read_bytes())
    assert PACKAGE.stat().st_size == SIZE
    assert digest(PACKAGE) == SHA256, 'Official release checksum mismatch'
    for prefix, slices in parts:
        prefix.unlink(missing_ok=True)
        for piece in slices:
            piece.unlink()


def main():
    for variable in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(variable, None)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    CACHE.mkdir(parents=True, exist_ok=True)
    download()
    subprocess.run(['dpkg-deb', '--extract', str(PACKAGE), str(PREFIX)], check=True)
    libraries = PREFIX / 'usr/lib/x86_64-linux-gnu'
    os.environ['LD_LIBRARY_PATH'] = str(libraries) + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
    dependencies = ['libsubid4=1:4.13+dfsg1-4ubuntu3.2', 'libtalloc2=2.4.2-1build2',
                    'uidmap=1:4.13+dfsg1-4ubuntu3.2', 'patchelf=0.18.0-1.1build1']
    subprocess.run(['apt-get', '-o', 'Acquire::http::Proxy=DIRECT', '-o', 'Acquire::https::Proxy=DIRECT',
                    'download', *dependencies], cwd=CACHE, check=True)
    for pattern in ['libsubid4_*.deb', 'libtalloc2_*.deb', 'uidmap_*.deb', 'patchelf_*.deb']:
        for package in CACHE.glob(pattern):
            subprocess.run(['dpkg-deb', '--extract', str(package), str(PREFIX)], check=True)
    # The Debian package splits /usr and /etc; relocated Apptainer expects
    # configuration and state below the relocated /usr prefix.
    for name in ['etc', 'var']:
        (PREFIX / name).mkdir(exist_ok=True)
        alias = PREFIX / 'usr' / name
        if not alias.exists():
            alias.symlink_to('../' + name, target_is_directory=True)
    patchelf = str(PREFIX / 'usr/bin/patchelf')
    modified = []
    binaries = [PREFIX / 'usr/bin' / name for name in ['apptainer', 'newuidmap', 'newgidmap']]
    binaries.extend((PREFIX / 'usr/libexec/apptainer/bin').iterdir())
    for binary in binaries:
        probe = subprocess.run([patchelf, '--print-needed', str(binary)], capture_output=True, text=True)
        if probe.returncode or not probe.stdout.strip():
            continue
        origin = '$ORIGIN/' + os.path.relpath(libraries, binary.parent)
        original = digest(binary)
        subprocess.run([patchelf, '--set-rpath', origin, str(binary)], check=True)
        modified.append({'path': str(binary.relative_to(ROOT)), 'original_sha256': original,
                         'installed_sha256': digest(binary), 'runpath': origin})
    # These helpers run as the current account; no setuid executable is needed.
    for name in ['newuidmap', 'newgidmap']:
        (PREFIX / 'usr/bin' / name).chmod(0o755)
    config = PREFIX / 'etc/apptainer/apptainer.conf'
    contents = config.read_text()
    contents = re.sub(r'^#?\s*binary path\s*=.*$', 'binary path = ' + str(PREFIX / 'usr/bin') + ':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', contents, flags=re.M)
    contents = re.sub(r'^mksquashfs procs\s*=.*$', 'mksquashfs procs = 2', contents, flags=re.M)
    contents += '\nmksquashfs mem = 256M\n'
    config.write_text(contents)
    version = subprocess.check_output([str(PREFIX / 'usr/bin/apptainer'), '--version'], text=True).strip()
    report = {'version': version, 'sha256': SHA256, 'package_bytes': SIZE,
              'official_asset_api': ASSET, 'installation_directory': str(PREFIX.relative_to(ROOT)),
              'ubuntu_runtime_dependencies': dependencies, 'relocated_binary_runpaths': modified}
    lock_path = ROOT / 'archive/harbor-local.lock.json'
    lock = json.loads(lock_path.read_text())
    lock['sandbox']['installation_record'] = report
    lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
