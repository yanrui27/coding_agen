#!/usr/bin/env python3
"""Resume verified OCI-layer downloads, then convert the three missing TB2 images locally."""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tb2', ROOT / 'scripts/eval_tb2.py')
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
RECORDS = tb.CACHE / 'records/terminal-bench-2.0'
DOWNLOADS = tb.CACHE / '.oci-downloads'
APPTAINER = ROOT / '.sandbox-tools/apptainer/usr/bin/apptainer'
REGISTRY = 'https://docker.1panel.live'
NAMES = ('qemu-alpine-ssh', 'mteb-leaderboard', 'mteb-retrieve')
GIB = 1024 ** 3
STOP = threading.Event()


def digest_bytes(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


class Repair:
    def __init__(self):
        self.state = json.loads((RECORDS / 'progress.json').read_text())
        self.rows = self.state['images']
        self.lock = threading.RLock()
        self.memory = threading.Condition()
        self.scratch_used = 0
        self.scratch_budget = 48 * GIB
        self.blob_dir = DOWNLOADS / 'blobs/sha256'
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.manifests = {}
        self.descriptors = {}
        self.verified = set()
        self.previous = {name: self.rows[name].copy() for name in NAMES}
        tb.save(RECORDS / 'resumable-repair.json', {
            'started_at': tb.now(), 'tasks': list(NAMES), 'previous': self.previous,
            'transport': 'aria2 HTTP/1.1 with persistent ranges and SHA256 verification',
            'download_cache': str(DOWNLOADS.relative_to(ROOT)), 'registry': REGISTRY,
            'conversion': 'offline OCI layout to SIF', 'scratch_budget_gib': 48,
            'script_sha256': tb.sha(Path(__file__)), 'starts_evaluation': False})

    def log(self, message):
        line = f'{tb.now()} {message}'
        print(line, flush=True)
        with self.lock:
            with (RECORDS / 'resumable-repair.log').open('a') as stream:
                stream.write(line + '\n')

    def update(self, name=None, **fields):
        with self.lock:
            if name:
                row = self.rows[name]
                if fields.get('status') in ('downloading', 'building', 'ready'):
                    row.pop('error', None)
                    row.pop('last_error', None)
                row.update(fields, updated_at=tb.now())
                tb.save(RECORDS / f'{name}.json', row)
            counts = {}
            for row in self.rows.values():
                counts[row['status']] = counts.get(row['status'], 0) + 1
            self.state.update(updated_at=tb.now(), complete=counts.get('ready', 0) == 89,
                stopped=STOP.is_set(), status_counts=counts, workers=2,
                transport='resumable verified OCI layers; local conversion',
                ready_sif_bytes=sum(row.get('sif_bytes', 0) for row in self.rows.values() if row['status'] == 'ready'),
                evaluation_running=subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0)
            tb.save(RECORDS / 'progress.json', self.state)

    def plan(self):
        for name in NAMES:
            metadata = self.rows[name]['metadata']
            sif = tb.CACHE / (metadata['original_image'].replace('/', '_').replace(':', '_') + '.sif')
            if self.rows[name]['status'] == 'ready' and sif.is_file():
                self.log(f'reusing previously verified SIF: {name}')
                continue
            repo = metadata['original_image'].rsplit(':', 1)[0]
            path = RECORDS / f'{name}.oci-manifest.json'
            if not path.exists():
                with httpx.Client(trust_env=False, timeout=30) as client:
                    response = client.get(f'{REGISTRY}/v2/{repo}/manifests/{metadata["platform_digest"]}',
                        headers={'Accept': tb.ACCEPT})
                    response.raise_for_status()
                    assert digest_bytes(response.content) == metadata['platform_digest']
                    path.write_bytes(response.content)
            data = path.read_bytes()
            assert digest_bytes(data) == metadata['platform_digest'], name
            manifest = json.loads(data)
            assert manifest['config']['digest'] == metadata['config_digest'], name
            self.manifests[name] = (manifest, data)
            for desc in [manifest['config'], *manifest['layers']]:
                descriptor = dict(desc, repo=repo)
                self.descriptors.setdefault(desc['digest'], descriptor)
            self.update(name, status='downloading', repair_started_at=tb.now(),
                        repair_transport='persistent aria2 chunks, pinned SHA256',
                        download_layers_total=len(manifest['layers']))
        input_path = DOWNLOADS / 'downloads.txt'
        blocks = []
        for digest, desc in self.descriptors.items():
            path = self.blob_dir / digest.removeprefix('sha256:')
            if path.is_file() and path.stat().st_size == desc['size'] and not Path(str(path) + '.aria2').exists():
                if 'sha256:' + tb.sha(path) == digest:
                    self.verified.add(digest)
                    continue
            blocks.append(f'{REGISTRY}/v2/{desc["repo"]}/blobs/{digest}\n'
                          f' dir={self.blob_dir}\n out={path.name}\n checksum=sha-256={path.name}\n')
        input_path.write_text('\n'.join(blocks))
        return input_path, bool(blocks)

    def verify_completed_downloads(self, notice_log, downloader_finished):
        completed = set(re.findall(r'Download complete: .*?/([0-9a-f]{64})(?:\s|$)', notice_log))
        for digest, desc in self.descriptors.items():
            if digest in self.verified:
                continue
            short = digest.removeprefix('sha256:')
            if short not in completed and not downloader_finished:
                continue
            path = self.blob_dir / short
            if not path.is_file() or Path(str(path) + '.aria2').exists() or path.stat().st_size != desc['size']:
                continue
            assert 'sha256:' + tb.sha(path) == digest, f'OCI blob checksum mismatch: {short}'
            self.verified.add(digest)
            self.log(f'verified layer {short[:16]}, {desc["size"] / GIB:.3f} GiB')

    def can_build(self, name):
        manifest, _ = self.manifests[name]
        return all(desc['digest'] in self.verified for desc in [manifest['config'], *manifest['layers']])

    def layout(self, name):
        manifest, data = self.manifests[name]
        path = DOWNLOADS / 'layouts' / name
        blobs = path / 'blobs/sha256'
        blobs.mkdir(parents=True, exist_ok=True)
        manifest_digest = digest_bytes(data)
        (blobs / manifest_digest.removeprefix('sha256:')).write_bytes(data)
        for desc in [manifest['config'], *manifest['layers']]:
            short = desc['digest'].removeprefix('sha256:')
            dest = blobs / short
            if not dest.exists():
                os.link(self.blob_dir / short, dest)
        config = json.loads((self.blob_dir / manifest['config']['digest'].removeprefix('sha256:')).read_text())
        assert config['architecture'] == 'amd64' and config['os'] == 'linux', name
        (RECORDS / f'{name}.oci-config.json').write_text(json.dumps(config, indent=2) + '\n')
        tb.save(path / 'oci-layout', {'imageLayoutVersion': '1.0.0'})
        tb.save(path / 'index.json', {'schemaVersion': 2, 'manifests': [
            {'mediaType': manifest.get('mediaType', 'application/vnd.docker.distribution.manifest.v2+json'),
             'digest': manifest_digest, 'size': len(data)}]})
        return path

    def build(self, name):
        amount = min(self.scratch_budget, self.rows[name]['metadata']['compressed_layers_bytes'] * 4 + 2 * GIB)
        with self.memory:
            while self.scratch_used + amount > self.scratch_budget and not STOP.is_set():
                self.memory.wait(timeout=1)
            if STOP.is_set():
                self.update(name, status='queued', error='conversion stopped before scratch allocation')
                return
            self.scratch_used += amount
        scratch = None
        partial = None
        try:
            self.update(name, status='building', layers_verified=True, conversion_source='local OCI layout')
            self.log(f'converting {name} locally; no registry transfer during build')
            layout = self.layout(name)
            image = self.rows[name]['metadata']['original_image']
            sif = tb.CACHE / (image.replace('/', '_').replace(':', '_') + '.sif')
            assert not sif.exists(), f'Never overwrite a previously prepared SIF: {sif}'
            partial = sif.with_suffix('.sif.repairing')
            scratch = Path(tempfile.mkdtemp(prefix=f'tb2-repair-{name}-', dir='/dev/shm'))
            env = tb.direct_env()
            env.update(APPTAINER_TMPDIR=str(scratch), SINGULARITY_TMPDIR=str(scratch), TMPDIR=str(scratch))
            command = [str(APPTAINER), 'build', '--force', '--mksquashfs-args',
                '-comp zstd -Xcompression-level 3 -processors 4 -mem 512M', str(partial), 'oci://' + str(layout)]
            with (RECORDS / f'{name}.repair-build.log').open('a') as output:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 7200
                while process.poll() is None:
                    if STOP.is_set() or time.monotonic() > deadline:
                        stop_process(process)
                        raise InterruptedError('local conversion stopped or exceeded two hours')
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f'Local conversion failed, exit={process.returncode}; see {name}.repair-build.log')
            subprocess.run([str(APPTAINER), 'sif', 'list', str(partial)], env=env,
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            checked = subprocess.run([str(APPTAINER), 'exec', '--no-mount', 'home,tmp,bind-paths,cwd',
                '--containall', '--fakeroot', '--pwd', '/', str(partial), '/bin/sh', '-c',
                'cat /etc/os-release'], env=env, capture_output=True, check=True, timeout=90)
            (RECORDS / f'{name}.os-release').write_bytes(checked.stdout)
            sha256 = tb.sha(partial)
            partial.replace(sif)
            self.update(name, status='ready', reused_cache=False, sif_path=str(sif.relative_to(ROOT)),
                        sif_bytes=sif.stat().st_size, sif_sha256=sha256, finished_at=tb.now(),
                        verification={'pinned_oci_manifest': True, 'all_blob_checksums': True,
                                      'sif_structure': True, 'sandbox_execution': True},
                        sif_compression='zstd, level 3')
            self.log(f'ready {name}, {sif.stat().st_size / GIB:.2f} GiB; sandbox execution verified')
        except Exception as error:
            self.update(name, status='queued' if STOP.is_set() else 'failed', error=str(error))
            self.log(f'{name}: {error}')
        finally:
            if partial:
                partial.unlink(missing_ok=True)
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
            with self.memory:
                self.scratch_used -= amount
                self.memory.notify_all()

    def run(self):
        input_path, pending = self.plan()
        log_path = RECORDS / 'resumable-download.log'
        process = None
        submitted = {}
        env = tb.direct_env()
        command = ['aria2c', '--no-conf=true', '--all-proxy=', '--http-proxy=', '--https-proxy=',
            '--disable-ipv6=true', '--continue=true', '--file-allocation=none', '--auto-file-renaming=false',
            '--check-integrity=true', '--check-certificate=true', '--http-accept-gzip=false',
            '--header=Accept-Encoding: identity', '--header=Cache-Control: no-cache',
            '--max-concurrent-downloads=3', '--split=4', '--max-connection-per-server=4', '--min-split-size=16M',
            '--connect-timeout=15', '--timeout=60', '--retry-wait=5', '--max-tries=100',
            '--auto-save-interval=1', '--summary-interval=5', '--console-log-level=notice', '--enable-color=false',
            '--input-file=' + str(input_path)]
        self.log(f'downloading {len(self.descriptors)} unique config/layer blobs, '
                 f'{sum(desc["size"] for desc in self.descriptors.values()) / GIB:.2f} GiB; partial files persist')
        try:
            with log_path.open('a') as output, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                if pending:
                    process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    if STOP.is_set():
                        if process:
                            stop_process(process)
                        break
                    finished = process is None or process.poll() is not None
                    self.verify_completed_downloads(log_path.read_text(errors='replace'), finished)
                    for name in self.manifests:
                        if name not in submitted and self.can_build(name):
                            submitted[name] = pool.submit(self.build, name)
                    if finished:
                        break
                    self.update()
                    time.sleep(5)
                for future in submitted.values():
                    future.result()
                for name in self.manifests:
                    if name not in submitted:
                        self.update(name, status='queued' if STOP.is_set() else 'failed',
                            error='Download interrupted; verified layers and aria2 control files retained for resume')
        finally:
            if process:
                stop_process(process)
        self.update()
        success = all(self.rows[name]['status'] == 'ready' for name in NAMES)
        receipt = json.loads((RECORDS / 'resumable-repair.json').read_text())
        receipt.update(finished_at=tb.now(), complete=success,
                       results={name: self.rows[name] for name in NAMES})
        tb.save(RECORDS / 'resumable-repair.json', receipt)
        if success:
            assert DOWNLOADS.resolve() == tb.CACHE / '.oci-downloads'
            shutil.rmtree(DOWNLOADS)
            self.log('All three repaired images verified; temporary OCI layer copies removed. All existing SIFs retained.')
        return 0 if success else 1


def main():
    os.chdir(ROOT)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    with (RECORDS / 'builder.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Repair().run()


if __name__ == '__main__':
    raise SystemExit(main())
