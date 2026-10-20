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
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tb2', ROOT / 'scripts/eval_tb2.py')
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
RECORDS = tb.CACHE / 'records/terminal-bench-2.0'
DOWNLOADS = tb.CACHE / '.oci-downloads'
APPTAINER = ROOT / '.sandbox-tools/apptainer/usr/bin/apptainer'
REGISTRY = 'https://docker.1panel.live'
RANGE_REGISTRY = 'https://docker.amingg.com'
NAMES = ('qemu-alpine-ssh', 'mteb-leaderboard', 'mteb-retrieve')
GIB = 1024 ** 3
STOP = threading.Event()


def digest_bytes(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def aria_prefix(path, expected_size):
    """Read only the consecutive completed pieces from an aria2 v1 control file."""
    control = Path(str(path) + '.aria2')
    if not control.exists():
        return 0
    data = control.read_bytes()
    if len(data) < 34 or data[:2] != b'\0\1':
        return 0
    offset = 10 + struct.unpack_from('>I', data, 6)[0]
    if len(data) < offset + 24:
        return 0
    piece, length = struct.unpack_from('>IQ', data, offset)
    count = struct.unpack_from('>I', data, offset + 20)[0]
    if length != expected_size or not piece or count != (length + piece * 8 - 1) // (piece * 8):
        return 0
    field = data[offset + 24:offset + 24 + count]
    if len(field) != count:
        return 0
    done = 0
    for byte in field:
        for bit in range(7, -1, -1):
            if not byte & (1 << bit):
                return min(done * piece, length)
            done += 1
    return length


def aria_downloaded(path, expected_size):
    """Count persisted completed pieces, including pieces after the first gap."""
    control = Path(str(path) + '.aria2')
    if not control.exists():
        return min(path.stat().st_size, expected_size) if path.exists() else 0
    try:
        data = control.read_bytes()
        if len(data) < 34 or data[:2] != b'\0\1':
            return 0
        offset = 10 + struct.unpack_from('>I', data, 6)[0]
        piece, length = struct.unpack_from('>IQ', data, offset)
        count = struct.unpack_from('>I', data, offset + 20)[0]
        if length != expected_size or not piece or count != (length + piece * 8 - 1) // (piece * 8):
            return 0
        field = data[offset + 24:offset + 24 + count]
        if len(field) != count:
            return 0
        completed = sum(byte.bit_count() for byte in field) * piece
        last = (length - 1) // piece
        if field[last // 8] & (1 << (7 - last % 8)):
            completed -= piece - (length - last * piece)
        return completed
    except (OSError, struct.error):
        # aria2 may be replacing its checkpoint while this status read runs.
        return 0


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
        self.download_progress = {}
        self.previous = {name: self.rows[name].copy() for name in NAMES}
        prior = RECORDS / 'resumable-repair.json'
        history = []
        if prior.exists():
            previous = json.loads(prior.read_text())
            history = previous.pop('previous_runs', []) + [previous]
        tb.save(RECORDS / 'resumable-repair.json', {
            'started_at': tb.now(), 'tasks': list(NAMES), 'previous': self.previous,
            'transport': 'aria2 HTTP/1.1 verified Range source; persistent pieces; pinned SHA256',
            'download_cache': str(DOWNLOADS.relative_to(ROOT)), 'registry': REGISTRY,
            'range_registry': RANGE_REGISTRY, 'connections_per_blob': 4, 'previous_runs': history,
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
                transport='persistent verified OCI layers; local conversion',
                ready_sif_bytes=sum(row.get('sif_bytes', 0) for row in self.rows.values() if row['status'] == 'ready'),
                evaluation_running=subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0)
            tb.save(RECORDS / 'progress.json', self.state)
            tb.save(RECORDS / 'resumable-download-progress.json', self.download_progress)

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
                        repair_transport='persistent HTTP range pieces, pinned SHA256',
                        download_layers_total=len(manifest['layers']))
        for digest, desc in self.descriptors.items():
            path = self.blob_dir / digest.removeprefix('sha256:')
            if path.is_file() and path.stat().st_size == desc['size'] and not Path(str(path) + '.aria2').exists():
                if 'sha256:' + tb.sha(path) == digest:
                    self.verified.add(digest)
                    continue

    def download_blob(self, digest):
        desc = self.descriptors[digest]
        target = self.blob_dir / digest.removeprefix('sha256:')
        partial = target.with_suffix('.stream')
        if partial.exists():
            # A previous full-body download has a longer consecutive prefix than
            # the legacy aria2 pieces. Replace only that redundant temporary file.
            assert aria_prefix(target, desc['size']) <= partial.stat().st_size <= desc['size']
            partial.replace(target)
            Path(str(target) + '.aria2').unlink(missing_ok=True)
            self.log(f'reusing {target.stat().st_size / 1024**2:.1f} MiB prefix for {digest[7:23]}')
        log_path = RECORDS / f'{desc["repo"].split("/")[-1]}.range-download.log'
        for attempt in range(50):
            if STOP.is_set():
                raise InterruptedError('download stopped; aria2 pieces retained')
            process = None
            try:
                url = f'{RANGE_REGISTRY}/v2/{desc["repo"]}/blobs/{digest}'
                with httpx.Client(trust_env=False, timeout=httpx.Timeout(30, connect=10)) as client:
                    response = client.get(RANGE_REGISTRY + '/token', params={
                        'service': 'registry.docker.io', 'scope': f'repository:{desc["repo"]}:pull'})
                    response.raise_for_status()
                    body = response.json()
                    # This is an anonymous token for public-image pulls; no local
                    # Docker/Hugging Face credentials are sent to the mirror.
                    authorization = 'Authorization: Bearer ' + (body.get('token') or body['access_token'])
                    last = desc['size'] - 1
                    with client.stream('GET', url, headers={
                            'Authorization': authorization.split(': ', 1)[1],
                            'Range': f'bytes={last}-{last}', 'Accept-Encoding': 'identity'}) as probe:
                        probe.raise_for_status()
                        assert probe.status_code == 206, 'Mirror must honor byte ranges'
                        assert probe.headers.get('content-range') == f'bytes {last}-{last}/{desc["size"]}'
                        assert len(probe.read()) == 1
                command = ['aria2c', '--no-conf=true', '--all-proxy=', '--http-proxy=', '--https-proxy=',
                    '--disable-ipv6=true', '--continue=true', '--file-allocation=none',
                    '--auto-file-renaming=false', '--check-integrity=true', '--check-certificate=true',
                    '--http-accept-gzip=false', '--header=Accept-Encoding: identity', '--header=' + authorization,
                    '--checksum=sha-256=' + digest.removeprefix('sha256:'),
                    '--split=4', '--max-connection-per-server=4', '--min-split-size=16M',
                    '--connect-timeout=15', '--timeout=60', '--retry-wait=3', '--max-tries=3',
                    '--auto-save-interval=1', '--summary-interval=10', '--console-log-level=notice',
                    '--enable-color=false', '--dir=' + str(target.parent), '--out=' + target.name, url]
                self.log(f'ranged download {digest[7:23]} from {RANGE_REGISTRY}; 4 connections; attempt {attempt + 1}')
                with log_path.open('a') as output:
                    process = subprocess.Popen(command, cwd=ROOT, env=tb.direct_env(),
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    checkpoint = time.monotonic()
                    while process.poll() is None:
                        if STOP.is_set():
                            raise InterruptedError('download stopped; aria2 pieces retained')
                        received = aria_downloaded(target, desc['size'])
                        with self.lock:
                            self.download_progress[digest] = {'received_bytes': received, 'total_bytes': desc['size'],
                                'registry': RANGE_REGISTRY, 'attempt': attempt + 1, 'connections': 4, 'verified': False}
                        if time.monotonic() - checkpoint >= 30:
                            self.log(f'progress {digest[7:23]}: {received / GIB:.3f} / {desc["size"] / GIB:.3f} GiB')
                            checkpoint = time.monotonic()
                        STOP.wait(5)
                    if process.returncode:
                        raise RuntimeError(f'aria2 exit {process.returncode}; completed pieces retained; see {log_path.name}')
                assert target.stat().st_size == desc['size']
                if 'sha256:' + tb.sha(target) != digest:
                    target.unlink()
                    Path(str(target) + '.aria2').unlink(missing_ok=True)
                    raise ValueError('OCI layer checksum mismatch; reject and re-download this layer')
                Path(str(target) + '.aria2').unlink(missing_ok=True)
                with self.lock:
                    self.verified.add(digest)
                    self.download_progress[digest] = {'received_bytes': desc['size'], 'total_bytes': desc['size'], 'verified': True}
                self.log(f'verified layer {digest[7:23]}, {desc["size"] / GIB:.3f} GiB')
                return
            except InterruptedError:
                raise
            except Exception as error:
                self.log(f'layer {digest[7:23]} attempt {attempt + 1}: {type(error).__name__}: {error}')
                if attempt == 49:
                    raise
                STOP.wait(2)
            finally:
                if process:
                    stop_process(process)

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
        self.plan()
        submitted = {}
        self.log(f'downloading {len(self.descriptors)} unique config/layer blobs, '
                 f'{sum(desc["size"] for desc in self.descriptors.values()) / GIB:.2f} GiB; partial files persist')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as downloads, \
             concurrent.futures.ThreadPoolExecutor(max_workers=2) as builds:
            pending = {downloads.submit(self.download_blob, digest): digest
                       for digest, desc in sorted(self.descriptors.items(), key=lambda item: item[1]['size'])
                       if digest not in self.verified}
            while True:
                for future in list(pending):
                    if future.done():
                        digest = pending.pop(future)
                        try:
                            future.result()
                        except Exception as error:
                            self.log(f'download incomplete {digest[7:23]}: {error}')
                for name in self.manifests:
                    if name not in submitted and self.can_build(name):
                        submitted[name] = builds.submit(self.build, name)
                if not pending:
                    break
                self.update()
                if STOP.is_set():
                    time.sleep(0.25)
                else:
                    STOP.wait(5)
            for future in submitted.values():
                future.result()
            for name in self.manifests:
                if name not in submitted:
                    self.update(name, status='queued' if STOP.is_set() else 'failed',
                                error='Download incomplete; saved prefixes retained for resume')
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
