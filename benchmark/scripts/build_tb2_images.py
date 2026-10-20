#!/usr/bin/env python3
"""Prepare all fixed TB2.0 SIF images in parallel; never run an agent or evaluator."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib

import eval_tb2 as tb

RECORDS = tb.CACHE / 'records/terminal-bench-2.0'
GIB = 1024 ** 3
STOP = threading.Event()


class Builder:
    def __init__(self, workers, scratch_gib, registry, stall_timeout):
        self.workers = workers
        self.budget = scratch_gib * GIB
        self.registry = registry.rstrip('/')
        self.stall_timeout = stall_timeout
        self.used = 0
        self.lock = threading.RLock()
        self.memory = threading.Condition()
        self.ids = (tb.ROOT / 'archive/manifests/terminal-bench-2.0-task-ids.txt').read_text().splitlines()
        assert len(self.ids) == len(set(self.ids)) == 89
        self.specs = {n: tomllib.loads((tb.DATA / n / 'task.toml').read_text()) for n in self.ids}
        assert len({s['environment']['docker_image'] for s in self.specs.values()}) == 89
        RECORDS.mkdir(parents=True, exist_ok=True)
        self.rows = {n: {'status': 'queued', 'image': self.specs[n]['environment']['docker_image']} for n in self.ids}
        run_path = RECORDS / 'run.json'
        history = []
        if run_path.exists():
            previous = json.loads(run_path.read_text())
            history = previous.pop('previous_runs', []) + [previous]
        tb.save(run_path, {'started_at': tb.now(), 'dataset_commit': '69671fbaac6d67a7ef0dfec016cc38a64ef7a77c',
            'task_ids': self.ids, 'workers': workers, 'scratch_budget_gib': scratch_gib,
            'scratch_directory': '/dev/shm', 'cache_eviction': False, 'starts_evaluation': False,
            'script_sha256': tb.sha(Path(__file__)), 'registry': self.registry,
            'fallback_registry': tb.MIRROR, 'http2_disabled': True,
            'stall_timeout_sec': stall_timeout, 'previous_runs': history})
        self.update()

    def log(self, message):
        line = f'{tb.now()} {message}'
        print(line, flush=True)
        with self.lock:
            with (RECORDS / 'builder.log').open('a') as f:
                f.write(line + '\n')

    def update(self, name=None, **fields):
        with self.lock:
            if name:
                self.rows[name].update(fields, updated_at=tb.now())
                tb.save(RECORDS / f'{name}.json', self.rows[name])
            counts = {s: sum(r['status'] == s for r in self.rows.values()) for s in {r['status'] for r in self.rows.values()}}
            tb.save(RECORDS / 'progress.json', {'updated_at': tb.now(), 'tasks_total': 89,
                'complete': counts.get('ready', 0) == 89, 'stopped': STOP.is_set(),
                'status_counts': counts, 'workers': self.workers,
                'ready_sif_bytes': sum(r.get('sif_bytes', 0) for r in self.rows.values() if r['status'] == 'ready'),
                'evaluation_running': False, 'images': self.rows})

    def prior_receipt(self, name, image):
        candidates = [RECORDS / f'{name}.json',
            tb.RESULTS / 'seed42' / name / 'image.json',
            tb.RESULTS / 'seed42-pilot' / name / 'image.json']
        if name == 'fix-git':
            candidates.append(tb.RESULTS / 'image-and-protocol.json')
        for path in candidates:
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            metadata = dict(data.get('metadata', data))
            if metadata.get('original_image') != image:
                continue
            if 'oci_manifest_digest' in metadata:
                metadata['platform_digest'] = metadata['oci_manifest_digest']
                metadata['index_digest'] = metadata['oci_manifest_digest']
                metadata['platform'] = 'linux/amd64'
                metadata['registry'] = metadata['download_registry']
            if metadata.get('platform_digest'):
                metadata['source_receipt'] = str(path.relative_to(tb.ROOT))
                expected = data.get('sif_sha256') or metadata.get('sif_sha256')
                return metadata, expected
        return None, None

    def reserve(self, amount):
        with self.memory:
            while self.used + amount > self.budget and not STOP.is_set():
                self.memory.wait(timeout=1)
            if STOP.is_set():
                raise InterruptedError('image preparation stopped')
            self.used += amount

    def release(self, amount):
        with self.memory:
            self.used -= amount
            self.memory.notify_all()

    @staticmethod
    def validate(path):
        subprocess.run([str(tb.ROOT / '.sandbox-tools/apptainer/usr/bin/apptainer'), 'sif', 'list', str(path)],
                       env=tb.direct_env(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        return tb.sha(path)

    def prepare(self, name):
        if STOP.is_set():
            return
        image = self.specs[name]['environment']['docker_image']
        sif = tb.CACHE / (image.replace('/', '_').replace(':', '_') + '.sif')
        metadata, expected = self.prior_receipt(name, image)
        try:
            self.update(name, status='checking_cache' if sif.exists() else 'resolving', started_at=tb.now())
            if metadata is None:
                metadata = tb.image_metadata(image)
            self.update(name, metadata=metadata)
            if sif.exists():
                digest = self.validate(sif)
                if expected and digest != expected:
                    raise ValueError(f'existing SIF checksum differs from its recorded receipt: {sif.name}')
                self.update(name, status='ready', reused_cache=True, sif_path=str(sif.relative_to(tb.ROOT)),
                            sif_bytes=sif.stat().st_size, sif_sha256=digest, finished_at=tb.now())
                self.log(f'cached {name}')
                return
            # Scratch includes OCI layers and unpacked rootfs. Limit simultaneous RAM use
            # independently of worker count; the service adds a hard memory boundary.
            reserve = min(self.budget, max(2 * GIB, metadata['compressed_layers_bytes'] * 4 + 2 * GIB))
            self.update(name, status='waiting_for_memory', scratch_reservation_bytes=reserve)
            self.reserve(reserve)
            try:
                for attempt in range(1, 5):
                    if STOP.is_set():
                        raise InterruptedError('image preparation stopped')
                    scratch = Path(tempfile.mkdtemp(prefix=f'tb2-images-{name}-', dir='/dev/shm'))
                    partial = sif.with_suffix('.sif.building')
                    env = tb.direct_env()
                    env.update(APPTAINER_TMPDIR=str(scratch), SINGULARITY_TMPDIR=str(scratch), TMPDIR=str(scratch))
                    debug = [part for part in env.get('GODEBUG', '').split(',') if part and not part.startswith('http2client=')]
                    env['GODEBUG'] = ','.join(debug + ['http2client=0'])
                    registry = self.registry if attempt % 2 else tb.MIRROR
                    reference = f"docker://{registry.removeprefix('https://')}/{image.rsplit(':', 1)[0]}@{metadata['platform_digest']}"
                    self.update(name, status='building', attempt=attempt, pull_reference=reference)
                    self.log(f'building {name}, attempt {attempt}, registry {registry}, OCI {metadata["compressed_layers_bytes"] / GIB:.2f} GiB')
                    try:
                        with (RECORDS / f'{name}.log').open('a') as output:
                            process = subprocess.Popen([str(tb.ROOT / '.sandbox-tools/apptainer/usr/bin/apptainer'),
                                'pull', '--force', str(partial), reference], env=env,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                            deadline = time.monotonic() + 3600
                            activity_at = time.monotonic()
                            activity = None
                            while process.poll() is None:
                                try:
                                    io = dict(line.split(':', 1) for line in Path(f'/proc/{process.pid}/io').read_text().splitlines())
                                    marker = (int(io['wchar']), partial.stat().st_size if partial.exists() else 0)
                                except (OSError, KeyError):
                                    marker = activity
                                if marker != activity:
                                    activity = marker
                                    activity_at = time.monotonic()
                                stalled = time.monotonic() - activity_at > self.stall_timeout
                                if STOP.is_set() or time.monotonic() > deadline or stalled:
                                    os.killpg(process.pid, signal.SIGKILL)
                                    process.wait(timeout=30)
                                    if STOP.is_set():
                                        raise InterruptedError('image preparation stopped')
                                    if stalled:
                                        raise TimeoutError(f'no download/output activity for {self.stall_timeout} seconds')
                                    raise TimeoutError('image pull/conversion exceeded 3600 seconds')
                                time.sleep(1)
                            if process.returncode:
                                raise RuntimeError(f'Apptainer exited {process.returncode}; see {name}.log')
                        digest = self.validate(partial)
                        partial.replace(sif)
                        self.update(name, status='ready', reused_cache=False, sif_path=str(sif.relative_to(tb.ROOT)),
                                    sif_bytes=sif.stat().st_size, sif_sha256=digest, finished_at=tb.now())
                        self.log(f'ready {name}, {sif.stat().st_size / GIB:.2f} GiB')
                        return
                    except InterruptedError:
                        raise
                    except Exception as error:
                        self.log(f'{name} attempt {attempt} failed: {error}')
                        self.update(name, last_error=str(error))
                        if attempt == 4:
                            raise
                    finally:
                        partial.unlink(missing_ok=True)
                        shutil.rmtree(scratch, ignore_errors=True)
            finally:
                self.release(reserve)
        except InterruptedError:
            self.update(name, status='queued', stopped_at=tb.now())
        except Exception as error:
            self.update(name, status='failed', error=str(error), finished_at=tb.now())
            self.log(f'failed {name}: {error}')

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self.prepare, name) for name in self.ids]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        self.update()
        self.log('image preparation stopped' if STOP.is_set() else 'image preparation finished; evaluation remains stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--scratch-gib', type=int, default=80)
    parser.add_argument('--registry', default='https://docker.1panel.live')
    parser.add_argument('--stall-timeout', type=int, default=300)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    os.chdir(tb.ROOT)
    if args.status:
        path = RECORDS / 'progress.json'
        data = json.loads(path.read_text()) if path.exists() else {'status': 'not started'}
        print(json.dumps({k: v for k, v in data.items() if k != 'images'}, ensure_ascii=False, indent=2))
        return
    if not 1 <= args.workers <= 16 or not 2 <= args.scratch_gib <= 128:
        parser.error('workers must be 1..16 and scratch-gib must be 2..128')
    if args.registry not in ('https://docker.1panel.live', tb.MIRROR) or args.stall_timeout < 60:
        parser.error('use a supported direct registry and stall-timeout >= 60')
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0:
        parser.error('stop TB2.0 evaluation before starting image-only preparation')
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    RECORDS.mkdir(parents=True, exist_ok=True)
    with (RECORDS / 'builder.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Builder(args.workers, args.scratch_gib, args.registry, args.stall_timeout).run()


if __name__ == '__main__':
    main()
