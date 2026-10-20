#!/usr/bin/env python3
"""Run the fixed TB2.0 task queue with rolling actual trials and bounded prefetch.

This is the requested 4 CPU / 32 GiB, seed-42 pilot, not the reference protocol protocol.
Official task files, tests and Terminus-2 are left unchanged.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import tarfile
import threading
import time
import tomllib
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/terminal-bench-2.0'
RESULTS = ROOT / 'results/Qwen3.6-35B-A3B/terminal-bench-2.0'
CACHE = ROOT / '.cache/harbor-images'
ASSETS = ROOT / '.cache/harbor-assets'
MIRROR = 'https://docker.1ms.run'
ACCEPT = ', '.join([
    'application/vnd.oci.image.index.v1+json',
    'application/vnd.docker.distribution.manifest.list.v2+json',
    'application/vnd.oci.image.manifest.v1+json',
    'application/vnd.docker.distribution.manifest.v2+json',
])
TERMINAL = {'passed', 'failed', 'timeout', 'infrastructure_error'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def direct_env():
    env = os.environ.copy()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env.update(NO_PROXY='*', no_proxy='*', GOMAXPROCS='2',
               PIP_PROXY='', PIP_CONFIG_FILE='/dev/null')
    return env


def image_metadata(image):
    """Resolve linux/amd64 and record the exact manifest before pulling."""
    repo, tag = image.rsplit(':', 1)
    with httpx.Client(trust_env=False, timeout=40, follow_redirects=True) as c:
        url = f'{MIRROR}/v2/{repo}/manifests/{tag}'
        headers = {'Accept': ACCEPT}
        response = c.get(url, headers=headers)
        if response.status_code == 401:
            challenge = response.headers['www-authenticate']
            fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
            query = urlencode({'service': fields.get('service', 'docker.1ms.run'),
                               'scope': f'repository:{repo}:pull'})
            token_response = c.get(fields['realm'] + '?' + query)
            token_response.raise_for_status()
            token_data = token_response.json()
            headers['Authorization'] = 'Bearer ' + (token_data.get('token') or token_data['access_token'])
            response = c.get(url, headers=headers)
        response.raise_for_status()
        index_digest = 'sha256:' + hashlib.sha256(response.content).hexdigest()
        manifest = response.json()
        if 'manifests' in manifest:
            selected = next(x for x in manifest['manifests']
                            if x.get('platform', {}).get('os') == 'linux'
                            and x.get('platform', {}).get('architecture') == 'amd64')
            response = c.get(f"{MIRROR}/v2/{repo}/manifests/{selected['digest']}", headers=headers)
            response.raise_for_status()
            assert 'sha256:' + hashlib.sha256(response.content).hexdigest() == selected['digest']
            manifest = response.json()
        digest = 'sha256:' + hashlib.sha256(response.content).hexdigest()
        reported = response.headers.get('docker-content-digest')
        if reported:
            assert reported == digest, 'registry manifest checksum mismatch'
        return {'original_image': image, 'registry': MIRROR,
                'index_digest': index_digest, 'platform_digest': digest,
                'platform': 'linux/amd64',
                'compressed_layers_bytes': sum(x['size'] for x in manifest['layers']),
                'config_digest': manifest['config']['digest'], 'recorded_at': now()}


class Runner:
    def __init__(self, args):
        self.args = args
        self.base = RESULTS / 'seed42-pilot'
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.protected = set()
        self.control_path = self.base / 'control.json'
        if args.concurrency is not None or not self.control_path.exists():
            save(self.control_path, {'concurrency': args.concurrency or 8, 'updated_at': now()})
        self.task_ids = (ROOT / 'archive/manifests/terminal-bench-2.0-task-ids.txt').read_text().splitlines()
        assert len(self.task_ids) == len(set(self.task_ids)) == 89
        self.tasks = {name: tomllib.loads((DATA / name / 'task.toml').read_text()) for name in self.task_ids}
        self.template = json.loads((RESULTS / 'attempt4-config.json').read_text())
        # Keep the original smoke configuration intact; new trials use the requested 80 Ki budget.
        self.template['agents'][0]['kwargs']['model_info']['max_output_tokens'] = 81920
        self.template['agents'][0]['kwargs']['llm_call_kwargs']['max_tokens'] = 81920
        self.fingerprint = {
            'dataset_commit': '69671fbaac6d67a7ef0dfec016cc38a64ef7a77c',
            'task_ids': self.task_ids, 'seed': 42,
            'agent': self.template['agents'][0],
            'resources': {'cpus_per_trial': 4, 'memory_mb_per_trial': 32768, 'concurrent_trials': self.concurrency()},
            'initial_tasks': ['adaptive-rejection-sampler', 'cancel-async-tasks',
                              'log-summary-date-ranges', 'openssl-selfsigned-cert'],
            'verifier_timeout_sec': 300, 'prefetch_tasks': 4,
            'image_cache_limit_gib': args.image_cache_gib,
            'writable_tmpfs_max_mib': 6144,
            'scope': 'full task queue, single-seed pilot with reduced resources and budgets; not reference-protocol-compliant',
        }
        previous = self.base / 'run.json'
        scripts = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'run_tb2.sh', ROOT / 'scripts/harbor_env.sh']}
        if previous.exists():
            record = json.loads(previous.read_text())
            old_protocol = json.loads(json.dumps(record['protocol']))
            assert old_protocol['resources']['memory_mb_per_trial'] in (8192, 32768)
            assert old_protocol['agent']['kwargs']['model_info']['max_output_tokens'] in (8192, 81920)
            assert old_protocol['agent']['kwargs']['llm_call_kwargs']['max_tokens'] in (8192, 81920)
            old_protocol['resources']['concurrent_trials'] = self.concurrency()
            old_protocol['resources']['memory_mb_per_trial'] = 32768
            old_protocol['agent']['kwargs']['model_info']['max_output_tokens'] = 81920
            old_protocol['agent']['kwargs']['llm_call_kwargs']['max_tokens'] = 81920
            assert old_protocol == self.fingerprint, 'changed run protocol; use a separate result directory'
            if record['protocol'] != self.fingerprint:
                record.setdefault('protocol_revisions', []).append({
                    'changed_at': now(), 'previous_protocol': record['protocol'],
                    'reason': 'User requested 8 rolling trials, 4 CPU / 32 GiB per task and an 81920-token output budget. Earlier attempt configurations and results remain unchanged; this pilot includes explicitly recorded budget revisions.'})
                record['protocol'] = self.fingerprint
            record.setdefault('launches', []).append({'started_at': now(), 'scripts': scripts})
            save(previous, record)
        else:
            save(previous, {'created_at': now(), 'protocol': self.fingerprint,
                            'scripts': scripts})
        progress = self.base / 'progress.json'
        self.rows = json.loads(progress.read_text())['tasks'] if progress.exists() else {name: {'status': 'queued'} for name in self.task_ids}
        self.max_active_agent_trials = json.loads(progress.read_text()).get('max_active_agent_trials', 0) if progress.exists() else 0
        # A coordinator restart must not replay trials that finished in their independent scopes.
        for name, row in self.rows.items():
            if row['status'] == 'running':
                native_paths = list((self.base / name / f"attempt{row['attempt']}").glob('*/result.json'))
                if native_paths and json.loads(native_paths[0].read_text()).get('finished_at'):
                    runtime_path = self.base / name / 'runtime.json'
                    runtime = json.loads(runtime_path.read_text())
                    row.update(self.outcome(native_paths[0], runtime.get('expects_ctrf', False)), finished_at=now())
                else:
                    scope = row.get('resource_scope')
                    if scope:
                        shown = subprocess.run(['systemctl', 'show', scope, '-p', 'Result', '-p', 'MemoryPeak',
                                                '-p', 'MemoryMax', '-p', 'ControlGroup'], capture_output=True, text=True)
                        properties = dict(line.split('=', 1) for line in shown.stdout.splitlines() if '=' in line)
                        if properties.get('Result') == 'oom-kill':
                            save(self.base / name / f"attempt{row['attempt']}-scope-result.json", properties)
                            row.update(status='infrastructure_error', finished_at=now(), reward=None,
                                       error='systemd scope reported oom-kill; no valid verifier result',
                                       scope_result=properties)
        baseline = RESULTS / 'fix-git-seed42-attempt4/fix-git__hiMSCd2/result.json'
        native = json.loads(baseline.read_text())
        assert native['verifier_result']['rewards']['reward'] == 1 and native['exception_info'] is None
        self.rows['fix-git'] = {'status': 'passed', 'reward': 1.0, 'inherited_smoke_trial': True,
                                'result_path': str(baseline.relative_to(ROOT)),
                                'note': 'Earlier seed42 smoke trial; its original runtime settings remain recorded separately.'}
        control = json.loads(self.control_path.read_text())
        for name in control.get('retry_tasks', []):
            assert name in self.rows, f'Unknown retry task: {name}'
            row = self.rows[name]
            if row['status'] == 'queued' and row.get('retry_requested_at'):
                continue
            assert row['status'] == 'infrastructure_error', f'Retry requires an infrastructure failure: {name}'
            self.rows[name] = {'status': 'queued', 'retry_requested_at': now(),
                               'retry_reason': 'User increased memory after the earlier OOM; preserve that attempt and rerun with the new budget.',
                               'previous_attempt': row}
        self.update()
        if control.get('retry_tasks'):
            control.pop('retry_tasks')
            control['updated_at'] = now()
            save(self.control_path, control)

    def concurrency(self):
        value = int(json.loads(self.control_path.read_text())['concurrency'])
        if not 1 <= value <= 32:
            raise ValueError('concurrency must be between 1 and 32')
        return value

    @staticmethod
    def outcome(path, expects_ctrf):
        native = json.loads(path.read_text())
        exception = native.get('exception_info')
        reward = (native.get('verifier_result') or {}).get('rewards', {}).get('reward')
        status = 'passed' if reward == 1 and not exception else 'failed'
        if exception:
            kind = exception.get('exception_type', '')
            status = 'timeout' if 'AgentTimeout' in kind else 'infrastructure_error'
        elif reward not in (0, 1):
            status = 'infrastructure_error'
        if reward == 0 and expects_ctrf and not (path.parent / 'verifier/ctrf.json').exists():
            status = 'infrastructure_error'
        return {'status': status, 'reward': reward, 'exception_info': exception,
                'result_path': str(path.relative_to(ROOT)), 'agent_usage': native.get('agent_result')}

    def log(self, message):
        line = f'{now()} {message}'
        print(line, flush=True)
        with (self.base / 'runner.log').open('a') as f:
            f.write(line + '\n')

    def update(self, name=None, **fields):
        with self.lock:
            if name:
                self.rows[name].update(fields, updated_at=now())
            for task_name, row in self.rows.items():
                if row['status'] != 'running':
                    continue
                trial_root = self.base / task_name / f"attempt{row['attempt']}"
                if list(trial_root.glob('*/verifier/test-stdout.txt')):
                    row['phase'] = 'verifier'
                elif list(trial_root.glob('*/agent/recording.cast')):
                    row['phase'] = 'agent'
                else:
                    row['phase'] = 'environment_setup'
            counts = {key: sum(row['status'] == key for row in self.rows.values()) for key in sorted({x['status'] for x in self.rows.values()})}
            complete = all(row['status'] in TERMINAL for row in self.rows.values())
            successes = counts.get('passed', 0)
            active_agents = sum(x.get('phase') == 'agent' and x['status'] == 'running' for x in self.rows.values())
            self.max_active_agent_trials = max(self.max_active_agent_trials, active_agents)
            save(self.base / 'progress.json', {'updated_at': now(), 'complete': complete,
                 'tasks_total': 89, 'status_counts': counts,
                 'trial_concurrency': self.concurrency(),
                 'resources_per_trial': {'cpus': 4, 'memory_mib': 32768},
                 'new_trial_sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
                                        'max_tokens': 81920, 'context_tokens': 262144},
                 'active_agent_trials': active_agents,
                 'max_active_agent_trials': self.max_active_agent_trials,
                 'pilot_success_rate_percent': 100 * successes / 89 if complete else None,
                 'official_benchmark_score': None, 'tasks': self.rows})

    def sif(self, name):
        image = self.tasks[name]['environment']['docker_image']
        return CACHE / (image.replace('/', '_').replace(':', '_') + '.sif')

    def prune(self, reserve=0):
        with self.lock:
            files = [self.sif(name) for name in self.task_ids if self.sif(name).is_file()]
            total = sum(p.stat().st_size for p in files)
            limit = self.args.image_cache_gib * 1024**3
            for path in sorted(files, key=lambda p: p.stat().st_mtime):
                if total + reserve <= limit:
                    break
                if path in self.protected or path == self.sif('fix-git'):
                    continue
                total -= path.stat().st_size
                path.unlink()
                self.log(f'evicted inactive image {path.name}; hash receipt retained')
            if total + reserve > limit:
                raise RuntimeError('image cache budget exceeded by active/prefetched images')

    def prepare(self, name):
        taskdir = self.base / name
        taskdir.mkdir(exist_ok=True)
        sif = self.sif(name)
        with self.lock:
            self.protected.add(sif)
        try:
            self.update(name, status='prefetching', started_at=now(), error=None)
            receipt = image_metadata(self.tasks[name]['environment']['docker_image'])
            receipt['preexisting_sif'] = sif.exists()
            save(taskdir / 'image.json', receipt)
            if not sif.exists():
                self.prune(reserve=receipt['compressed_layers_bytes'])
                temporary = sif.with_suffix('.sif.partial')
                if receipt['compressed_layers_bytes'] <= 128 * 1024**2:
                    # Small OCI rootfs trees are very slow to extract on shared NFS.
                    # This RAM scratch space remains charged to the 4 GiB preparer cgroup.
                    build_tmp = Path(tempfile.mkdtemp(prefix='coding-agent-tb2-', dir='/dev/shm'))
                elif receipt['compressed_layers_bytes'] <= 512 * 1024**2 and shutil.disk_usage('/tmp').free > 8 * 1024**3:
                    build_tmp = Path(tempfile.mkdtemp(prefix='coding-agent-tb2-build-', dir='/tmp'))
                else:
                    build_tmp = ROOT / '.cache/tmp/tb2-build' / name
                    build_tmp.mkdir(parents=True, exist_ok=True)
                pull_env = direct_env()
                pull_env.update(APPTAINER_TMPDIR=str(build_tmp), SINGULARITY_TMPDIR=str(build_tmp), TMPDIR=str(build_tmp))
                reference = f"docker://{MIRROR.removeprefix('https://')}/{receipt['original_image'].rsplit(':', 1)[0]}@{receipt['platform_digest']}"
                receipt['pull_reference'] = reference
                self.log(f'pulling {name}: {receipt["compressed_layers_bytes"] / 1024**2:.1f} MiB OCI layers')
                with (taskdir / 'image-pull.log').open('a') as output:
                    pull = subprocess.Popen(['singularity', 'pull', '--force', str(temporary), reference],
                        env=pull_env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = pull.wait(timeout=1200)
                    except subprocess.TimeoutExpired:
                        os.killpg(pull.pid, signal.SIGKILL)
                        pull.wait()
                        temporary.unlink(missing_ok=True)
                        raise
                    finally:
                        shutil.rmtree(build_tmp, ignore_errors=True)
                if code:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f'image pull failed, exit={code}; see image-pull.log')
                temporary.replace(sif)
            receipt.update(sif_path=str(sif.relative_to(ROOT)), sif_bytes=sif.stat().st_size,
                           sif_sha256=sha(sif), ready_at=now())
            save(taskdir / 'image.json', receipt)
            runtime = taskdir / 'runtime'
            runtime.mkdir(exist_ok=True)
            inspect_cmd = ['singularity', 'exec', '--no-mount', 'home,tmp,bind-paths,cwd',
                           '--containall', '--fakeroot', '--pwd', '/', str(sif)]
            inspect_env = direct_env()
            inspect_env.pop('APPTAINER_BINDPATH', None)
            distro = subprocess.run(inspect_cmd + ['cat', '/etc/os-release'], env=inspect_env,
                                    capture_output=True, check=True, timeout=60).stdout.decode()
            (runtime / 'os-release').write_text(distro)
            archive = subprocess.run(inspect_cmd + ['sh', '-c',
                'if [ -f /etc/apt/sources.list ]; then tar -C /etc/apt -cf - sources.list sources.list.d; else tar -C /etc/apt -cf - sources.list.d; fi'],
                env=inspect_env, capture_output=True, check=True, timeout=60).stdout
            binds = ['/etc/resolv.conf:/etc/resolv.conf:ro',
                     f'{RESULTS}/runtime/apt-direct.conf:/etc/apt/apt.conf.d/99harbor-direct:ro',
                     f'{ASSETS}/ca-certificates.crt:/etc/ssl/certs/ca-certificates.crt:ro',
                     f'{ASSETS}:/opt/harbor-deps:ro']
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for member in tar.getmembers():
                    relative = Path(member.name)
                    if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                        continue
                    original = tar.extractfile(member).read().decode()
                    # Minimal Ubuntu/Debian images may not have CA certificates yet.
                    # APT still verifies the distribution's signed package indexes.
                    text = re.sub(r'https?://(?:[a-z]{2}\.)?archive\.ubuntu\.com/ubuntu/?', 'http://mirrors.aliyun.com/ubuntu/', original)
                    for source, target in [
                        ('http://security.ubuntu.com/ubuntu', 'http://mirrors.aliyun.com/ubuntu'),
                        ('https://security.ubuntu.com/ubuntu', 'http://mirrors.aliyun.com/ubuntu'),
                        ('http://deb.debian.org/debian', 'http://mirrors.aliyun.com/debian'),
                        ('https://deb.debian.org/debian', 'http://mirrors.aliyun.com/debian'),
                        ('http://security.debian.org/debian-security', 'http://mirrors.aliyun.com/debian-security'),
                        ('https://security.debian.org/debian-security', 'http://mirrors.aliyun.com/debian-security')]:
                        text = text.replace(source, target)
                    if text != original:
                        path = runtime / relative
                        path.parent.mkdir(exist_ok=True, parents=True)
                        path.write_text(text)
                        binds.append(f'{path}:/etc/apt/{relative}:ro')
            test_path = DATA / name / 'tests/test.sh'
            text = test_path.read_text() if test_path.exists() else ''
            uv_versions = re.findall(r'https://astral\.sh/uv/([^/\s"\']+)/install\.sh', text)
            uv_url = 'file:///opt/harbor-deps' if uv_versions and set(uv_versions) == {'0.9.5'} else None
            runtime_info = {'binds': binds, 'uv_download_url': uv_url,
                            'expects_ctrf': '--ctrf' in text,
                            'uv_python_install_mirror': 'https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download',
                            'uv_default_index': 'https://mirrors.aliyun.com/pypi/simple',
                            'ca_bundle_sha256': sha(ASSETS / 'ca-certificates.crt'),
                            'files_sha256': {str(x.relative_to(runtime)): sha(x) for x in runtime.rglob('*') if x.is_file()}}
            save(taskdir / 'runtime.json', runtime_info)
            self.update(name, status='ready', image_bytes=sif.stat().st_size)
            self.log(f'image ready: {name}, {sif.stat().st_size / 1024**2:.1f} MiB')
            return runtime_info
        except Exception as error:
            self.update(name, status='infrastructure_error', error=str(error), finished_at=now())
            self.log(f'{name}: image/environment preparation failed: {error}')
            with self.lock:
                self.protected.discard(sif)
            return None

    def evaluate(self, name, runtime):
        directory = self.base / name
        prior = sorted(directory.glob('attempt*-config.json'))
        attempt = max([int(p.name.split('-')[0].removeprefix('attempt')) for p in prior], default=0) + 1
        config = json.loads(json.dumps(self.template))
        config.update(job_name=f'attempt{attempt}', jobs_dir=str(directory.relative_to(ROOT)),
                      tasks=[{'path': str((DATA / name).relative_to(ROOT))}])
        config_path = directory / f'attempt{attempt}-config.json'
        save(config_path, config)
        save(directory / f'attempt{attempt}-runtime.json', {
            'runtime': runtime, 'created_at': now(),
            'resources': {'cpu_quota_cores': 4, 'memory_max_bytes': 34359738368},
            'runner_sha256': sha(Path(__file__)),
            'harbor_lock_sha256': sha(ROOT / 'archive/harbor-local.lock.json'),
            'config_sha256': sha(config_path)})
        env = direct_env()
        env['APPTAINER_BINDPATH'] = ','.join(runtime['binds'])
        env.update(APPTAINERENV_PIP_INDEX_URL='https://mirrors.aliyun.com/pypi/simple',
                   APPTAINERENV_PIP_PROXY='', APPTAINERENV_PIP_CONFIG_FILE='/dev/null',
                   APPTAINERENV_PIP_NO_CACHE_DIR='1',
                   APPTAINERENV_UV_PYTHON_INSTALL_MIRROR=runtime['uv_python_install_mirror'],
                   APPTAINERENV_UV_DEFAULT_INDEX=runtime['uv_default_index'])
        env.pop('APPTAINERENV_UV_DOWNLOAD_URL', None)
        if runtime['uv_download_url']:
            env['APPTAINERENV_UV_DOWNLOAD_URL'] = runtime['uv_download_url']
        unit = f'coding-agent-tb2-{name}-{attempt}.scope'
        command = ['systemd-run', '--quiet', '--scope', '--unit=' + unit,
                   '-p', 'CPUQuota=400%', '-p', 'MemoryMax=32G']
        if os.environ.get('TB2_IN_SYSTEMD_SERVICE') == '1':
            command += ['-p', 'PartOf=coding-agent-tb2.service', '-p', 'BindsTo=coding-agent-tb2.service']
        command += [str(ROOT / '.venv-harbor/bin/harbor'), 'run', '-c', str(config_path)]
        self.update(name, status='running', attempt=attempt, trial_started_at=now(), resource_scope=unit)
        self.log(f'starting trial: {name}, attempt {attempt}')
        with (directory / f'attempt{attempt}-launcher.log').open('w') as output:
            process = subprocess.Popen(command,
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 2400
            while process.poll() is None:
                if time.monotonic() > deadline:
                    subprocess.run(['systemctl', 'stop', unit], timeout=30, check=False)
                    process.wait(timeout=30)
                    break
                time.sleep(5)
                self.update()
        returncode = process.returncode
        results = list((directory / f'attempt{attempt}').glob('*/result.json'))
        if not results:
            self.update(name, status='infrastructure_error', error=f'No trial result; launcher exit={returncode}', finished_at=now())
            return
        outcome = self.outcome(results[0], runtime['expects_ctrf'])
        self.update(name, **outcome, finished_at=now())
        self.log(f'finished {name}: {outcome["status"]}, reward={outcome["reward"]}')

    def evaluate_safely(self, name, runtime):
        try:
            self.evaluate(name, runtime)
        except Exception as error:
            self.update(name, status='infrastructure_error', error=str(error), finished_at=now())
            self.log(f'{name}: evaluation launcher error: {error}')
        finally:
            with self.lock:
                self.protected.discard(self.sif(name))

    def run(self):
        if self.args.wait_for_prefetch_unit:
            self.log('waiting for initial image downloads: ' + self.args.wait_for_prefetch_unit)
            while subprocess.run(['systemctl', 'is-active', '--quiet', self.args.wait_for_prefetch_unit]).returncode == 0:
                time.sleep(5)
                self.update()
        order = list(dict.fromkeys(self.fingerprint['initial_tasks'] + self.task_ids))
        pending = [name for name in order if self.rows[name]['status'] not in TERMINAL]
        self.log(f'queue started: {len(pending)} tasks pending; {self.concurrency()} actual trials, 4 CPU / 32 GiB each; max_tokens=81920, context=262144')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prep, \
             concurrent.futures.ThreadPoolExecutor(max_workers=32) as trials:
            # Prepare the initial batch first; image conversion is not trial concurrency.
            warm_count = min(len(pending), self.concurrency())
            warm = [(name, prep.submit(self.prepare, name)) for name in pending[:warm_count]]
            ready = [(name, future.result()) for name, future in warm]
            remaining = iter(pending[warm_count:])
            active = {}
            preparing = {}
            exhausted = False
            while ready or active or preparing or not exhausted:
                while ready and len(active) < self.concurrency():
                    name, runtime = ready.pop(0)
                    if runtime:
                        active[trials.submit(self.evaluate_safely, name, runtime)] = name
                while not exhausted and len(ready) + len(preparing) < 4:
                    next_name = next(remaining, None)
                    if next_name:
                        preparing[prep.submit(self.prepare, next_name)] = next_name
                    else:
                        exhausted = True
                waiting = list(active)
                waiting.extend(preparing)
                if not waiting:
                    break
                done, _ = concurrent.futures.wait(waiting, timeout=5,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                for future in list(active):
                    if future in done:
                        future.result()
                        del active[future]
                for future in list(preparing):
                    if future in done:
                        ready.append((preparing.pop(future), future.result()))
                self.update()
        self.update()
        self.log('queue finished; all 89 tasks retained in progress.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-cache-gib', type=int, default=20)
    parser.add_argument('--concurrency', type=int, default=None,
                        help='Actual trial slots; default is the saved control value, initially 8.')
    parser.add_argument('--resume-after-active', action='store_true',
                        help='Finish independent active scopes, then restart the paused older coordinator.')
    parser.add_argument('--wait-for-prefetch-unit', default=None,
                        help='Wait for an already-running initial image download scope.')
    args = parser.parse_args()
    if args.image_cache_gib < 1:
        parser.error('image cache limit must be positive')
    if args.concurrency is not None and not 1 <= args.concurrency <= 32:
        parser.error('concurrency must be between 1 and 32')
    os.chdir(ROOT)
    if args.resume_after_active:
        base = RESULTS / 'seed42-pilot'
        record_path = base / 'rescale-drain.json'
        record = json.loads(record_path.read_text())
        record['status'] = 'waiting_for_existing_trials'
        save(record_path, record)
        if record.get('requested_resources'):
            save(base / 'budget-upgrade.json', record)
        deadline = time.monotonic() + 1500
        while True:
            incomplete = []
            for name, row in record['tasks'].items():
                active = subprocess.run(['systemctl', 'is-active', '--quiet', row['resource_scope']]).returncode == 0
                if active:
                    incomplete.append(name)
            preparation_unit = record.get('image_preparation_unit')
            if preparation_unit and subprocess.run(['systemctl', 'is-active', '--quiet', preparation_unit]).returncode == 0:
                incomplete.append('image_preparation')
            if not incomplete:
                break
            record.update(waiting_for=incomplete, updated_at=now())
            save(record_path, record)
            if record.get('requested_resources'):
                save(base / 'budget-upgrade.json', record)
            if time.monotonic() > deadline:
                subprocess.run(['systemctl', 'kill', '--kill-whom=main', '--signal=SIGCONT', 'coding-agent-tb2.service'], check=False)
                raise RuntimeError('Existing scopes did not finish; no task was interrupted for rescaling.')
            time.sleep(5)
        subprocess.run(['systemctl', 'stop', 'coding-agent-tb2.service'], check=True, timeout=40)
        subprocess.run(['bash', str(ROOT / 'run_tb2.sh'), 'start'], check=True)
        if record.get('requested_resources'):
            for _ in range(30):
                progress = json.loads((base / 'progress.json').read_text())
                active = subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0
                pid = subprocess.check_output(['systemctl', 'show', 'coding-agent-tb2.service', '-p', 'MainPID', '--value'], text=True).strip()
                if (active and pid != str(record['coordinator_pid'])
                        and progress.get('resources_per_trial', {}).get('memory_mib') == 32768
                        and progress.get('new_trial_sampling', {}).get('max_tokens') == 81920):
                    break
                time.sleep(1)
            else:
                raise RuntimeError('New coordinator did not confirm the requested memory and output budgets.')
        record.update(status='completed', finished_at=now())
        record.pop('waiting_for', None)
        save(record_path, record)
        if record.get('requested_resources'):
            save(base / 'budget-upgrade.json', record)
        return
    lockpath = RESULTS / 'seed42-pilot/runner.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Runner(args).run()


if __name__ == '__main__':
    main()
