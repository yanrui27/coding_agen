#!/usr/bin/env python3
"""Run the fixed TB2.0 task queue with rolling actual trials and bounded prefetch.

Run the selected task manifest, with 3-hour agents and no custom turn limit.
The requested 4 CPU / 32 GiB sandbox differs from the reference protocol hardware.
Official task files and tests are unchanged. Terminus-2 uses the disclosed
local context-recovery backport recorded in archive/harbor-local.lock.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import tarfile
import threading
import time
import tomllib
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/terminal-bench-2.0'
RESULTS = ROOT / 'results/Qwen3.6-35B-A3B/terminal-bench-2.0'
CACHE = ROOT / '.cache/harbor-images'
ASSETS = ROOT / '.cache/harbor-assets'
SUITE = RESULTS
TASK_MANIFEST = ROOT / 'archive/manifests/terminal-bench-2.0-task-ids.txt'
ENDPOINT_ASSIGNMENT = 'least-active service at task start; fixed for the entire task attempt'
AGENT_TIMEOUT_SEC = 3 * 60 * 60
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


def expose_trial_files(taskdir, attempt):
    """Expose the native conversation and score without duplicating their data."""
    for filename, pattern in [('trajectory.json', '*/agent/trajectory.json'), ('result.json', '*/result.json')]:
        target = next((taskdir / f'attempt{attempt}').glob(pattern), None)
        if target is None:
            continue
        link = taskdir / filename
        relative = target.relative_to(taskdir)
        if link.is_symlink():
            if link.readlink() == relative:
                continue
            link.unlink()
        elif link.exists():
            continue
        link.symlink_to(relative)


def direct_env():
    env = os.environ.copy()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env.update(NO_PROXY='*', no_proxy='*', GOMAXPROCS='2',
               PIP_PROXY='', PIP_CONFIG_FILE='/dev/null')
    return env


def build_template(seed, args, base_url=None):
    """Use a fresh environment and the same agent settings for every seed."""
    kwargs = {
        'api_base': base_url or args.base_url[0], 'temperature': args.temperature,
        'model_info': {'max_input_tokens': args.context_limit, 'max_output_tokens': args.max_output_tokens,
                       'input_cost_per_token': 0, 'output_cost_per_token': 0,
                       'litellm_provider': 'openai', 'mode': 'chat'},
        'llm_kwargs': {'api_key': 'EMPTY', 'timeout': args.agent_timeout, 'max_retries': 0},
        'llm_call_kwargs': {'max_tokens': args.max_output_tokens, 'top_p': args.top_p, 'seed': seed,
                           'extra_body': {'top_k': args.top_k,
                                          'chat_template_kwargs': {'enable_thinking': args.thinking == 'on'}}},
    }
    if args.max_turns is not None:
        kwargs['max_turns'] = args.max_turns
    return {
        'n_concurrent_trials': 1, 'n_attempts': 1, 'debug': True, 'retry': {'max_retries': 0},
        'agents': [{'name': args.agent, 'model_name': f'openai/{args.model}',
                    'override_timeout_sec': args.agent_timeout, 'override_setup_timeout_sec': 300,
                    'kwargs': kwargs}],
        'verifier': {},  # Use each official task's timeout.
        'environment': {'type': args.environment, 'override_cpus': args.cpus,
                        'override_memory_mb': args.memory_mib,
                        'cpu_enforcement_policy': 'ignore', 'memory_enforcement_policy': 'ignore',
                        'kwargs': {'singularity_image_cache_dir': str(CACHE.relative_to(ROOT)) if CACHE.is_relative_to(ROOT) else str(CACHE),
                                   'singularity_no_mount': 'home,tmp,bind-paths,cwd',
                                   'singularity_startup_timeout_sec': 300}},
    }


def protocol_settings(agent, resources):
    kwargs = agent['kwargs']
    call = kwargs['llm_call_kwargs']
    return {'resources_per_trial': {'cpus': resources['cpus_per_trial'],
                                    'memory_mib': resources['memory_mb_per_trial']},
            'agent_timeout_sec': agent['override_timeout_sec'],
            'custom_max_turns': kwargs.get('max_turns'),
            'sampling': {'temperature': kwargs['temperature'], 'top_p': call['top_p'],
                         'top_k': call['extra_body']['top_k'], 'max_tokens': call['max_tokens'],
                         'context_tokens': kwargs['model_info']['max_input_tokens'],
                         'thinking': call['extra_body']['chat_template_kwargs']['enable_thinking']}}


def recorded_settings():
    for seed in configured_seeds():
        path = SUITE / f'seed{seed}/run.json'
        if path.exists():
            protocol = json.loads(path.read_text())['protocol']
            return protocol_settings(protocol['agent'], protocol['resources'])
    path = SUITE / 'run.json'
    if path.exists():
        protocol = json.loads(path.read_text())['protocol']
        return protocol_settings(protocol['agent_configs'][0],
            {'cpus_per_trial': protocol['cpu_quota_cores'], 'memory_mb_per_trial': protocol['memory_mib_per_trial']})
    args = argument_parser().parse_args([])
    return protocol_settings(build_template(42, args)['agents'][0],
                             {'cpus_per_trial': args.cpus, 'memory_mb_per_trial': args.memory_mib})


def configured_seeds():
    path = SUITE / 'control.json'
    seeds = json.loads(path.read_text()).get('seeds', [42]) if path.exists() else [42]
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError('seeds must be distinct unsigned 32-bit integers')
    return tuple(seeds)


def result_composition():
    """Disclose historical results reused when completing a selected task set."""
    path = SUITE / 'provenance/full-coverage-plan.json'
    if not path.exists():
        return {}
    return {'result_composition': json.loads(path.read_text())['result_composition']}


def suite_summary(write=True):
    """Keep each full-run rate separate; do not score incomplete runs."""
    runs = []
    total_counts = {}
    seeds = configured_seeds()
    record = SUITE / 'run.json'
    selected_total = (json.loads(record.read_text())['protocol']['tasks_per_seed'] if record.exists()
                      else len(TASK_MANIFEST.read_text().splitlines()))
    for seed in seeds:
        path = SUITE / f'seed{seed}/progress.json'
        progress = json.loads(path.read_text()) if path.exists() else {}
        counts = progress.get('status_counts', {'queued': selected_total})
        complete = progress.get('complete', False)
        passed = counts.get('passed', 0)
        runs.append({'seed': seed, 'tasks_total': selected_total, 'complete': complete,
                     'status_counts': counts, 'successful_tasks': passed,
                     'success_rate_percent': 100 * passed / selected_total if complete else None,
                     'progress': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)})
        for status, count in counts.items():
            total_counts[status] = total_counts.get(status, 0) + count
    complete = all(run['complete'] for run in runs)
    rates = [run['success_rate_percent'] for run in runs] if complete else []
    control = SUITE / 'control.json'
    concurrency = json.loads(control.read_text())['concurrency'] if control.exists() else 8
    settings = recorded_settings()
    result = {'updated_at': now(), 'complete': complete, 'seeds': list(seeds),
              'completed_seeds': sum(run['complete'] for run in runs),
              'tasks_per_seed': selected_total, 'trials_total': selected_total * len(seeds),
              'full_dataset_tasks_total': 89, 'is_full_task_set': selected_total == 89,
              'completed_trials': sum(total_counts.get(status, 0) for status in TERMINAL),
              'status_counts': total_counts, 'trial_concurrency': concurrency,
              **settings, **result_composition(),
              'runs': runs,
              'mean_success_rate_percent': statistics.fmean(rates) if complete else None,
              'stddev_success_rate_percentage_points': statistics.pstdev(rates) if complete and len(rates) > 1 else None,
              'stddev_method': 'population standard deviation, ddof=0; withheld for a single run',
              'sample_stddev_percentage_points': statistics.stdev(rates) if complete and len(rates) > 1 else None,
              'success_rate_denominator': f'all {selected_total} selected tasks per seed, including errors and timeouts',
              'reference_hardware_compliant': False,
              'hardware_deviation': f"Local limits: {settings['resources_per_trial']['cpus']} CPU / "
                                    f"{settings['resources_per_trial']['memory_mib']} MiB; "
                                    'reference protocol specifies 32 CPU / 48 GB'}
    if write:
        save(SUITE / 'summary.json', result)
    return result


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
        self.seed = args.seed
        self.base = SUITE / f'seed{self.seed}'
        self.base.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.protected = set()
        self.control_path = SUITE / 'control.json'
        if not self.control_path.exists():
            save(self.control_path, {'concurrency': args.concurrency or 8, 'updated_at': now()})
        self.task_ids = TASK_MANIFEST.read_text().splitlines()
        official_ids = (ROOT / 'archive/manifests/terminal-bench-2.0-task-ids.txt').read_text().splitlines()
        assert self.task_ids and len(self.task_ids) == len(set(self.task_ids)) and set(self.task_ids) <= set(official_ids)
        self.tasks = {name: tomllib.loads((DATA / name / 'task.toml').read_text()) for name in self.task_ids}
        self.template = build_template(self.seed, args)
        self.fingerprint = {
            'dataset_commit': '69671fbaac6d67a7ef0dfec016cc38a64ef7a77c',
            'task_ids': self.task_ids, 'seed': self.seed, 'suite_seeds': list(configured_seeds()),
            'agent': self.template['agents'][0],
            'base_urls': args.base_url, 'endpoint_assignment': ENDPOINT_ASSIGNMENT,
            'resources': {'cpus_per_trial': args.cpus, 'memory_mb_per_trial': args.memory_mib,
                          'concurrent_trials': self.concurrency()},
            'initial_tasks': ['bn-fit-modify', 'adaptive-rejection-sampler',
                              'cancel-async-tasks', 'log-summary-date-ranges', 'openssl-selfsigned-cert'],
            'verifier_timeout_source': 'official per-task task.toml', 'prefetch_tasks': 4,
            'image_cache_limit_gib': args.image_cache_gib,
            'image_build_concurrency': 2,
            'writable_tmpfs_max_mib': 6144,
            'custom_max_turns': args.max_turns,
            'seed_scope': 'LLM sampling seed and Python hash seed; task programs retain their native RNG defaults',
            'scope': 'complete seeded runs with modified sandbox resources; not reference-hardware-compliant',
        }
        self.settings = protocol_settings(self.template['agents'][0], self.fingerprint['resources'])
        previous = self.base / 'run.json'
        scripts = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'run_tb2.sh', ROOT / 'scripts/harbor_env.sh']}
        if previous.exists():
            record = json.loads(previous.read_text())
            old_protocol = json.loads(json.dumps(record['protocol']))
            old_protocol['resources']['concurrent_trials'] = self.concurrency()
            old_protocol['suite_seeds'] = list(configured_seeds())
            old_protocol['image_build_concurrency'] = 2
            old_protocol['image_cache_limit_gib'] = args.image_cache_gib
            old_protocol['base_urls'] = args.base_url
            old_protocol['endpoint_assignment'] = ENDPOINT_ASSIGNMENT
            old_protocol['agent']['kwargs']['api_base'] = args.base_url[0]
            if old_protocol.get('scope') == 'five complete runs with modified sandbox resources; not reference-hardware-compliant':
                old_protocol['scope'] = self.fingerprint['scope']
            assert old_protocol == self.fingerprint, 'changed run protocol; use a separate result directory'
            if record['protocol'] != self.fingerprint:
                record.setdefault('protocol_revisions', []).append({
                    'changed_at': now(), 'previous_protocol': record['protocol'],
                    'reason': 'User updated seeds, image preparation or replica endpoints; model and agent budgets unchanged.'})
                record['protocol'] = self.fingerprint
            record.setdefault('launches', []).append({'started_at': now(), 'scripts': scripts})
            save(previous, record)
        else:
            save(previous, {'created_at': now(), 'protocol': self.fingerprint,
                            'scripts': scripts})
        progress = self.base / 'progress.json'
        self.rows = json.loads(progress.read_text())['tasks'] if progress.exists() else {name: {'status': 'queued'} for name in self.task_ids}
        assert set(self.rows) == set(self.task_ids), 'saved task set does not match the complete dataset'
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
                        elif subprocess.run(['systemctl', 'is-active', '--quiet', scope]).returncode == 0:
                            raise RuntimeError(f'Previous trial scope is still active: {scope}')
                        else:
                            save(self.base / name / f"attempt{row['attempt']}-interrupted.json",
                                 {'recorded_at': now(), 'previous_progress': row,
                                  'reason': 'coordinator restart after an interrupted trial; no completed native result'})
                            row.update(status='queued', error=None)
        self.update()

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
            status = 'timeout' if 'Timeout' in kind else 'infrastructure_error'
        elif reward not in (0, 1):
            status = 'infrastructure_error'
        if reward == 0 and expects_ctrf and not (path.parent / 'verifier/ctrf.json').exists():
            status = 'infrastructure_error'
        return {'status': status, 'reward': reward, 'exception_info': exception,
                'result_path': str(path.relative_to(ROOT)), 'agent_usage': native.get('agent_result')}

    def log(self, message):
        line = f'{now()} seed={self.seed} {message}'
        print(line, flush=True)
        with (self.base / 'runner.log').open('a') as f:
            f.write(line + '\n')

    def update(self, name=None, **fields):
        with self.lock:
            if name:
                self.rows[name].update(fields, updated_at=now())
                if 'attempt' in self.rows[name]:
                    expose_trial_files(self.base / name, self.rows[name]['attempt'])
            for task_name, row in self.rows.items():
                if row['status'] != 'running':
                    continue
                expose_trial_files(self.base / task_name, row['attempt'])
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
                 'seed': self.seed, 'tasks_total': len(self.task_ids), 'status_counts': counts,
                 'full_dataset_tasks_total': 89, 'is_full_task_set': len(self.task_ids) == 89,
                 'base_urls': self.args.base_url, 'endpoint_assignment': ENDPOINT_ASSIGNMENT,
                 'trial_concurrency': self.concurrency(),
                 **self.settings, 'new_trial_sampling': self.settings['sampling'],
                 'active_agent_trials': active_agents,
                 'max_active_agent_trials': self.max_active_agent_trials,
                 'success_rate_percent': 100 * successes / len(self.task_ids) if complete else None,
                 'reference_hardware_compliant': False, 'tasks': self.rows})
            suite_summary()

    def sif(self, name):
        image = self.tasks[name]['environment']['docker_image']
        return CACHE / (image.replace('/', '_').replace(':', '_') + '.sif')

    def prune(self, reserve=0):
        if self.args.image_cache_gib == 0:
            return
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
            cached_receipt = CACHE / 'records/terminal-bench-2.0' / f'{name}.json'
            if sif.exists() and cached_receipt.exists():
                cached = json.loads(cached_receipt.read_text())
                receipt = dict(cached['metadata'])
                assert receipt['original_image'] == self.tasks[name]['environment']['docker_image']
                receipt['source_receipt'] = str(cached_receipt.relative_to(ROOT))
            else:
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
                     f'{ASSETS}/apt-direct.conf:/etc/apt/apt.conf.d/99harbor-direct:ro',
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

    def evaluate(self, name, runtime, base_url):
        directory = self.base / name
        prior = sorted(directory.glob('attempt*-config.json'))
        attempt = max([int(p.name.split('-')[0].removeprefix('attempt')) for p in prior], default=0) + 1
        config = json.loads(json.dumps(self.template))
        config['agents'][0]['kwargs']['api_base'] = base_url
        config.update(job_name=f'attempt{attempt}', jobs_dir=str(directory.relative_to(ROOT)),
                      tasks=[{'path': str((DATA / name).relative_to(ROOT))}])
        verifier_timeout = self.tasks[name].get('verifier', {}).get('timeout_sec', 600)
        # The watchdog includes setup, the full agent budget, official judging, and cleanup.
        # It must never reintroduce the old 40-minute wall-clock cap.
        watchdog_timeout = (self.args.agent_timeout + verifier_timeout
                            + self.tasks[name]['environment'].get('build_timeout_sec', 600)
                            + config['agents'][0]['override_setup_timeout_sec'] + 300)
        config_path = directory / f'attempt{attempt}-config.json'
        save(config_path, config)
        save(directory / f'attempt{attempt}-runtime.json', {
            'runtime': runtime, 'created_at': now(),
            'base_url': base_url, 'endpoint_assignment': ENDPOINT_ASSIGNMENT,
            'seed': self.seed, 'agent_timeout_sec': self.args.agent_timeout,
            'verifier_timeout_sec': verifier_timeout, 'watchdog_timeout_sec': watchdog_timeout,
            'custom_max_turns': self.args.max_turns,
            'resources': {'cpu_quota_cores': self.args.cpus, 'memory_max_bytes': self.args.memory_mib * 1024**2},
            'runner_sha256': sha(Path(__file__)),
            'harbor_lock_sha256': sha(ROOT / 'archive/harbor-local.lock.json'),
            'config_sha256': sha(config_path)})
        env = direct_env()
        env.update(PYTHONHASHSEED=str(self.seed), APPTAINERENV_PYTHONHASHSEED=str(self.seed))
        env['APPTAINER_BINDPATH'] = ','.join(runtime['binds'])
        env.update(APPTAINERENV_PIP_INDEX_URL='https://mirrors.aliyun.com/pypi/simple',
                   APPTAINERENV_PIP_PROXY='', APPTAINERENV_PIP_CONFIG_FILE='/dev/null',
                   APPTAINERENV_PIP_NO_CACHE_DIR='1',
                   APPTAINERENV_UV_PYTHON_INSTALL_MIRROR=runtime['uv_python_install_mirror'],
                   APPTAINERENV_UV_DEFAULT_INDEX=runtime['uv_default_index'])
        env.pop('APPTAINERENV_UV_DOWNLOAD_URL', None)
        if runtime['uv_download_url']:
            env['APPTAINERENV_UV_DOWNLOAD_URL'] = runtime['uv_download_url']
        unit = f'coding-agent-tb2-s{self.seed}-{name}-{attempt}.scope'
        command = ['systemd-run', '--quiet', '--scope', '--unit=' + unit,
                   '-p', f'CPUQuota={self.args.cpus * 100}%', '-p', f'MemoryMax={self.args.memory_mib}M']
        if self.args.service_unit:
            command += ['-p', f'PartOf={self.args.service_unit}', '-p', f'BindsTo={self.args.service_unit}']
        command += [str(ROOT / '.venv-harbor/bin/harbor'), 'run', '-c', str(config_path)]
        self.update(name, status='running', attempt=attempt, trial_started_at=now(), resource_scope=unit,
                    base_url=base_url)
        self.log(f'starting trial: {name}, attempt {attempt}, service {base_url}')
        with (directory / f'attempt{attempt}-launcher.log').open('w') as output:
            process = subprocess.Popen(command,
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + watchdog_timeout
            watchdog_expired = False
            while process.poll() is None:
                if time.monotonic() > deadline:
                    watchdog_expired = True
                    subprocess.run(['systemctl', 'stop', unit], timeout=30, check=False)
                    process.wait(timeout=30)
                    break
                time.sleep(5)
                self.update()
        returncode = process.returncode
        results = list((directory / f'attempt{attempt}').glob('*/result.json'))
        if not results:
            shown = subprocess.run(['systemctl', 'show', unit, '-p', 'Result', '-p', 'MemoryPeak',
                                    '-p', 'MemoryMax', '-p', 'ControlGroup'], capture_output=True, text=True)
            properties = dict(line.split('=', 1) for line in shown.stdout.splitlines() if '=' in line)
            save(directory / f'attempt{attempt}-scope-result.json', properties)
            self.update(name, status='timeout' if watchdog_expired else 'infrastructure_error', reward=None,
                        error=f'No trial result; launcher exit={returncode}; scope={properties.get("Result")}; watchdog_expired={watchdog_expired}',
                        scope_result=properties, finished_at=now())
            return
        outcome = self.outcome(results[0], runtime['expects_ctrf'])
        self.update(name, **outcome, finished_at=now())
        self.log(f'finished {name}: {outcome["status"]}, reward={outcome["reward"]}')

    def evaluate_safely(self, name, runtime, base_url):
        try:
            self.evaluate(name, runtime, base_url)
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
        pending = [name for name in order if name in self.rows and self.rows[name]['status'] not in TERMINAL]
        self.log(f'queue started: {len(pending)} tasks pending; {self.concurrency()} actual trials, '
                 f'{self.args.cpus} CPU / {self.args.memory_mib} MiB each; agent_timeout={self.args.agent_timeout}, '
                 f'max_turns={self.args.max_turns}, max_tokens={self.args.max_output_tokens}, context={self.args.context_limit}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prep, \
             concurrent.futures.ThreadPoolExecutor(max_workers=32) as trials:
            # Prepare the initial batch first; image conversion is not trial concurrency.
            warm_count = min(len(pending), self.concurrency())
            warm = [(name, prep.submit(self.prepare, name)) for name in pending[:warm_count]]
            ready = [(name, future.result()) for name, future in warm]
            remaining = iter(pending[warm_count:])
            active = {}
            service_load = dict.fromkeys(self.args.base_url, 0)
            preparing = {}
            exhausted = False
            while ready or active or preparing or not exhausted:
                while ready and len(active) < self.concurrency():
                    name, runtime = ready.pop(0)
                    if runtime:
                        base_url = min(service_load, key=service_load.get)
                        service_load[base_url] += 1
                        active[trials.submit(self.evaluate_safely, name, runtime, base_url)] = (name, base_url)
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
                        _, base_url = active.pop(future)
                        service_load[base_url] -= 1
                for future in list(preparing):
                    if future in done:
                        ready.append((preparing.pop(future), future.result()))
                self.update()
        self.update()
        self.log(f'queue finished; all {len(self.task_ids)} selected tasks retained in progress.json')
        write_seed_report(self.seed)


def write_seed_report(seed):
    directory = SUITE / f'seed{seed}'
    progress = json.loads((directory / 'progress.json').read_text())
    assert progress['complete'], 'a complete result requires every selected task'
    rows = progress['tasks']
    protocol = json.loads((directory / 'run.json').read_text())['protocol']
    manifest = protocol['task_ids']
    assert set(rows) == set(manifest) and len(rows) == len(manifest)
    assert all(row['status'] in TERMINAL for row in rows.values())
    counts = progress['status_counts']
    settings = protocol_settings(protocol['agent'], protocol['resources'])
    model = protocol['agent']['model_name'].removeprefix('openai/')
    usage = {'input_tokens': 0, 'output_tokens': 0, 'cached_tokens': 0}
    with (directory / 'tasks.csv').open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['task_id', 'status', 'reward', 'error', 'input_tokens', 'output_tokens', 'cached_tokens', 'base_url', 'result_path'])
        for name in manifest:
            row = rows[name]
            agent = row.get('agent_usage') or {}
            input_tokens = agent.get('n_input_tokens') or 0
            output_tokens = agent.get('n_output_tokens') or 0
            cached_tokens = agent.get('n_cache_tokens') or 0
            usage['input_tokens'] += input_tokens
            usage['output_tokens'] += output_tokens
            usage['cached_tokens'] += cached_tokens
            error = row.get('error') or json.dumps(row.get('exception_info'), ensure_ascii=False)
            writer.writerow([name, row['status'], row.get('reward'), error,
                             input_tokens, output_tokens, cached_tokens, row.get('base_url', ''), row.get('result_path', '')])
    result = {'created_at': now(), 'model': model, 'benchmark': 'terminal-bench-2.0',
              'seed': seed, 'complete': True, 'tasks_total': len(manifest), 'status_counts': counts,
              'full_dataset_tasks_total': 89, 'is_full_task_set': len(manifest) == 89,
              'success_rate_percent': 100 * counts.get('passed', 0) / len(manifest),
              'base_urls': protocol['base_urls'], 'endpoint_assignment': ENDPOINT_ASSIGNMENT,
              **settings, **result_composition(),
              'resources': {'cpus_per_trial': protocol['resources']['cpus_per_trial'],
                            'memory_mib_per_trial': protocol['resources']['memory_mb_per_trial']},
              'token_usage_from_available_agent_results': usage,
              'tasks_csv': 'tasks.csv', 'progress': 'progress.json',
              'reference_hardware_compliant': False,
              'note': 'Single seed over the explicitly selected task manifest; a subset is not a full TB2 score. Hardware differs from reference protocol.'}
    save(directory / 'summary.json', result)
    composition = result.get('result_composition')
    composition_note = (
        f"结果来源：复用 {composition['reused_tasks']} 个未出现压缩失败的历史任务，"
        f"新运行 {composition['fresh_tasks']} 个任务（补齐 {composition['missing_tasks']} 题、"
        f"重跑 {composition['compression_bug_reruns']} 个压缩失败任务）。"
        "各任务的 Agent 版本和原始结果路径见 "
        "[逐题来源](../provenance/full-coverage-plan.json)。\n\n"
        if composition else ''
    )
    (directory / 'report.md').write_text(
        f"# {model} · Terminal-Bench 2.0 · seed {seed}\n\n"
        f"本轮已选任务：{len(manifest)} / {len(manifest)}；完整数据集为 89 题。"
        f"本轮成功率：**{result['success_rate_percent']:.2f}%**。\n\n"
        f"通过 {counts.get('passed', 0)}；模型失败 {counts.get('failed', 0)}；"
        f"超时 {counts.get('timeout', 0)}；环境/评测错误 {counts.get('infrastructure_error', 0)}。"
        f"成功率分母包含本轮全部 {len(manifest)} 个已选任务；子集结果不作为完整 TB2.0 成绩。\n\n"
        f"每任务 {settings['resources_per_trial']['cpus']} CPU / {settings['resources_per_trial']['memory_mib']} MiB，"
        f"Agent 预算 {settings['agent_timeout_sec']} 秒，步数上限 {settings['custom_max_turns']}（None 表示不设自定义上限）。"
        "这是单种子的已选任务结果；硬件与参考配置的 32 CPU / 48 GB 不同，未估计多种子波动。\n\n"
        f"{composition_note}"
        "[逐任务结果](tasks.csv) · [原始进度与结果索引](progress.json) · [汇总 JSON](summary.json)\n")
    return result


def watch_seed(seed):
    """Apply a seed stop boundary to a coordinator started before live seed selection existed."""
    receipt = SUITE / 'first-seed-stop.json'
    record = {'started_at': now(), 'seed': seed, 'status': 'watching',
              'reason': 'User requested only seed 42 for now; stop before subsequent seeds.'}
    save(receipt, record)
    while True:
        path = SUITE / f'seed{seed}/progress.json'
        progress = json.loads(path.read_text()) if path.exists() else {}
        if progress.get('complete'):
            if subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0:
                subprocess.run(['systemctl', 'stop', 'coding-agent-tb2.service'], check=True, timeout=45)
            report = write_seed_report(seed)
            suite_summary()
            record.update(status='completed_and_stopped', finished_at=now(),
                          success_rate_percent=report['success_rate_percent'])
            save(receipt, record)
            return
        if subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode != 0:
            record.update(status='service_stopped_before_seed_completed', finished_at=now())
            save(receipt, record)
            return
        time.sleep(0.5)


def resume_after_active():
    record_path = SUITE / 'transition.json'
    record = json.loads(record_path.read_text())
    record.update(status='waiting_for_existing_trials', updated_at=now())
    save(record_path, record)
    deadline = time.monotonic() + 3600
    while True:
        incomplete = [name for name, row in record['tasks'].items()
                      if subprocess.run(['systemctl', 'is-active', '--quiet', row['resource_scope']]).returncode == 0]
        if not incomplete:
            break
        record.update(waiting_for=incomplete, updated_at=now())
        save(record_path, record)
        if time.monotonic() > deadline:
            record.update(status='failed', error='Old trials are still active; no trial was interrupted.')
            save(record_path, record)
            raise RuntimeError(record['error'])
        time.sleep(5)
    # Reconcile old pilot output without importing any of it into the five-seed run.
    path = RESULTS / 'seed42-pilot/progress.json'
    pilot = json.loads(path.read_text())
    for name, row in pilot['tasks'].items():
        if row['status'] != 'running':
            continue
        directory = path.parent / name
        native = list((directory / f"attempt{row['attempt']}").glob('*/result.json'))
        if native and json.loads(native[0].read_text()).get('finished_at'):
            runtime = json.loads((directory / 'runtime.json').read_text())
            row.update(Runner.outcome(native[0], runtime.get('expects_ctrf', False)), finished_at=now())
        else:
            row.update(status='infrastructure_error', reward=None, finished_at=now(),
                       error='Pilot scope ended without a completed native result; raw output preserved.')
    pilot.update(stopped_for_new_protocol_at=now(), superseded_by=str(SUITE.relative_to(ROOT)),
                 status_counts={status: sum(row['status'] == status for row in pilot['tasks'].values())
                                for status in {row['status'] for row in pilot['tasks'].values()}},
                 active_agent_trials=0)
    save(path, pilot)
    subprocess.run(['systemctl', 'stop', 'coding-agent-tb2.service'], check=True, timeout=40)
    subprocess.run(['bash', str(ROOT / 'archive/scripts/manage_tb2.sh'), 'start'], check=True)
    for _ in range(45):
        path = SUITE / 'seed42/run.json'
        active = subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0
        pid = subprocess.check_output(['systemctl', 'show', 'coding-agent-tb2.service', '-p', 'MainPID', '--value'], text=True).strip()
        if active and pid != str(record['coordinator_pid']) and path.exists():
            protocol = json.loads(path.read_text())['protocol']
            if protocol['agent']['override_timeout_sec'] == AGENT_TIMEOUT_SEC and protocol['custom_max_turns'] is None:
                break
        time.sleep(1)
    else:
        record.update(status='failed', error='New coordinator did not confirm the five-seed protocol.')
        save(record_path, record)
        raise RuntimeError(record['error'])
    record.update(status='completed', finished_at=now())
    record.pop('waiting_for', None)
    save(record_path, record)


def turn_limit(value):
    if value.lower() in ('none', 'unlimited', '0'):
        return None
    try:
        limit = int(value)
        if limit > 0:
            return limit
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('use a positive turn count or none')


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen3.6-35B-A3B')
    parser.add_argument('--base-url', nargs='+', default=['http://10.119.27.14:8200/v1'],
                        help='Same-model services; each task stays on one service')
    parser.add_argument('--dataset-path', type=Path, default=Path('datasets/terminal-bench-2.0'))
    parser.add_argument('--results-dir', type=Path, default=Path('results/Qwen3.6-35B-A3B/terminal-bench-2.0'))
    parser.add_argument('--task-manifest', type=Path,
                        default=Path('archive/manifests/terminal-bench-2.0-task-ids.txt'))
    parser.add_argument('--image-cache-dir', type=Path, default=Path('.cache/harbor-images'))
    parser.add_argument('--agent', choices=['terminus-2'], default='terminus-2')
    parser.add_argument('--environment', choices=['singularity'], default='singularity')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--cpus', type=int, default=4, help='CPU quota per task')
    parser.add_argument('--memory-mib', type=int, default=32768, help='Memory limit per task')
    parser.add_argument('--agent-timeout', type=int, default=AGENT_TIMEOUT_SEC)
    parser.add_argument('--max-turns', type=turn_limit, default=None, help='none means no custom step limit')
    parser.add_argument('--context-limit', type=int, default=262144)
    parser.add_argument('--max-output-tokens', type=int, default=81920)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--thinking', choices=['on', 'off'], default='on')
    parser.add_argument('--service-unit', default=None, help='Bind task scopes to this background service')
    parser.add_argument('--print-config', action='store_true', help='Print the effective plan without starting evaluation')
    parser.add_argument('--image-cache-gib', type=int, default=0, help='0 preserves all prepared images')
    parser.add_argument('--concurrency', type=int, default=None,
                        help='Actual trial slots; default is the saved control value, initially 8.')
    parser.add_argument('--resume-after-active', action='store_true',
                        help='Finish the independent pilot scopes, then start the new five-seed coordinator.')
    parser.add_argument('--wait-for-prefetch-unit', default=None,
                        help='Wait for an already-running initial image download scope.')
    parser.add_argument('--summary', action='store_true', help='Summarize only the currently selected seeds.')
    parser.add_argument('--watch-seed', type=int,
                        help='Stop an older coordinator when this seed completes and export its full result.')
    return parser


def main():
    global DATA, RESULTS, CACHE, SUITE, TASK_MANIFEST
    parser = argument_parser()
    args = parser.parse_args()
    args.base_url = list(dict.fromkeys(url.rstrip('/') for url in args.base_url))
    if any(urlsplit(url).scheme not in ('http', 'https') or not urlsplit(url).netloc for url in args.base_url):
        parser.error('base-url must contain one or more HTTP(S) service URLs')
    if args.image_cache_gib < 0:
        parser.error('image cache limit must be nonnegative; 0 preserves all images')
    if args.concurrency is not None and not 1 <= args.concurrency <= 32:
        parser.error('concurrency must be between 1 and 32')
    if min(args.cpus, args.memory_mib, args.agent_timeout, args.context_limit, args.max_output_tokens) <= 0:
        parser.error('resources, timeouts and token limits must be positive')
    if args.max_output_tokens >= args.context_limit:
        parser.error('output token limit must be smaller than the total context limit')
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error('invalid sampling settings')
    if args.seeds is not None and (len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds)):
        parser.error('seeds must be distinct unsigned 32-bit integers')
    args.model = args.model.removeprefix('openai/')
    DATA = (ROOT / args.dataset_path).resolve()
    RESULTS = (ROOT / args.results_dir).resolve()
    CACHE = (ROOT / args.image_cache_dir).resolve()
    SUITE = RESULTS
    TASK_MANIFEST = (ROOT / args.task_manifest).resolve()
    task_ids = TASK_MANIFEST.read_text().splitlines()
    official_ids = (ROOT / 'archive/manifests/terminal-bench-2.0-task-ids.txt').read_text().splitlines()
    if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= set(official_ids):
        parser.error('task manifest must contain unique IDs from the fixed 89-task dataset')
    os.chdir(ROOT)
    if args.print_config:
        seeds = args.seeds or configured_seeds()
        print(json.dumps({'model': args.model, 'dataset_path': str(DATA), 'results_dir': str(RESULTS),
                          'image_cache_dir': str(CACHE), 'image_cache_limit_gib': args.image_cache_gib,
                          'seeds': list(seeds), 'trial_concurrency': args.concurrency or 8,
                          'base_urls': args.base_url, 'endpoint_assignment': ENDPOINT_ASSIGNMENT,
                          'tasks_total': len(task_ids), 'full_dataset_tasks_total': 89,
                          'task_manifest': str(TASK_MANIFEST), 'is_full_task_set': len(task_ids) == 89,
                          'resources_per_trial': {'cpus': args.cpus, 'memory_mib': args.memory_mib},
                          'harbor_job_templates': {str(seed): build_template(seed, args) for seed in seeds}},
                         indent=2, ensure_ascii=False))
        return
    if args.summary:
        result = suite_summary(write=False)
        save(SUITE / 'summary.json', result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.resume_after_active:
        resume_after_active()
        return
    if args.watch_seed is not None:
        watch_seed(args.watch_seed)
        return
    lockpath = SUITE / 'runner.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = SUITE / 'control.json'
        if args.concurrency is not None or args.seeds is not None or not control.exists():
            settings = json.loads(control.read_text()) if control.exists() else {'seeds': [42]}
            settings.update(concurrency=args.concurrency or 8, updated_at=now())
            if args.seeds is not None:
                settings['seeds'] = args.seeds
            save(control, settings)
        seeds = configured_seeds()
        protocol = {'seeds': list(seeds), 'tasks_per_seed': len(task_ids), 'trials_total': len(task_ids) * len(seeds),
                    'task_ids': task_ids, 'full_dataset_tasks_total': 89, 'is_full_task_set': len(task_ids) == 89,
                    'base_urls': args.base_url, 'endpoint_assignment': ENDPOINT_ASSIGNMENT,
                    'agent_timeout_sec': args.agent_timeout, 'custom_max_turns': args.max_turns,
                    'verifier_timeout_source': 'official per-task task.toml',
                    'cpu_quota_cores': args.cpus, 'memory_mib_per_trial': args.memory_mib,
                    'dataset_commit': '69671fbaac6d67a7ef0dfec016cc38a64ef7a77c',
                    'task_manifest_sha256': sha(TASK_MANIFEST),
                    'agent_configs': [build_template(seed, args)['agents'][0] for seed in seeds],
                    'image_build_concurrency': 2,
                    'run_order': 'complete one seed at a time, rolling concurrency within each seed',
                    'prior_pilot_results_included': False,
                    'reference_hardware_compliant': False}
        path = SUITE / 'run.json'
        record = json.loads(path.read_text()) if path.exists() else {'created_at': now(), 'protocol': protocol}
        old_protocol = json.loads(json.dumps(record['protocol']))
        for key in ('seeds', 'trials_total', 'agent_configs', 'image_build_concurrency', 'base_urls'):
            old_protocol[key] = protocol[key]
        assert old_protocol == protocol, 'changed suite protocol; keep results in a separate directory'
        if record['protocol'] != protocol:
            record.setdefault('protocol_revisions', []).append({'changed_at': now(),
                'previous_protocol': record['protocol'],
                'reason': 'User updated seeds, replica endpoints or parallel preparation; task selection and budgets unchanged.'})
            record['protocol'] = protocol
        record.setdefault('launches', []).append({'started_at': now(),
            'scripts': {str(p.relative_to(ROOT)): sha(p) for p in
                        [Path(__file__), ROOT / 'run_tb2.sh', ROOT / 'scripts/harbor_env.sh']},
            'harbor_lock_sha256': sha(ROOT / 'archive/harbor-local.lock.json')})
        save(path, record)
        suite_summary()
        for seed in list(configured_seeds()):
            if seed not in configured_seeds():
                continue
            args.seed = seed
            Runner(args).run()
        suite_summary()


if __name__ == '__main__':
    main()
