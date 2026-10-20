#!/usr/bin/env python3
"""Run pinned SWE Pro HARD-51 with offline host-mini agents and pristine regrading."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/swe-bench-pro-v2-hard51'
TOOLING = ROOT / 'third_party/swe-bench-pro/v2/tooling'
RUNTIME = ROOT / '.cache/swe-agent-runtime'
TERMINAL = {'passed', 'failed', 'infrastructure_error'}
PROXY_KEYS = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def direct_env():
    env = os.environ.copy()
    for key in PROXY_KEYS:
        env.pop(key, None)
    env.update(NO_PROXY='*', no_proxy='*', PIP_PROXY='', PIP_CONFIG_FILE='/dev/null',
               LITELLM_LOCAL_MODEL_COST_MAP='True', HARBOR_TELEMETRY='0',
               DOCKER_HOST='unix:///run/coding-agent-swe-docker/docker.sock',
               DOCKER_CONFIG=str(ROOT / '.cache/swe-build/docker-client'),
               MSWEA_GLOBAL_CONFIG_DIR=str(ROOT / '.cache/swe-agent-config'),
               MSWEA_SILENT_STARTUP='1', MSWEA_COST_TRACKING='ignore_errors')
    return env


def capture(command, *, timeout=120, check=True):
    return subprocess.run(command, env=direct_env(), capture_output=True, text=True,
                          errors='replace', timeout=timeout, check=check)


def docker(*args, timeout=120, check=True):
    return capture(['docker', *map(str, args)], timeout=timeout, check=check)


def ensure_runtime():
    if getattr(ensure_runtime, 'ready', False):
        return
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    os.environ.update(direct_env())
    sys.path.insert(0, str(RUNTIME))
    sys.path.insert(0, str(TOOLING))
    import minisweagent
    if minisweagent.__version__ != '2.4.6':
        raise RuntimeError('Expected isolated mini-swe-agent 2.4.6')
    ensure_runtime.ready = True


class DockerSandbox:
    """Only bash execution enters the offline container; the LLM stays on the host."""
    def __init__(self, container, cwd, deadline=None):
        self.container, self.cwd, self.deadline = container, cwd, deadline

    async def exec(self, command, cwd=None, env=None, user=None, timeout_sec=None, **kwargs):
        timeout = max(1, int(timeout_sec or 60))
        if self.deadline:
            timeout = min(timeout, max(1, int(self.deadline - time.time())))
        args = ['exec', '-w', cwd or self.cwd]
        if user:
            args += ['-u', str(user)]
        for key, value in (env or {}).items():
            args += ['-e', f'{key}={value}']
        prefix = 'unset ' + ' '.join(PROXY_KEYS) + "; export NO_PROXY='*' no_proxy='*'; "
        args += [self.container, 'timeout', '-k', '5', str(timeout),
                 'bash', '-lc', prefix + command]
        result = await asyncio.to_thread(docker, *args, timeout=timeout + 15, check=False)
        return SimpleNamespace(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)

    async def upload_file(self, source_path, target_path):
        await asyncio.to_thread(docker, 'cp', str(source_path), f'{self.container}:{target_path}')


def create_container(image, phase, args, task):
    name = f'coding-agent-swe-{phase}-{uuid.uuid4().hex[:12]}'
    argv = ['create', '--name', name, '--pull', 'never', '--network',
            'none' if phase == 'agent' else 'host', '--cpus', str(args.cpus),
            '--memory', f'{args.memory_mib}m', '--memory-swap', f'{args.memory_mib}m',
            '--pids-limit', '2048', '--ulimit', 'nofile=65536:65536',
            '--label', 'coding-agent.suite=swe-hard51', '--label', f'coding-agent.task={task}',
            '--label', f'coding-agent.phase={phase}', '--entrypoint', '/bin/bash']
    for key in PROXY_KEYS:
        argv += ['-e', f'{key}=']
    argv += ['-e', 'NO_PROXY=*', '-e', 'no_proxy=*', '-e', 'PIP_PROXY=',
             '-e', 'PIP_CONFIG_FILE=/dev/null', image, '-c', 'sleep infinity']
    container = docker(*argv).stdout.strip()
    try:
        docker('start', container)
        inspection = json.loads(docker('inspect', container).stdout)[0]
        if inspection['HostConfig']['NetworkMode'] != ('none' if phase == 'agent' else 'host'):
            raise RuntimeError('Unexpected sandbox network policy')
        if inspection['Image'] != image:
            raise RuntimeError('Sandbox did not use the pinned image ID')
        return container, inspection
    except BaseException:
        docker('rm', '-f', container, check=False)
        raise


def checked_exec(environment, command, timeout=120):
    result = asyncio.run(environment.exec(command, timeout_sec=timeout))
    if result.return_code:
        raise RuntimeError(f'Container command failed ({result.return_code}): {result.stderr[-2000:]} {result.stdout[-2000:]}')
    return result.stdout


def preflight(args):
    ids = (ROOT / 'archive/manifests/swe-hard51-task-ids.txt').read_text().splitlines()
    upstream = (ROOT / 'third_party/swe-bench-pro/v2/hard51_ids.txt').read_text().splitlines()
    if len(ids) != 51 or len(set(ids)) != 51 or set(ids) != set(upstream):
        raise RuntimeError('The pinned HARD-51 manifest differs from upstream')
    cache = json.loads((ROOT / '.cache/swe-build/manifest.json').read_text())
    if cache['status'] != 'complete' or cache['ready_task_images'] != 51:
        raise RuntimeError('SWE images are not complete')
    commit = capture(['git', '-C', str(TOOLING.parents[1]), 'rev-parse', 'HEAD']).stdout.strip()
    if commit != cache['code_commit']:
        raise RuntimeError('SWE tooling commit changed')
    store = docker('info', '--format', '{{.DockerRootDir}}').stdout.strip()
    if store != str(ROOT / '.cache/docker-images/swe-bench-pro-v2-hard'):
        raise RuntimeError('Refusing to use a different Docker daemon')
    images = json.loads(docker('image', 'inspect', *[cache['tasks'][task]['official_ref'] for task in ids], timeout=180).stdout)
    tasks = {}
    for task, image in zip(ids, images):
        record = cache['tasks'][task]
        if image['Id'] != record['image_id'] or image['Id'] != record['registry']['config_digest']:
            raise RuntimeError(f'Pinned image changed: {task}')
        if sha(DATA / task / 'task.toml') != record['task_config_sha256']:
            raise RuntimeError(f'Task configuration changed: {task}')
        if not (DATA / task / 'instruction.md').is_file():
            raise RuntimeError(f'Missing instruction: {task}')
        tasks[task] = {'image_id': image['Id'], 'repo': record['checks']['repository'],
                       'base_commit': record['checks']['base_commit'],
                       'instruction_sha256': sha(DATA / task / 'instruction.md'),
                       'verifier_files_sha256': {str(p.relative_to(DATA / task)): sha(p)
                                                 for p in (DATA / task / 'tests').rglob('*') if p.is_file()}}
    ensure_runtime()
    from minisweagent.agents.default import DefaultAgent  # noqa: F401
    from host_mini_swe import HostMiniSweAgent  # noqa: F401
    return ids, tasks


def agent_config(args):
    import yaml
    config = yaml.safe_load((TOOLING / 'configs/mini_textbased.yaml').read_text())
    config['agent'].update(cost_limit=0, step_limit=0, wall_time_limit_seconds=args.agent_timeout)
    config['model'].update(cost_tracking='ignore_errors')
    config['model']['model_kwargs'] = {
        'api_base': args.base_url, 'api_key': 'EMPTY', 'timeout': args.agent_timeout,
        'max_retries': 0, 'max_tokens': 81920, 'temperature': 1.0, 'top_p': 0.95,
        'seed': 42, 'extra_body': {'top_k': 20, 'chat_template_kwargs': {'enable_thinking': True}}}
    return config


def run_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    ensure_runtime()
    from host_mini_swe import HostMiniSweAgent
    from harbor.models.agent.context import AgentContext
    adapter = DockerSandbox(spec['container'], spec['repo'], spec['deadline'])
    agent = HostMiniSweAgent(logs_dir=Path(spec['logs_dir']), model_name=spec['model'],
                            config_file=spec['config'], cwd=spec['repo'], cost_limit=0,
                            command_timeout=60, deterministic_outputs_file=spec.get('deterministic_outputs'))
    asyncio.run(agent.run(Path(spec['instruction']).read_text(), adapter, AgentContext()))


def score_outcome(reward, parsed_tests, grade_exit):
    if grade_exit in (124, 137) or not parsed_tests:
        return 'infrastructure_error'
    if reward == 1 and grade_exit == 0:
        return 'passed'
    if reward == 0 and parsed_tests:
        return 'failed'
    return 'infrastructure_error'


def evaluate(task, metadata, args, suite, update):
    directory = suite / 'seed42' / task
    directory.mkdir(parents=True, exist_ok=True)
    containers = []
    row = {'task_name': task, 'status': 'running', 'started_at': now(), 'phase': 'agent_setup',
           'image_id': metadata['image_id'], 'agent_network': 'none', 'grader_network': 'host',
           'base_commit': metadata['base_commit']}
    update(task, row)
    try:
        container, inspect = create_container(metadata['image_id'], 'agent', args, task)
        containers.append(container)
        save(directory / 'agent-container.json', inspect)
        env = DockerSandbox(container, metadata['repo'])
        if checked_exec(env, 'git rev-parse HEAD').strip() != metadata['base_commit']:
            raise RuntimeError('Agent sandbox base commit changed')
        row.update(phase='agent', agent_container=container)
        update(task, row)
        logs = directory / 'agent'
        logs.mkdir(exist_ok=True)
        spec = {'container': container, 'repo': metadata['repo'], 'deadline': time.time() + args.agent_timeout,
                'logs_dir': str(logs), 'model': f'openai/{args.model}',
                'config': str(suite / 'provenance/agent-config.yaml'),
                'instruction': str(DATA / task / 'instruction.md')}
        save(directory / 'worker.json', spec)
        with (directory / 'agent-worker.log').open('w') as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker-spec', str(directory / 'worker.json')],
                                       env=direct_env(), stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=args.agent_timeout)
                row['agent_exit_code'] = process.returncode
                row['agent_timeout'] = False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                time.sleep(6)  # final in-container command has at most five seconds to stop
                row.update(agent_exit_code=process.returncode, agent_timeout=True)
        # Capture after timeout too; only this diff crosses into the pristine grader.
        patch = checked_exec(env, 'git add -A && git diff --cached && git reset -q', timeout=180)
        (logs / 'model.patch').write_text(patch)
        row.update(patch_sha256=sha(logs / 'model.patch'), patch_bytes=(logs / 'model.patch').stat().st_size,
                   phase='fresh_sandbox_regrade')
        save(directory / 'result.json', row)  # upstream PatchReplayAgent finds captured source trials here
        docker('rm', '-f', container)
        containers.remove(container)
        update(task, row)
        grader, inspect = create_container(metadata['image_id'], 'grader', args, task)
        containers.append(grader)
        if grader == container:
            raise RuntimeError('Agent and grader must use different containers')
        row['grader_container'] = grader
        save(directory / 'grader-container.json', inspect)
        grade = directory / 'regrade'
        save(grade / 'config.json', {'task': {'path': str(DATA / task)}})
        ensure_runtime()
        from patch_replay import PatchReplayAgent
        from harbor.models.agent.context import AgentContext
        grader_env = DockerSandbox(grader, metadata['repo'])
        replay = PatchReplayAgent(logs_dir=grade / 'agent', model_name='replay', source_job=str(suite / 'seed42'))
        asyncio.run(replay.run('', grader_env, AgentContext()))
        docker('cp', str(DATA / task / 'tests') + '/.', f'{grader}:/tests')
        result = asyncio.run(grader_env.exec('mkdir -p /logs/verifier && bash /tests/test.sh',
                                             timeout_sec=args.verifier_timeout))
        (grade / 'test-stdout.txt').write_text(result.stdout)
        (grade / 'test-stderr.txt').write_text(result.stderr)
        docker('cp', f'{grader}:/logs/verifier', str(grade / 'verifier'), check=False)
        reward_file = grade / 'verifier/reward.txt'
        try:
            reward = int(reward_file.read_text().strip())
        except (OSError, ValueError):
            reward = None
        try:
            parsed_tests = isinstance(json.loads((grade / 'verifier/output.json').read_text()).get('tests'), list)
        except (OSError, ValueError, AttributeError):
            parsed_tests = False
        row.update(reward=reward, grade_exit_code=result.return_code,
                   status=score_outcome(reward, parsed_tests, result.return_code),
                   phase='finished', finished_at=now())
    except Exception as exc:
        row.update(status='infrastructure_error', error=f'{type(exc).__name__}: {exc}', finished_at=now())
        logging.exception('SWE task failed: %s', task)
    finally:
        for container in containers:
            docker('rm', '-f', container, check=False)
    save(directory / 'result.json', row)
    update(task, row)


def run(args):
    import yaml
    ids, tasks = preflight(args)
    suite = args.results_dir.resolve()
    suite.mkdir(parents=True, exist_ok=True)
    with (suite / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = agent_config(args)
        protocol = {'benchmark': 'swe-bench-pro-v2-hard51', 'task_ids': ids, 'seed': 42,
                    'model': args.model, 'base_url': args.base_url, 'parallel': args.parallel,
                    'cpus': args.cpus, 'memory_mib': args.memory_mib,
                    'agent_timeout': args.agent_timeout, 'verifier_timeout': args.verifier_timeout,
                    'mini_swe_agent_version': '2.4.6', 'agent_config': config,
                    'tasks': tasks, 'runner_sha256': sha(Path(__file__)),
                    'upstream_tooling_sha256': {name: sha(TOOLING / name) for name in ['host_mini_swe.py', 'patch_replay.py']},
                    'score_source': 'unchanged official verifier on a fresh pinned-image sandbox; captured git diff only'}
        previous = suite / 'run.json'
        if previous.exists() and json.loads(previous.read_text())['protocol'] != protocol:
            raise ValueError('SWE protocol changed; choose a separate result directory')
        save(previous, {'started_at': now(), 'protocol': protocol})
        (suite / 'provenance').mkdir(exist_ok=True)
        (suite / 'provenance/agent-config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run(['systemctl', 'set-property', '--runtime', 'coding-agent-swe.slice',
                        f'CPUQuota={args.cpus * args.parallel * 100:g}%',
                        f'MemoryMax={args.memory_mib * args.parallel}M'], check=True)
        progress_path = suite / 'progress.json'
        rows = json.loads(progress_path.read_text())['tasks'] if progress_path.exists() else {task: {'status': 'queued'} for task in ids}
        if set(rows) != set(ids):
            raise ValueError('SWE saved task set differs from HARD-51')
        guard = threading.Lock()

        def update(task=None, row=None):
            with guard:
                if task:
                    rows[task] = dict(row)
                counts = dict(Counter(row['status'] for row in rows.values()))
                completed = sum(counts.get(status, 0) for status in TERMINAL)
                summary = {'updated_at': now(), 'benchmark': 'swe-bench-pro-v2-hard51', 'seed': 42,
                           'tasks_total': 51, 'completed_tasks': completed, 'complete': completed == 51,
                           'status_counts': counts,
                           'resolved_percent': 100 * counts.get('passed', 0) / 51 if completed == 51 else None,
                           'score_denominator': 'all 51 fixed HARD-51 tasks, including errors',
                           'agent': 'upstream host-mini-swe-agent, 2.4.6, textbased config',
                           'agent_network': 'none', 'grading': 'pristine Docker sandbox per captured patch',
                           'hardware': {'cpus': args.cpus, 'memory_mib': args.memory_mib}}
                save(progress_path, {**summary, 'tasks': rows})
                save(suite / 'summary.json', summary)

        update()
        pending = [task for task in ids if rows[task]['status'] not in TERMINAL]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(evaluate, task, tasks[task], args, suite, update) for task in pending]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        update()


def smoke(args):
    """Check offline execution and diff-only transfer without calling a model or running hidden tests."""
    import tempfile
    import yaml
    ids, tasks = preflight(args)
    task = ids[0]
    metadata = tasks[task]
    containers = []
    try:
        agent_id, agent_inspect = create_container(metadata['image_id'], 'agent', args, 'smoke')
        containers.append(agent_id)
        env = DockerSandbox(agent_id, metadata['repo'])
        # The image's private test payload is never mounted into this container.
        checked_exec(env, "test ! -e /tests/test_patch.patch; test ! -e SWE_SMOKE_FIXTURE.txt; "
                          "printf 'isolated marker' > /tmp/swe-smoke-agent-only")
        with tempfile.TemporaryDirectory(prefix='swe-smoke-', dir=ROOT / '.cache/tmp') as tmp:
            work = Path(tmp)
            (work / 'instruction.txt').write_text('Write the smoke fixture and submit.')
            (work / 'config.yaml').write_text(yaml.safe_dump(agent_config(args)))
            outputs = [{'role': 'assistant', 'content': 'fixture', 'extra': {'cost': 0, 'actions': [{'command': "printf 'smoke\\n' > SWE_SMOKE_FIXTURE.txt"}]}},
                       {'role': 'assistant', 'content': 'submit', 'extra': {'cost': 0, 'actions': [{'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}]}}]
            save(work / 'outputs.json', outputs)
            save(work / 'spec.json', {'container': agent_id, 'repo': metadata['repo'], 'deadline': time.time() + 120,
                                     'logs_dir': str(work / 'agent'), 'model': 'deterministic',
                                     'config': str(work / 'config.yaml'), 'instruction': str(work / 'instruction.txt'),
                                     'deterministic_outputs': str(work / 'outputs.json')})
            capture([sys.executable, str(Path(__file__).resolve()), '--worker-spec', str(work / 'spec.json')], timeout=120)
            patch = work / 'agent/model.patch'
            if 'SWE_SMOKE_FIXTURE.txt' not in patch.read_text():
                raise RuntimeError('Upstream host agent failed to capture its diff')
            grader_id, grader_inspect = create_container(metadata['image_id'], 'grader', args, 'smoke')
            containers.append(grader_id)
            grader = DockerSandbox(grader_id, metadata['repo'])
            checked_exec(grader, 'test ! -e SWE_SMOKE_FIXTURE.txt && test ! -e /tmp/swe-smoke-agent-only')
            # Exercise the same upstream replay agent used for authoritative grading.
            source = work / 'source' / task
            (source / 'agent').mkdir(parents=True)
            (source / 'agent/model.patch').write_bytes(patch.read_bytes())
            save(source / 'result.json', {'task_name': task})
            save(work / 'regrade/config.json', {'task': {'path': str(DATA / task)}})
            from patch_replay import PatchReplayAgent
            from harbor.models.agent.context import AgentContext
            replay = PatchReplayAgent(logs_dir=work / 'regrade/agent', model_name='replay', source_job=str(source.parent))
            asyncio.run(replay.run('', grader, AgentContext()))
            replayed = json.loads((work / 'regrade/agent/replay.json').read_text())
            if replayed.get('apply_rc') != 0:
                raise RuntimeError('Upstream patch replay did not apply the captured diff')
            checked_exec(grader, 'grep -q smoke SWE_SMOKE_FIXTURE.txt')
            receipt = {'checked_at': now(), 'model_calls': 0, 'offline_agent_network': agent_inspect['HostConfig']['NetworkMode'],
                       'fresh_grader': agent_id != grader_id, 'same_pinned_image': agent_inspect['Image'] == grader_inspect['Image'],
                       'diff_captured': True, 'agent_only_state_absent_in_grader': True, 'diff_replayed': True,
                       'task_images_verified': 51, 'hidden_tests_not_exposed_to_agent': True,
                       'runner_sha256': sha(Path(__file__))}
            save(ROOT / 'archive/validation/swe-locked-docker-smoke.json', receipt)
            print(json.dumps(receipt, indent=2))
    finally:
        for container in containers:
            docker('rm', '-f', container, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['run', 'check', 'smoke'], default='run')
    parser.add_argument('--model', default='Qwen3.6-35B-A3B')
    parser.add_argument('--base-url', default='http://10.119.27.14:8200/v1')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/Qwen3.6-35B-A3B/swe-bench-pro-v2-hard51')
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--cpus', type=float, default=4)
    parser.add_argument('--memory-mib', type=int, default=32768)
    parser.add_argument('--agent-timeout', type=int, default=3000)
    parser.add_argument('--verifier-timeout', type=int, default=3000)
    parser.add_argument('--worker-spec', type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        return run_worker(args.worker_spec)
    if not 1 <= args.parallel <= 8 or min(args.cpus, args.memory_mib, args.agent_timeout, args.verifier_timeout) <= 0:
        parser.error('Positive resource/time limits and parallel in 1..8 required')
    if args.mode == 'check':
        ids, tasks = preflight(args)
        print(json.dumps({'status': 'ready', 'tasks': len(ids), 'images': len(tasks),
                          'agent_network': 'none', 'fresh_sandbox_regrade': True, 'base_url': args.base_url}))
    elif args.mode == 'smoke':
        smoke(args)
    else:
        run(args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
