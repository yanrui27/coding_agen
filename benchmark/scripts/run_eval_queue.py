#!/usr/bin/env python3
"""Durable sequential handoff: completed TB2 -> all ASI conditions -> SWE HARD-51."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/Qwen3.6-35B-A3B'
QUEUE = RESULTS / 'evaluation-queue'
PRIORITY = QUEUE / 'priority.json'
TB2 = RESULTS / 'terminal-bench-2.0/full-seed42-context-fixed'
ASI = RESULTS / 'asi-bench'
SWE = RESULTS / 'swe-bench-pro-v2-hard51'
PROXIES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def direct_env():
    env = os.environ.copy()
    for key in PROXIES:
        env.pop(key, None)
    env.update(NO_PROXY='*', no_proxy='*', PIP_PROXY='', PIP_CONFIG_FILE='/dev/null')
    return env


def tb2_finished(summary, coordinator_active):
    return (summary.get('complete') is True and summary.get('tasks_per_seed') == 89
            and summary.get('completed_trials') == 89 and not coordinator_active)


def asi_coverage(directory=ASI):
    expected_tasks = {task.removesuffix('__seed42') for task in
                      (ROOT / 'archive/manifests/asi-seed42-task-ids.txt').read_text().splitlines()}
    expected = {(task, level) for task in expected_tasks for level in ('b1', 'b2', 'b3', 'b4')}
    latest = {}
    for path in directory.glob('*/*.json'):
        try:
            data = read(path)
            if not isinstance(data, dict) or not {'task_id', 'instance_id', 'prompt_level', 'attempt'} <= data.keys():
                continue
            key = (data['task_id'], data['prompt_level'])
            if key not in expected:
                continue
            if key not in latest or data['attempt'] > latest[key]['attempt']:
                latest[key] = data
        except (OSError, ValueError):
            continue
    return {'expected_conditions': len(expected), 'attempted_conditions': len(latest),
            'execution_complete': set(latest) == expected,
            'agent_status_counts': dict(Counter((data.get('agent_output') or {}).get('status', 'missing_agent_output')
                                                for data in latest.values())),
            'scoring_complete': False, 'official_score': None,
            'scoring_status': 'pending private-reference ASI Portal scoring'}


def priority_progress(config):
    """Keep follow-on benchmarks idle until the requested TB2 runs finish."""
    runs = []
    for item in config.get('runs', []):
        path = ROOT / item['progress']
        progress = read(path)
        active = subprocess.run(['systemctl', 'is-active', '--quiet', item['service']]).returncode == 0
        complete = (progress.get('complete') is True
                    and progress.get('tasks_total') == item['expected_tasks']
                    and not active)
        runs.append({'progress': item['progress'], 'service': item['service'],
                     'expected_tasks': item['expected_tasks'], 'complete': complete,
                     'service_active': active, 'status_counts': progress.get('status_counts', {}),
                     'external_inputs_pending': [name for name, row in progress.get('tasks', {}).items()
                                                 if row.get('model_started') is False and row.get('status') == 'infrastructure_error']})
    return bool(runs) and all(row['complete'] for row in runs), runs


def wait_for_priority(update):
    config = read(PRIORITY)
    while config.get('enabled'):
        complete, runs = priority_progress(config)
        update('tb2_environment_retest', status='finished' if complete else 'priority_running',
               reason=config.get('reason'), runs=runs,
               follow_on_benchmarks='paused until the runnable TB2 environmental reruns finish')
        if complete:
            config.update(enabled=False, completed_at=now(), completion=runs)
            save(PRIORITY, config)
            return
        time.sleep(15)
        config = read(PRIORITY)


def execute(command, log_name):
    with (QUEUE / log_name).open('a') as log:
        log.write('\n' + now() + ' starting ' + json.dumps(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=direct_env(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        def stop(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            return process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    state_path = QUEUE / 'progress.json'
    if args.status:
        print(json.dumps(read(state_path), ensure_ascii=False, indent=2))
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(state_path) or {'created_at': now(), 'order': ['tb2', 'asi', 'swe'],
                                    'model': 'Qwen3.6-35B-A3B', 'base_url': 'http://10.119.27.14:8200/v1',
                                    'stages': {}}
        def update(stage, **fields):
            state['stages'].setdefault(stage, {}).update(fields)
            state.update(current_stage=stage, updated_at=now())
            save(state_path, state)
        update('tb2', status='waiting', output=str(TB2.relative_to(ROOT)))
        while True:
            summary = read(TB2 / 'summary.json')
            active = subprocess.run(['systemctl', 'is-active', '--quiet', 'coding-agent-tb2.service']).returncode == 0
            if tb2_finished(summary, active):
                break
            if not active:
                raise RuntimeError('TB2 stopped before all 89 tasks completed; follow-on benchmarks were not started')
            update('tb2', completed_trials=summary.get('completed_trials'), status_counts=summary.get('status_counts'))
            time.sleep(20)
        update('tb2', status='finished', finished_at=now(), summary=summary)

        wait_for_priority(update)
        coverage = asi_coverage()
        if not coverage['execution_complete']:
            update('asi', status='running', started_at=now(), output=str(ASI.relative_to(ROOT)), expected_conditions=240)
            for attempt in range(2):
                command = ['bash', 'run_asi.sh']
                if (ASI / 'launcher.json').exists():
                    command += ['--resume', str(ASI)]
                returncode = execute(command, 'asi.log')
                coverage = asi_coverage()
                update('asi', returncode=returncode, **coverage)
                if coverage['execution_complete']:
                    break
        update('asi', status='outputs_collected' if coverage['execution_complete'] else 'incomplete',
               finished_at=now(), **coverage)
        save(QUEUE / 'asi-execution-summary.json', coverage)
        if coverage['execution_complete']:
            package_rc = execute(['bash', 'scripts/manage_asi.sh', 'cli', 'submit',
                                  '--results-dir', str(ASI), '--benchmark-repo',
                                  'Apexintelligence-AI/ASI-Bench-seed42', '--no-upload'], 'asi-package.log')
            update('asi', local_submission_package_returncode=package_rc)

        wait_for_priority(update)
        summary = read(SWE / 'summary.json')
        if not (summary.get('complete') and summary.get('tasks_total') == 51):
            update('swe', status='running', started_at=now(), output=str(SWE.relative_to(ROOT)), expected_tasks=51)
            returncode = execute(['bash', 'run_swe.sh', 'run'], 'swe.log')
            summary = read(SWE / 'summary.json')
            update('swe', returncode=returncode)
        swe_complete = bool(summary.get('complete') and summary.get('tasks_total') == 51)
        update('swe', status='finished' if swe_complete else 'incomplete', finished_at=now(), summary=summary)
        state.update(status='finished' if coverage['execution_complete'] and swe_complete else 'finished_with_incomplete_stages',
                     finished_at=now(), asi_official_scoring='pending private-reference Portal scoring')
        save(state_path, state)
        if not coverage['execution_complete'] or not swe_complete:
            raise RuntimeError('At least one follow-on benchmark did not cover its complete fixed task set; see queue progress')


if __name__ == '__main__':
    main()
