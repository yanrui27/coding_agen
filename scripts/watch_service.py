#!/usr/bin/env python3
"""Watch this deployment and record its endpoint after a real inference probe."""
import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
import requests

for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(key, None)
PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(json.loads((PROJECT / 'logs/current-service.json').read_text())['output_dir'])
JOB = ROOT.name
ACP = '/mnt/data/rui.yan/skills/sensecore-acp-training-job/scripts/acp.py'
session = requests.Session()
session.trust_env = False


def save_json(path, data):
    temporary = path.with_name(path.name + f'.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def watch_replicas(manifest_path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    expected_model = manifest['model']

    def check(replica):
        if replica.get('ready'):
            return replica
        folder = Path(replica['output_dir'])
        replica['checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            process = subprocess.run(['python3', ACP, 'get', replica['name'], '--json'],
                capture_output=True, text=True, timeout=45)
            if process.returncode:
                raise RuntimeError('ACP read failed')
            record = json.loads(process.stdout)
            save_json(folder / 'latest-job.json', record)
            replica['state'] = record['job']['state']
            replica['workers'] = record.get('workers', {}).get('workers', [])
            replica['terminal'] = replica['state'] in ('FAILED', 'DELETED', 'SUCCEEDED')
            workers = [worker for worker in replica['workers'] if worker.get('ip') and worker.get('phase') == 'Running']
            if not workers:
                replica['error'] = 'Waiting for a running Worker with an IP'
                return replica
            base = f"http://{workers[0]['ip']}:8200"
            replica['base_url'] = base + '/v1'
            client = requests.Session()
            client.trust_env = False
            client.get(base + '/health', timeout=5).raise_for_status()
            response = client.get(base + '/v1/models', timeout=5)
            response.raise_for_status()
            models = response.json()
            save_json(folder / 'models.json', models)
            matching = [row for row in models.get('data', []) if row.get('id') == expected_model]
            if not matching or matching[0].get('max_model_len') != 262144:
                raise RuntimeError('Model identity or context window differs from the original deployment')
            probe = client.post(base + '/v1/chat/completions', json={
                'model': expected_model, 'messages': [{'role': 'user', 'content': 'Reply with exactly: ready'}],
                'temperature': 0, 'seed': 42, 'max_tokens': 64,
                'chat_template_kwargs': {'enable_thinking': False}}, timeout=120)
            probe.raise_for_status()
            result = probe.json()
            save_json(folder / 'text-probe.json', result)
            if not result.get('choices', [{}])[0].get('message', {}).get('content'):
                raise RuntimeError('Probe returned no final answer')
            replica.update(ready=True, ready_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), error=None)
        except Exception as error:
            replica['error'] = f'{type(error).__name__}: {error}'[:1000]
        finally:
            save_json(folder / 'readiness.json', replica)
        return replica

    for _ in range(2880):
        with ThreadPoolExecutor(max_workers=3) as executor:
            manifest['replicas'] = list(executor.map(check, manifest['replicas']))
        manifest['checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_json(manifest_path, manifest)
        print(json.dumps({'checked_at': manifest['checked_at'], 'replicas': [
            {key: replica.get(key) for key in ('name', 'state', 'ready', 'base_url', 'error')}
            for replica in manifest['replicas']]}, ensure_ascii=False), flush=True)
        if all(replica.get('ready') for replica in manifest['replicas']):
            return 0
        if any(replica.get('terminal') for replica in manifest['replicas']):
            return 2
        time.sleep(30)
    return 2


def save(name, data):
    target = ROOT / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)


def main():
    for _ in range(2880):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            process = subprocess.run(['python3', ACP, 'get', JOB, '--json'],
                                     capture_output=True, text=True, timeout=45)
            if process.returncode:
                raise RuntimeError('ACP read failed')
            record = json.loads(process.stdout)
            save('latest-job.json', record)
            state = record['job']['state']
            workers = record.get('workers', {}).get('workers', [])
            worker = workers[0] if workers else {}
            status = {'checked_at': timestamp, 'job': JOB, 'job_state': state,
                      'worker_phase': worker.get('phase'), 'worker_ip': worker.get('ip'),
                      'ready': False}
            if state in ('FAILED', 'DELETED', 'SUCCEEDED'):
                save('readiness.json', status)
                print(json.dumps(status), flush=True)
                return
            save('readiness.json', status)
            ip = worker.get('ip')
            if ip and worker.get('phase') == 'Running':
                base = f'http://{ip}:8200'
                session.get(base + '/health', timeout=5).raise_for_status()
                response = session.get(base + '/v1/models', timeout=5)
                response.raise_for_status()
                save('models.json', response.json())
                probe = session.post(base + '/v1/chat/completions', json={
                    'model': 'Qwen3.6-35B-A3B',
                    'messages': [{'role': 'user', 'content': 'Reply with exactly: ready'}],
                    'temperature': 0, 'max_tokens': 64,
                    'chat_template_kwargs': {'enable_thinking': False}}, timeout=120)
                probe.raise_for_status()
                result = probe.json()
                save('text-probe.json', result)
                if not result.get('choices', [{}])[0].get('message', {}).get('content'):
                    raise RuntimeError('Probe returned no answer')
                status.update(ready=True, base_url=base + '/v1',
                              served_model_name='Qwen3.6-35B-A3B')
                save('readiness.json', status)
                print(json.dumps(status), flush=True)
                return
            save('readiness.json', status)
            print(json.dumps(status), flush=True)
        except Exception as error:
            print(json.dumps({'checked_at': timestamp, 'ready': False,
                              'error': type(error).__name__}), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Watch all replicas in a deployment manifest')
    args = parser.parse_args()
    raise SystemExit(watch_replicas(args.manifest) if args.manifest else main())
