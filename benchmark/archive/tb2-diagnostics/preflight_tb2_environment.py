#!/usr/bin/env python3
"""Exercise the actual Apptainer/Harbor adapter without model calls."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import tomllib
import uuid

from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'archive/validation/tb2-environment-repair-20261018'


async def run(name, regrade_headless=False, verifier_assets=False, public_sources=False, output_dir=None,
              headless_source=None, qemu_dependencies=False, control_cleanup=None):
    directory = (output_dir or OUT) / name
    directory.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'datasets/terminal-bench-2.0' / name
    conf = tomllib.loads((data / 'task.toml').read_text())['environment']
    image = ROOT / '.cache/harbor-images' / (conf['docker_image'].replace('/', '_').replace(':', '_') + '.sif')
    conf['docker_image'] = str(image)
    prior = ROOT / 'results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed/seed42' / name / 'runtime.json'
    runtime = json.loads(prior.read_text())
    if qemu_dependencies:
        binds = []
        for bind in runtime['binds']:
            parts = bind.split(':')
            original = Path(parts[0])
            if len(parts) >= 2 and parts[1].startswith('/etc/apt/') and original.is_file() and 'sources.list' in parts[1]:
                copy = directory/'runtime-apt'/original.name
                copy.parent.mkdir(parents=True, exist_ok=True)
                text = original.read_text().replace('http://mirrors.aliyun.com/debian-security',
                    'https://archive.debian.org/debian-security').replace('https://mirrors.aliyun.com/debian-security',
                    'https://archive.debian.org/debian-security')
                copy.write_text(text)
                bind = str(copy.resolve())+':'+parts[1]+':rw'
            binds.append(bind)
        runtime['binds'] = binds
    os.environ['APPTAINER_BINDPATH'] = ','.join(runtime['binds'])
    os.environ.update(APPTAINERENV_PIP_INDEX_URL='https://mirrors.aliyun.com/pypi/simple',
        APPTAINERENV_PIP_PROXY='', APPTAINERENV_PIP_CONFIG_FILE='/dev/null',
        APPTAINERENV_UV_DEFAULT_INDEX=runtime['uv_default_index'],
        APPTAINERENV_UV_PYTHON_INSTALL_MIRROR=runtime['uv_python_install_mirror'],
        APPTAINERENV_HF_ENDPOINT='https://hf-mirror.com', APPTAINERENV_UV_HTTP_TIMEOUT='180')
    os.environ['APPTAINERENV_UV_FIND_LINKS'] = 'file:///opt/harbor-deps/wheels'
    if verifier_assets or public_sources:
        os.environ['HARBOR_VERIFIER_ASSET_MANIFEST'] = str(ROOT/'.cache/tb2-verifier-assets/manifest.json')
    if runtime.get('uv_download_url'):
        os.environ['APPTAINERENV_UV_DOWNLOAD_URL'] = runtime['uv_download_url']
    mounts = []
    for role in ('agent', 'verifier', 'artifacts'):
        path = directory / role
        path.mkdir(exist_ok=True)
        mounts.append({'type': 'bind', 'source': str(path), 'target': f'/logs/{role}'})
    log = logging.getLogger('tb2-preflight-' + name)
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.FileHandler(directory / 'environment.log'))
    env = SingularityEnvironment(environment_dir=data / 'environment', environment_name=name,
        session_id='preflight-' + uuid.uuid4().hex, trial_paths=TrialPaths(directory),
        task_env_config=EnvironmentConfig(**conf), logger=log, mounts=mounts,
        override_cpus=8, override_memory_mb=32768,
        cpu_enforcement_policy='ignore', memory_enforcement_policy='ignore',
        singularity_no_mount='home,tmp,bind-paths,cwd', singularity_startup_timeout_sec=300)
    started = time.monotonic()
    result = {'task': name, 'model_calls': 0, 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    try:
        await asyncio.wait_for(env.start(force_build=False), timeout=330)
        result['started'] = True
        if control_cleanup:
            # Destructive process probes run only inside this disposable
            # Apptainer PID namespace, never in a model trial or on the host.
            response = await env.exec(
                "touch /app/control-cleanup-sentinel; "
                "tmux new-session -d -s cleanup-probe; "
                "python3 -c 'import time; time.sleep(180)' >/dev/null 2>&1 & "
                "echo $! > /app/control-cleanup-child.pid", timeout_sec=10)
            assert response.return_code == 0, response
            response = await env.exec("ps -eo pid,comm,args", timeout_sec=10)
            result['controller_before'] = [line for line in response.stdout.splitlines()
                if '/staging/_hbexec.py' in line]
            cleanup = "kill -9 $(pgrep -f 'qemu|nc|python3') 2>/dev/null; sleep 1"
            result['original_agent_cleanup_command'] = cleanup
            import shlex
            try:
                await env.exec('tmux send-keys -t cleanup-probe -- '+shlex.quote(cleanup)+' Enter', timeout_sec=10)
            except Exception as exc:
                result['send_exception'] = type(exc).__name__
            await asyncio.sleep(2)
            try:
                health = await env._http_client.get(f'http://localhost:{env._server_port}/health', timeout=3)
                alive = health.status_code == 200
            except Exception as exc:
                alive = False
                result['health_exception'] = type(exc).__name__
            result['controller_survived_cleanup'] = alive
            if control_cleanup == 'baseline':
                assert not alive, 'baseline did not reproduce the controller disconnect'
            else:
                assert alive, 'control service was killed by ordinary Python cleanup'
                response = await env.exec("test -f /app/control-cleanup-sentinel && "
                    "tmux has-session -t cleanup-probe && "
                    "ps -o stat= -p $(cat /app/control-cleanup-child.pid)", timeout_sec=10)
                assert not response.stdout or response.stdout.startswith('Z'), response
                # A separate comm-name cleanup must also leave control working.
                response = await env.exec("pkill -9 python3 2>/dev/null; echo CONTROL_OK", timeout_sec=10)
                assert response.return_code == 0 and 'CONTROL_OK' in response.stdout, response
                response = await env.exec("test -f /app/control-cleanup-sentinel && tmux has-session -t cleanup-probe", timeout_sec=10)
                assert response.return_code == 0, response
                result['ordinary_python_process_killed'] = True
                result['comm_name_cleanup_survived'] = True
                result['filesystem_and_terminal_preserved'] = True
            result['probe_mode'] = control_cleanup
            result['passed'] = True
            return
        code = "import socket,json; ss=[]\nfor p in [25,80,8000,8080]:\n s=socket.socket();s.bind(('127.0.0.1',p));ss.append(s)\nprint(json.dumps({'bound_ports':[25,80,8000,8080]}))"
        import shlex
        response = await env.exec('python3 -c ' + shlex.quote(code), timeout_sec=10)
        assert response.return_code == 0, response
        result['port_bind_check'] = json.loads(response.stdout)
        response = await env.exec("sleep 1; echo ENCLOSING_BUDGET_OK")
        assert response.return_code == 0 and 'ENCLOSING_BUDGET_OK' in response.stdout, response
        result['exec_without_hidden_transport_limit'] = True
        if public_sources:
            manifest = json.loads(Path(os.environ['HARBOR_VERIFIER_ASSET_MANIFEST']).read_text())
            spec = manifest['agent_public_sources'][name]
            original = spec['git_mirrors'][0]['original_url']
            probe = ('test ! -e /app/pyknotid && test ! -e /opt/harbor-verifier-assets && '
                     'git -c http.proxy= -c https.proxy= clone --depth 1 --branch '+shlex.quote(spec['upstream_ref'])+' '+shlex.quote(original)+' /app/pyknotid && '
                     'test -d /app/pyknotid/.git && git -C /app/pyknotid rev-parse HEAD')
            response = await env.exec(probe, timeout_sec=120)
            assert response.return_code == 0 and spec['commit'] in response.stdout, response
            result['public_source_original_clone_passed'] = True
            result['public_source_commit'] = spec['commit']
            result['solution_not_precreated'] = True
            result['verifier_inputs_hidden_until_phase'] = True
        if verifier_assets:
            manifest = json.loads(Path(os.environ['HARBOR_VERIFIER_ASSET_MANIFEST']).read_text())
            spec = manifest['tasks'][name]
            assert spec['ready'] is True
            if name == 'reshard-c4-data':
                absent = '/root/.cache/huggingface/datasets/allenai___c4/default-c33333bd6ac7ada4'
            else:
                absent = spec['directories'][0]['target']
            response = await env.exec('test ! -e ' + shlex.quote(absent), timeout_sec=10)
            assert response.return_code == 0, 'verifier inputs were exposed before verifier preparation'
            prepared = await env.prepare_verifier_dependencies()
            if spec.get('git_mirrors'):
                original = spec['git_mirrors'][0]['original_url']
                path = '/app/verifier-cache-check'
                probe = 'git clone '+shlex.quote(original)+' '+path+' && cd '+path+' && git checkout '+shlex.quote(spec['upstream_ref'])+' && git rev-parse HEAD'
                response = await env.exec(probe, env=prepared, timeout_sec=120)
                assert response.return_code == 0 and spec['commit'] in response.stdout, response
            else:
                code = "from datasets import load_dataset; d=load_dataset('allenai/c4',data_files={'train':['en/c4-train.00009-of-01024.json.gz']},split='train'); assert len(d)==356318; print('ORIGINAL_C4_LOAD_OK',len(d))"
                response = await env.exec('uv run --python 3.13 --with datasets==3.6.0 python -c '+shlex.quote(code), env=prepared, timeout_sec=600)
                assert response.return_code == 0 and 'ORIGINAL_C4_LOAD_OK' in response.stdout, response
            result['verifier_inputs_hidden_until_phase'] = True
            result['original_external_input_request_passed'] = True
        if qemu_dependencies:
            assert name == 'qemu-startup'
            original = (data/'tests/test.sh').read_text()
            setup = original.split('uvx \\', 1)[0]
            check = ('\ncommand -v curl && command -v expect && command -v uvx && '
                     'uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 '
                     'python -c "import pytest,importlib.metadata; '
                     'assert importlib.metadata.version(\'pytest-json-ctrf\')==\'0.3.5\'; '
                     'print(\'ORIGINAL_GRADER_DEPENDENCIES_OK\')"')
            response = await env.exec('bash -c '+shlex.quote(setup+check)+' > /logs/verifier/dependency-stdout.txt 2>&1', timeout_sec=900)
            assert response.return_code == 0, (directory/'verifier/dependency-stdout.txt').read_text()[-3000:]
            result['original_grader_setup_prefix_sha256'] = hashlib.sha256(setup.encode()).hexdigest()
            result['original_grader_dependencies_passed'] = True
        if regrade_headless or headless_source:
            assert name == 'headless-terminal'
            audit = ROOT / 'results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed/provenance/failure-root-cause-audit-20261018'
            row = next(r for r in json.loads((audit / 'raw-evidence.json').read_text())['tasks'] if r['task'] == name)
            # The audit recorded trial paths from the repo root before benchmark/ moved out of data/.
            trial = ROOT / row['trial'].removeprefix('data/benchmark/')
            trajectory = json.loads((trial / 'agent/trajectory.json').read_text())
            writes = [c['arguments']['keystrokes'] for s in trajectory['steps'] for c in s.get('tool_calls', []) if c.get('arguments', {}).get('keystrokes', '').startswith("cat > /app/headless_terminal.py << 'PYEOF'\n")]
            source = (Path(headless_source).read_text() if headless_source else
                      writes[-1].split('\n', 1)[1].rsplit('\nPYEOF', 1)[0])
            path = directory / 'saved_headless_terminal.py'
            path.write_text(source)
            await env.upload_file(path, '/app/headless_terminal.py')
            # Restore only the model's runtime dependency; use the original
            # seven-test verifier unchanged, after the saved implementation.
            response = await env.exec("python -m pip install --proxy '' pexpect==4.9.0", timeout_sec=120)
            assert response.return_code == 0, response
            await env.upload_dir(data / 'tests', '/tests')
            response = await asyncio.wait_for(env.exec('bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1'), timeout=900)
            assert response.return_code == 0, response
            reward = (directory / 'verifier/reward.txt').read_text().strip()
            result['saved_implementation_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            result['official_verifier_reward'] = float(reward)
            assert reward == '1', (directory / 'verifier/test-stdout.txt').read_text()[-2500:]
        result['passed'] = True
    except BaseException as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        await env.stop(delete=True)
        result['elapsed_seconds'] = round(time.monotonic()-started, 3)
        (directory / 'receipt.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--regrade-headless', action='store_true')
    parser.add_argument('--verifier-assets', action='store_true')
    parser.add_argument('--public-sources', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--headless-source', type=Path)
    parser.add_argument('--qemu-dependencies', action='store_true')
    parser.add_argument('--control-cleanup', choices=['baseline', 'protected'])
    args = parser.parse_args()
    asyncio.run(run(args.task, args.regrade_headless, args.verifier_assets, args.public_sources, args.output_dir,
                    args.headless_source, args.qemu_dependencies, args.control_cleanup))
