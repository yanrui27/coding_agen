"""Regression checks for actual PTY delivery and transport budget ownership."""
import asyncio
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time
import uuid

from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.environments.singularity.singularity import SingularityEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'archive/validation/tb2-environment-repair-20261018'
LIB = ROOT / '.cache/tb2-audit-tools/asciinema-2.2.0'


class LocalTerminal:
    def __init__(self, env):
        self.env = env
        self.session_id = 'local-pty-regression'

    async def exec(self, command, user=None, **kwargs):
        process = await asyncio.create_subprocess_exec('bash', '-c', command, env=self.env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        return ExecResult(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())


async def check_terminal():
    receipts = []
    for size in (35000, 75000, 262144):
        with tempfile.TemporaryDirectory(prefix='tb2-repaired-terminal-', dir='/tmp') as temp:
            directory = Path(temp)
            name = 'repair-' + uuid.uuid4().hex
            env = os.environ.copy()
            env.update(TMUX_TMPDIR=temp, TERM='xterm-256color', SHELL='/bin/bash',
                PYTHONPATH=str(LIB), ASCIINEMA_CONFIG_HOME=str(directory / 'config'))
            local = LocalTerminal(env)
            # Only the installed Harbor send_keys implementation is exercised;
            # the old recorder is deliberately retained to catch regressions.
            session = TmuxSession(name, local, directory / 'pane.log', None, None)
            command = f"python3 -m asciinema rec -q --stdin -c 'bash --noprofile --norc' {shlex.quote(str(directory / 'recording.cast'))}"
            start = await local.exec(f"tmux -f /dev/null new-session -d -s {name} -x 160 -y 40 {shlex.quote(command)}")
            assert start.return_code == 0, start
            try:
                deadline = time.monotonic()+10
                while 'bash-5.2#' not in (await session.capture_pane()):
                    assert time.monotonic() < deadline, 'recorder shell did not start'
                    await asyncio.sleep(0.1)
                line = '# harmless Unicode input 中文 ' + 'x'*80 + '\n'
                body = line * (size // len(line.encode()))
                destination = directory / 'payload.txt'
                marker = directory / 'done'
                key = f"cat > {shlex.quote(str(destination))} << 'TESTEOF'\n{body}TESTEOF\nprintf DONE > {shlex.quote(str(marker))}\n"
                await asyncio.wait_for(session.send_keys(key), timeout=45)
                deadline = time.monotonic()+5
                while not marker.exists() and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                assert marker.exists(), (await session.capture_pane())[-1000:]
                assert destination.read_bytes() == body.encode(), 'terminal payload was altered or truncated'
                await session.send_keys('sleep 30\n', min_timeout_sec=0.2)
                await session.send_keys('C-c', min_timeout_sec=0.2)
                await session.send_keys(f"printf RECOVERED > {shlex.quote(str(directory / 'recovered'))}\n", min_timeout_sec=0.3)
                assert (directory / 'recovered').read_text() == 'RECOVERED'
                receipts.append({'input_bytes': len(key.encode()), 'payload_intact': True,
                    'ctrl_c_recovered': True, 'asciinema': '2.2.0', 'actual_installed_TmuxSession': True})
                print(json.dumps(receipts[-1]), flush=True)
            finally:
                # Only this test's private tmux socket is destroyed.
                await local.exec('tmux kill-server')
    return receipts


async def check_transport():
    import httpx
    class FakeClient:
        def __init__(self):
            self.calls = []
        async def post(self, url, **kwargs):
            self.calls.append(kwargs)
            if kwargs['json']['command'] == 'long':
                await asyncio.sleep(1)
            return httpx.Response(200, json={'stdout': 'OK', 'stderr': None, 'return_code': 0}, request=httpx.Request('POST', url))
    with tempfile.TemporaryDirectory(prefix='tb2-budget-test-') as temp:
        env = SingularityEnvironment(environment_dir=ROOT/'datasets/terminal-bench-2.0/headless-terminal/environment',
            environment_name='budget', session_id='budget', trial_paths=TrialPaths(Path(temp)),
            task_env_config=EnvironmentConfig(docker_image=str(ROOT/'.cache/harbor-images/alexgshaw_headless-terminal_20251031.sif')),
            cpu_enforcement_policy='ignore', memory_enforcement_policy='ignore')
        client = FakeClient()
        env._http_client = client
        env._server_port = 12345
        await env.exec('fast')
        assert client.calls[-1]['timeout'] is None
        await env.exec('fast', timeout_sec=17)
        assert client.calls[-1]['timeout'] == 27
        try:
            await asyncio.wait_for(env.exec('long'), timeout=0.02)
        except TimeoutError:
            pass
        else:
            raise AssertionError('the enclosing verifier budget was not enforced')
        return {'unbounded_exec_transport_timeout': None, 'explicit_exec_17s_transport_timeout': 27,
                'enclosing_budget_still_enforced': True}


async def main():
    result = {'model_calls': 0, 'terminal': await check_terminal(), 'transport': await check_transport()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'regression-tests.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
