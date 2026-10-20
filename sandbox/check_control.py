#!/usr/bin/env python3
"""Check a running sandbox: real processes, persistent sessions, cancel/timeout, crash recovery and reset.

Destructive for the sandbox under test: it freezes and resets the in-sandbox agent.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import time
import traceback
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', default='check')
    args = parser.parse_args()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    credentials = json.loads(args.endpoint_file.read_text())
    url, token = credentials['url'], credentials['token']
    opener = build_opener(ProxyHandler({}))
    calls, checks, latencies = [], {}, []
    report = {'complete': False, 'endpoint': url, 'label': args.label,
              'checks': checks, 'calls': calls, 'control_health_latencies_seconds': latencies}

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json', 'Connection': 'close'}
        call = Request(url + path, data=data, headers=headers)
        with opener.open(call, timeout=12) as response:
            result = json.load(response)
        calls.append({'path': path, 'result': result})
        return result

    def health():
        started = time.monotonic()
        result = request('/health')
        latencies.append(time.monotonic() - started)
        assert result['control_boot_id'] == first['control_boot_id']
        assert latencies[-1] < 3, latencies[-1]
        return result

    def wait(job, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = request('/jobs/' + job['job_id'])
            if result['state'] != 'running':
                return result
            time.sleep(0.05)
        raise TimeoutError('Job did not finish')

    def command(code, timeout=20):
        return request('/commands', {'command': code, 'timeout_seconds': timeout})

    def execute(session, code, timeout=20):
        return request('/sessions/' + session['session_id'] + '/execute', {'code': code, 'timeout_seconds': timeout})

    def python_command(code):
        return '/opt/sandbox-python/bin/python3.12 -u -c ' + shlex.quote(code)

    def heartbeat_code(path):
        return ("from pathlib import Path\nimport time\np=Path(" + repr(path) + ")\n"
                "q=p.with_suffix('.tmp')\n"
                "for i in range(600):\n    q.write_text(str(i))\n    q.replace(p)\n    time.sleep(0.03)\n")

    def counter_probe(path, changed):
        code = ("from pathlib import Path\nimport time\np=Path(" + repr(path) + ")\n"
                "before=p.read_text()\ntime.sleep(0.25)\nafter=p.read_text()\n"
                + ("assert before != after" if changed else "assert before == after") + "\nprint('CHECKED')")
        result = wait(command(python_command(code)))
        assert result['returncode'] == 0 and result['stdout'].strip() == 'CHECKED', result

    try:
        first = request('/health')
        assert not first['agent']['gpu_device_visible']
        assert first['agent']['rootfs_marker'].startswith('sandbox-')
        report['health_before'] = first

        job = wait(command(python_command('import os; print("REAL_PROCESS", os.getpid()); raise SystemExit(7)')))
        assert job['returncode'] == 7 and 'REAL_PROCESS' in job['stdout']
        checks['raw_command_runs_real_os_process_with_exit_status'] = True

        python = request('/sessions', {'kind': 'python'})
        assert python['process_pid'] != python['agent_pid']
        assert wait(execute(python, "x = 10\nitems = ['a']\nprint(x)"))['stdout'].strip() == '10'
        assert wait(execute(python, "x += 5\nitems.append('b')\nprint(x, items)"))['stdout'].strip() == "15 ['a', 'b']"
        checks['python_state_in_separate_persistent_process'] = True

        bash = request('/sessions', {'kind': 'bash'})
        assert wait(execute(bash, 'cd /tmp\nexport SANDBOX_KEEP=retained\nprintf ready'))['stdout'] == 'ready'
        assert wait(execute(bash, 'printf "%s:%s" "$PWD" "$SANDBOX_KEEP"'))['stdout'] == '/tmp:retained'
        checks['bash_cwd_and_environment_persist'] = True

        # Shell + real child are an OS process group. Stopping that group must
        # leave both the supervisor and other sessions operational.
        child_counter = '/tmp/' + args.label + '-cancel-heartbeat'
        parent_code = ("import subprocess,sys,time\nfrom pathlib import Path\n"
                       "p=Path(" + repr(child_counter) + ")\np.unlink(missing_ok=True)\n"
                       "child=subprocess.Popen([sys.executable,'-u','-c'," + repr(heartbeat_code(child_counter)) + "])\n"
                       "while not p.exists(): time.sleep(0.02)\n"
                       "print('CHILD',child.pid,flush=True)\ntime.sleep(60)")
        busy = command(python_command(parent_code), timeout=90)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            health()
            state = request('/jobs/' + busy['job_id'])
            if 'CHILD' in state['stdout']:
                break
            assert state['state'] == 'running', state
            time.sleep(0.05)
        assert state['state'] == 'running' and 'CHILD' in state['stdout'], state
        counter_probe(child_counter, changed=True)
        stopped = request('/jobs/' + busy['job_id'] + '/cancel', {})
        assert stopped['state'] == 'cancelled'
        deadline = time.monotonic() + 3
        while request('/jobs/' + busy['job_id'])['process_running'] and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not request('/jobs/' + busy['job_id'])['process_running']
        counter_probe(child_counter, changed=False)
        health()
        assert wait(execute(python, 'print(x)'))['stdout'].strip() == '15'
        checks['cancel_command_process_group_preserves_controller_and_other_session'] = True
        checks['cancel_also_stops_actual_child_heartbeat'] = True

        # A real CPU-bound infinite loop in the Python worker must not execute
        # in either the HTTP server or the sandbox process agent.
        loop = execute(python, 'while True:\n    pass', timeout=90)
        time.sleep(0.2)
        for _ in range(3):
            health()
        assert request('/jobs/' + loop['job_id'])['state'] == 'running'
        assert request('/jobs/' + loop['job_id'] + '/cancel', {})['state'] == 'cancelled'
        health()
        assert wait(command('printf after-cancel'))['stdout'] == 'after-cancel'
        checks['cpu_infinite_loop_does_not_block_external_control'] = True
        checks['infinite_loop_can_be_cancelled_externally'] = True

        timed = request('/sessions', {'kind': 'python'})
        job = execute(timed, 'while True:\n    pass', timeout=0.6)
        assert wait(job)['state'] == 'timed_out'
        health()
        checks['supervisor_enforces_task_timeout'] = True

        exited = request('/sessions', {'kind': 'python'})
        job = wait(execute(exited, 'import os\nos._exit(23)'))
        assert job['returncode'] == 23
        health()
        checks['user_process_exit_does_not_exit_controller'] = True

        crashed = request('/sessions', {'kind': 'python'})
        job = wait(execute(crashed, 'import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)'))
        assert job['returncode'] == -11, job
        health()
        checks['native_process_crash_does_not_crash_controller'] = True

        replacement = request('/sessions', {'kind': 'python'})
        assert wait(execute(replacement, 'print(6 * 7)'))['stdout'].strip() == '42'
        checks['replacement_session_works_after_crash'] = True
        assert health()['agent']['agent_boot_id'] == first['agent']['agent_boot_id']
        checks['sandbox_process_agent_survives_user_failures'] = True

        # Reset is a container/process-namespace lifecycle operation. The HTTP
        # controller stays outside and retains its identity.
        path = '/tmp/' + args.label + '-survives-reset'
        assert wait(command('printf kept > ' + path))['returncode'] == 0
        detached_counter = '/tmp/' + args.label + '-detached-heartbeat'
        detached_code = ("import subprocess,sys,time\nfrom pathlib import Path\n"
                         "p=Path(" + repr(detached_counter) + ")\np.unlink(missing_ok=True)\n"
                         "child=subprocess.Popen([sys.executable,'-u','-c'," + repr(heartbeat_code(detached_counter))
                         + "],start_new_session=True)\n"
                         "while not p.exists(): time.sleep(0.02)\nprint('DETACHED',child.pid)")
        detached = wait(command(python_command(detached_code)))
        assert detached['returncode'] == 0 and 'DETACHED' in detached['stdout']
        counter_probe(detached_counter, changed=True)
        old_agent = health()['agent']['agent_boot_id']
        request('/reset', {})
        after_reset = health()
        assert after_reset['agent']['agent_boot_id'] != old_agent
        assert wait(command('cat ' + path))['stdout'] == 'kept'
        assert wait(command('printf after-reset'))['stdout'] == 'after-reset'
        counter_probe(detached_counter, changed=False)
        checks['sandbox_can_restart_without_restarting_outer_controller'] = True
        checks['private_filesystem_survives_sandbox_process_restart'] = True
        checks['sandbox_restart_stops_detached_child_in_old_pid_namespace'] = True

        # Even freezing the in-sandbox process agent must leave the outer
        # control endpoint available to reset the sandbox namespace.
        agent_pid = after_reset['agent']['agent_pid']
        freeze_code = f'import os,signal,time\ntime.sleep(0.2)\nos.kill({agent_pid},signal.SIGSTOP)\ntime.sleep(60)'
        frozen_job = command(python_command(freeze_code))
        deadline = time.monotonic() + 12
        frozen = False
        while time.monotonic() < deadline:
            health()
            try:
                state = request('/jobs/' + frozen_job['job_id'])
                assert state['state'] == 'running', state
            except HTTPError as error:
                assert error.code == 503, error
                calls.append({'path': '/jobs/' + frozen_job['job_id'], 'expected_http_status': 503})
                frozen = True
                break
            time.sleep(0.05)
        assert frozen, 'In-sandbox agent never became unresponsive'
        health()
        frozen_agent_boot = health()['agent']['agent_boot_id']
        request('/reset', {})
        assert health()['agent']['agent_boot_id'] != frozen_agent_boot
        assert wait(command('printf recovered-from-agent-freeze'))['stdout'] == 'recovered-from-agent-freeze'
        checks['outer_controller_responds_when_sandbox_agent_is_frozen'] = True
        checks['outer_controller_can_reset_frozen_sandbox_agent'] = True
        report['health_after'] = health()
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'calls'}, indent=2))
    if not report['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
