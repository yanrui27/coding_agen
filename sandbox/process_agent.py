#!/usr/bin/env python3
"""Process manager inside the Apptainer sandbox; user code runs only in child processes.

control_server.py starts this file once per sandbox and talks to it through stdin/stdout JSON lines.
Commands, persistent Python/Bash sessions and PTY terminals are separate OS process groups, so a hung
or crashed task never takes this manager or the outer control server down with it.
"""

import argparse
import base64
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import traceback
import uuid

CONTROL_DIR = Path('/opt/sandbox-control')
PYTHON_HOME = '/opt/sandbox-python'
WORK_DIR = Path('/tmp/sandbox-process-agent')
CHUNK_BYTES = 32768


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value) + '\n')
    temporary.replace(path)


def user_environment(overrides=None):
    # Tasks see the image's own environment, not the control runtime's Python.
    environment = dict(os.environ)
    for name in ('PYTHONHOME', 'PYTHONPATH', 'LD_LIBRARY_PATH'):
        environment.pop(name, None)
    environment['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    image_manifest = CONTROL_DIR / 'image.json'
    if image_manifest.exists():
        for item in json.loads(image_manifest.read_text()).get('image_env', []):
            name, _, value = item.partition('=')
            environment[name] = value
    environment.update(overrides or {})
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(name, None)
    environment.update(NO_PROXY='*', no_proxy='*')
    return environment


def run_session_worker(kind, directory):
    namespace = {'__name__': '__sandbox_session__'}
    shell = None
    if kind == 'bash':
        shell = subprocess.Popen(['/bin/bash', '--noprofile', '--norc'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
                                 env=user_environment())
    while True:
        request_path = directory / 'request.json'
        if not request_path.exists():
            time.sleep(0.02)
            continue
        request = json.loads(request_path.read_text())
        request_path.unlink()
        job_id = request['job_id']
        result = {'returncode': 0, 'worker_pid': os.getpid(), 'shell_pid': shell.pid if shell else None}
        with (directory / (job_id + '.stdout')).open('w', buffering=1) as stdout, \
                (directory / (job_id + '.stderr')).open('w', buffering=1) as stderr:
            try:
                if kind == 'python':
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exec(compile(request['code'], '<sandbox-session>', 'exec'), namespace, namespace)
                else:
                    marker = 'SANDBOX_DONE_' + uuid.uuid4().hex
                    payload = request['code'] + '\nSANDBOX_RC=$?\nprintf "\\n' + marker + ':%s\\n" "$SANDBOX_RC"\n'
                    shell.stdin.write(payload.encode())
                    shell.stdin.flush()
                    data = b''
                    pattern = re.compile(rb'\n' + marker.encode() + rb':(-?\d+)\r?\n')
                    while True:
                        part = os.read(shell.stdout.fileno(), 4096)
                        if not part:
                            raise RuntimeError('Persistent Bash exited before command completion')
                        data += part
                        match = pattern.search(data)
                        if match:
                            stdout.write(data[:match.start()].decode(errors='replace'))
                            result['returncode'] = int(match.group(1))
                            break
            except BaseException:
                traceback.print_exc(file=stderr)
                result['returncode'] = 1
        atomic_json(directory / (job_id + '.result.json'), result)


def run_manager():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    os.environ['PATH'] = PYTHON_HOME + '/bin:' + os.environ.get('PATH', '')
    os.environ['PYTHONUNBUFFERED'] = '1'
    sessions, jobs, terminals = {}, {}, {}
    lock = threading.RLock()
    boot_id = str(uuid.uuid4())

    def kill_group(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def refresh():
        with lock:
            for job in jobs.values():
                if job['state'] != 'running':
                    continue
                process = job['process']
                result_path = job.get('result_path')
                if result_path and result_path.exists():
                    result = json.loads(result_path.read_text())
                    job.update(state='finished', returncode=result['returncode'], worker_result=result)
                elif process.poll() is not None:
                    job.update(state='exited', returncode=process.returncode)
                    if job.get('session_id'):
                        sessions[job['session_id']]['invalid'] = True
                elif time.monotonic() >= job['deadline']:
                    kill_group(process)
                    job.update(state='timed_out', returncode=-9)
                    if job.get('session_id'):
                        sessions[job['session_id']]['invalid'] = True
                if job['state'] != 'running' and job.get('session_id'):
                    sessions[job['session_id']]['active_job'] = None

    def snapshot(job):
        hidden = ('process', 'result_path', 'stdout_path', 'stderr_path', 'deadline')
        result = {key: value for key, value in job.items() if key not in hidden}
        for stream in ('stdout', 'stderr'):
            path = job[stream + '_path']
            if path.exists():
                with path.open('rb') as handle:
                    handle.seek(max(0, path.stat().st_size - 65536))
                    result[stream] = handle.read().decode(errors='replace')
            else:
                result[stream] = ''
        result['process_pid'] = job['process'].pid
        result['process_running'] = job['process'].poll() is None
        return result

    def dispatch(method, data):
        refresh()
        with lock:
            if method == 'health':
                image_manifest = CONTROL_DIR / 'image.json'
                image = json.loads(image_manifest.read_text()).get('image') if image_manifest.exists() else None
                return {'agent_pid': os.getpid(), 'agent_boot_id': boot_id, 'image': image,
                        'rootfs_marker': (CONTROL_DIR / 'rootfs-marker').read_text().strip(),
                        'gpu_device_visible': Path('/dev/nvidia0').exists(),
                        'proc_available': Path('/proc/self').exists()}
            if method in ('file_read', 'file_write'):
                path = Path(data['path'])
                if not path.is_absolute():
                    raise ValueError('File path must be absolute inside the sandbox')
                offset = int(data.get('offset', 0))
                if offset < 0:
                    raise ValueError('Negative offset')
                if method == 'file_read':
                    limit = min(int(data.get('limit', CHUNK_BYTES)), CHUNK_BYTES)
                    with path.open('rb') as handle:
                        handle.seek(offset)
                        payload = handle.read(limit)
                    return {'data_base64': base64.b64encode(payload).decode(), 'offset': offset + len(payload),
                            'eof': offset + len(payload) >= path.stat().st_size}
                payload = base64.b64decode(data['data_base64'], validate=True)
                if len(payload) > CHUNK_BYTES:
                    raise ValueError('File chunk too large')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('wb' if offset == 0 else 'r+b') as handle:
                    handle.seek(offset)
                    handle.write(payload)
                if data.get('mode') is not None:
                    path.chmod(int(data['mode']))
                return {'written': len(payload), 'offset': offset + len(payload)}
            if method == 'terminal_start':
                master, slave = os.openpty()
                terminal_id = uuid.uuid4().hex
                output_path = WORK_DIR / (terminal_id + '.terminal')
                argv = data.get('argv') or ['/bin/bash', '--noprofile', '--norc', '-i']
                environment = user_environment(data.get('env'))
                environment.update(PS1='', TERM='xterm')
                # The fresh child acquires the controlling terminal before exec,
                # avoiding preexec_fn in this multi-threaded manager.
                worker_environment = dict(environment, PYTHONHOME=PYTHON_HOME, LD_LIBRARY_PATH=PYTHON_HOME + '/lib')
                worker_environment['SANDBOX_TERMINAL_ENV'] = json.dumps(environment)
                command = [sys.executable, __file__, '--terminal-worker', '--terminal-command', *argv]
                process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                                           cwd=data.get('cwd') or '/', env=worker_environment,
                                           start_new_session=True)
                os.close(slave)
                os.set_blocking(master, False)
                terminals[terminal_id] = {'process': process, 'master': master, 'output': output_path}

                def read_terminal():
                    with output_path.open('wb', buffering=0) as output:
                        while True:
                            try:
                                if not select.select([master], [], [], 0.1)[0]:
                                    if process.poll() is not None:
                                        return
                                    continue
                                part = os.read(master, 8192)
                                if not part:
                                    return
                                output.write(part)
                            except (OSError, ValueError):
                                return

                threading.Thread(target=read_terminal, daemon=True).start()
                return {'terminal_id': terminal_id, 'process_pid': process.pid}
            if method in ('terminal_read', 'terminal_write', 'terminal_close'):
                terminal = terminals[data['terminal_id']]
                if method == 'terminal_write':
                    payload = base64.b64decode(data['data_base64'], validate=True)
                    try:
                        written = os.write(terminal['master'], payload)
                    except BlockingIOError:
                        written = 0
                    return {'written': written}
                if method == 'terminal_close':
                    # Guard against non-positive groups when the PTY closes right after creation.
                    try:
                        foreground = os.tcgetpgrp(terminal['master'])
                        if foreground > 0 and foreground != terminal['process'].pid:
                            os.killpg(foreground, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                    kill_group(terminal['process'])
                    return {'closed': True}
                offset = int(data.get('offset', 0))
                payload = b''
                if terminal['output'].exists():
                    with terminal['output'].open('rb') as handle:
                        handle.seek(offset)
                        payload = handle.read(CHUNK_BYTES)
                return {'data_base64': base64.b64encode(payload).decode(), 'offset': offset + len(payload),
                        'running': terminal['process'].poll() is None, 'returncode': terminal['process'].poll()}
            if method == 'session_create':
                kind = data['kind']
                if kind not in ('python', 'bash'):
                    raise ValueError('kind must be python or bash')
                session_id = uuid.uuid4().hex
                session_dir = WORK_DIR / session_id
                session_dir.mkdir()
                with (session_dir / 'worker.log').open('w') as log:
                    command = [sys.executable, __file__, '--session-worker', kind, '--directory', str(session_dir)]
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               start_new_session=True)
                sessions[session_id] = {'process': process, 'directory': session_dir, 'active_job': None,
                                        'invalid': False, 'kind': kind}
                return {'session_id': session_id, 'kind': kind, 'process_pid': process.pid, 'agent_pid': os.getpid()}
            if method == 'session_execute':
                session_id = data['session_id']
                session = sessions[session_id]
                if session['invalid'] or session['process'].poll() is not None:
                    raise ValueError('Session exited; create a replacement session')
                if session['active_job']:
                    raise ValueError('Session is busy; cancel or wait for its active execution')
                job_id = uuid.uuid4().hex
                session_dir = session['directory']
                job = {'job_id': job_id, 'kind': session['kind'], 'session_id': session_id,
                       'state': 'running', 'returncode': None, 'process': session['process'],
                       'deadline': time.monotonic() + float(data.get('timeout_seconds', 30)),
                       'stdout_path': session_dir / (job_id + '.stdout'),
                       'stderr_path': session_dir / (job_id + '.stderr'),
                       'result_path': session_dir / (job_id + '.result.json')}
                jobs[job_id] = job
                session['active_job'] = job_id
                atomic_json(session_dir / 'request.json', {'job_id': job_id, 'code': data['code']})
                return snapshot(job)
            if method == 'command_start':
                job_id = uuid.uuid4().hex
                job_dir = WORK_DIR / job_id
                job_dir.mkdir()
                stdout_path, stderr_path = job_dir / 'stdout', job_dir / 'stderr'
                with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                    process = subprocess.Popen(['/bin/bash', '-c', data['command']], cwd=data.get('cwd', '/'),
                                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               start_new_session=True, env=user_environment(data.get('env')))
                job = {'job_id': job_id, 'kind': 'command', 'state': 'running', 'returncode': None,
                       'process': process, 'deadline': time.monotonic() + float(data.get('timeout_seconds', 30)),
                       'stdout_path': stdout_path, 'stderr_path': stderr_path}
                jobs[job_id] = job
                return snapshot(job)
            if method == 'job_status':
                return snapshot(jobs[data['job_id']])
            if method == 'job_cancel':
                job = jobs[data['job_id']]
                if job['state'] == 'running':
                    kill_group(job['process'])
                    job.update(state='cancelled', returncode=-9)
                    if job.get('session_id'):
                        sessions[job['session_id']].update(invalid=True, active_job=None)
                return snapshot(job)
            if method == 'shutdown':
                for record in list(jobs.values()) + list(sessions.values()) + list(terminals.values()):
                    if record['process'].poll() is None:
                        kill_group(record['process'])
                return {'stopped': True}
            raise ValueError('Unknown method')

    def monitor():
        while True:
            refresh()
            time.sleep(0.03)

    threading.Thread(target=monitor, daemon=True).start()
    print(json.dumps({'event': 'ready', **dispatch('health', {})}), flush=True)
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                result = dispatch(request['method'], request.get('data', {}))
                response = {'id': request['id'], 'ok': True, 'result': result}
            except BaseException as error:
                response = {'id': request.get('id'), 'ok': False, 'error': repr(error)}
            print(json.dumps(response), flush=True)
            if request.get('method') == 'shutdown':
                break
    finally:
        dispatch('shutdown', {})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-worker', choices=['python', 'bash'])
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--terminal-worker', action='store_true')
    parser.add_argument('--terminal-command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.terminal_worker:
        # Ctrl-C must reach the foreground job even if the server was started with SIGINT ignored (nohup ... &).
        for signal_number in (signal.SIGINT, signal.SIGQUIT):
            signal.signal(signal_number, signal.SIG_DFL)
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        os.tcsetpgrp(0, os.getpgrp())
        environment = json.loads(os.environ['SANDBOX_TERMINAL_ENV'])
        os.execvpe(args.terminal_command[0], args.terminal_command, environment)
    elif args.session_worker:
        run_session_worker(args.session_worker, args.directory)
    else:
        run_manager()
