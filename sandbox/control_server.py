#!/usr/bin/env python3
"""Start one Apptainer sandbox on this host and serve its HTTP control API from outside the sandbox.

Run on an ACP CPU Worker (or the preparation host for local checks). The rootfs is copied to a private
local directory, process_agent.py runs inside Apptainer, and this server only forwards requests to it, so
user code never executes here. The URL and token go to <output-dir>/endpoint.private.json; the server
stops on TTL or SIGTERM and then removes the sandbox, the rootfs copy and the private endpoint file.
"""

import argparse
import contextlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

SANDBOX_DIR = Path(__file__).resolve().parent
APPTAINER_PREFIX = SANDBOX_DIR / 'runtime/apptainer-1.5.4-jammy'
# Read-only Python for process_agent.py, independent of the task image's own Python.
PYTHON_RUNTIME = Path('/mnt/data/rui.yan/venvs/OmniRL/runtime/cpython-3.12.14-linux-x86_64-gnu')
PROXY_VARIABLES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')


class SandboxAgent:
    """The process_agent.py instance inside Apptainer, driven over stdin/stdout JSON lines."""

    def __init__(self, command, environment, log_path):
        self.command, self.environment, self.log_path = command, environment, log_path
        self.process = None
        self.info = {}
        self.pending = {}
        self.guard = threading.Lock()
        self.lifecycle = threading.Lock()

    def start(self):
        self.info = {}
        ready = threading.Event()
        requests = queue.Queue()
        self.requests = requests
        with open(self.log_path, 'a') as stderr:
            process = subprocess.Popen(self.command, env=self.environment, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                                       start_new_session=True)
        self.process = process

        def read():
            for line in process.stdout:
                try:
                    response = json.loads(line)
                except ValueError:
                    continue
                if response.get('event') == 'ready':
                    if self.process is process:
                        self.info = response
                    ready.set()
                    continue
                with self.guard:
                    future = self.pending.get(response.get('id'))
                if future:
                    future['response'] = response
                    future['event'].set()
            ready.set()

        def write():
            while True:
                message = requests.get()
                if message is None:
                    return
                try:
                    process.stdin.write(json.dumps(message) + '\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    return

        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=write, daemon=True).start()
        if not ready.wait(60) or not self.info:
            raise RuntimeError('Sandbox process agent did not become ready')

    def rpc(self, method, data=None, timeout=3):
        request_id = uuid.uuid4().hex
        future = {'event': threading.Event()}
        with self.guard:
            self.pending[request_id] = future
        try:
            if self.process.poll() is not None:
                raise RuntimeError('Sandbox process agent exited')
            self.requests.put({'id': request_id, 'method': method, 'data': data or {}})
            if not future['event'].wait(timeout):
                raise TimeoutError('Sandbox agent did not respond; outer control service remains available')
            response = future['response']
            if not response['ok']:
                raise ValueError(response['error'])
            return response['result']
        finally:
            with self.guard:
                self.pending.pop(request_id, None)

    def stop(self):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                self.rpc('shutdown', timeout=0.5)
            except (OSError, RuntimeError, ValueError, TimeoutError):
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self.requests.put(None)

    def reset(self):
        # Kills the old PID namespace (and every task in it); the private rootfs files stay.
        with self.lifecycle:
            self.stop()
            self.start()
            return self.info


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def make_handler(agent, token, control_boot_id, started):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, value):
            body = json.dumps(value).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            if hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                return True
            self.respond(401, {'error': 'unauthorized'})
            return False

        def call(self, method, data):
            try:
                self.respond(200, agent.rpc(method, data))
            except (ValueError, KeyError) as error:
                self.respond(400, {'error': str(error)})
            except (OSError, RuntimeError, TimeoutError) as error:
                self.respond(503, {'error': str(error)})

        def do_GET(self):
            if not self.authorized():
                return
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            offset = int(query.get('offset', ['0'])[0])
            if parsed.path == '/health':
                # No call into the sandbox: hung user code or a frozen agent cannot block health.
                self.respond(200, {'control_boot_id': control_boot_id, 'control_pid': os.getpid(),
                                   'uptime_seconds': time.monotonic() - started,
                                   'agent_process_running': agent.process.poll() is None, 'agent': agent.info})
            elif parsed.path.startswith('/jobs/'):
                self.call('job_status', {'job_id': parsed.path.split('/')[2]})
            elif parsed.path.startswith('/terminals/'):
                self.call('terminal_read', {'terminal_id': parsed.path.split('/')[2], 'offset': offset})
            elif parsed.path == '/files/read':
                self.call('file_read', {'path': query.get('path', [''])[0], 'offset': offset})
            else:
                self.respond(404, {'error': 'not_found'})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('body must be 1..65536 bytes')
                data = json.loads(self.rfile.read(size))
                parts = self.path.split('/')
                if self.path == '/reset':
                    self.respond(200, agent.reset())
                elif self.path == '/commands':
                    self.call('command_start', data)
                elif self.path == '/sessions':
                    self.call('session_create', data)
                elif self.path == '/terminals':
                    self.call('terminal_start', data)
                elif self.path == '/files/write':
                    self.call('file_write', data)
                elif self.path.startswith('/terminals/') and self.path.endswith('/write'):
                    self.call('terminal_write', dict(data, terminal_id=parts[2]))
                elif self.path.startswith('/terminals/') and self.path.endswith('/close'):
                    self.call('terminal_close', {'terminal_id': parts[2]})
                elif self.path.startswith('/sessions/') and self.path.endswith('/execute'):
                    self.call('session_execute', dict(data, session_id=parts[2]))
                elif self.path.startswith('/jobs/') and self.path.endswith('/cancel'):
                    self.call('job_cancel', {'job_id': parts[2]})
                else:
                    self.respond(404, {'error': 'not_found'})
            except (KeyError, TypeError, ValueError) as error:
                self.respond(400, {'error': str(error)})
            except (OSError, RuntimeError, TimeoutError) as error:
                self.respond(503, {'error': str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rootfs', type=Path, required=True, help='rootfs tar archive or unpacked directory')
    parser.add_argument('--image-manifest', type=Path, help='JSON with the task image name and image_env')
    parser.add_argument('--ttl-seconds', type=int, default=7200)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in PROXY_VARIABLES:
        os.environ.pop(name, None)
    environment = dict(os.environ, NO_PROXY='*', no_proxy='*', GOMAXPROCS='2')
    library_dirs = f'{APPTAINER_PREFIX}/usr/lib/x86_64-linux-gnu:{APPTAINER_PREFIX}/lib/x86_64-linux-gnu'
    environment['LD_LIBRARY_PATH'] = library_dirs
    binary_dirs = f'{APPTAINER_PREFIX}/usr/bin:{APPTAINER_PREFIX}/usr/libexec/apptainer/bin:'
    environment['PATH'] = binary_dirs + environment.get('PATH', '')
    for name in ('APPTAINER_BIND', 'APPTAINER_BINDPATH', 'SINGULARITY_BIND', 'SINGULARITY_BINDPATH'):
        environment.pop(name, None)
    apptainer = str(APPTAINER_PREFIX / 'usr/bin/apptainer')
    private_endpoint = output / 'endpoint.private.json'
    control_boot_id = str(uuid.uuid4())
    stop_reason = 'ttl'

    def handle_sigterm(*_args):
        nonlocal stop_reason
        stop_reason = 'sigterm'
        # A second SIGTERM must not interrupt the cleanup below.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        # ACP Worker runs as root without a subuid mapping; fakeroot needs one.
        subprocess.run([apptainer, 'config', 'fakeroot', '--add', 'root'], env=environment, check=True, timeout=15)
        with contextlib.ExitStack() as resources:
            work = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix='sandbox-')))
            rootfs = work / 'rootfs'
            rootfs.mkdir()
            # Worker-local copy: extracting onto the shared disk is far too slow for GiB-sized images.
            if args.rootfs.is_dir():
                copy = ['cp', '-a', '--reflink=auto', f'{args.rootfs}/.', str(rootfs)]
                subprocess.run(copy, check=True, timeout=600)
            else:
                subprocess.run(['tar', '-xf', str(args.rootfs), '-C', str(rootfs)], check=True, timeout=600)
            control_dir = rootfs / 'opt/sandbox-control'
            control_dir.mkdir(parents=True, exist_ok=True)
            (rootfs / 'opt/sandbox-python').mkdir(parents=True, exist_ok=True)
            shutil.copy2(SANDBOX_DIR / 'process_agent.py', control_dir / 'process_agent.py')
            if args.image_manifest:
                shutil.copy2(args.image_manifest, control_dir / 'image.json')
            rootfs_marker = 'sandbox-' + str(uuid.uuid4())
            (control_dir / 'rootfs-marker').write_text(rootfs_marker)

            # ACP Workers have no /dev/fuse (no SIF/overlay mounts) and refuse new procfs mounts, so the
            # sandbox is a writable private directory with proc/sys mounts disabled.
            command = [apptainer, 'exec', '--writable', '--fakeroot', '--containall', '--pid', '--cleanenv']
            for mount in ('home', 'tmp', 'bind-paths', 'cwd', 'proc', 'sys'):
                command.extend(['--no-mount', mount])
            command.extend(['--pwd', '/', '--bind', f'{PYTHON_RUNTIME}:/opt/sandbox-python:ro',
                            '--env', 'PYTHONHOME=/opt/sandbox-python',
                            '--env', 'LD_LIBRARY_PATH=/opt/sandbox-python/lib',
                            '--env', 'PYTHONUNBUFFERED=1', str(rootfs),
                            '/opt/sandbox-python/bin/python3.12', '/opt/sandbox-control/process_agent.py'])
            agent = SandboxAgent(command, environment, output / 'sandbox-agent.log')
            # Stop every sandbox process before the private rootfs is deleted.
            resources.callback(agent.stop)
            agent.start()
            if agent.info['rootfs_marker'] != rootfs_marker:
                raise RuntimeError('Sandbox agent reports a different rootfs')
            if agent.info['gpu_device_visible']:
                raise RuntimeError('GPU device is visible inside the sandbox')

            token = secrets.token_urlsafe(32)
            handler = make_handler(agent, token, control_boot_id, time.monotonic())
            server = ThreadingHTTPServer(('0.0.0.0', 0), handler)
            server.daemon_threads = True
            resources.callback(server.server_close)
            port = server.server_address[1]
            address_infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
            addresses = sorted({info[4][0] for info in address_infos if not info[4][0].startswith('127.')})
            if not addresses:
                raise RuntimeError('Could not discover a non-loopback Worker IP')
            url = f'http://{addresses[0]}:{port}'
            resources.callback(private_endpoint.unlink, missing_ok=True)
            private_endpoint.write_text(json.dumps({'url': url, 'token': token}) + '\n')
            private_endpoint.chmod(0o600)
            public = {'url': url, 'worker_addresses': addresses, 'port': port,
                      'worker_hostname': socket.gethostname(), 'rootfs_marker': rootfs_marker,
                      'control_boot_id': control_boot_id, 'control_pid': os.getpid(),
                      'ttl_seconds': args.ttl_seconds, 'agent': agent.info, 'agent_command': command}
            write_json(output / 'endpoint.json', public)
            print(json.dumps({'event': 'server_ready', **public}), flush=True)
            timer = threading.Timer(args.ttl_seconds, server.shutdown)
            timer.daemon = True
            timer.start()
            server.serve_forever(poll_interval=0.2)
    except BaseException as error:
        if not isinstance(error, SystemExit):
            stop_reason = 'error: ' + repr(error)
        raise
    finally:
        print(json.dumps({'event': 'server_stopped', 'reason': stop_reason,
                          'private_endpoint_removed': not private_endpoint.exists()}), flush=True)


if __name__ == '__main__':
    main()
