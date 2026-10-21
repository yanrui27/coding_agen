#!/usr/bin/env python3
"""Sandbox host (layer 3): run up to --slots sandboxes on one ACP CPU Worker behind one HTTP API.

Each image is unpacked once from the shared image cache (images.py) onto Worker-local disk; every sandbox
gets a private writable copy of that base and its own process_agent.py, started by launch.py as container
root mapped to the slot's own unprivileged Worker uid (see launch.py for why uids carry the isolation). This
server only forwards requests, so user code never runs here. The controller (controller.py) creates, renews and deletes
sandboxes with the host token; a client talks to <host url>/sandboxes/<id> with that sandbox's own token,
using the same API a single sandbox always had. A create request may add read-only binds of directories under
--bind-root (shared tools, not task data). A sandbox whose lease is not renewed is deleted. The host
exits after --idle-exit-seconds without sandboxes or requests, at --ttl-seconds, or on SIGTERM, and removes
every sandbox on the way out. The URL and host token go to <output-dir>/endpoint.private.json.
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
import stat
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import images

SANDBOX_DIR = Path(__file__).resolve().parent
# Only its unsquashfs is used, to unpack SIF images.
APPTAINER_PREFIX = SANDBOX_DIR / 'runtime/apptainer-1.5.4-jammy'
# Read-only Python for process_agent.py, independent of the task image's own Python.
PYTHON_RUNTIME = Path('/mnt/data/rui.yan/venvs/OmniRL/runtime/cpython-3.12.14-linux-x86_64-gnu')
PROXY_VARIABLES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
# One 1 MiB file chunk, base64-encoded inside JSON.
MAX_BODY_BYTES = 4 << 20
# Slot i runs as Worker uid/gid SLOT_HOST_ID + i (container root) and SLOT_SUBORDINATE_BASE + i * 65536 onward
# (container ids 1..65535); both lie above the Worker's own users.
SLOT_HOST_ID, SLOT_SUBORDINATE_BASE, SUBORDINATE_COUNT = 20000, 200000, 65535
AGENT_ENVIRONMENT = ('PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'HOME=/root',
                     'PYTHONHOME=/opt/sandbox-python', 'LD_LIBRARY_PATH=/opt/sandbox-python/lib', 'PYTHONUNBUFFERED=1')


def kill_owned(host_id, subordinate_base):
    """SIGKILL every Worker process running as one of a slot's uids, until none is left."""
    for _ in range(100):
        found = False
        for entry in os.scandir('/proc'):
            if not entry.name.isdigit():
                continue
            try:
                with open(f'/proc/{entry.name}/status') as status:
                    uids = next(line for line in status if line.startswith('Uid:')).split()[1:]
                if any(int(uid) == host_id or 0 <= int(uid) - subordinate_base < SUBORDINATE_COUNT for uid in uids):
                    found = True
                    os.kill(int(entry.name), signal.SIGKILL)
            except (OSError, StopIteration):
                pass
        if not found:
            return
        time.sleep(0.05)
    raise RuntimeError(f'Processes of uid {host_id} survive SIGKILL')


class SandboxAgent:
    """The process_agent.py instance inside the sandbox, driven over stdin/stdout JSON lines."""

    def __init__(self, command, owner_ids, log_path):
        self.command, self.owner_ids, self.log_path = command, owner_ids, log_path
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
            process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
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
        # Detached tasks (daemons, tmux servers) left the agent's process group; the slot's uids catch them.
        kill_owned(*self.owner_ids)
        self.requests.put(None)

    def reset(self):
        # Kills every process of the sandbox; the private rootfs files stay.
        with self.lifecycle:
            self.stop()
            self.start()
            return self.info


class HostFull(Exception):
    pass


class LocalImages:
    """Worker-local unpacked base rootfs per image, built once from the shared cache; idle bases are evicted LRU."""

    def __init__(self, cache, directory, limit, environment):
        self.cache, self.directory, self.limit, self.environment = cache, directory, limit, environment
        self.entries = {}
        self.guard = threading.Lock()

    def acquire(self, image):
        key = images.image_key(image)
        with self.guard:
            entry = self.entries.get(key)
            builder = entry is None
            if builder:
                entry = {'image': image, 'path': self.directory / key, 'users': 0, 'error': None,
                         'ready': threading.Event(), 'last_used': time.monotonic()}
                self.entries[key] = entry
            entry['users'] += 1
        if builder:
            try:
                manifest = images.load(self.cache, image)
                entry['image_env'] = manifest['image_env']
                entry['path'].mkdir(parents=True)
                started = time.monotonic()
                images.unpack(manifest, entry['path'], self.environment)
                entry['unpack_seconds'] = round(time.monotonic() - started, 1)
            except BaseException as error:
                entry['error'] = error
                with self.guard:
                    self.entries.pop(key, None)
                subprocess.run(['rm', '-rf', str(entry['path'])], timeout=600)
                raise
            finally:
                entry['ready'].set()
        entry['ready'].wait()
        if entry['error']:
            raise RuntimeError(f'Preparing image {image} failed: {entry["error"]!r}')
        return entry

    def release(self, entry):
        with self.guard:
            entry['users'] -= 1
            entry['last_used'] = time.monotonic()
            idle = sorted((item for item in self.entries.values() if item['users'] == 0 and item['ready'].is_set()),
                          key=lambda item: item['last_used'])
            victims = idle[:max(0, len(self.entries) - self.limit)]
            for victim in victims:
                del self.entries[images.image_key(victim['image'])]
        for victim in victims:
            subprocess.run(['rm', '-rf', str(victim['path'])], timeout=600)

    def names(self):
        with self.guard:
            return sorted(entry['image'] for entry in self.entries.values() if entry['ready'].is_set())


def shift_owners(rootfs, host_id, subordinate_base):
    """Re-own a rootfs copied with image ids so that the slot's user namespace shows those ids again.

    launch.py maps container id 0 to Worker id host_id and container id n >= 1 to subordinate_base + n - 1;
    a file left at Worker uid 0 or 42 (apt's _apt) would show as nobody in there, closed even to container root.
    """
    for directory, subdirectories, files in os.walk(rootfs):
        for name in subdirectories + files:
            path = os.path.join(directory, name)
            info = os.lstat(path)
            os.lchown(path, subordinate_base + info.st_uid - 1 if info.st_uid else host_id,
                      subordinate_base + info.st_gid - 1 if info.st_gid else host_id)
            # chown clears the setuid/setgid bits of regular files.
            if not stat.S_ISLNK(info.st_mode) and info.st_mode & (stat.S_ISUID | stat.S_ISGID):
                os.chmod(path, stat.S_IMODE(info.st_mode))
    os.lchown(rootfs, host_id, host_id)


def bind_arguments(rootfs, binds, bind_roots):
    """Validate read-only binds and create their mount points inside a fresh rootfs."""
    arguments = []
    for bind in binds:
        source = Path(bind['source']).resolve()
        if not source.exists() or not any(source.is_relative_to(root) for root in bind_roots):
            raise ValueError(f'Bind source {source} is missing or outside the allowed bind roots')
        target = Path(bind['target'])
        if not target.is_absolute() or '..' in target.parts or ':' in str(target) + str(source):
            raise ValueError(f'Bad bind target {target}')
        # Image symlinks are absolute paths inside the image; following one here would leave the rootfs.
        path = rootfs
        for part in target.parts[1:]:
            path = path / part
            if path.is_symlink():
                raise ValueError(f'Bind target {target} crosses a symlink in the image')
            path.mkdir(exist_ok=True)
        arguments.extend(['--bind', f'{source}:{target}'])
    return arguments


class Host:
    def __init__(self, work, output, local_images, slots, min_free_bytes, bind_roots):
        self.work, self.output, self.local_images = work, output, local_images
        self.slots, self.min_free_bytes, self.bind_roots = slots, min_free_bytes, bind_roots
        self.sandboxes = {}
        self.free_slots = list(range(slots))
        self.reserved = 0
        self.guard = threading.Lock()
        self.last_activity = time.monotonic()
        (output / 'sandboxes').mkdir(exist_ok=True)

    def touch(self):
        self.last_activity = time.monotonic()

    def create(self, image, ttl_seconds, binds):
        with self.guard:
            if not self.free_slots:
                raise HostFull(f'All {self.slots} slots are in use')
            if shutil.disk_usage(self.work).free < self.min_free_bytes:
                raise HostFull('Worker-local disk is nearly full')
            slot = self.free_slots.pop(0)
            self.reserved += 1
        owner_ids = (SLOT_HOST_ID + slot, SLOT_SUBORDINATE_BASE + slot * (SUBORDINATE_COUNT + 1))
        started = time.monotonic()
        sandbox_id = 'sbx-' + uuid.uuid4().hex[:16]
        directory = self.work / 'sandboxes' / sandbox_id
        rootfs = directory / 'rootfs'
        agent = None
        try:
            base = self.local_images.acquire(image)
            try:
                rootfs.mkdir(parents=True)
                subprocess.run(['cp', '-a', '--reflink=auto', f'{base["path"]}/.', str(rootfs)], check=True, timeout=1800)
            finally:
                self.local_images.release(base)
            # Like Docker, give the sandbox the Worker's resolver and hosts files; images ship empty placeholders.
            for name in ('resolv.conf', 'hosts'):
                target = rootfs / 'etc' / name
                target.unlink(missing_ok=True)
                shutil.copyfile(f'/etc/{name}', target)
            control_dir = rootfs / 'opt/sandbox-control'
            control_dir.mkdir(parents=True, exist_ok=True)
            for mount_point in ('proc', 'sys', 'dev', 'opt/sandbox-python'):
                (rootfs / mount_point).mkdir(parents=True, exist_ok=True)
            shutil.copy2(SANDBOX_DIR / 'process_agent.py', control_dir / 'process_agent.py')
            (control_dir / 'image.json').write_text(json.dumps({'image': image, 'image_env': base['image_env']}))
            rootfs_marker = 'sandbox-' + str(uuid.uuid4())
            (control_dir / 'rootfs-marker').write_text(rootfs_marker)
            command = [sys.executable, str(SANDBOX_DIR / 'launch.py'), '--rootfs', str(rootfs),
                       '--host-id', str(owner_ids[0]), '--subordinate-base', str(owner_ids[1]),
                       '--bind', f'{PYTHON_RUNTIME}:/opt/sandbox-python', *bind_arguments(rootfs, binds, self.bind_roots)]
            for variable in AGENT_ENVIRONMENT:
                command.extend(['--env', variable])
            command.extend(['/opt/sandbox-python/bin/python3.12', '/opt/sandbox-control/process_agent.py'])
            shift_owners(rootfs, *owner_ids)
            agent = SandboxAgent(command, owner_ids, self.output / 'sandboxes' / f'{sandbox_id}.log')
            agent.start()
            if agent.info['rootfs_marker'] != rootfs_marker:
                raise RuntimeError('Sandbox agent reports a different rootfs')
            if agent.info['gpu_device_visible']:
                raise RuntimeError('GPU device is visible inside the sandbox')
            record = {'id': sandbox_id, 'image': image, 'token': secrets.token_urlsafe(32), 'agent': agent, 'slot': slot,
                      'directory': directory, 'created_at': time.time(), 'expires': time.monotonic() + ttl_seconds,
                      'create_seconds': round(time.monotonic() - started, 2)}
            with self.guard:
                self.sandboxes[sandbox_id] = record
        except BaseException:
            if agent:
                agent.stop()
            subprocess.run(['rm', '-rf', str(directory)], timeout=600)
            with self.guard:
                self.free_slots.append(slot)
            raise
        finally:
            with self.guard:
                self.reserved -= 1
        print(json.dumps({'event': 'sandbox_created', 'sandbox_id': sandbox_id, 'image': image,
                          'create_seconds': record['create_seconds'],
                          'unpack_seconds': base.get('unpack_seconds')}), flush=True)
        return dict(self.describe(record), token=record['token'])

    def describe(self, record):
        return {'sandbox_id': record['id'], 'image': record['image'], 'created_at': record['created_at'],
                'expires_in_seconds': round(record['expires'] - time.monotonic(), 1),
                'create_seconds': record['create_seconds'],
                'agent_process_running': record['agent'].process.poll() is None, 'agent': record['agent'].info}

    def renew(self, sandbox_id, ttl_seconds):
        with self.guard:
            record = self.sandboxes[sandbox_id]
            record['expires'] = time.monotonic() + ttl_seconds
        return self.describe(record)

    def delete(self, sandbox_id, reason):
        with self.guard:
            record = self.sandboxes.pop(sandbox_id, None)
        if record is None:
            return False
        record['agent'].stop()
        subprocess.run(['rm', '-rf', str(record['directory'])], timeout=600)
        with self.guard:
            self.free_slots.append(record['slot'])
        print(json.dumps({'event': 'sandbox_deleted', 'sandbox_id': sandbox_id, 'reason': reason}), flush=True)
        return True

    def expire(self):
        now = time.monotonic()
        with self.guard:
            expired = [sandbox_id for sandbox_id, record in self.sandboxes.items() if record['expires'] < now]
        for sandbox_id in expired:
            self.delete(sandbox_id, 'lease_expired')

    def delete_all(self, reason):
        with self.guard:
            sandbox_ids = list(self.sandboxes)
        for sandbox_id in sandbox_ids:
            self.delete(sandbox_id, reason)

    def status(self):
        with self.guard:
            records = list(self.sandboxes.values())
            used = len(records) + self.reserved
        return {'slots': self.slots, 'used_slots': used, 'free_slots': self.slots - used,
                'disk_free_bytes': shutil.disk_usage(self.work).free, 'images': self.local_images.names(),
                'sandboxes': [self.describe(record) for record in records]}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def make_handler(host, token, control_boot_id, started):
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

        def body(self):
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_BODY_BYTES:
                raise ValueError(f'body must be 1..{MAX_BODY_BYTES} bytes')
            return json.loads(self.rfile.read(size))

        def route(self, method):
            parsed = urlsplit(self.path)
            parts = [part for part in parsed.path.split('/') if part]
            authorization = self.headers.get('Authorization', '')
            host_authorized = hmac.compare_digest(authorization, 'Bearer ' + token)
            try:
                if len(parts) >= 2 and parts[0] == 'sandboxes':
                    record = host.sandboxes.get(parts[1])
                    sandbox_authorized = record and hmac.compare_digest(authorization, 'Bearer ' + record['token'])
                    if not (host_authorized or sandbox_authorized):
                        return self.respond(401, {'error': 'unauthorized'})
                    if record is None:
                        return self.respond(404, {'error': 'sandbox_not_found'})
                    host.touch()
                    rest = '/' + '/'.join(parts[2:])
                    if method == 'DELETE' and rest == '/':
                        if not host_authorized:
                            return self.respond(403, {'error': 'release sandboxes through the controller'})
                        return self.respond(200, {'deleted': host.delete(record['id'], 'released')})
                    if method == 'POST' and rest == '/renew':
                        if not host_authorized:
                            return self.respond(403, {'error': 'renew leases through the controller'})
                        return self.respond(200, host.renew(record['id'], int(self.body()['ttl_seconds'])))
                    return self.sandbox_api(method, record, rest, parse_qs(parsed.query))
                if not host_authorized:
                    return self.respond(401, {'error': 'unauthorized'})
                host.touch()
                if method == 'GET' and parts == ['health']:
                    self.respond(200, dict(host.status(), control_boot_id=control_boot_id, control_pid=os.getpid(),
                                           uptime_seconds=time.monotonic() - started))
                elif method == 'GET' and parts == ['sandboxes']:
                    self.respond(200, {'sandboxes': host.status()['sandboxes']})
                elif method == 'POST' and parts == ['sandboxes']:
                    data = self.body()
                    self.respond(200, host.create(data['image'], int(data.get('ttl_seconds', 3600)),
                                                  data.get('binds', [])))
                else:
                    self.respond(404, {'error': 'not_found'})
            except HostFull as error:
                self.respond(409, {'error': 'host_full', 'detail': str(error)})
            except FileNotFoundError as error:
                self.respond(422, {'error': 'image_unavailable', 'detail': str(error)})
            except (KeyError, TypeError, ValueError) as error:
                self.respond(400, {'error': str(error)})
            except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
                self.respond(503, {'error': str(error)})

        def sandbox_api(self, method, record, path, query):
            agent = record['agent']
            parts = path.split('/')
            if method == 'GET':
                offset = int(query.get('offset', ['0'])[0])
                if path == '/health':
                    # No call into the sandbox: hung user code or a frozen agent cannot block health.
                    self.respond(200, dict(host.describe(record), control_boot_id=control_boot_id,
                                           control_pid=os.getpid(), uptime_seconds=time.monotonic() - started))
                elif path.startswith('/jobs/'):
                    self.respond(200, agent.rpc('job_status', {'job_id': parts[2]}))
                elif path.startswith('/terminals/'):
                    self.respond(200, agent.rpc('terminal_read', {'terminal_id': parts[2], 'offset': offset}))
                elif path == '/files/read':
                    self.respond(200, agent.rpc('file_read', {'path': query.get('path', [''])[0], 'offset': offset}))
                else:
                    self.respond(404, {'error': 'not_found'})
                return
            if method != 'POST':
                return self.respond(404, {'error': 'not_found'})
            data = self.body()
            if path == '/reset':
                self.respond(200, agent.reset())
            elif path == '/commands':
                self.respond(200, agent.rpc('command_start', data))
            elif path == '/sessions':
                self.respond(200, agent.rpc('session_create', data))
            elif path == '/terminals':
                self.respond(200, agent.rpc('terminal_start', data))
            elif path == '/files/write':
                self.respond(200, agent.rpc('file_write', data))
            elif path.startswith('/terminals/') and path.endswith('/write'):
                self.respond(200, agent.rpc('terminal_write', dict(data, terminal_id=parts[2])))
            elif path.startswith('/terminals/') and path.endswith('/close'):
                self.respond(200, agent.rpc('terminal_close', {'terminal_id': parts[2]}))
            elif path.startswith('/sessions/') and path.endswith('/execute'):
                self.respond(200, agent.rpc('session_execute', dict(data, session_id=parts[2])))
            elif path.startswith('/jobs/') and path.endswith('/cancel'):
                self.respond(200, agent.rpc('job_cancel', {'job_id': parts[2]}))
            else:
                self.respond(404, {'error': 'not_found'})

        def do_GET(self):
            self.route('GET')

        def do_POST(self):
            self.route('POST')

        def do_DELETE(self):
            self.route('DELETE')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--image-cache', type=Path, default=images.DEFAULT_CACHE)
    parser.add_argument('--slots', type=int, default=4)
    parser.add_argument('--local-images', type=int, default=16, help='unpacked image bases kept on local disk')
    parser.add_argument('--min-free-gib', type=float, default=20)
    parser.add_argument('--idle-exit-seconds', type=int, default=1800)
    parser.add_argument('--ttl-seconds', type=int, default=43200, help='lifetime of the whole host')
    parser.add_argument('--bind-root', type=Path, action='append', default=[],
                        help='directory whose subdirectories create requests may bind read-only; repeatable')
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
        with contextlib.ExitStack() as resources:
            work = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix='sandbox-host-')))
            local_images = LocalImages(args.image_cache, work / 'images', args.local_images, environment)
            host = Host(work, output, local_images, args.slots, int(args.min_free_gib * (1 << 30)),
                        [root.resolve() for root in args.bind_root])
            # Stop every sandbox process before the private rootfs copies are deleted.
            resources.callback(lambda: host.delete_all('host_stopped'))

            token = secrets.token_urlsafe(32)
            started = time.monotonic()
            server = ThreadingHTTPServer(('0.0.0.0', 0), make_handler(host, token, control_boot_id, started))
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
            public = {'url': url, 'worker_addresses': addresses, 'port': port, 'worker_hostname': socket.gethostname(),
                      'control_boot_id': control_boot_id, 'control_pid': os.getpid(), 'slots': args.slots,
                      'image_cache': str(args.image_cache), 'idle_exit_seconds': args.idle_exit_seconds,
                      'ttl_seconds': args.ttl_seconds}
            write_json(output / 'endpoint.json', public)
            print(json.dumps({'event': 'server_ready', **public}), flush=True)

            def maintain():
                nonlocal stop_reason
                while True:
                    time.sleep(2)
                    host.expire()
                    if time.monotonic() - started > args.ttl_seconds:
                        stop_reason = 'ttl'
                        break
                    if not host.sandboxes and time.monotonic() - host.last_activity > args.idle_exit_seconds:
                        stop_reason = 'idle'
                        break
                server.shutdown()

            threading.Thread(target=maintain, daemon=True).start()
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
