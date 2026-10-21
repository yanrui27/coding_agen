#!/usr/bin/env python3
"""Sandbox controller (layer 2): lease sandboxes to clients and keep a pool of sandbox hosts on ACP.

A client asks for a sandbox by image (POST /sandboxes). The controller picks a ready host with a free slot,
preferring one that already unpacked that image, creates the sandbox there and returns its direct URL and
token, so command and file traffic goes straight to the host (host_server.py), never through here. Leases
expire unless renewed. While requests wait for capacity the controller submits more host jobs (ACP CPU
TrainingJobs) through the sensecore-acp-training-job CLI, up to --max-hosts; hosts without sandboxes for
--host-idle-seconds are deleted down to --min-hosts. After MAX_START_FAILURES host jobs in a row end before
becoming ready, no more are submitted, and while no host is ready waiting requests fail at once. On SIGTERM/SIGINT every host job it manages is
deleted (unless --keep-hosts-on-exit). State lives in --state-dir, so a restarted controller adopts the
hosts and leases that are still alive. The URL and token go to <state-dir>/controller.private.json.
"""

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

import images
from host_server import PROXY_VARIABLES, PYTHON_RUNTIME, SANDBOX_DIR

ACP_CLI = Path('/mnt/data/rui.yan/skills/sensecore-acp-training-job/scripts/acp.py')
ACP_ACTIVE_STATES = {'WAITING', 'INIT', 'QUEUEING', 'CREATING', 'STARTING', 'RUNNING', 'RESTARTING',
                     'SUSPENDING', 'RECOVERING'}
ACP_IMAGE = ('registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-'
             'torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image')
ACP_RESOURCE_POOL = {'name': 'kunlun-intel6348', 'zone': 'cn-sh-01a', 'vpc_id': '019ce515-b557-7028-b0c3-786e81826b87'}
# The host keeps a sandbox slightly longer than its lease, so the controller stays the one that reaps it.
HOST_LEASE_GRACE_SECONDS = 60
ORPHAN_GRACE_SECONDS = 120
MAX_START_FAILURES = 3
MAX_CREATE_FAILURES = 3
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class HostError(Exception):
    def __init__(self, status, detail):
        super().__init__(f'HTTP {status}: {detail}')
        self.status = status


class ClientError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def call(url, token, method, path, body=None, timeout=10):
    """One request to a host. Raises HostError for HTTP errors and OSError/URLError for transport failures."""
    request = urllib.request.Request(url + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
                                              'Connection': 'close'})
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise HostError(error.code, error.read(2048).decode(errors='replace')) from error


def acp(*arguments, timeout=120):
    # The skill CLI loads its own credentials; they never pass through this process's arguments or output.
    result = subprocess.run([sys.executable, str(ACP_CLI), *arguments], capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def acp_state(name):
    code, output = acp('get', name, '--json', timeout=60)
    if code == 0:
        return json.loads(output[output.index('{'):])['job']['state']
    if 'HTTP 404' in output:
        return 'NOT_FOUND'
    raise RuntimeError(f'acp get {name} failed: {output[-300:]}')


def log(event, **fields):
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'event': event, **fields}, ensure_ascii=False),
          flush=True)


class Pool:
    def __init__(self, args):
        self.args = args
        self.state_dir = args.state_dir
        self.hosts = {}
        self.sandboxes = {}
        self.waiting = 0
        self.start_failures = 0
        self.condition = threading.Condition()
        self.save_lock = threading.Lock()
        self.stopping = False

    # ----- persistence -----

    def save(self):
        with self.condition:
            state = {'hosts': {name: {key: host[key] for key in ('managed', 'output_dir', 'created_at')}
                               for name, host in self.hosts.items()},
                     'sandboxes': {sandbox_id: {key: record[key] for key in
                                                ('host', 'image', 'label', 'created_at', 'expires_at')}
                                   for sandbox_id, record in self.sandboxes.items()}}
        path = self.state_dir / 'controller-state.json'
        # A failed save only weakens restart recovery; it must not fail the request that changed the state.
        try:
            with self.save_lock:
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, indent=1) + '\n')
                temporary.replace(path)
        except OSError as error:
            log('state_save_failed', error=repr(error))

    def add_host(self, name, managed, output_dir, created_at):
        self.hosts[name] = {'name': name, 'managed': managed, 'output_dir': output_dir, 'created_at': created_at,
                            'state': 'starting', 'url': None, 'token': None, 'slots': 0, 'reserved': 0,
                            'images': [], 'failures': 0, 'idle_since': None, 'full_until': 0, 'acp_state': None,
                            'acp_checked': 0}

    def load(self):
        path = self.state_dir / 'controller-state.json'
        if path.exists():
            state = json.loads(path.read_text())
            for name, host in state['hosts'].items():
                if host['managed']:
                    self.add_host(name, True, host['output_dir'], host['created_at'])
            for sandbox_id, record in state['sandboxes'].items():
                if record['host'] in self.hosts:
                    self.sandboxes[sandbox_id] = dict(record, id=sandbox_id)
            log('state_loaded', hosts=list(self.hosts), sandboxes=len(self.sandboxes))
        for index, endpoint in enumerate(self.args.static_host):
            self.add_host(f'static-{index}', False, str(endpoint.resolve().parent), time.time())

    # ----- allocation -----

    def free_slots(self, host):
        used = sum(record['host'] == host['name'] for record in self.sandboxes.values())
        return host['slots'] - used - host['reserved']

    def pick(self, image):
        now = time.monotonic()
        candidates = [host for host in self.hosts.values()
                      if host['state'] == 'ready' and host['full_until'] < now and self.free_slots(host) > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda host: (image not in host['images'], -self.free_slots(host)))

    def allocate(self, image, ttl_seconds, wait_seconds, label, binds):
        started = time.monotonic()
        deadline = started + wait_seconds
        attempts = []
        create_failures = 0
        while True:
            with self.condition:
                self.waiting += 1
                self.condition.notify_all()
                try:
                    host = self.pick(image)
                    while host is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or self.stopping:
                            raise ClientError(503, f'No sandbox capacity within {wait_seconds}s; attempts={attempts}')
                        if self.start_failures >= MAX_START_FAILURES and not any(
                                item['state'] == 'ready' for item in self.hosts.values()):
                            raise ClientError(503, f'{self.start_failures} host jobs in a row failed to start; see '
                                                   f'{self.state_dir}/hosts/*/worker.log')
                        self.condition.wait(min(remaining, 2))
                        host = self.pick(image)
                finally:
                    self.waiting -= 1
                host['reserved'] += 1
                waited = time.monotonic() - started
            try:
                body = {'image': image, 'ttl_seconds': ttl_seconds + HOST_LEASE_GRACE_SECONDS, 'binds': binds}
                result = call(host['url'], host['token'], 'POST', '/sandboxes', body, timeout=1800)
            except (HostError, OSError, urllib.error.URLError) as error:
                attempts.append(f'{host["name"]}: {error}'[:300])
                with self.condition:
                    host['reserved'] -= 1
                    if isinstance(error, HostError) and error.status == 409:
                        host['full_until'] = time.monotonic() + 10
                    elif isinstance(error, HostError) and error.status in (400, 422):
                        raise ClientError(error.status, str(error)) from error
                    else:
                        host['failures'] += 1
                        create_failures += 1
                        # A sandbox that fails to start fails fast and the same way again; give up after a few.
                        if create_failures >= MAX_CREATE_FAILURES:
                            raise ClientError(503, f'Sandbox creation failed; attempts={attempts}') from error
                    self.condition.notify_all()
                continue
            now = time.time()
            record = {'id': result['sandbox_id'], 'host': host['name'], 'image': image, 'label': label,
                      'created_at': now, 'expires_at': now + ttl_seconds}
            with self.condition:
                host['reserved'] -= 1
                host['idle_since'] = None
                if image not in host['images']:
                    host['images'].append(image)
                self.sandboxes[record['id']] = record
                url = host['url']
            self.save()
            log('sandbox_created', sandbox_id=record['id'], host=host['name'], image=image, label=label,
                wait_seconds=round(waited, 2), create_seconds=result['create_seconds'])
            return dict(self.describe(record), url=f'{url}/sandboxes/{record["id"]}', token=result['token'],
                        wait_seconds=round(waited, 2), create_seconds=result['create_seconds'])

    def describe(self, record):
        return {'sandbox_id': record['id'], 'host': record['host'], 'image': record['image'], 'label': record['label'],
                'created_at': record['created_at'], 'expires_at': record['expires_at'],
                'expires_in_seconds': round(record['expires_at'] - time.time(), 1)}

    def release(self, sandbox_id, reason):
        with self.condition:
            record = self.sandboxes.pop(sandbox_id, None)
            host = self.hosts.get(record['host']) if record else None
            self.condition.notify_all()
        if record is None:
            raise ClientError(404, 'sandbox_not_found')
        if host and host['state'] == 'ready':
            try:
                call(host['url'], host['token'], 'DELETE', f'/sandboxes/{sandbox_id}', timeout=60)
            except HostError as error:
                if error.status != 404:
                    log('release_failed', sandbox_id=sandbox_id, error=str(error))
            except (OSError, urllib.error.URLError) as error:
                log('release_failed', sandbox_id=sandbox_id, error=repr(error))
        self.save()
        log('sandbox_released', sandbox_id=sandbox_id, host=record['host'], reason=reason)
        return self.describe(record)

    def renew(self, sandbox_id, ttl_seconds):
        with self.condition:
            record = self.sandboxes.get(sandbox_id)
            if record is None:
                raise ClientError(404, 'sandbox_not_found')
            host = self.hosts[record['host']]
        call(host['url'], host['token'], 'POST', f'/sandboxes/{sandbox_id}/renew',
             {'ttl_seconds': ttl_seconds + HOST_LEASE_GRACE_SECONDS}, timeout=30)
        with self.condition:
            record['expires_at'] = time.time() + ttl_seconds
        self.save()
        return self.describe(record)

    # ----- host pool -----

    def host_payload(self, name, output_dir):
        args = self.args
        command = [str(PYTHON_RUNTIME / 'bin/python3.12'), str(SANDBOX_DIR / 'host_server.py'),
                   '--output-dir', output_dir, '--image-cache', str(args.image_cache),
                   '--slots', str(args.slots_per_host), '--idle-exit-seconds', str(args.host_idle_exit_seconds),
                   '--ttl-seconds', str(args.host_ttl_seconds)]
        for root in args.bind_root:
            command.extend(['--bind-root', str(root)])
        script = '\n'.join([
            'set -uo pipefail', 'unset ' + ' '.join(PROXY_VARIABLES),
            "export NO_PROXY='*' no_proxy='*' PYTHONUNBUFFERED=1",
            'exec ' + shlex.join(command) + ' > ' + shlex.quote(output_dir + '/worker.log') + ' 2>&1', ''])
        return {'name': name, 'display_name': name, 'framework': 'PYTORCH',
                'roles': [{'name': 'Worker', 'resource_spec': [{'name': args.host_spec}], 'total_replicas': 1,
                           'image_path': ACP_IMAGE, 'startup_script': 'bash -c ' + shlex.quote(script)}],
                'resource_pool': ACP_RESOURCE_POOL, 'scheduling': {'priority': 'HIGHEST', 'quota_type': 'RESERVED'},
                'root_mapping': True, 'fault_tolerance': {'backoff_limit': 0}}

    def submit_host(self):
        name = f'{self.args.job_prefix}-{time.strftime("%m%d-%H%M%S")}-{uuid.uuid4().hex[:4]}'
        output_dir = self.state_dir / 'hosts' / name
        output_dir.mkdir(parents=True)
        payload = output_dir / 'payload.json'
        payload.write_text(json.dumps(self.host_payload(name, str(output_dir)), indent=1, ensure_ascii=False) + '\n')
        with self.condition:
            self.add_host(name, True, str(output_dir), time.time())
        self.save()
        try:
            code, output = acp('submit', '--payload', str(payload), '--confirm')
        except subprocess.TimeoutExpired:
            code, output = 1, 'timeout'
        # POST is never retried: an ambiguous result is settled by looking the exact name up.
        state = 'SUBMITTED' if code == 0 else acp_state(name)
        if state == 'NOT_FOUND':
            with self.condition:
                self.hosts.pop(name, None)
                self.condition.notify_all()
            self.save()
            log('host_submit_failed', host=name, output=output[-500:])
            return
        log('host_submitted', host=name, spec=self.args.host_spec, slots=self.args.slots_per_host)

    def delete_host(self, host, reason):
        with self.condition:
            self.hosts.pop(host['name'], None)
            failed_start = host['managed'] and host['state'] == 'starting' and not self.stopping
            self.start_failures += failed_start
            dropped = [sandbox_id for sandbox_id, record in self.sandboxes.items() if record['host'] == host['name']]
            for sandbox_id in dropped:
                del self.sandboxes[sandbox_id]
            self.condition.notify_all()
        if host['managed']:
            code, output = acp('delete', host['name'], '--confirm')
            if code != 0 and 'HTTP 404' not in output:
                log('host_delete_failed', host=host['name'], output=output[-500:])
        self.save()
        log('host_deleted', host=host['name'], reason=reason, dropped_sandboxes=dropped)
        if failed_start and self.start_failures == MAX_START_FAILURES:
            log('host_submission_stopped', start_failures=self.start_failures)

    def check_host(self, host):
        """Health-check one host without holding the lock; returns its sandbox list when reachable."""
        now = time.monotonic()
        if host['state'] == 'starting':
            endpoint = Path(host['output_dir']) / 'endpoint.private.json'
            if endpoint.exists():
                credentials = json.loads(endpoint.read_text())
                host['url'], host['token'] = credentials['url'], credentials['token']
        if host['url']:
            try:
                health = call(host['url'], host['token'], 'GET', '/health', timeout=5)
                with self.condition:
                    if host['state'] != 'ready':
                        log('host_ready', host=host['name'], url=host['url'], slots=health['slots'],
                            startup_seconds=round(time.time() - host['created_at'], 1))
                    host.update(state='ready', slots=health['slots'], images=health['images'], failures=0)
                    self.start_failures = 0
                    self.condition.notify_all()
                return health['sandboxes']
            except (HostError, OSError, urllib.error.URLError, ValueError) as error:
                host['failures'] += 1
                host['last_error'] = repr(error)[:300]
        if host['managed'] and (host['failures'] >= 3 or host['state'] == 'starting') and now - host['acp_checked'] > 15:
            host['acp_checked'] = now
            try:
                host['acp_state'] = acp_state(host['name'])
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                log('acp_query_failed', host=host['name'], error=str(error)[:300])
            if host['acp_state'] and host['acp_state'] not in ACP_ACTIVE_STATES:
                self.delete_host(host, f'acp_state={host["acp_state"]}')
            elif host['state'] == 'starting' and time.time() - host['created_at'] > self.args.host_start_timeout:
                self.delete_host(host, 'start_timeout')
            elif host['state'] == 'ready' and host['failures'] >= 3:
                self.delete_host(host, 'unreachable: ' + host.get('last_error', ''))
        elif not host['managed'] and host['failures'] >= 3:
            with self.condition:
                host['state'] = 'starting'
        return None

    def reconcile(self):
        with self.condition:
            hosts = list(self.hosts.values())
        for host in hosts:
            if host['name'] not in self.hosts:
                continue
            reported = self.check_host(host)
            if reported is None:
                continue
            on_host = {item['sandbox_id']: item for item in reported}
            with self.condition:
                missing = [sandbox_id for sandbox_id, record in self.sandboxes.items()
                           if record['host'] == host['name'] and sandbox_id not in on_host
                           and time.time() - record['created_at'] > ORPHAN_GRACE_SECONDS]
                for sandbox_id in missing:
                    del self.sandboxes[sandbox_id]
                orphans = [sandbox_id for sandbox_id, item in on_host.items() if sandbox_id not in self.sandboxes
                           and time.time() - item['created_at'] > ORPHAN_GRACE_SECONDS]
            for sandbox_id in missing:
                log('sandbox_lost', sandbox_id=sandbox_id, host=host['name'])
            for sandbox_id in orphans:
                try:
                    call(host['url'], host['token'], 'DELETE', f'/sandboxes/{sandbox_id}', timeout=60)
                    log('orphan_deleted', sandbox_id=sandbox_id, host=host['name'])
                except (HostError, OSError, urllib.error.URLError) as error:
                    log('orphan_delete_failed', sandbox_id=sandbox_id, error=repr(error)[:300])

        with self.condition:
            expired = [sandbox_id for sandbox_id, record in self.sandboxes.items() if record['expires_at'] < time.time()]
        for sandbox_id in expired:
            try:
                self.release(sandbox_id, 'lease_expired')
            except ClientError:
                pass

        with self.condition:
            managed = [host for host in self.hosts.values() if host['managed']]
            free = sum(max(self.free_slots(host), 0) for host in self.hosts.values() if host['state'] == 'ready')
            free += sum(self.args.slots_per_host for host in managed if host['state'] == 'starting')
            shortage = max(self.waiting - free, 0)
            wanted = max(math.ceil(shortage / self.args.slots_per_host), self.args.min_hosts - len(managed))
            new_hosts = max(min(wanted, self.args.max_hosts - len(managed)), 0)
            if self.stopping or self.start_failures >= MAX_START_FAILURES:
                new_hosts = 0
            idle_victims = []
            now = time.monotonic()
            for host in managed:
                busy = host['reserved'] or any(record['host'] == host['name'] for record in self.sandboxes.values())
                if host['state'] != 'ready' or busy or self.waiting:
                    host['idle_since'] = None
                elif host['idle_since'] is None:
                    host['idle_since'] = now
                elif now - host['idle_since'] > self.args.host_idle_seconds:
                    idle_victims.append(host)
            idle_victims = idle_victims[:max(len(managed) - self.args.min_hosts, 0)]
        for _ in range(new_hosts):
            self.submit_host()
        for host in idle_victims:
            self.delete_host(host, 'idle')

    def run(self):
        while not self.stopping:
            try:
                self.reconcile()
            except Exception as error:  # the loop must survive one bad round
                log('reconcile_error', error=repr(error)[:500])
            with self.condition:
                self.condition.wait(3)

    def status(self):
        with self.condition:
            hosts = [{key: host.get(key) for key in ('name', 'managed', 'state', 'url', 'slots', 'reserved', 'images',
                                                     'failures', 'acp_state', 'created_at', 'last_error')}
                     | {'free_slots': self.free_slots(host) if host['state'] == 'ready' else 0,
                        'sandboxes': sum(record['host'] == host['name'] for record in self.sandboxes.values())}
                     for host in self.hosts.values()]
            return {'waiting': self.waiting, 'hosts': hosts,
                    'sandboxes': [self.describe(record) for record in self.sandboxes.values()]}


def make_handler(pool, token, boot_id, started):
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

        def route(self, method):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                return self.respond(401, {'error': 'unauthorized'})
            parts = [part for part in self.path.split('?')[0].split('/') if part]
            try:
                data = {}
                if method == 'POST':
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 65536:
                        raise ValueError('body must be 1..65536 bytes')
                    data = json.loads(self.rfile.read(size))
                if method == 'GET' and parts == ['health']:
                    self.respond(200, {'controller_boot_id': boot_id, 'uptime_seconds': time.monotonic() - started,
                                       'hosts': len(pool.hosts), 'sandboxes': len(pool.sandboxes)})
                elif method == 'GET' and parts == ['status']:
                    self.respond(200, pool.status())
                elif method == 'POST' and parts == ['sandboxes']:
                    image = data['image']
                    if not isinstance(image, str) or not image:
                        raise ValueError('image must be a nonempty string')
                    if not (pool.args.image_cache / images.image_key(image) / 'image.json').exists():
                        raise ClientError(422, f'image_unavailable: {image} is not in {pool.args.image_cache}')
                    binds = data.get('binds', [])
                    if not all(isinstance(bind, dict) and set(bind) == {'source', 'target'} for bind in binds):
                        raise ValueError('binds must be a list of {"source", "target"}')
                    self.respond(200, pool.allocate(image, int(data.get('ttl_seconds', 3600)),
                                                    float(data.get('wait_seconds', 900)), str(data.get('label', '')),
                                                    binds))
                elif method == 'GET' and len(parts) == 2 and parts[0] == 'sandboxes':
                    with pool.condition:
                        record = pool.sandboxes.get(parts[1])
                    if record is None:
                        raise ClientError(404, 'sandbox_not_found')
                    self.respond(200, pool.describe(record))
                elif method == 'POST' and len(parts) == 3 and parts[0] == 'sandboxes' and parts[2] == 'renew':
                    self.respond(200, pool.renew(parts[1], int(data['ttl_seconds'])))
                elif method == 'DELETE' and len(parts) == 2 and parts[0] == 'sandboxes':
                    self.respond(200, pool.release(parts[1], 'released'))
                else:
                    self.respond(404, {'error': 'not_found'})
            except ClientError as error:
                self.respond(error.status, {'error': str(error)})
            except (KeyError, TypeError, ValueError) as error:
                self.respond(400, {'error': repr(error)})
            except (HostError, OSError, urllib.error.URLError) as error:
                self.respond(503, {'error': repr(error)[:500]})

        def do_GET(self):
            self.route('GET')

        def do_POST(self):
            self.route('POST')

        def do_DELETE(self):
            self.route('DELETE')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--image-cache', type=Path, default=images.DEFAULT_CACHE)
    parser.add_argument('--host-spec', default='N3lS.Ii.I60.32c128g')
    parser.add_argument('--slots-per-host', type=int, default=8)
    parser.add_argument('--min-hosts', type=int, default=0)
    parser.add_argument('--max-hosts', type=int, default=8)
    parser.add_argument('--host-idle-seconds', type=int, default=600, help='delete a host this long without sandboxes')
    parser.add_argument('--host-start-timeout', type=int, default=1200)
    parser.add_argument('--host-idle-exit-seconds', type=int, default=900,
                        help='a host exits by itself after this long without sandboxes or controller requests')
    parser.add_argument('--host-ttl-seconds', type=int, default=86400)
    parser.add_argument('--job-prefix', default='sandbox-host')
    parser.add_argument('--static-host', type=Path, action='append', default=[],
                        help='endpoint.private.json of a host started by hand; never created or deleted here')
    parser.add_argument('--keep-hosts-on-exit', action='store_true')
    parser.add_argument('--bind-root', type=Path, action='append', default=[],
                        help='passed to every host: directories sandboxes may bind read-only; repeatable')
    args = parser.parse_args()
    for name in PROXY_VARIABLES:
        os.environ.pop(name, None)
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    pool = Pool(args)
    pool.load()

    token = secrets.token_urlsafe(32)
    boot_id = str(uuid.uuid4())
    server = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(pool, token, boot_id, time.monotonic()))
    server.daemon_threads = True
    address_infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    address = sorted({info[4][0] for info in address_infos if not info[4][0].startswith('127.')})[0]
    url = f'http://{address}:{server.server_address[1]}'
    private_endpoint = args.state_dir / 'controller.private.json'
    private_endpoint.write_text(json.dumps({'url': url, 'token': token}) + '\n')
    private_endpoint.chmod(0o600)
    log('controller_ready', url=url, boot_id=boot_id, pid=os.getpid(), state_dir=str(args.state_dir),
        host_spec=args.host_spec, slots_per_host=args.slots_per_host, min_hosts=args.min_hosts,
        max_hosts=args.max_hosts)

    def stop(signal_number, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        log('controller_stopping', signal=signal_number)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    reconciler = threading.Thread(target=pool.run, daemon=True)
    reconciler.start()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        with pool.condition:
            pool.stopping = True
            pool.condition.notify_all()
        reconciler.join(timeout=30)
        server.server_close()
        private_endpoint.unlink(missing_ok=True)
        if not args.keep_hosts_on_exit:
            for host in list(pool.hosts.values()):
                if host['managed']:
                    pool.delete_host(host, 'controller_stopped')
        log('controller_stopped', kept_hosts=[name for name, host in pool.hosts.items() if host['managed']])


if __name__ == '__main__':
    main()
