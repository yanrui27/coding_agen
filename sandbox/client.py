#!/usr/bin/env python3
"""Client for the sandbox controller and for a leased sandbox (standard library only).

    controller = Controller.from_file('<state-dir>/controller.private.json')
    with controller.create('docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001661') as sandbox:
        result = sandbox.run('python -m pytest -q', cwd='/workspace/repo', timeout=600)
        sandbox.write_file('/tmp/patch.diff', patch_bytes)

Leaving the `with` block releases the sandbox. Commands and files go straight to the sandbox host; only
create, renew and release go through the controller. The CLI's `create --endpoint-out FILE` writes the
{url, token} file that mimo-rl's remote_apptainer backend and check_control.py read.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

CHUNK_BYTES = 1 << 20
opener = build_opener(ProxyHandler({}))


def request(url, token, method, path, body=None, timeout=30):
    call = Request(url + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                   headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
                            'Connection': 'close'})
    try:
        with opener.open(call, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f'{method} {path}: HTTP {error.code} {error.read(2048).decode(errors="replace")}') from error


class Sandbox:
    def __init__(self, controller, info):
        self.controller, self.info = controller, info
        self.sandbox_id, self.url, self.token = info['sandbox_id'], info['url'], info['token']

    def call(self, method, path, body=None, timeout=30):
        return request(self.url, self.token, method, path, body, timeout)

    def run(self, command, cwd='/', env=None, timeout=300, poll_interval=0.2):
        """Run a bash command to completion; returns the job (state, returncode, stdout/stderr tails)."""
        job = self.call('POST', '/commands', {'command': command, 'cwd': cwd, 'env': env or {},
                                               'timeout_seconds': timeout})
        deadline = time.monotonic() + timeout + 30
        while job['state'] == 'running':
            if time.monotonic() > deadline:
                self.call('POST', f'/jobs/{job["job_id"]}/cancel', {})
                raise TimeoutError(f'Command did not finish after {timeout}s')
            time.sleep(poll_interval)
            job = self.call('GET', f'/jobs/{job["job_id"]}')
        return job

    def write_file(self, path, data, mode=0o644):
        offset = 0
        while True:
            chunk = data[offset:offset + CHUNK_BYTES]
            self.call('POST', '/files/write', {'path': path, 'offset': offset, 'mode': mode,
                                                'data_base64': base64.b64encode(chunk).decode()})
            offset += len(chunk)
            if offset >= len(data):
                return

    def read_file(self, path):
        output, offset = bytearray(), 0
        while True:
            result = self.call('GET', '/files/read?' + urlencode({'path': path, 'offset': offset}))
            output.extend(base64.b64decode(result['data_base64']))
            offset = result['offset']
            if result['eof']:
                return bytes(output)

    def renew(self, ttl_seconds):
        return self.controller.renew(self.sandbox_id, ttl_seconds)

    def release(self):
        return self.controller.release(self.sandbox_id)

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.release()


class Controller:
    def __init__(self, url, token):
        self.url, self.token = url, token

    @classmethod
    def from_file(cls, path):
        credentials = json.loads(Path(path).read_text())
        return cls(credentials['url'], credentials['token'])

    def create(self, image, ttl_seconds=3600, wait_seconds=900, label='', binds=()):
        """`binds`: [{'source': shared dir under a host --bind-root, 'target': path in the sandbox}], read-only."""
        body = {'image': image, 'ttl_seconds': ttl_seconds, 'wait_seconds': wait_seconds, 'label': label,
                'binds': list(binds)}
        # Creation can wait for a new host job and then unpack the image on it.
        return Sandbox(self, request(self.url, self.token, 'POST', '/sandboxes', body, timeout=wait_seconds + 1900))

    def renew(self, sandbox_id, ttl_seconds):
        return request(self.url, self.token, 'POST', f'/sandboxes/{sandbox_id}/renew', {'ttl_seconds': ttl_seconds})

    def release(self, sandbox_id):
        return request(self.url, self.token, 'DELETE', f'/sandboxes/{sandbox_id}', timeout=90)

    def status(self):
        return request(self.url, self.token, 'GET', '/status')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--controller', type=Path, required=True, help='controller.private.json')
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('--image', required=True)
    create.add_argument('--ttl-seconds', type=int, default=3600)
    create.add_argument('--wait-seconds', type=float, default=900)
    create.add_argument('--label', default='')
    create.add_argument('--endpoint-out', type=Path, required=True)
    for name in ('release', 'renew'):
        command = commands.add_parser(name)
        command.add_argument('sandbox_id')
        if name == 'renew':
            command.add_argument('--ttl-seconds', type=int, default=3600)
    commands.add_parser('status')
    args = parser.parse_args()
    controller = Controller.from_file(args.controller)
    if args.command == 'create':
        sandbox = controller.create(args.image, args.ttl_seconds, args.wait_seconds, args.label)
        args.endpoint_out.write_text(json.dumps({'url': sandbox.url, 'token': sandbox.token,
                                                 'sandbox_id': sandbox.sandbox_id}) + '\n')
        args.endpoint_out.chmod(0o600)
        print(json.dumps({key: value for key, value in sandbox.info.items() if key != 'token'}, indent=1))
    elif args.command == 'release':
        print(json.dumps(controller.release(args.sandbox_id), indent=1))
    elif args.command == 'renew':
        print(json.dumps(controller.renew(args.sandbox_id, args.ttl_seconds), indent=1))
    else:
        print(json.dumps(controller.status(), indent=1))


if __name__ == '__main__':
    for variable in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(variable, None)
    main()
