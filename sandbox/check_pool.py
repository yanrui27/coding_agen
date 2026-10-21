#!/usr/bin/env python3
"""Check the controller and its hosts end to end: concurrent leases, isolation, tokens, renew, release, expiry.

Creates --count sandboxes at once through the controller, so with fewer free slots it also exercises scaling.
Each sandbox writes its own marker through a command and through the file API and must read back only its own;
a sandbox token must not reach another sandbox or the host API. Every sandbox is released at the end. With
--expiry-seconds, one more sandbox is leased for that long, never renewed, and must disappear on its own.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import traceback

from client import Controller, request


def expect_http_error(code, function):
    try:
        function()
    except RuntimeError as error:
        if f'HTTP {code}' in str(error):
            return True
        raise
    raise AssertionError(f'expected HTTP {code}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path, required=True)
    parser.add_argument('--image', action='append', required=True, help='repeatable; sandboxes cycle through them')
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--wait-seconds', type=float, default=1200)
    parser.add_argument('--expiry-seconds', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    controller = Controller.from_file(args.controller)
    checks = {}
    report = {'complete': False, 'checks': checks, 'sandboxes': []}
    sandboxes = []

    def lease(sandbox_id):
        return request(controller.url, controller.token, 'GET', f'/sandboxes/{sandbox_id}')

    def save():
        args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')

    def create(index):
        started = time.monotonic()
        sandbox = controller.create(args.image[index % len(args.image)], ttl_seconds=1800,
                                    wait_seconds=args.wait_seconds, label=f'check-{index}')
        sandbox.client_seconds = round(time.monotonic() - started, 2)
        return sandbox

    def exercise(sandbox):
        marker = sandbox.sandbox_id
        job = sandbox.run(f'printf %s {marker} > /tmp/marker && sleep 1 && cat /tmp/marker && id -u', timeout=60)
        assert job['returncode'] == 0 and job['stdout'].startswith(marker), job
        payload = (marker * 5000).encode()  # several 32 KiB chunks
        sandbox.write_file('/tmp/file-api/marker.bin', payload)
        assert sandbox.read_file('/tmp/file-api/marker.bin') == payload
        assert sandbox.run('cat /tmp/marker', timeout=30)['stdout'] == marker
        return {'sandbox_id': marker, 'host': sandbox.info['host'], 'image': sandbox.info['image'],
                'client_seconds': sandbox.client_seconds, 'wait_seconds': sandbox.info['wait_seconds'],
                'create_seconds': sandbox.info['create_seconds'], 'uid_inside': job['stdout'][len(marker):].strip()}

    try:
        started = time.monotonic()
        with ThreadPoolExecutor(args.count) as pool:
            futures = [pool.submit(create, index) for index in range(args.count)]
        # Keep every sandbox that was created, so the finally block releases them even if another failed.
        sandboxes.extend(future.result() for future in futures if not future.exception())
        for future in futures:
            future.result()
        report['all_created_seconds'] = round(time.monotonic() - started, 2)
        checks['concurrent_create'] = len(sandboxes) == args.count
        save()
        with ThreadPoolExecutor(args.count) as pool:
            report['sandboxes'] = list(pool.map(exercise, sandboxes))
        checks['commands_and_files_isolated'] = True
        report['hosts_used'] = sorted({item['host'] for item in report['sandboxes']})
        save()

        first, second = sandboxes[0], sandboxes[-1]
        host_url = first.url.rsplit('/sandboxes/', 1)[0]
        if len(sandboxes) > 1:
            expect_http_error(401, lambda: request(second.url, first.token, 'GET', '/health'))
            checks['sandbox_token_cannot_reach_other_sandbox'] = True
        expect_http_error(401, lambda: request(host_url, first.token, 'GET', '/health'))
        expect_http_error(403, lambda: request(first.url, first.token, 'DELETE', ''))
        checks['sandbox_token_cannot_use_host_api'] = True

        before = first.call('GET', '/health')['expires_in_seconds']
        renewed = first.renew(3600)
        after = first.call('GET', '/health')['expires_in_seconds']
        assert renewed['expires_in_seconds'] > 3500 and after > before + 1000, (before, renewed, after)
        checks['renew_extends_controller_and_host_lease'] = True
        report['status_before_release'] = controller.status()
        save()

        for sandbox in sandboxes:
            sandbox.release()
        released = list(sandboxes)
        sandboxes.clear()
        for sandbox in released:
            expect_http_error(404, lambda: lease(sandbox.sandbox_id))
            try:
                sandbox.call('GET', '/health')
                raise AssertionError('released sandbox still answers')
            except RuntimeError as error:
                assert 'HTTP 401' in str(error) or 'HTTP 404' in str(error), error
        checks['release_removes_sandbox'] = True
        save()

        if args.expiry_seconds:
            sandbox = controller.create(args.image[0], ttl_seconds=args.expiry_seconds, wait_seconds=args.wait_seconds,
                                        label='check-expiry')
            sandboxes.append(sandbox)
            deadline = time.monotonic() + args.expiry_seconds + 60
            while time.monotonic() < deadline:
                time.sleep(3)
                try:
                    lease(sandbox.sandbox_id)
                except RuntimeError as error:
                    if 'HTTP 404' in str(error):
                        break
            expect_http_error(404, lambda: lease(sandbox.sandbox_id))
            sandboxes.clear()
            checks['unrenewed_lease_expires'] = True
        report['status_after'] = controller.status()
        report['complete'] = True
    except BaseException as error:
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        for sandbox in sandboxes:
            try:
                sandbox.release()
            except Exception:
                pass
        save()
        print(json.dumps({'complete': report['complete'], 'checks': checks,
                          'hosts_used': report.get('hosts_used')}, indent=1))


if __name__ == '__main__':
    main()
