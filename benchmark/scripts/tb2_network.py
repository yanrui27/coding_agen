#!/usr/bin/env python3
"""Run one Harbor trial in its own network namespace with outbound connectivity.

slirp4netns provides user-mode NAT, without changing host routes or firewall.
Harbor and its Apptainer children share the namespace, so localhost control
requests keep working and privileged task ports belong exclusively to a trial.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / '.cache/tb2-runtime-net/root'


def direct_env():
    env = os.environ.copy()
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    env.update(NO_PROXY='*', no_proxy='*', PIP_PROXY='', PIP_CONFIG_FILE='/dev/null')
    return env


def worker(go_fd, dns_file, command):
    if os.read(go_fd, 1) != b'1':
        raise RuntimeError('network supervisor exited before readiness')
    os.close(go_fd)
    subprocess.run(['ip', 'link', 'set', 'lo', 'up'], check=True)
    # Apptainer fakeroot capabilities belong to a child user namespace. Allow
    # task ports 25/80 in this private network, without touching the host sysctl.
    Path('/proc/sys/net/ipv4/ip_unprivileged_port_start').write_text('0\n')
    # This bind occurs only inside the private mount namespace. Apptainer's
    # existing /etc/resolv.conf bind consequently receives this trial's DNS.
    subprocess.run(['mount', '--bind', dns_file, '/etc/resolv.conf'], check=True)
    os.execvpe(command[0], command, direct_env())


def launch(command, receipt=None):
    binary = NET / 'usr/bin/slirp4netns'
    if not binary.is_file():
        raise RuntimeError(f'missing isolated networking runtime: {binary}')
    env = direct_env()
    lib = str(NET / 'usr/lib/x86_64-linux-gnu')
    helper_env = dict(env, LD_LIBRARY_PATH=lib + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''))
    with tempfile.TemporaryDirectory(prefix='tb2-net-') as temp:
        dns = Path(temp) / 'resolv.conf'
        dns.write_text('nameserver 10.0.2.3\noptions timeout:3 attempts:2\n')
        go_r, go_w = os.pipe()
        ready_r, ready_w = os.pipe()
        child = helper = None
        previous_handlers = {}

        def stop(signum, frame):
            if child is not None and child.poll() is None:
                child.send_signal(signum)
            raise SystemExit(128 + signum)

        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, stop)
            child = subprocess.Popen(['unshare', '--net', '--mount', '--propagation', 'private',
                sys.executable, str(Path(__file__).resolve()), '--worker', str(go_r), str(dns), '--', *command],
                env=env, pass_fds=(go_r,))
            os.close(go_r)
            go_r = -1
            original_ns = os.readlink('/proc/self/ns/net')
            deadline = time.monotonic() + 10
            while os.readlink(f'/proc/{child.pid}/ns/net') == original_ns:
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('private network namespace did not become ready')
                time.sleep(0.02)
            # Host loopback is disabled: task localhost never forwards to an
            # unrelated host service. Normal public/internal egress uses NAT.
            helper = subprocess.Popen([str(binary), '--configure', '--mtu=1500',
                '--disable-host-loopback', '--ready-fd', str(ready_w), str(child.pid), 'tap0'],
                env=helper_env, pass_fds=(ready_w,), stdout=subprocess.DEVNULL)
            os.close(ready_w)
            ready_w = -1
            if not select.select([ready_r], [], [], 15)[0] or os.read(ready_r, 1) != b'1':
                raise RuntimeError('slirp4netns did not report readiness')
            if receipt:
                data = {'network_mode': 'per-trial namespace with slirp4netns NAT',
                    'parent_network_namespace': original_ns,
                    'trial_network_namespace': os.readlink(f'/proc/{child.pid}/ns/net'),
                    'dns': '10.0.2.3', 'host_loopback_forwarding': False,
                    'runtime': str(binary), 'command': command}
                path = Path(receipt)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, indent=2) + '\n')
            os.write(go_w, b'1')
            os.close(go_w)
            go_w = -1
            return child.wait()
        finally:
            for fd in (go_r, go_w, ready_r, ready_w):
                if fd >= 0:
                    os.close(fd)
            for process in (child, helper):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(int(sys.argv[2]), sys.argv[3], sys.argv[5:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a command after --')
    raise SystemExit(launch(command, args.receipt))


if __name__ == '__main__':
    main()
