#!/usr/bin/env python3
"""Start one sandbox for host_server.py: run a command as container root in a private rootfs.

Run as Worker root. A forked child unshares user, mount and IPC namespaces; this parent writes its id maps
(container root -> --host-id, container ids 1..65535 -> --subordinate-base onward), then waits for it. The
child mounts the rootfs, the Worker's /proc and /sys, a private /dev and the read-only --bind directories,
pivots into the rootfs and execs the command as container root, i.e. as an unprivileged Worker uid.

ACP Workers lack CAP_SYS_ADMIN and mask parts of /proc, so no new procfs can be mounted: the sandbox shares
the Worker's PID namespace and sees its /proc, which keeps PIDs and /proc consistent for ps, kill and uv.
Isolation then rests on uids: a sandbox reaches /proc/<pid>/root, environ or signals only for processes of
its own uids, and no process of those uids lives outside its rootfs (unlike Apptainer, whose starter stays
outside the container under the caller's uid). The parent stays Worker root.
"""

import argparse
import ctypes
import os
from pathlib import Path
import sys

MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 1, 2, 4, 8
MS_REMOUNT, MS_NOATIME, MS_NODIRATIME, MS_BIND, MS_REC = 32, 1024, 2048, 4096, 16384
MS_PRIVATE, MS_RELATIME = 1 << 18, 1 << 21
MNT_DETACH = 2
SYS_PIVOT_ROOT = 155
SUBORDINATE_COUNT = 65535
DEVICES = ('null', 'zero', 'full', 'random', 'urandom', 'tty')
# statvfs flag -> mount flag for the flags a less privileged mount namespace may not clear.
LOCKED_FLAGS = ((os.ST_NOSUID, MS_NOSUID), (os.ST_NODEV, MS_NODEV), (os.ST_NOEXEC, MS_NOEXEC),
                (os.ST_NOATIME, MS_NOATIME), (os.ST_NODIRATIME, MS_NODIRATIME), (os.ST_RELATIME, MS_RELATIME))
libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p)


def check(result, action):
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f'{action}: {os.strerror(error)}')


def mount(source, target, filesystem, flags, options=None):
    check(libc.mount(source and os.fsencode(source), os.fsencode(target), filesystem and filesystem.encode(),
                     flags, options and options.encode()), f'mount {source} on {target}')


def bind_read_only(source, target):
    mount(source, target, None, MS_BIND | MS_REC)
    locked = sum(mount_flag for statvfs_flag, mount_flag in LOCKED_FLAGS if os.statvfs(target).f_flag & statvfs_flag)
    mount(None, target, None, MS_REMOUNT | MS_BIND | MS_RDONLY | locked)


def child(args, ready_write, mapped_read):
    os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNS | os.CLONE_NEWIPC)
    os.write(ready_write, b'1')
    if os.read(mapped_read, 1) != b'1':
        os._exit(111)
    rootfs = args.rootfs
    mount(None, '/', None, MS_REC | MS_PRIVATE)
    mount(rootfs, rootfs, None, MS_BIND | MS_REC)
    # Recursive binds keep the Worker's masks over /proc/kcore, /proc/sys and the like.
    mount('/proc', f'{rootfs}/proc', None, MS_BIND | MS_REC)
    bind_read_only('/sys', f'{rootfs}/sys')
    # Bind sources may sit below directories only Worker root can enter, so bind them while still Worker uid 0.
    for bind in args.bind:
        source, target = bind.split(':')
        bind_read_only(source, rootfs + target)
    # Worker uid 0 is unmapped in here and cannot create files on the new tmpfs mounts below; the Worker
    # directories above the rootfs are closed to the slot uid, so continue relative to the rootfs.
    os.chdir(rootfs)
    os.setgroups([])
    os.setresgid(0, 0, 0)
    os.setresuid(0, 0, 0)
    mount('tmpfs', 'dev', 'tmpfs', MS_NOSUID, 'mode=755,size=65536k')
    for name in DEVICES:
        Path(f'dev/{name}').touch()
        mount(f'/dev/{name}', f'dev/{name}', None, MS_BIND)
    os.mkdir('dev/pts')
    mount('devpts', 'dev/pts', 'devpts', MS_NOSUID | MS_NOEXEC, 'newinstance,ptmxmode=0666,mode=0620')
    os.symlink('pts/ptmx', 'dev/ptmx')
    os.mkdir('dev/shm')
    mount('tmpfs', 'dev/shm', 'tmpfs', MS_NOSUID | MS_NODEV, 'mode=1777')
    for index, name in enumerate(('stdin', 'stdout', 'stderr')):
        os.symlink(f'/proc/self/fd/{index}', f'dev/{name}')
    os.symlink('/proc/self/fd', 'dev/fd')
    check(libc.syscall(SYS_PIVOT_ROOT, b'.', b'.'), 'pivot_root')
    check(libc.umount2(b'.', MNT_DETACH), 'detach the old root')
    os.chdir('/')
    environment = dict(item.split('=', 1) for item in args.env)
    os.execve(args.command[0], args.command, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rootfs', required=True, help='writable rootfs with proc, sys, dev and bind mount points')
    parser.add_argument('--host-id', type=int, required=True, help='Worker uid and gid of container root')
    parser.add_argument('--subordinate-base', type=int, required=True, help='Worker id of container id 1')
    parser.add_argument('--bind', action='append', default=[], help='SOURCE:TARGET, read-only; repeatable')
    parser.add_argument('--env', action='append', default=[], help='NAME=VALUE for the command; repeatable')
    parser.add_argument('command', nargs='+')
    args = parser.parse_args()
    ready_read, ready_write = os.pipe()
    mapped_read, mapped_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(ready_read)
            os.close(mapped_write)
            child(args, ready_write, mapped_read)
        except BaseException as error:
            print(f'launch.py: {error!r}', file=sys.stderr, flush=True)
        os._exit(111)
    os.close(ready_write)
    os.close(mapped_read)
    if os.read(ready_read, 1) == b'1':
        mapping = f'0 {args.host_id} 1\n1 {args.subordinate_base} {SUBORDINATE_COUNT}\n'
        for name in ('uid_map', 'gid_map'):
            Path(f'/proc/{pid}/{name}').write_text(mapping)
        os.write(mapped_write, b'1')
    os.close(mapped_write)
    _, status = os.waitpid(pid, 0)
    sys.exit(os.waitstatus_to_exitcode(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status))


if __name__ == '__main__':
    main()
