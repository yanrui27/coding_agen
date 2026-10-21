"""Harbor environment backed by a sandbox leased from sandbox/controller.py (Apptainer on ACP CPU Workers).

Load it with environment.import_path = "harbor_acp_sandbox:AcpSandboxEnvironment" (benchmark/scripts on
PYTHONPATH) and environment.kwargs.controller_file = <controller.private.json>. The task's docker_image must
be registered in the controller's image cache (sandbox/images.py add --rootfs <its SIF>).

Each trial gets what eval_tb2.py and the singularity backend give it on real-241: the task image, the shared
tools in .cache/harbor-assets bound read-only at /opt/harbor-deps, the pinned CA bundle, apt sources moved to
mirrors.aliyun.com, the pip/uv/Hugging Face mirrors in every command's environment, and verifier assets staged
only after the agent has finished. Not reproduced: the per-trial network namespace (sandboxes on one Worker share
its network), CPU/memory caps (ACP cgroups are read-only) and /proc (ACP refuses new procfs mounts).
"""

import asyncio
import base64
import importlib.util
import os
from pathlib import Path
import re
import shlex
import tarfile
import tempfile
import uuid

from harbor.environments.base import BaseEnvironment, ExecResult, transfer_tar_filter
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.environments.singularity.singularity import SingularityEnvironment

BENCHMARK = Path(__file__).resolve().parents[1]
ASSETS = BENCHMARK / '.cache/harbor-assets'
_spec = importlib.util.spec_from_file_location('sandbox_client', BENCHMARK.parent / 'sandbox/client.py')
sandbox_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sandbox_client)

# Same mirrors eval_tb2.py passes to Apptainer as APPTAINERENV_*.
MIRROR_ENV = {
    'DEBIAN_FRONTEND': 'noninteractive',
    'PIP_INDEX_URL': 'https://mirrors.aliyun.com/pypi/simple', 'PIP_PROXY': '', 'PIP_CONFIG_FILE': '/dev/null',
    'PIP_NO_CACHE_DIR': '1',
    'UV_PYTHON_INSTALL_MIRROR': 'https://releases.astral.sh/github/python-build-standalone/releases/download',
    'UV_DEFAULT_INDEX': 'https://mirrors.aliyun.com/pypi/simple',
    'UV_HTTP_TIMEOUT': '180', 'UV_HTTP_RETRIES': '3', 'UV_FIND_LINKS': 'file:///opt/harbor-deps/wheels',
    'HF_ENDPOINT': 'https://hf-mirror.com',
}
APT_MIRRORS = [
    ('http://security.ubuntu.com/ubuntu', 'http://mirrors.aliyun.com/ubuntu'),
    ('https://security.ubuntu.com/ubuntu', 'http://mirrors.aliyun.com/ubuntu'),
    ('http://deb.debian.org/debian', 'http://mirrors.aliyun.com/debian'),
    ('https://deb.debian.org/debian', 'http://mirrors.aliyun.com/debian'),
    ('http://security.debian.org/debian-security', 'http://mirrors.aliyun.com/debian-security'),
    ('https://security.debian.org/debian-security', 'http://mirrors.aliyun.com/debian-security')]
# From bootstrap.sh of the singularity backend: the apt index refresh and the pinned Python 3.9 stdlib repair.
BOOTSTRAP = r'''
if command -v apt-get >/dev/null 2>&1; then apt-get update -qq 2>/dev/null || true; fi
if [ -x /usr/bin/python3 ] \
    && /usr/bin/python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,9) else 1)' \
    && ! /usr/bin/python3 -c 'import distutils.cmd' 2>/dev/null \
    && [ -d /opt/harbor-deps/python39-stdlib/usr/lib/python3.9/distutils ]; then
  mkdir -p /usr/lib/python3.9/distutils /usr/lib/python3.9/lib2to3
  cp -r /opt/harbor-deps/python39-stdlib/usr/lib/python3.9/distutils/. /usr/lib/python3.9/distutils/
  cp -r /opt/harbor-deps/python39-stdlib/usr/lib/python3.9/lib2to3/. /usr/lib/python3.9/lib2to3/
fi
'''
# The process agent inlines only this much of a command's stdout/stderr.
INLINE_OUTPUT_BYTES = 65536
UNBOUNDED_SECONDS = 7 * 86400
LEASE_SECONDS = 3600
RENEW_SECONDS = 600


def apt_sources(text, distro):
    """eval_tb2.py's rewrite of one apt sources file onto mirrors.aliyun.com."""
    text = re.sub(r'https?://(?:[a-z]{2}\.)?archive\.ubuntu\.com/ubuntu/?', 'http://mirrors.aliyun.com/ubuntu/', text)
    for source, target in APT_MIRRORS:
        text = text.replace(source, target)
    if 'ID=debian' in distro and 'VERSION_ID="11"' in distro:
        # Bullseye security packages have moved to the signed Debian archive; live repositories return 404.
        text = text.replace('http://mirrors.aliyun.com/debian-security', 'https://archive.debian.org/debian-security')
        text = text.replace('https://mirrors.aliyun.com/debian-security', 'https://archive.debian.org/debian-security')
    return text


class AcpSandboxEnvironment(BaseEnvironment):
    def __init__(self, environment_dir, environment_name, session_id, trial_paths, task_env_config, logger=None,
                 controller_file=None, wait_seconds=3600, python_hash_seed=None, **kwargs):
        if not controller_file:
            raise ValueError('acp-sandbox environment needs kwargs.controller_file')
        self._controller = sandbox_client.Controller.from_file(controller_file)
        self._wait_seconds = float(wait_seconds)
        self._python_hash_seed = python_hash_seed
        self._sandbox = None
        self._renewal = None
        super().__init__(environment_dir=environment_dir, environment_name=environment_name, session_id=session_id,
                         trial_paths=trial_paths, task_env_config=task_env_config, logger=logger, **kwargs)
        self._workdir = SingularityEnvironment._resolve_workdir(self)
        test_script = self.environment_dir.parent / 'tests/test.sh'
        test_text = test_script.read_text() if test_script.exists() else ''
        uv_versions = set(re.findall(r'https://astral\.sh/uv/([^/\s"\']+)/install\.sh', test_text))
        self._runtime_env = dict(MIRROR_ENV)
        if uv_versions == {'0.9.5'}:
            # The pinned uv release ships in /opt/harbor-deps; its install script fetches from here.
            self._runtime_env['UV_DOWNLOAD_URL'] = 'file:///opt/harbor-deps'
        if python_hash_seed is not None:
            self._runtime_env['PYTHONHASHSEED'] = str(python_hash_seed)

    @staticmethod
    def type():
        return 'acp-sandbox'

    @classmethod
    def resource_capabilities(cls):
        return EnvironmentResourceCapabilities()

    @property
    def capabilities(self):
        return EnvironmentCapabilities()

    @property
    def _dockerfile_path(self):
        return self.environment_dir / 'Dockerfile'

    def _validate_definition(self):
        if not self.task_env_config.docker_image:
            raise ValueError('acp-sandbox environment requires docker_image in task.toml [environment]')

    # Public, pinned dependencies staged by the singularity backend's local patch; reused unchanged.
    prepare_public_source_dependencies = SingularityEnvironment.prepare_public_source_dependencies
    prepare_verifier_dependencies = SingularityEnvironment.prepare_verifier_dependencies
    _stage_dependency_entry = SingularityEnvironment._stage_dependency_entry

    async def _call(self, method, path, body=None, timeout=60):
        return await asyncio.to_thread(self._sandbox.call, method, path, body, timeout)

    async def _renew_forever(self):
        while True:
            await asyncio.sleep(RENEW_SECONDS)
            try:
                await asyncio.to_thread(self._sandbox.renew, LEASE_SECONDS)
            except Exception as error:
                self.logger.warning(f'Sandbox lease renewal failed: {error}')

    async def start(self, force_build):
        binds = [{'source': str(ASSETS), 'target': '/opt/harbor-deps'}]
        self._sandbox = await asyncio.to_thread(
            self._controller.create, self.task_env_config.docker_image, LEASE_SECONDS, self._wait_seconds,
            self.session_id, binds)
        self.logger.debug(f'Leased sandbox {self._sandbox.sandbox_id} on {self._sandbox.info["host"]}')
        self._renewal = asyncio.create_task(self._renew_forever())
        distro = (await self.exec('cat /etc/os-release', cwd='/')).stdout or ''
        listing = await self.exec('cd /etc/apt 2>/dev/null && find sources.list sources.list.d -type f', cwd='/')
        for name in (listing.stdout or '').split():
            path = '/etc/apt/' + name
            original = (await asyncio.to_thread(self._sandbox.read_file, path)).decode()
            text = apt_sources(original, distro)
            if text != original:
                await asyncio.to_thread(self._sandbox.write_file, path, text.encode())
        setup = ('mkdir -p /etc/ssl/certs ' + shlex.quote(self._workdir)
                 + ' && cp /opt/harbor-deps/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt'
                 + ' && if [ -d /etc/apt/apt.conf.d ]; then'
                 + ' cp /opt/harbor-deps/apt-direct.conf /etc/apt/apt.conf.d/99harbor-direct; fi' + BOOTSTRAP)
        result = await self.exec(setup, cwd='/', timeout_sec=900)
        if result.return_code != 0:
            raise RuntimeError(f'Sandbox runtime setup failed: {result.stderr or result.stdout}')
        # Nothing is mounted from the trial directory, so the log directories Harbor expects must exist here.
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()
        await self.prepare_public_source_dependencies()

    async def stop(self, delete):
        if self._renewal:
            self._renewal.cancel()
        if self._sandbox:
            try:
                await asyncio.to_thread(self._sandbox.release)
            except Exception as error:
                self.logger.warning(f'Sandbox release failed (its lease will expire): {error}')
            self._sandbox = None

    async def _read_stream(self, job, stream):
        if job[stream + '_bytes'] <= INLINE_OUTPUT_BYTES:
            return job[stream]
        data = await asyncio.to_thread(self._sandbox.read_file, job[stream + '_path'])
        return data.decode(errors='replace')

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self._sandbox is None:
            raise RuntimeError('acp-sandbox environment not started')
        resolved_user = self._resolve_user(user)
        # Sandbox commands already run as (fake)root; su would only add a PAM dependency on the missing /proc.
        if resolved_user not in (None, 'root', 0):
            if isinstance(resolved_user, int):
                user_argument = f'$(getent passwd {resolved_user} | cut -d: -f1)'
            else:
                user_argument = shlex.quote(str(resolved_user))
            command = f'su {user_argument} -s /bin/bash -c {shlex.quote(command)}'
        environment = dict(self._runtime_env, **(self._merge_env(env) or {}))
        job = await self._call('POST', '/commands', {
            'command': command, 'cwd': cwd or self._workdir, 'env': environment,
            'timeout_seconds': timeout_sec or UNBOUNDED_SECONDS})
        delay = 0.02
        try:
            while job['state'] == 'running':
                await asyncio.sleep(delay)
                delay = min(delay * 2, 0.5)
                job = await self._call('GET', f'/jobs/{job["job_id"]}')
        except asyncio.CancelledError:
            # The caller's own budget ran out (e.g. the agent timeout); stop the command in the sandbox too.
            await asyncio.shield(self._call('POST', f'/jobs/{job["job_id"]}/cancel', {}))
            raise
        stdout, stderr = await self._read_stream(job, 'stdout'), await self._read_stream(job, 'stderr')
        if job['state'] == 'timed_out':
            # Same result the singularity server gives through `timeout`: exit code 124 and the output so far.
            stderr += f'\nCommand timed out after {timeout_sec} seconds (timeout exit code 124)'
            return ExecResult(stdout=stdout, stderr=stderr, return_code=124)
        return ExecResult(stdout=stdout, stderr=stderr, return_code=job['returncode'])

    async def _upload_path(self, source, target, mode):
        # The first write truncates the target, so an empty file is created as well.
        with open(source, 'rb') as handle:
            offset, chunk = 0, handle.read(sandbox_client.CHUNK_BYTES)
            while True:
                await self._call('POST', '/files/write', {'path': target, 'offset': offset, 'mode': mode,
                                                          'data_base64': base64.b64encode(chunk).decode()})
                offset += len(chunk)
                chunk = handle.read(sandbox_client.CHUNK_BYTES)
                if not chunk:
                    return

    async def upload_file(self, source_path, target_path):
        source = Path(source_path)
        await self._upload_path(source, str(target_path), source.stat().st_mode & 0o777)

    async def upload_dir(self, source_dir, target_dir):
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f'Source directory not found: {source}')
        remote = f'/tmp/.hb-transfer-{uuid.uuid4().hex}.tar'
        with tempfile.TemporaryDirectory() as local:
            archive = Path(local) / 'upload.tar'
            # Uncompressed: the dev container must not spend CPU on gzip.
            with tarfile.open(archive, 'w') as tar:
                tar.add(source, arcname='.')
            await self._upload_path(archive, remote, 0o600)
        result = await self.exec(f'mkdir -p {shlex.quote(target_dir)} && tar -xf {remote} -C {shlex.quote(target_dir)}'
                                 f' --no-same-owner; status=$?; rm -f {remote}; exit $status', cwd='/', user='root')
        if result.return_code != 0:
            raise RuntimeError(f'Failed to upload directory: {result.stderr or result.stdout}')

    async def download_file(self, source_path, target_path):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await asyncio.to_thread(self._sandbox.read_file, str(source_path)))

    async def download_dir(self, source_dir, target_dir):
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        remote = f'/tmp/.hb-transfer-{uuid.uuid4().hex}.tar'
        result = await self.exec(f'tar -cf {remote} -C {shlex.quote(source_dir)} .', cwd='/', user='root')
        try:
            if result.return_code != 0:
                raise RuntimeError(f'Failed to download directory: {result.stderr or result.stdout}')
            with tempfile.TemporaryDirectory() as local:
                archive = Path(local) / 'download.tar'
                archive.write_bytes(await asyncio.to_thread(self._sandbox.read_file, remote))
                with tarfile.open(archive) as tar:
                    tar.extractall(path=target, filter=transfer_tar_filter)
        finally:
            await self.exec(f'rm -f {remote}', cwd='/', user='root')
