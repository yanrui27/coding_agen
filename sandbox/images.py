#!/usr/bin/env python3
"""Shared image cache for sandbox hosts: one directory per task image, holding image.json and its layers.

<cache>/<image key>/image.json = {"image": name, "image_env": ["K=V", ...], "layers": [absolute paths]}.
A layer is a tar archive (plain or gzip) applied in order with OCI whiteouts, one unpacked rootfs directory,
or one SIF file (Apptainer image; its squashfs partition is extracted). Hosts unpack an image once onto
Worker-local disk (host_server.py) and copy that base per sandbox.

  python3 images.py add --image NAME --rootfs TAR_DIR_OR_SIF [--image-env K=V ...]   register an existing rootfs
  python3 images.py pull --image docker.io/OWNER/REPO:TAG                         download layers via a mirror
  python3 images.py list
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_CACHE = Path(__file__).resolve().parent / 'runtime/images'
# Docker Hub itself is unreachable from ACP and the dev container; this mirror serves it directly.
DEFAULT_MIRROR = 'dockerproxy.net'
# Some mirrors sit behind Cloudflare, which rejects Python's default User-Agent.
USER_AGENT = {'User-Agent': 'docker/24.0.7 go/go1.20.10 os/linux arch/amd64'}
MANIFEST_TYPES = ('application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, '
                  'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json')
CHUNK_BYTES = 8 << 20
SIF_PARTITION, SIF_JSON, SIF_PRIMARY_SYSTEM = 0x4004, 0x4006, 2


def image_key(image):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', image)


def load(cache, image):
    path = cache / image_key(image) / 'image.json'
    if not path.exists():
        raise FileNotFoundError(f'Image {image} is not in the cache {cache}; add or pull it first')
    manifest = json.loads(path.read_text())
    if manifest['image'] != image:
        raise ValueError(f'Cache entry {path} belongs to {manifest["image"]}')
    return manifest


def save(cache, manifest):
    directory = cache / image_key(manifest['image'])
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'image.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=1) + '\n')
    temporary.replace(directory / 'image.json')
    return directory


def sif_objects(path):
    """Used data objects of a SIF file, from its global header and descriptor table (sylabs/sif v2 layout)."""
    with open(path, 'rb') as handle:
        header = handle.read(128)
        if header[32:41] != b'SIF_MAGIC':
            raise ValueError(f'{path} is not a SIF file')
        total, offset = struct.unpack_from('<qq', header, 88)
        handle.seek(offset)
        objects = []
        for _ in range(total):
            fields = struct.unpack('<i?IIIqqqqqqq128s384s', handle.read(585))
            if fields[1]:
                filesystem, partition = struct.unpack_from('<ii', fields[13])
                objects.append({'type': fields[0], 'offset': fields[5], 'size': fields[6],
                                'name': fields[12].rstrip(b'\0').decode(), 'filesystem': filesystem,
                                'partition': partition})
        return objects


def sif_config(path):
    """The OCI image config stored in a SIF built from a Docker image (Env, WorkingDir, ...), or {}."""
    for item in sif_objects(path):
        if item['type'] == SIF_JSON and item['name'] == 'oci-config.json':
            with open(path, 'rb') as handle:
                handle.seek(item['offset'])
                return json.loads(handle.read(item['size']))
    return {}


def unpack(manifest, destination, environment=None):
    """Build the image rootfs in the empty directory `destination`; `environment` must find unsquashfs for SIF."""
    for index, layer in enumerate(manifest['layers']):
        layer = Path(layer)
        if layer.suffix == '.sif':
            # Extracting the squashfs partition directly needs neither FUSE nor a user namespace.
            partition = next(item for item in sif_objects(layer)
                             if item['type'] == SIF_PARTITION and item['partition'] == SIF_PRIMARY_SYSTEM)
            subprocess.run(['unsquashfs', '-force', '-no-xattrs', '-quiet', '-dest', str(destination),
                            '-offset', str(partition['offset']), str(layer)],
                           check=True, timeout=1800, env=environment, stdout=subprocess.DEVNULL)
            continue
        if layer.is_dir():
            subprocess.run(['cp', '-a', '--reflink=auto', f'{layer}/.', str(destination)], check=True, timeout=1800)
            continue
        if index:
            # Whiteouts delete files of lower layers; a first layer has nothing below it to delete.
            with tarfile.open(layer) as archive:
                for member in archive:
                    parent, name = os.path.split(member.name)
                    if not name.startswith('.wh.'):
                        continue
                    target = destination / parent
                    if name == '.wh..wh..opq':
                        victims = list(target.iterdir()) if target.is_dir() else []
                    else:
                        victims = [target / name[len('.wh.'):]]
                    for victim in victims:
                        if victim.is_dir() and not victim.is_symlink():
                            shutil.rmtree(victim)
                        elif victim.exists() or victim.is_symlink():
                            victim.unlink()
        subprocess.run(['tar', '-xf', str(layer), '-C', str(destination), '--exclude=.wh.*', '--exclude=*/.wh.*'],
                       check=True, timeout=1800)


class Registry:
    """Anonymous pull from a Docker Hub mirror, following redirects without leaking the token to the CDN."""

    def __init__(self, mirror, repository):
        self.base = f'https://{mirror}/v2/{repository}'
        self.repository = repository
        self.headers = dict(USER_AGENT)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_args, **_kwargs):
                return None

        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)

    def open(self, path, headers=None, timeout=60):
        request = urllib.request.Request(self.base + path, headers={**self.headers, **(headers or {})})
        for _ in range(5):
            try:
                return self.opener.open(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                if error.code == 401 and 'Authorization' not in self.headers:
                    challenge = dict(re.findall(r'(\w+)="([^"]*)"', error.headers['WWW-Authenticate']))
                    query = urllib.parse.urlencode({'service': challenge.get('service', ''),
                                                    'scope': f'repository:{self.repository}:pull'})
                    token_request = urllib.request.Request(challenge['realm'] + '?' + query, headers=USER_AGENT)
                    with self.opener.open(token_request, timeout=timeout) as response:
                        token = json.load(response)
                    self.headers['Authorization'] = 'Bearer ' + (token.get('token') or token['access_token'])
                    request = urllib.request.Request(self.base + path, headers={**self.headers, **(headers or {})})
                elif error.code in (301, 302, 303, 307, 308):
                    location = urllib.parse.urljoin(request.full_url, error.headers['Location'])
                    request = urllib.request.Request(location, headers={**USER_AGENT, **(headers or {})})
                else:
                    raise
        raise RuntimeError(f'Too many redirects for {path}')

    def download(self, digest, size, path, connections):
        """Parallel ranged download into `path`, verified against the digest."""
        with open(path, 'wb') as handle:
            handle.truncate(size)

        def fetch(start):
            end = min(start + CHUNK_BYTES, size) - 1
            for attempt in range(5):
                try:
                    with self.open(f'/blobs/{digest}', {'Range': f'bytes={start}-{end}'}, timeout=120) as response:
                        data = response.read()
                    if len(data) != end - start + 1:
                        raise OSError(f'short read at {start}')
                    with open(path, 'r+b') as handle:
                        handle.seek(start)
                        handle.write(data)
                    return
                except (OSError, urllib.error.URLError):
                    if attempt == 4:
                        raise

        with ThreadPoolExecutor(connections) as pool:
            list(pool.map(fetch, range(0, size, CHUNK_BYTES)))
        sha256 = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                sha256.update(block)
        if 'sha256:' + sha256.hexdigest() != digest:
            raise ValueError(f'Digest mismatch for {digest}')


def pull(cache, image, mirror, connections):
    name, _, tag = image.rpartition(':')
    repository = name.removeprefix('docker.io/')
    registry = Registry(mirror, repository)
    with registry.open(f'/manifests/{tag}', {'Accept': MANIFEST_TYPES}) as response:
        manifest = json.load(response)
    if 'manifests' in manifest:
        choice = next(item for item in manifest['manifests']
                      if item.get('platform', {}).get('architecture') == 'amd64'
                      and item.get('platform', {}).get('os') == 'linux')
        with registry.open(f'/manifests/{choice["digest"]}', {'Accept': MANIFEST_TYPES}) as response:
            manifest = json.load(response)
    with registry.open(f'/blobs/{manifest["config"]["digest"]}') as response:
        config = json.load(response).get('config') or {}
    directory = cache / image_key(image)
    (directory / 'layers').mkdir(parents=True, exist_ok=True)
    layers = []
    for layer in manifest['layers']:
        path = directory / 'layers' / (layer['digest'].replace(':', '-') + '.tar')
        if not (path.exists() and path.stat().st_size == layer['size']):
            partial = path.with_suffix('.partial')
            registry.download(layer['digest'], layer['size'], partial, connections)
            partial.replace(path)
        layers.append(str(path))
    return save(cache, {'image': image, 'image_env': config.get('Env', []), 'layers': layers,
                        'source': f'{mirror}/{repository}:{tag}', 'manifest_digest_layers':
                        [layer['digest'] for layer in manifest['layers']]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest='command', required=True)
    add = commands.add_parser('add')
    add.add_argument('--image', required=True)
    add.add_argument('--rootfs', type=Path, required=True)
    add.add_argument('--image-env', action='append', default=[], help='K=V; repeatable')
    pull_parser = commands.add_parser('pull')
    pull_parser.add_argument('--image', required=True)
    pull_parser.add_argument('--mirror', default=DEFAULT_MIRROR)
    pull_parser.add_argument('--connections', type=int, default=8)
    commands.add_parser('list')
    args = parser.parse_args()
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    if args.command == 'add':
        rootfs = args.rootfs.resolve(strict=True)
        image_env = args.image_env or (sif_config(rootfs).get('Env', []) if rootfs.suffix == '.sif' else [])
        print(save(args.cache, {'image': args.image, 'image_env': image_env, 'layers': [str(rootfs)],
                                'source': str(rootfs)}))
    elif args.command == 'pull':
        print(pull(args.cache, args.image, args.mirror, args.connections))
    else:
        for path in sorted(args.cache.glob('*/image.json')):
            manifest = json.loads(path.read_text())
            print(manifest['image'], len(manifest['layers']), manifest.get('source', ''), sep='\t')


if __name__ == '__main__':
    main()
