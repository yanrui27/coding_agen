#!/usr/bin/env bash
set -euo pipefail
ASI_BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ASI_BUILD_WORK="$ASI_BENCH_ROOT/.cache/asi-build"
export DOCKER_HOST=unix:///run/coding-agent-asi-docker/docker.sock
export AI4SCI_DOCKER_NETWORK=host
export ASI_MODEL_API_KEY="${ASI_MODEL_API_KEY:-EMPTY}"
export PYTHONPATH="$ASI_BENCH_ROOT/third_party/asi-bench${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
cd "$ASI_BENCH_ROOT"

asi_start_docker() {
  if [[ "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" == \
      "$ASI_BENCH_ROOT/.cache/docker-images/asi-bench" ]]; then
    return
  fi
  mkdir -p "$ASI_BUILD_WORK" /run/coding-agent-asi-docker /run/coding-agent-asi-containerd
  local asi_store="$ASI_BENCH_ROOT/.cache/docker-images/asi-bench"
  local asi_disk="$ASI_BENCH_ROOT/.cache/docker-images/asi-bench.ext4"
  mkdir -p "$asi_store"
  if ! mountpoint -q "$asi_store"; then
    if [[ -n "$(ls -A -- "$asi_store")" ]]; then
      echo "Refusing to mount over a nonempty ASI Docker directory: $asi_store" >&2
      return 1
    fi
    if [[ ! -f "$asi_disk" ]]; then
      truncate -s 32G "$asi_disk"
      mkfs.ext4 -q -F -m 0 -E lazy_itable_init=1,lazy_journal_init=1 "$asi_disk"
    fi
    mount -o loop,noatime "$asi_disk" "$asi_store"
  fi
  "$ASI_BENCH_ROOT/.venv-harbor/bin/python" -B - "$ASI_BENCH_ROOT" <<'PY'
from pathlib import Path
import json, sys
root=Path(sys.argv[1]); work=root/'.cache/asi-build'; store=root/'.cache/docker-images/asi-bench'
store.mkdir(parents=True,exist_ok=True)
config={'data-root':str(store),'exec-root':'/run/coding-agent-asi-docker',
 'pidfile':'/run/coding-agent-asi-docker/docker.pid',
 'hosts':['unix:///run/coding-agent-asi-docker/docker.sock'],
 'storage-driver':'overlay2','bridge':'none','iptables':False,'ip6tables':False,
 'ip-forward':False,'ip-masq':False,'exec-opts':['native.cgroupdriver=systemd'],
 'cgroup-parent':'coding-agent-asi.slice','features':{'containerd-snapshotter':False},
 'containerd':'/run/coding-agent-asi-containerd/containerd.sock',
 'containerd-namespace':'coding-agent-asi','containerd-plugins-namespace':'coding-agent-asi-plugins',
 'max-concurrent-downloads':2}
(work/'dockerd.json').write_text(json.dumps(config,indent=2)+'\n')
(work/'containerd.toml').write_text('version = 2\nroot = '+json.dumps(str(store/'containerd-runtime'))+
 '\nstate = "/run/coding-agent-asi-containerd/state"\n'
 'disabled_plugins = ["io.containerd.grpc.v1.cri"]\n'
 '[grpc]\n  address = "/run/coding-agent-asi-containerd/containerd.sock"\n')
PY
  if ! systemctl is-active --quiet coding-agent-asi-containerd.service; then
    systemd-run --quiet --unit=coding-agent-asi-containerd --slice=coding-agent-asi.slice \
      -p Restart=on-failure -p RestartSec=5 \
      -p "StandardOutput=append:$ASI_BUILD_WORK/containerd.log" \
      -p "StandardError=append:$ASI_BUILD_WORK/containerd.log" \
      /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      NO_PROXY='*' no_proxy='*' /usr/bin/containerd --config "$ASI_BUILD_WORK/containerd.toml"
  fi
  if ! systemctl is-active --quiet coding-agent-asi-docker.service; then
    systemd-run --quiet --unit=coding-agent-asi-docker --slice=coding-agent-asi.slice \
      -p Requires=coding-agent-asi-containerd.service -p After=coding-agent-asi-containerd.service \
      -p CPUQuota=400% -p MemoryMax=32G -p Restart=on-failure -p RestartSec=5 \
      -p "StandardOutput=append:$ASI_BUILD_WORK/dockerd.log" \
      -p "StandardError=append:$ASI_BUILD_WORK/dockerd.log" \
      /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      NO_PROXY='*' no_proxy='*' \
      PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      /usr/bin/dockerd --config-file="$ASI_BUILD_WORK/dockerd.json"
  fi
  systemctl set-property --runtime coding-agent-asi.slice CPUQuota=400% MemoryMax=32G
  for _asi_attempt in {1..30}; do
    if [[ "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" == \
        "$ASI_BENCH_ROOT/.cache/docker-images/asi-bench" ]]; then
      return
    fi
    sleep 1
  done
  echo "ASI Docker failed to start; see $ASI_BUILD_WORK/dockerd.log" >&2
  return 1
}

