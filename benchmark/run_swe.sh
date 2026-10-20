#!/usr/bin/env bash
set -euo pipefail
SWE_BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWE_BUILD_WORK="$SWE_BENCH_ROOT/.cache/swe-build"
export DOCKER_HOST=unix:///run/coding-agent-swe-docker/docker.sock
export DOCKER_CONFIG="$SWE_BUILD_WORK/docker-client"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
cd "$SWE_BENCH_ROOT"
mkdir -p "$DOCKER_CONFIG"
if [[ ! -f "$DOCKER_CONFIG/config.json" ]]; then
  printf '%s\n' '{"auths":{},"proxies":{"default":{"httpProxy":"","httpsProxy":"","allProxy":"","noProxy":"*"}}}' > "$DOCKER_CONFIG/config.json"
fi

start_docker() {
  mkdir -p "$SWE_BUILD_WORK" /run/coding-agent-swe-docker /run/coding-agent-swe-containerd
  local swe_store="$SWE_BENCH_ROOT/.cache/docker-images/swe-bench-pro-v2-hard"
  local swe_disk="$SWE_BENCH_ROOT/.cache/docker-images/swe-bench-pro-v2-hard.ext4"
  mkdir -p "$swe_store"
  if ! mountpoint -q "$swe_store"; then
    if [[ -n "$(ls -A -- "$swe_store")" ]]; then
      echo "Refusing to mount over a nonempty SWE Docker directory: $swe_store" >&2
      return 1
    fi
    if [[ ! -f "$swe_disk" ]]; then
      truncate -s 32G "$swe_disk"
      mkfs.ext4 -q -F -m 0 -E lazy_itable_init=1,lazy_journal_init=1 "$swe_disk"
    fi
    mount -o loop,noatime "$swe_disk" "$swe_store"
  fi
  "$SWE_BENCH_ROOT/.venv-harbor/bin/python" -B - "$SWE_BENCH_ROOT" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]); work = root/'.cache/swe-build'
store = root/'.cache/docker-images/swe-bench-pro-v2-hard'
config = {
 'data-root': str(store), 'exec-root': '/run/coding-agent-swe-docker',
 'pidfile': '/run/coding-agent-swe-docker/docker.pid',
 'hosts': ['unix:///run/coding-agent-swe-docker/docker.sock'],
 'storage-driver': 'overlay2', 'bridge': 'none', 'iptables': False, 'ip6tables': False,
 'ip-forward': False, 'ip-masq': False, 'exec-opts': ['native.cgroupdriver=systemd'],
 'cgroup-parent': 'coding-agent-swe.slice', 'features': {'containerd-snapshotter': False},
 'containerd': '/run/coding-agent-swe-containerd/containerd.sock',
 'containerd-namespace': 'coding-agent-swe',
 'containerd-plugins-namespace': 'coding-agent-swe-plugins',
 'max-concurrent-downloads': 4}
(work/'dockerd.json').write_text(json.dumps(config, indent=2)+'\n')
(work/'containerd.toml').write_text(
 'version = 2\nroot = '+json.dumps(str(store/'containerd-runtime'))+
 '\nstate = "/run/coding-agent-swe-containerd/state"\n'
 'disabled_plugins = ["io.containerd.grpc.v1.cri"]\n'
 '[grpc]\n  address = "/run/coding-agent-swe-containerd/containerd.sock"\n')
PY
  if ! systemctl is-active --quiet coding-agent-swe-containerd.service; then
    systemd-run --quiet --unit=coding-agent-swe-containerd --slice=coding-agent-swe.slice \
      -p Restart=on-failure -p RestartSec=5 \
      -p "StandardOutput=append:$SWE_BUILD_WORK/containerd.log" \
      -p "StandardError=append:$SWE_BUILD_WORK/containerd.log" \
      /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      NO_PROXY='*' no_proxy='*' /usr/bin/containerd --config "$SWE_BUILD_WORK/containerd.toml"
  fi
  if ! systemctl is-active --quiet coding-agent-swe-docker.service; then
    systemd-run --quiet --unit=coding-agent-swe-docker --slice=coding-agent-swe.slice \
      -p Requires=coding-agent-swe-containerd.service -p After=coding-agent-swe-containerd.service \
      -p CPUQuota=400% -p MemoryMax=32G -p Restart=on-failure -p RestartSec=5 \
      -p "StandardOutput=append:$SWE_BUILD_WORK/dockerd.log" \
      -p "StandardError=append:$SWE_BUILD_WORK/dockerd.log" \
      /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      NO_PROXY='*' no_proxy='*' PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      /usr/bin/dockerd --config-file="$SWE_BUILD_WORK/dockerd.json"
  fi
  systemctl set-property --runtime coding-agent-swe.slice CPUQuota=400% MemoryMax=32G
  for _swe_attempt in {1..30}; do
    if [[ "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" == "$swe_store" ]]; then
      return
    fi
    sleep 1
  done
  echo "SWE Docker failed to start; see $SWE_BUILD_WORK/dockerd.log" >&2
  return 1
}

if [[ "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" != \
    "$SWE_BENCH_ROOT/.cache/docker-images/swe-bench-pro-v2-hard" ]]; then
  start_docker
fi

case "${1:-run}" in
  run|check|smoke)
    swe_action="${1:-run}"
    if [[ "$#" -gt 0 ]]; then shift; fi
    exec "$SWE_BENCH_ROOT/.venv-harbor/bin/python" -u scripts/eval_swe.py \
      --mode "$swe_action" --model Qwen3.6-35B-A3B \
      --base-url http://10.119.27.14:8200/v1 \
      --results-dir results/Qwen3.6-35B-A3B/swe-bench-pro-v2-hard51 \
      --parallel 8 --cpus 4 --memory-mib 32768 \
      --agent-timeout 3000 --verifier-timeout 3000 "$@"
    ;;
  docker) shift; exec docker "$@" ;;
  prepare-images)
    shift
    if [[ "${1:-}" == --background ]]; then
      shift
      if systemctl is-active --quiet coding-agent-swe-images.service; then
        echo "SWE image preparation is already running."
        exit 0
      fi
      systemctl reset-failed coding-agent-swe-images.service 2>/dev/null || true
      systemd-run --quiet --unit=coding-agent-swe-images --slice=coding-agent-swe.slice \
        --description='Prepare official SWE-bench Pro V2 HARD-51 images' \
        --working-directory="$SWE_BENCH_ROOT" -p CPUQuota=400% -p MemoryMax=32G \
        -p "StandardOutput=append:$SWE_BUILD_WORK/prepare.log" \
        -p "StandardError=append:$SWE_BUILD_WORK/prepare.log" \
        /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
        NO_PROXY='*' no_proxy='*' /usr/bin/bash "$SWE_BENCH_ROOT/run_swe.sh" prepare-images "$@"
      echo "SWE image preparation started; log: $SWE_BUILD_WORK/prepare.log"
    else
      exec "$SWE_BENCH_ROOT/.venv-harbor/bin/python" -B -u \
        "$SWE_BENCH_ROOT/scripts/prepare_swe_images.py" "$@"
    fi
    ;;
  status)
    exec "$SWE_BENCH_ROOT/.venv-harbor/bin/python" -B -c '
import json
from pathlib import Path
p = Path(".cache/swe-build/manifest.json")
if not p.exists():
    print("Image preparation has not started.")
else:
    m = json.loads(p.read_text())
    keys = ("status", "ready_task_images", "task_count", "failed_task_images", "active_tasks", "updated_at", "runtime_store")
    print(json.dumps({k:m.get(k) for k in keys}, ensure_ascii=False, indent=2))
'
    ;;
  *) echo 'Usage: bash run_swe.sh {run|check|smoke|docker ARGS...|prepare-images [--background]|status}' >&2; exit 2 ;;
esac
