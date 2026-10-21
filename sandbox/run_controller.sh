#!/usr/bin/env bash
# 沙盒控制器的常驻入口：按需在 ACP 上建删宿主（32 vCPU / 128 GiB 纯 CPU Worker，每台 8 个沙盒，最多 8 台 = 64 路）。
# 前台运行；后台用 nohup bash run_controller.sh &。客户端读 runtime/controller/controller.private.json。
set -euo pipefail
SANDBOX_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE_DIR=$SANDBOX_DIR/runtime/controller
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONUNBUFFERED=1
mkdir -p "$STATE_DIR"
exec /mnt/data/rui.yan/venvs/OmniRL/runtime/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12 \
    "$SANDBOX_DIR/controller.py" --state-dir "$STATE_DIR" \
    --host-spec N3lS.Ii.I60.32c128g --slots-per-host 8 --min-hosts 0 --max-hosts 8 \
    --host-idle-seconds 600 --host-start-timeout 1200 --host-idle-exit-seconds 900 --host-ttl-seconds 86400 \
    >> "$STATE_DIR/controller.log" 2>&1
