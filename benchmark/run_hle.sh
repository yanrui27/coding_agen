#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/benchmark

if [[ -z "${VIDEO_JUDGE_KEY:-}" && -f /mnt/data/rui.yan/.secrets/skyrouter.env ]]; then
  set -a
  source /mnt/data/rui.yan/.secrets/skyrouter.env
  set +a
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# 固定 216 题；仅在显式追加 --full 时评测全部 2158 题。
exec .venv-harbor/bin/python scripts/eval_hle.py run \
  --model Qwen3.6-35B-A3B \
  --base-url http://10.119.27.14:8200/v1 \
    http://10.119.28.50:8200/v1 \
    http://10.119.28.161:8200/v1 \
    http://10.119.29.55:8200/v1 \
  --subset text-only-10pct --workers 32 \
  --context-limit 262144 --max-output-tokens 131072 \
  --temperature 1.0 --top-p 1.0 --top-k 20 \
  --seed 42 --thinking on \
  --judge-base-url https://sr-us.singularity-ai.com/v1 \
  --judge-model gpt-6.1-sol --judge-api-key-env VIDEO_JUDGE_KEY \
  "$@"
