#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/data/benchmark
source scripts/harbor_env.sh

exec .venv-harbor/bin/python -u scripts/eval_tb2.py \
  --model Qwen3.6-35B-A3B \
  --base-url http://10.119.27.14:8200/v1 \
  --dataset-path datasets/terminal-bench-2.0 \
  --results-dir results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed \
  --task-manifest results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed/task-ids.txt \
  --agent terminus-2 --environment singularity \
  --seeds 42 --concurrency 8 \
  --cpus 4 --memory-mib 32768 \
  --agent-timeout 10800 --max-turns none \
  --context-limit 262144 --max-output-tokens 81920 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --thinking on \
  --image-cache-dir .cache/harbor-images --image-cache-gib 0 \
  "$@"
