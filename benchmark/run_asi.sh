#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/asi_env.sh

exec .venv-harbor/bin/python -u scripts/eval_asi.py \
  --cpus 4 --memory-mib 32768 \
  run \
  --agent opencode_cli \
  --agent-config '{
    "model": "Qwen3.6-35B-A3B",
    "api_base": "http://10.119.27.14:8200/v1",
    "api_protocol": "openai",
    "api_key_env": "ASI_MODEL_API_KEY"
  }' \
  --instances-dir datasets/asi-bench-seed42/tasks \
  --tasks-dir third_party/asi-bench/tasks \
  --output-dir results/Qwen3.6-35B-A3B/asi-bench \
  --tasks all --prompt-levels b1,b2,b3,b4 --seed 42 \
  --sandbox os --parallel 8 --timeout 10800 \
  --tool-mode restricted --retries 1 --retry-strategy all \
  --write-batch-records --fail-on-agent-error \
  "$@"
