#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
source scripts/harbor_env.sh

# Require all completed no-model checks before spending model inference time.
.venv-harbor/bin/python - <<'PY'
import json
from pathlib import Path
base=Path('archive/validation/tb2-environment-repair-20261018')
for name in ('qemu-startup','qemu-alpine-ssh','headless-terminal','build-cython-ext','fix-ocaml-gc','reshard-c4-data'):
    receipt=json.loads((base/name/'receipt.json').read_text())
    assert receipt['passed'] is True, (name,receipt)
checks=json.loads((base/'regression-tests.json').read_text())
assert len(checks['terminal']) == 3 and all(r['payload_intact'] and r['ctrl_c_recovered'] for r in checks['terminal'])
assert checks['transport']['enclosing_budget_still_enforced']
PY

exec .venv-harbor/bin/python -u scripts/eval_tb2.py \
  --model Qwen3.6-35B-A3B --base-url http://10.119.27.14:8200/v1 \
  --dataset-path datasets/terminal-bench-2.0 \
  --results-dir results/Qwen3.6-35B-A3B/terminal-bench-2.0/apptainer-environment-retest17 \
  --task-manifest archive/manifests/terminal-bench-2.0-environment-retest17.txt \
  --preflight-report archive/validation/tb2-environment-repair-20261018/rerun-gates.json \
  --scope-prefix coding-agent-tb2-envfix --service-unit coding-agent-tb2-envfix.service \
  --agent terminus-2 --environment singularity --network-isolation per-trial \
  --seeds 42 --concurrency 4 --cpus 8 --memory-mib 32768 \
  --agent-timeout 10800 --max-turns none --context-limit 262144 --max-output-tokens 81920 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --thinking on \
  --image-cache-dir .cache/harbor-images --image-cache-gib 0 "$@"
