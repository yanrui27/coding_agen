#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
source scripts/harbor_env.sh

python - <<'PY'
import json
from pathlib import Path
p=Path('archive/validation/tb2-public-source-repair-20261018/build-cython-ext/receipt.json')
r=json.loads(p.read_text())
assert r['passed'] and r['public_source_original_clone_passed']
assert r['solution_not_precreated'] and r['verifier_inputs_hidden_until_phase']
PY

exec bash archive/tb2-diagnostics/run_tb2_environment_retest.sh \
  --task-manifest archive/manifests/terminal-bench-2.0-public-source-retry1.txt \
  --results-dir results/Qwen3.6-35B-A3B/terminal-bench-2.0/apptainer-environment-public-source-retry1 \
  --scope-prefix coding-agent-tb2-public-source \
  --service-unit coding-agent-tb2-public-source.service \
  --concurrency 1 "$@"
