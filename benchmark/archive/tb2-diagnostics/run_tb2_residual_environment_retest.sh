#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
source scripts/harbor_env.sh

python - <<'PY'
import datetime, json, time
from pathlib import Path
checks=Path('archive/validation/tb2-residual-environment-20261018')
headless=json.loads((checks/'headless-prompt-validation.json').read_text())
qemu=json.loads((checks/'qemu-v2/qemu-startup/receipt.json').read_text())
assert headless['runtime_prompt_normalization_passed']
assert qemu['passed'] and qemu['original_grader_dependencies_passed']
base=Path('results/Qwen3.6-35B-A3B/terminal-bench-2.0')
# The original cohort is at its final four tasks. Start these two only after
# two slots become free and no original queued/preparing tasks can fill them.
while True:
    progress=json.loads((base/'apptainer-environment-retest17/seed42/progress.json').read_text())
    rows=progress['tasks'].values()
    running=sum(row['status']=='running' for row in rows)
    pending=sum(row['status'] not in {'running','passed','failed','timeout','infrastructure_error'} for row in rows)
    if pending==0 and running<=2:
        break
    state={'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'status':'waiting_for_two_trial_slots','original_running':running,
           'original_pending':pending,'combined_concurrency_limit':4}
    path=base/'apptainer-environment-residual-retest2/waiting.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(state,indent=2)+'\n')
    time.sleep(15)
PY

exec bash archive/tb2-diagnostics/run_tb2_environment_retest.sh \
  --task-manifest archive/manifests/terminal-bench-2.0-residual-environment-retest2.txt \
  --results-dir results/Qwen3.6-35B-A3B/terminal-bench-2.0/apptainer-environment-residual-retest2 \
  --scope-prefix coding-agent-tb2-residual \
  --service-unit coding-agent-tb2-residual.service \
  --concurrency 2 "$@"
