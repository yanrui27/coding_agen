#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
source scripts/harbor_env.sh

python - <<'PY'
import datetime, json, time
from pathlib import Path
checks=Path('archive/validation/tb2-gap-investigation-20261018')
before=json.loads((checks/'control-baseline/install-windows-3.11/receipt.json').read_text())
after=json.loads((checks/'control-protected/install-windows-3.11/receipt.json').read_text())
assert before['passed'] and before['controller_survived_cleanup'] is False
assert after['passed'] and after['controller_survived_cleanup'] is True
assert after['ordinary_python_process_killed'] and after['filesystem_and_terminal_preserved']
base=Path('results/Qwen3.6-35B-A3B/terminal-bench-2.0')
predecessors=['apptainer-environment-retest17','apptainer-environment-residual-retest2']
while True:
    states={name:json.loads(path.read_text()) if path.exists() else {}
            for name in predecessors
            for path in [base/name/'seed42/progress.json']}
    if all(state.get('complete') for state in states.values()):
        break
    path=base/'apptainer-control-cleanup-retest2/waiting.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'waiting_for_preceding_tb2_cohorts','combined_concurrency_limit':4,
        'predecessors':{name:{'complete':state.get('complete',False),
                             'status_counts':state.get('status_counts',{})}
                        for name,state in states.items()}},indent=2)+'\n')
    time.sleep(15)
PY

exec bash archive/tb2-diagnostics/run_tb2_environment_retest.sh \
  --task-manifest archive/manifests/terminal-bench-2.0-control-cleanup-retest2.txt \
  --results-dir results/Qwen3.6-35B-A3B/terminal-bench-2.0/apptainer-control-cleanup-retest2 \
  --scope-prefix coding-agent-tb2-control \
  --service-unit coding-agent-tb2-control.service \
  --concurrency 2 "$@"
