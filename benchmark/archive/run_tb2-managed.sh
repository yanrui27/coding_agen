#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/data/benchmark
# Only sandbox paths, dependency runtime settings and direct-network settings.
source scripts/harbor_env.sh

run_evaluation() {
  exec .venv-harbor/bin/python -u scripts/eval_tb2.py \
    --model Qwen3.6-35B-A3B \
    --base-url http://10.119.27.14:8200/v1 \
    --dataset-path datasets/terminal-bench-2.0 \
    --results-dir results/Qwen3.6-35B-A3B/terminal-bench-2.0 \
    --agent terminus-2 --environment singularity \
    --seeds 42 --concurrency 8 \
    --cpus 4 --memory-mib 32768 \
    --agent-timeout 10800 --max-turns none \
    --context-limit 262144 --max-output-tokens 81920 \
    --temperature 1.0 --top-p 0.95 --top-k 20 --thinking on \
    --image-cache-dir .cache/harbor-images --image-cache-gib 0 \
    "$@"
}

unit=coding-agent-tb2
case "${1:-}" in
  --print-config|--help) run_evaluation "$@" ;;
  eval-worker)
    shift
    run_evaluation "$@"
    ;;
  harbor)
    shift
    exec systemd-run --quiet --scope --unit="coding-agent-harbor-$(date +%s%N)" \
      -p CPUQuota=400% -p MemoryMax=8G \
      "$BENCHMARK_ROOT/.venv-harbor/bin/harbor" "$@"
    ;;
  images)
    shift
    image_action=${1:-start}
    if [[ "$#" -gt 0 ]]; then shift; fi
    case "$image_action" in
      start)
        if [[ "$#" == 0 ]] && [[ "$(systemctl show coding-agent-tb2-images.service -p LoadState --value)" == loaded ]]; then
          exec systemctl start coding-agent-tb2-images.service
        fi
        exec systemd-run --unit=coding-agent-tb2-images --service-type=exec \
          --working-directory="$BENCHMARK_ROOT" \
          -p CPUQuota=1600% -p MemoryMax=96G -p KillMode=control-group -p TimeoutStopSec=30s \
          /bin/bash "$BENCHMARK_ROOT/run_tb2.sh" images-worker "$@"
        ;;
      status) exec "$BENCHMARK_ROOT/.venv-harbor/bin/python" scripts/build_tb2_images.py --status ;;
      stop) exec systemctl stop coding-agent-tb2-images.service ;;
      *) printf 'Usage: bash run_tb2.sh images {start|status|stop}\n' >&2; exit 2 ;;
    esac
    ;;
  images-worker)
    shift
    exec .venv-harbor/bin/python -u scripts/build_tb2_images.py \
      --workers 8 --scratch-gib 80 \
      --registry https://docker.1panel.live --stall-timeout 300 \
      "$@"
    ;;
  start)
    shift
    if [[ "$#" == 0 ]] && [[ "$(systemctl show "$unit.service" -p LoadState --value)" == loaded ]]; then
      exec systemctl start "$unit.service"
    fi
    exec systemd-run --unit="$unit" --service-type=exec \
      --working-directory="$BENCHMARK_ROOT" \
      -p CPUQuota=200% -p MemoryMax=4G -p KillMode=control-group -p TimeoutStopSec=15s \
      /bin/bash "$BENCHMARK_ROOT/run_tb2.sh" eval-worker \
      --service-unit coding-agent-tb2.service "$@"
    ;;
  parallel)
    shift
    "$BENCHMARK_ROOT/.venv-harbor/bin/python" - "${1:?Provide the actual trial concurrency}" <<'PY'
import datetime,json,sys
from pathlib import Path
n=int(sys.argv[1])
assert 1 <= n <= 32, 'concurrency must be between 1 and 32'
p=Path('results/Qwen3.6-35B-A3B/terminal-bench-2.0/five-seed/control.json')
p.parent.mkdir(parents=True,exist_ok=True)
r=json.loads(p.read_text()) if p.exists() else {}
r.update(concurrency=n,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
t=p.with_suffix('.json.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
print(f'Actual trial slots set to {n}; finished tasks are replaced automatically.')
PY
    exit
    ;;
  status) exec systemctl status "$unit" --no-pager ;;
  summary)
    exec "$BENCHMARK_ROOT/.venv-harbor/bin/python" scripts/eval_tb2.py --summary
    ;;
  stop)
    for helper in coding-agent-tb2-rescale.service coding-agent-tb2-budget-switch.service coding-agent-tb2-protocol-switch.service coding-agent-tb2-first-seed.service; do
      if systemctl is-active --quiet "$helper"; then
        systemctl stop "$helper"
      fi
    done
    exec systemctl stop "$unit"
    ;;
esac
if [[ "${TB2_IN_SYSTEMD_SERVICE:-}" == 1 ]]; then
  # Compatibility with the already-created transient service; new launches use CLI arguments.
  run_evaluation --service-unit coding-agent-tb2.service "$@"
fi
exec systemd-run --quiet --scope --unit="coding-agent-tb2-$(date +%s%N)" \
  -p CPUQuota=200% -p MemoryMax=4G \
  /bin/bash "$BENCHMARK_ROOT/run_tb2.sh" eval-worker "$@"
