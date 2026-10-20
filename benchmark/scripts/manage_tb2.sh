#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
# Only sandbox paths, dependency runtime settings and direct-network settings.
source scripts/harbor_env.sh

unit=coding-agent-tb2
case "${1:-}" in
  --print-config|--help) exec /bin/bash "$BENCHMARK_ROOT/run_tb2.sh" "$@" ;;
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
          /bin/bash "$BENCHMARK_ROOT/scripts/manage_tb2.sh" images-worker "$@"
        ;;
      status) exec "$BENCHMARK_ROOT/.venv-harbor/bin/python" scripts/build_tb2_images.py --status ;;
      repair)
        if [[ "$(systemctl show coding-agent-tb2-image-repair.service -p LoadState --value)" == loaded ]]; then
          exec systemctl start coding-agent-tb2-image-repair.service
        fi
        exec systemd-run --unit=coding-agent-tb2-image-repair --service-type=exec \
          --working-directory="$BENCHMARK_ROOT" \
          -p CPUQuota=800% -p MemoryMax=64G -p KillMode=control-group -p TimeoutStopSec=30s \
          /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
          /bin/bash -c 'source scripts/harbor_env.sh
exec .venv-harbor/bin/python -u scripts/repair_tb2_images.py'
        ;;
      stop)
        for image_unit in coding-agent-tb2-images.service coding-agent-tb2-image-repair.service; do
          if systemctl is-active --quiet "$image_unit"; then systemctl stop "$image_unit"; fi
        done
        exit
        ;;
      *) printf 'Usage: bash scripts/manage_tb2.sh images {start|repair|status|stop}\n' >&2; exit 2 ;;
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
      /bin/bash "$BENCHMARK_ROOT/run_tb2.sh" \
      --service-unit coding-agent-tb2.service "$@"
    ;;
  status) exec systemctl status "$unit" --no-pager ;;
  summary)
    exec /bin/bash "$BENCHMARK_ROOT/run_tb2.sh" --summary
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
printf 'Usage: bash scripts/manage_tb2.sh {start|stop|status|summary|images start|images status|images stop|harbor ...}\n' >&2
exit 2
