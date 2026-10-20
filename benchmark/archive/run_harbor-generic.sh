#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/harbor_env.sh"
cd "$BENCHMARK_ROOT"
exec systemd-run --quiet --scope \
  --unit="coding-agent-harbor-$(date +%s%N)" \
  -p CPUQuota=400% -p MemoryMax=8G \
  "$BENCHMARK_ROOT/.venv-harbor/bin/harbor" "$@"
