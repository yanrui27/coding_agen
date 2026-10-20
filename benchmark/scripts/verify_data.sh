#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/harbor_env.sh"
cd "$BENCHMARK_ROOT"
exec systemd-run --quiet --scope \
  --unit="coding-agent-benchmark-verify-$(date +%s%N)" \
  -p CPUQuota=200% -p MemoryMax=4G \
  env PYTHONPATH="$BENCHMARK_ROOT/.data-tools" \
  "$BENCHMARK_ROOT/.venv-harbor/bin/python" "$BENCHMARK_ROOT/scripts/verify_benchmarks.py"
