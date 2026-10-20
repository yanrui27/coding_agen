#!/usr/bin/env bash
# Serialize OCI-to-SIF builds, including calls from already-running coordinators.
set -euo pipefail
tb2_script=$(readlink -f -- "${BASH_SOURCE[0]}")
tb2_root=$(cd -- "$(dirname -- "$tb2_script")/.." && pwd)
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
case "${1:-}" in
  pull|build)
    mkdir -p "$tb2_root/.cache/tmp"
    exec 9>"$tb2_root/.cache/tmp/image-build.lock"
    # The shared FUSE filesystem returns a conflict instead of blocking in flock.
    while true; do
      if flock --exclusive --nonblock --conflict-exit-code 75 9; then
        break
      else
        tb2_lock_status=$?
        if [[ "$tb2_lock_status" != 75 ]]; then
          exit "$tb2_lock_status"
        fi
      fi
      sleep 0.25
    done
    ;;
esac
exec "$tb2_root/.sandbox-tools/apptainer/usr/bin/apptainer" "$@"
