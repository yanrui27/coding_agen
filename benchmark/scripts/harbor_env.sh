#!/usr/bin/env bash
# Source this file before using the project-local Harbor and sandbox runtime.
export BENCHMARK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$BENCHMARK_ROOT/.venv-harbor/bin:$BENCHMARK_ROOT/.sandbox-tools/apptainer/usr/bin:$PATH"
export APPTAINER_TMPDIR="$BENCHMARK_ROOT/.cache/tmp"
export APPTAINER_CACHEDIR="$APPTAINER_TMPDIR/apptainer"
export APPTAINER_CONFIGDIR="$APPTAINER_TMPDIR/apptainer-config"
# Harbor retains the final SIF separately; avoid a second persistent OCI cache.
export APPTAINER_DISABLE_CACHE=true SINGULARITY_DISABLE_CACHE=true
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export TMPDIR="$BENCHMARK_ROOT/.cache/tmp"
export PIP_CACHE_DIR="$BENCHMARK_ROOT/.cache/pip"
export PIP_NO_CACHE_DIR=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export HARBOR_TELEMETRY=0
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
export APPTAINERENV_HTTP_PROXY='' APPTAINERENV_HTTPS_PROXY='' APPTAINERENV_ALL_PROXY=''
export APPTAINERENV_http_proxy='' APPTAINERENV_https_proxy='' APPTAINERENV_all_proxy=''
export APPTAINERENV_NO_PROXY='*' APPTAINERENV_no_proxy='*'
mkdir -p "$APPTAINER_TMPDIR" "$BENCHMARK_ROOT/.cache/harbor-images"
