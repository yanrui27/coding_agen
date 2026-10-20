#!/usr/bin/env bash
# Based on scratch/serve_qwen38-27b-vllm-tp8-0929063124.sh.
# Run inside a GPU ACP Worker with the model and vLLM venv available.
set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

MODEL=${MODEL:-/mnt/model/ckpts/Qwen3.6-35B-A3B}
VLLM_PYTHON=${VLLM_PYTHON:-/mnt/data/rui.yan/venvs/vllm/bin/python}

[[ -d "$MODEL" ]] || { echo "Model directory missing: $MODEL" >&2; exit 1; }
[[ -x "$VLLM_PYTHON" ]] || { echo "vLLM Python missing: $VLLM_PYTHON" >&2; exit 1; }

LOG_FILE="${LOG_DIR:-/mnt/data/rui.yan/projects/coding_agent/logs/vllm}/${SERVED_MODEL_NAME:-$(basename "$MODEL")}-$(date +%Y%m%d%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[serve] started $(date -Is) host=$(hostname) model=$MODEL log=$LOG_FILE"

export PYTHONUNBUFFERED=1
# Keep the total input/output budget within the model's native 256K.
unset VLLM_ALLOW_LONG_MAX_MODEL_LEN
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "${SERVED_MODEL_NAME:-$(basename "$MODEL")}" \
  --host 0.0.0.0 --port 8200 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-4}" \
  --data-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --no-enable-log-requests \
  --limit-mm-per-prompt '{"image":64,"video":16}' \
  --mm-encoder-tp-mode data \
  --allowed-local-media-path / \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  "$@"
