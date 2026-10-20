#!/usr/bin/env bash
# Agent训练：两个版本独立从同一基座开始，2 节点 × 8 卡。
set -euo pipefail
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=/mnt/data/rui.yan/projects/coding_agent
MODEL=/mnt/model/ckpts/Qwen3.6-35B-A3B
MAX_LENGTH=262144
VARIANT="${1:?Usage: train_agent.sh positive|full RUN_DIR}"
RUN_DIR="${2:?Run directory is required}"
case "${VARIANT}" in
  positive) DATASETS=(rts rlmath naturebench) ;;
  full) DATASETS=(mindforge rts_all rlmath_all naturebench_all) ;;
  *) echo 'Expected positive or full' >&2; exit 1 ;;
esac

# 只从平台读取节点编号和主节点地址，训练超参数不使用环境变量覆盖。
export NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-${GROUP_RANK:-${RANK:--1}}}}"
export NNODES=2 NPROC_PER_NODE=8
: "${MASTER_ADDR:?Platform MASTER_ADDR is required}"
export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29541}"
[[ "${NODE_RANK}" == 0 || "${NODE_RANK}" == 1 ]] || { echo 'Expected node rank 0 or 1'; exit 1; }
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK PYTHONPATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MODELSCOPE_OFFLINE=1 PIP_NO_INDEX=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_DEVICE_MAX_CONNECTIONS=1
mkdir -p "${RUN_DIR}/logs"
trap 'status=$?; if (( status )); then python3 "${CODE}/runtime.py" failed "${RUN_DIR}" "${NODE_RANK}" "${status}"; fi' EXIT

SCRATCH="$(python3 "${CODE}/runtime.py" scratch "${RUN_DIR}" "${NODE_RANK}" "${MODEL}" "${MAX_LENGTH}" "${DATASETS[@]}")"
export MEGATRON_SWIFT_LOCAL_ENV="${SCRATCH}/environment" TMPDIR="${SCRATCH}/tmp"
export HF_HOME="${SCRATCH}/huggingface" HF_DATASETS_CACHE="${SCRATCH}/huggingface/datasets"
export MODELSCOPE_CACHE="${SCRATCH}/modelscope"
source /mnt/code/yuhao.feng/qwen36_omni/training/qwen36_audio_swift/scripts/cluster_env.sh
qwen36_audio_setup_nccl WARN
source /mnt/code/yuhao.feng/qwen36_omni/training/qwen36_audio_swift/activate_megatron_swift.sh

python3 "${CODE}/runtime.py" status "${RUN_DIR}" "${NODE_RANK}" encoding
python3 "${CODE}/runtime.py" cache "${RUN_DIR}" "${NODE_RANK}" "${MODEL}" "${MAX_LENGTH}" "${DATASETS[@]}"
mapfile -t CACHED_DATASETS < "${RUN_DIR}/cache_paths.txt"
python3 "${CODE}/runtime.py" status "${RUN_DIR}" "${NODE_RANK}" training
if [[ "${NODE_RANK}" == 0 && -e "${RUN_DIR}/checkpoints" ]]; then
  echo 'Training output already exists; use a fresh run directory.' >&2
  exit 1
fi

# 每 100 步保留一份完整 checkpoint；不设置 save_total_limit，不自动删除。
megatron sft \
  --model "${MODEL}" --model_type qwen3_5_moe --template qwen3_5 \
  --cached_dataset "${CACHED_DATASETS[@]}" --split_dataset_ratio 0 --eval_iters 0 \
  --loss_scale default --max_length "${MAX_LENGTH}" --truncation_strategy delete \
  --tuner_type full --freeze_llm false --freeze_vit true --freeze_aligner true \
  --tensor_model_parallel_size 16 --pipeline_model_parallel_size 1 --context_parallel_size 1 \
  --expert_model_parallel_size 16 --expert_tensor_parallel_size 1 --sequence_parallel true \
  --moe_token_dispatcher_type alltoall --moe_grouped_gemm true \
  --micro_batch_size 1 --global_batch_size 8 --num_train_epochs 1 \
  --lr 1e-5 --min_lr 1e-6 --lr_decay_style cosine --lr_warmup_fraction 0.03 \
  --bf16 true --attention_backend flash \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --optimizer_cpu_offload true --optimizer_offload_fraction 1.0 --use_precision_aware_optimizer true \
  --main_grads_dtype bf16 --main_params_dtype fp32 --exp_avg_dtype fp32 --exp_avg_sq_dtype fp32 \
  --packing true --packing_length "${MAX_LENGTH}" --packing_num_proc 8 \
  --dataset_num_proc 8 --dataloader_num_workers 4 \
  --save_strategy steps --save_steps 100 --save_safetensors false \
  --no_save_optim false --no_save_rng false --async_save false --add_version false \
  --external_plugins "${CODE}/runtime.py" --callbacks disk_guard \
  --logging_steps 1 --report_to none --output_dir "${RUN_DIR}/checkpoints" --seed 42 \
  2>&1 | tee "${RUN_DIR}/logs/train_node${NODE_RANK}.log"

python3 "${CODE}/runtime.py" trained "${RUN_DIR}" "${NODE_RANK}"
if [[ "${NODE_RANK}" == 0 ]]; then
  python3 "${CODE}/runtime.py" status "${RUN_DIR}" 0 exporting
  CHECKPOINT="$(python3 "${CODE}/runtime.py" latest "${RUN_DIR}/checkpoints")"
  export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29543
  export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1
  megatron export \
    --to_hf true --model "${MODEL}" --model_type qwen3_5_moe --template qwen3_5 \
    --mcore_model "${CHECKPOINT}" --output_dir "${RUN_DIR}/hf_export" --exist_ok false --max_shard_size 5GB \
    --tuner_type full --tensor_model_parallel_size 8 --pipeline_model_parallel_size 1 --context_parallel_size 1 \
    --expert_model_parallel_size 8 --expert_tensor_parallel_size 1 --sequence_parallel true \
    --no_load_optim true --no_load_rng true --finetune true \
    2>&1 | tee "${RUN_DIR}/logs/hf_export.log"
  python3 "${CODE}/runtime.py" publish "${RUN_DIR}" "${MODEL}" 2>&1 | tee "${RUN_DIR}/logs/hf_verify.log"
else
  python3 "${CODE}/runtime.py" wait-hf "${RUN_DIR}"
fi
python3 "${CODE}/runtime.py" status "${RUN_DIR}" "${NODE_RANK}" complete
