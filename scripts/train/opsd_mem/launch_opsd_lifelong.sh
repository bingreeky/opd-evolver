#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../.."

if [ -x ".venv/bin/python" ]; then

  . ".venv/bin/activate"
fi

DATA_ROOT="${DATA_ROOT:-data/lifelong_agent_bench/processed}"
SPLIT="${SPLIT:-train}"
TASK_TYPES="${TASK_TYPES:-db,os}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/opsd_lifelong/executor}"

source scripts/train/common/resolve_hf_model_dir.sh
resolve_qwen35_9b_model_dir
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TRL_EXPERIMENTAL_SILENCE=1

export TMPDIR="${TMPDIR:-$PWD/workspace/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-$PWD/workspace/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/workspace/cache/triton}"
export WANDB_DIR="${WANDB_DIR:-$PWD/workspace/cache/wandb}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR"

USE_VLLM="${USE_VLLM:-true}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
LR="${LR:-1e-5}"
BS="${BS:-2}"
GA="${GA:-4}"
EPOCHS="${EPOCHS:-3}"
MAX_LEN="${MAX_LEN:-4096}"
MAX_COMP="${MAX_COMP:-1024}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-10}"
LOG_STEPS="${LOG_STEPS:-1}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-lifelong}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

MEMORY_DIR="${MEMORY_DIR:-workspace/memory/lifelong_agent_bench/gen_for_train}"
MEMORY_TOP_K="${MEMORY_TOP_K:-50}"
MEMORY_RETRIEVE_K="${MEMORY_RETRIEVE_K:-100}"
MEMORY_SELECT_K="${MEMORY_SELECT_K:-10}"
MEMORY_EMBEDDING_MODEL="${MEMORY_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
MEMORY_MIN_SCORE="${MEMORY_MIN_SCORE:-0.01}"
TEACHER_MODE="${TEACHER_MODE:-both}"

if [ "${GOLD_ONLY:-0}" = "1" ]; then
  TEACHER_MODE="gold"
  MEMORY_DIR=""
  MEMORY_TOP_K="0"
  MEMORY_RETRIEVE_K="0"
  MEMORY_SELECT_K="0"
  MEMORY_MIN_SCORE="0.0"
fi

CMD=(
  accelerate launch
    --config_file config/train/opsd_sql_accelerate.yaml
  scripts/train/opsd_mem/train_opsd_lifelong.py
    --model_name_or_path "${MODEL_DIR}"
    --trust_remote_code true
    --attn_implementation "${ATTN_IMPL}"
    --dtype bfloat16
    --use_peft true
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --learning_rate "${LR}"
    --per_device_train_batch_size "${BS}"
    --gradient_accumulation_steps "${GA}"
    --num_train_epochs "${EPOCHS}"
    --max_length "${MAX_LEN}"
    --max_completion_length "${MAX_COMP}"
    --gradient_checkpointing true
    --bf16 true
    --lr_scheduler_type cosine
    --warmup_ratio 0.05
    --weight_decay 0.01
    --max_grad_norm 1.0
    --temperature 1.0
    --beta 0.5
    --lmbda 1.0
    --fixed_teacher true
    --jsd_token_clip 0.05
    --top_p 0.95
    --top_k 50
    --output_dir "${OUTPUT_DIR}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 3
    --logging_steps "${LOG_STEPS}"
    --log_completions true
    --report_to "${REPORT_TO}"
    --data_root "${DATA_ROOT}"
    --split "${SPLIT}"
    --task_types "${TASK_TYPES}"
    --teacher_mode "${TEACHER_MODE}"
    --memory_top_k "${MEMORY_TOP_K}"
    --memory_retrieve_k "${MEMORY_RETRIEVE_K}"
    --memory_select_k "${MEMORY_SELECT_K}"
    --memory_embedding_model "${MEMORY_EMBEDDING_MODEL}"
    --memory_min_score "${MEMORY_MIN_SCORE}"
)

if [ "${USE_VLLM}" = "true" ]; then
  CMD+=(
    --use_vllm true
    --vllm_mode colocate
    --vllm_tensor_parallel_size 2
    --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}"
    --vllm_enable_sleep_mode false
  )
else
  CMD+=(--use_vllm false)
fi

[ -n "${RUN_NAME}" ] && CMD+=(--run_config "${RUN_NAME}")
[ -n "${MEMORY_DIR}" ] && CMD+=(--memory_store_dir "${MEMORY_DIR}")
if [ -n "${MAX_TRAIN_SAMPLES}" ] && [ "${MAX_TRAIN_SAMPLES}" -gt 0 ] 2>/dev/null; then
  CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

echo "================================================================"
echo "  LifelongAgentBench Executor OPD"
echo "  Model:      ${MODEL_DIR##*/}"
echo "  Data root:  ${DATA_ROOT}"
echo "  Split:      ${SPLIT}"
echo "  Tasks:      ${TASK_TYPES}"
echo "  Teacher:    ${TEACHER_MODE}"
echo "  Memory dir: ${MEMORY_DIR:-none}"
echo "  Output:     ${OUTPUT_DIR}"
echo "================================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '  %s\n' "${CMD[@]}"
  exit 0
fi

exec "${CMD[@]}"
