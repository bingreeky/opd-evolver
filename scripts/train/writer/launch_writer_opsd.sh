#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../.."

if [ -x ".venv/bin/python" ]; then

  . ".venv/bin/activate"
fi

source scripts/train/common/resolve_hf_model_dir.sh
resolve_qwen35_9b_model_dir

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export TMPDIR="${TMPDIR:-$PWD/workspace/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-$PWD/workspace/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/workspace/cache/triton}"
export WANDB_DIR="${WANDB_DIR:-$PWD/workspace/cache/wandb}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR"

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TRL_EXPERIMENTAL_SILENCE=1

export ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-0}"

_visible_gpu_count=0
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  _cvd_trim="${CUDA_VISIBLE_DEVICES// /}"
  if [ -n "${_cvd_trim}" ]; then
    IFS=',' read -ra _cvd_parts <<< "${_cvd_trim}"
    _visible_gpu_count="${#_cvd_parts[@]}"
  fi
fi
if [ -n "${ACCELERATE_CONFIG_FILE:-}" ]; then
  _NUM_PROCESSES_FOR_BATCH="${NUM_PROCESSES_FOR_BATCH:-2}"
elif [ "${SINGLE_GPU:-0}" = "1" ]; then
  ACCELERATE_CONFIG_FILE="config/train/selector_accelerate_1gpu.yaml"
  _NUM_PROCESSES_FOR_BATCH=1
else
  ACCELERATE_CONFIG_FILE="config/train/opsd_sql_accelerate.yaml"
  _NUM_PROCESSES_FOR_BATCH=2
fi

USE_VLLM="${USE_VLLM:-true}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

if [ "${USE_VLLM}" = "true" ]; then
  UTIL_PRESET="${UTIL_PRESET:-safe}"
else
  UTIL_PRESET="${UTIL_PRESET:-high}"
fi

case "${UTIL_PRESET}" in
  turbo)
    BS_DEFAULT=3
    GA_DEFAULT=3
    MAX_LEN_DEFAULT=4096
    MAX_COMP_DEFAULT=1024
    GRADIENT_CHECKPOINTING_DEFAULT=true
    DATALOADER_NUM_WORKERS_DEFAULT=4
    ;;
  safe)
    BS_DEFAULT=2
    GA_DEFAULT=4
    MAX_LEN_DEFAULT=4096
    MAX_COMP_DEFAULT=1024
    GRADIENT_CHECKPOINTING_DEFAULT=true
    DATALOADER_NUM_WORKERS_DEFAULT=4
    ;;
  high)
    BS_DEFAULT=6
    GA_DEFAULT=2
    MAX_LEN_DEFAULT=4096
    MAX_COMP_DEFAULT=1024
    GRADIENT_CHECKPOINTING_DEFAULT=false
    DATALOADER_NUM_WORKERS_DEFAULT=8
    ;;
  max)
    BS_DEFAULT=8
    GA_DEFAULT=2
    MAX_LEN_DEFAULT=4096
    MAX_COMP_DEFAULT=1024
    GRADIENT_CHECKPOINTING_DEFAULT=false
    DATALOADER_NUM_WORKERS_DEFAULT=12
    ;;
  *)
    echo "WARNING: Unknown UTIL_PRESET='${UTIL_PRESET}'. Falling back to 'safe'." >&2
    BS_DEFAULT=2
    GA_DEFAULT=4
    MAX_LEN_DEFAULT=4096
    MAX_COMP_DEFAULT=1024
    GRADIENT_CHECKPOINTING_DEFAULT=true
    DATALOADER_NUM_WORKERS_DEFAULT=4
    ;;
esac

LR="${LR:-1e-5}"
BS="${BS:-${BS_DEFAULT}}"
GA="${GA:-${GA_DEFAULT}}"
EPOCHS="${EPOCHS:-3}"
MAX_LEN="${MAX_LEN:-${MAX_LEN_DEFAULT}}"
MAX_COMP="${MAX_COMP:-${MAX_COMP_DEFAULT}}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-${GRADIENT_CHECKPOINTING_DEFAULT}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-${DATALOADER_NUM_WORKERS_DEFAULT}}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-10}"
LOG_STEPS="${LOG_STEPS:-1}"
RUN_NAME="${RUN_NAME:-lifelong_writer_opsd}"
REPORT_TO="${REPORT_TO:-wandb}"

DATASET_PATH="${DATASET_PATH:-workspace/memory/lifelong_agent_bench/gen_for_train/db_os/memory_writer_dataset_scored.jsonl}"
TOPK_RATIO="${TOPK_RATIO:-0.3}"
TOPK_SAMPLES="${TOPK_SAMPLES:-0}"
MIN_SCORE="${MIN_SCORE:-0.0}"
HARD_SCORE_THRESHOLD="${HARD_SCORE_THRESHOLD:--1e9}"
MAX_TRACE_CHARS="${MAX_TRACE_CHARS:-12000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/writer_opsd}"

FIXED_TEACHER="${FIXED_TEACHER:-true}"
USE_EMA_TEACHER="${USE_EMA_TEACHER:-false}"
EMA_DECAY="${EMA_DECAY:-0.999}"
USE_TINKER_LOSS="${USE_TINKER_LOSS:-false}"
JSD_TOKEN_CLIP="${JSD_TOKEN_CLIP:-0.05}"
if [ "${SINGLE_GPU:-0}" = "1" ]; then
  VLLM_TP="${VLLM_TP:-1}"
else
  VLLM_TP="${VLLM_TP:-2}"
fi

VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.38}"

VLLM_ENABLE_SLEEP_MODE="${VLLM_ENABLE_SLEEP_MODE:-false}"

if [ "${USE_VLLM}" = "true" ]; then
  if [ "${UTIL_PRESET}" = "safe" ]; then
    TOP_K_LOSS="${TOP_K_LOSS:-0}"
  else
    TOP_K_LOSS="${TOP_K_LOSS:-256}"
  fi
  GEN_TOP_K="${GEN_TOP_K:-50}"
else
  TOP_K_LOSS="${TOP_K_LOSS:-0}"
  GEN_TOP_K="${GEN_TOP_K:-0}"
fi

CMD=(
  uv run accelerate launch
    --config_file "${ACCELERATE_CONFIG_FILE}"
  scripts/train/writer/train_writer_opsd.py
    --model_name_or_path  "${MODEL_DIR}"
    --trust_remote_code   true
    --attn_implementation "${ATTN_IMPL}"
    --dtype               bfloat16
    --use_peft            true
    --lora_r              "${LORA_R}"
    --lora_alpha          "${LORA_ALPHA}"
    --lora_dropout        "${LORA_DROPOUT}"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --learning_rate       "${LR}"
    --per_device_train_batch_size "${BS}"
    --gradient_accumulation_steps "${GA}"
    --num_train_epochs    "${EPOCHS}"
    --max_length          "${MAX_LEN}"
    --max_completion_length "${MAX_COMP}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataloader_pin_memory "${DATALOADER_PIN_MEMORY}"
    --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
    --bf16                true
    --lr_scheduler_type   cosine
    --warmup_ratio        0.05
    --weight_decay        0.01
    --max_grad_norm       1.0
    --temperature         1.0
    --beta                0.5
    --lmbda               1.0
    --fixed_teacher       "${FIXED_TEACHER}"
    --use_ema_teacher     "${USE_EMA_TEACHER}"
    --ema_decay           "${EMA_DECAY}"
    --use_tinker_loss     "${USE_TINKER_LOSS}"
    --top_k_loss          "${TOP_K_LOSS}"
    --jsd_token_clip      "${JSD_TOKEN_CLIP}"
    --top_p               0.95
    --top_k               "${GEN_TOP_K}"
    --output_dir          "${OUTPUT_DIR}"
    --save_steps          "${SAVE_STEPS}"
    --save_total_limit    3
    --logging_steps       "${LOG_STEPS}"
    --log_completions     true
    --report_to           "${REPORT_TO}"
    --dataset_path        "${DATASET_PATH}"
    --topk_ratio          "${TOPK_RATIO}"
    --topk_samples        "${TOPK_SAMPLES}"

    "--min_score=${MIN_SCORE}"
    "--hard_score_threshold=${HARD_SCORE_THRESHOLD}"
    --max_trace_chars     "${MAX_TRACE_CHARS}"
)

if [ "${USE_VLLM}" = "true" ]; then
  CMD+=(
    --use_vllm                    true
    --vllm_mode                   colocate
    --vllm_tensor_parallel_size   "${VLLM_TP}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}"
    --vllm_enable_sleep_mode      "${VLLM_ENABLE_SLEEP_MODE}"
  )
else
  CMD+=(--use_vllm false)
fi

if [ -n "${RUN_NAME}" ]; then
  CMD+=(--run_name "${RUN_NAME}")
fi

if [ -n "${MAX_TRAIN_SAMPLES}" ] && [ "${MAX_TRAIN_SAMPLES}" -gt 0 ] 2>/dev/null; then
  CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

if [ -n "${MAX_STEPS:-}" ] && [ "${MAX_STEPS}" -gt 0 ] 2>/dev/null; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi

if [ "${SINGLE_GPU:-0}" != "1" ] && [ "${_visible_gpu_count}" -eq 1 ]; then
  echo "WARNING: Using 2-process DeepSpeed but CUDA_VISIBLE_DEVICES lists only 1 GPU." >&2
  echo "         Use CUDA_VISIBLE_DEVICES=0,1 (default) or set SINGLE_GPU=1." >&2
fi

echo "================================================================"
echo "  Writer OPD Training"
echo "  Model:      ${MODEL_DIR##*/}"
echo "  Dataset:    ${DATASET_PATH}"
echo "  Accelerate: ${ACCELERATE_CONFIG_FILE}"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"
if [ "${USE_VLLM}" = "true" ]; then
  echo "  vLLM:       colocate (TP=${VLLM_TP}, util=${VLLM_GPU_UTIL}, sleep_mode=${VLLM_ENABLE_SLEEP_MODE})"
else
  echo "  vLLM:       off (HF generate)"
fi
echo "  Trace cap:  ${MAX_TRACE_CHARS} chars (max_trace_chars)"
echo "  Preset:     ${UTIL_PRESET}"
echo "  LoRA:       r=${LORA_R} alpha=${LORA_ALPHA}"
echo "  Batch:      ${BS}x${GA}x${_NUM_PROCESSES_FOR_BATCH} = $((BS * GA * _NUM_PROCESSES_FOR_BATCH)) effective"
echo "  Loader:     workers=${DATALOADER_NUM_WORKERS} pin=${DATALOADER_PIN_MEMORY} persistent=${DATALOADER_PERSISTENT_WORKERS}"
echo "  GC:         ${GRADIENT_CHECKPOINTING}"
echo "  JSD top_k:  ${TOP_K_LOSS} (0 = full vocab)"
echo "  LR:         ${LR}  Epochs: ${EPOCHS}"
echo "  Top-k:      ratio=${TOPK_RATIO}, samples=${TOPK_SAMPLES}, min_score=${MIN_SCORE}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  report_to:  ${REPORT_TO}"
if [ -n "${MAX_TRAIN_SAMPLES}" ] && [ "${MAX_TRAIN_SAMPLES}" -gt 0 ] 2>/dev/null; then
  echo "  Train cap:  first ${MAX_TRAIN_SAMPLES} examples"
fi
echo "================================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ""
  echo "DRY RUN - command:"
  printf '  %s\n' "${CMD[@]}"
  exit 0
fi

exec "${CMD[@]}"
