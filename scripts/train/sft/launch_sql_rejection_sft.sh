#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

if [ -x ".venv/bin/python" ]; then

  . ".venv/bin/activate"
fi

DATASET="${DATASET:-workspace/logs/sql/rs_sft_exp/rejection_sft_steps.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sql_rejection_sft}"

source scripts/train/common/resolve_hf_model_dir.sh
resolve_qwen35_9b_model_dir
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

export TMPDIR="${TMPDIR:-$PWD/workspace/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/workspace/cache}"
export TORCH_HOME="${TORCH_HOME:-$PWD/workspace/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/workspace/cache/triton}"
export WANDB_DIR="${WANDB_DIR:-$PWD/workspace/cache/wandb}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$WANDB_DIR"

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
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
elif [ -n "${ACCELERATE_CONFIG:-}" ]; then
  ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG}"
  _NUM_PROCESSES_FOR_BATCH="${NUM_PROCESSES_FOR_BATCH:-2}"
elif [ "${SINGLE_GPU:-0}" = "1" ]; then
  ACCELERATE_CONFIG_FILE="config/train/selector_accelerate_1gpu.yaml"
  _NUM_PROCESSES_FOR_BATCH=1
else
  ACCELERATE_CONFIG_FILE="config/train/opsd_sql_accelerate.yaml"
  _NUM_PROCESSES_FOR_BATCH=2
fi

LR="${LR:-1e-5}"
BS="${BS:-2}"
GA="${GA:-4}"
EPOCHS="${EPOCHS:-1}"
MAX_LEN="${MAX_LEN:-8192}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_STEPS="${LOG_STEPS:-1}"
REPORT_TO="${REPORT_TO:-wandb}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
MIX_ORDER="${MIX_ORDER:-interleave_prefix}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
if [ "${ATTN_IMPL}" = "flash_attention_2" ] && ! .venv/bin/python -c "import flash_attn" >/dev/null 2>&1; then
  echo "Warning: flash_attn is not installed in .venv; falling back to sdpa"
  ATTN_IMPL="sdpa"
fi

CMD=(
  accelerate launch
    --config_file "${ACCELERATE_CONFIG_FILE}"
  scripts/train/sft/train_sql_rejection_sft.py
    --dataset "${DATASET}"
    --output-dir "${OUTPUT_DIR}"
    --model-dir "${MODEL_DIR}"
    --max-length "${MAX_LEN}"
    --per-device-train-batch-size "${BS}"
    --gradient-accumulation-steps "${GA}"
    --num-train-epochs "${EPOCHS}"
    --learning-rate "${LR}"
    --logging-steps "${LOG_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --attn-implementation "${ATTN_IMPL}"
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --report-to "${REPORT_TO}"
    --mix-order "${MIX_ORDER}"
    --shuffle-seed "${SHUFFLE_SEED}"
)

echo "================================================================"
echo "  SQL Rejection-SFT"
echo "  Model:      ${MODEL_DIR##*/}"
echo "  Dataset:    ${DATASET}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "  Batch:      ${BS} x ${GA} x ${_NUM_PROCESSES_FOR_BATCH} = $((BS * GA * _NUM_PROCESSES_FOR_BATCH)) effective"
echo "  Epochs:     ${EPOCHS}"
echo "  LR:         ${LR}"
echo "  Attn impl:  ${ATTN_IMPL}"
echo "  LoRA:       r=${LORA_R} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT}"
echo "  Mix order:  ${MIX_ORDER} (shuffle_seed=${SHUFFLE_SEED})"
echo "================================================================"

if [ "${SINGLE_GPU:-0}" != "1" ] && [ "${_visible_gpu_count}" -lt 2 ]; then
  echo "WARNING: dual-GPU launch selected but CUDA_VISIBLE_DEVICES exposes ${_visible_gpu_count} GPU(s)." >&2
  echo "         Use CUDA_VISIBLE_DEVICES=0,1 (default) or set SINGLE_GPU=1." >&2
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '  %s\n' "${CMD[@]}"
  exit 0
fi

exec "${CMD[@]}"
