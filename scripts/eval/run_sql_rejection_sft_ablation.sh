#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -x ".venv/bin/python" ]; then

  . ".venv/bin/activate"
fi

MODEL="${MODEL:-qwen/qwen3.5-9b}"
MODEL_DIR="${MODEL_DIR:-/path/to/models/Qwen3.5-9B}"
TRAIN_DATASET="${TRAIN_DATASET:?Set TRAIN_DATASET=/path/to/train_1200.json}"
TEST_DATASET="${TEST_DATASET:?Set TEST_DATASET=/path/to/test_300.json}"
EXP_NAME="${EXP_NAME:-sql_rejection_sft_$(date +%Y%m%d_%H%M%S)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-workspace/logs/sql/${EXP_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${OUTPUT_ROOT}/rollout_base_1200}"
BASE_TRAIN_DIR="${BASE_TRAIN_DIR:-${OUTPUT_ROOT}/eval_base_1200}"
BASE_TEST_DIR="${BASE_TEST_DIR:-${OUTPUT_ROOT}/eval_base_300}"
RS_TRAIN_DIR="${RS_TRAIN_DIR:-${OUTPUT_ROOT}/eval_rs_sft_1200}"
RS_TEST_DIR="${RS_TEST_DIR:-${OUTPUT_ROOT}/eval_rs_sft_300}"

SFT_DATASET="${SFT_DATASET:-${OUTPUT_ROOT}/rejection_sft_steps.jsonl}"
SFT_MANIFEST="${SFT_MANIFEST:-${SFT_DATASET}.manifest.json}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/sql_rejection_sft/${EXP_NAME}}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_ROOT}/summary.json}"

VLLM_GPUS="${VLLM_GPUS:-0,1}"
VLLM_TP="${VLLM_TP:-2}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.5}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-600}"
CONCURRENCY="${CONCURRENCY:-8}"
STEP_TIMEOUT="${STEP_TIMEOUT:-1800}"
MAX_STEPS="${MAX_STEPS:-30}"
LORA_ID="${LORA_ID:-rs_sft}"
LORA_R="${LORA_R:-32}"

REUSE_ROLLOUT_FOR_BASE_TRAIN_EVAL="${REUSE_ROLLOUT_FOR_BASE_TRAIN_EVAL:-1}"

mkdir -p "${OUTPUT_ROOT}"

run_cmd() {
  echo ""
  echo "+ $*"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    return 0
  fi
  "$@"
}

run_eval() {
  local data_path="$1"
  local output_dir="$2"
  shift 2

  run_cmd uv run scripts/eval/bench_simple_intercode.py \
    --env sql \
    --model "${MODEL}" \
    --memory false \
    --data-path "${data_path}" \
    --output-dir "${output_dir}" \
    --max-steps "${MAX_STEPS}" \
    --step-timeout "${STEP_TIMEOUT}" \
    --concurrency "${CONCURRENCY}" \
    --vllm \
    --vllm-model-dir "${MODEL_DIR}" \
    --vllm-gpus "${VLLM_GPUS}" \
    --vllm-tp "${VLLM_TP}" \
    --vllm-gpu-mem "${VLLM_GPU_MEM}" \
    --vllm-timeout "${VLLM_TIMEOUT}" \
    "$@"
}

echo "================================================================"
echo "  SQL Rejection-SFT Ablation"
echo "  EXP_NAME:        ${EXP_NAME}"
echo "  TRAIN_DATASET:   ${TRAIN_DATASET}"
echo "  TEST_DATASET:    ${TEST_DATASET}"
echo "  ROLLOUT_DIR:     ${ROLLOUT_DIR}"
echo "  ADAPTER_DIR:     ${ADAPTER_DIR}"
echo "  SUMMARY_CSV:     ${SUMMARY_CSV}"
echo "================================================================"

run_eval "${TRAIN_DATASET}" "${ROLLOUT_DIR}"

run_cmd python3 scripts/dataset/build_sql_rejection_sft_dataset.py \
  --dataset "${TRAIN_DATASET}" \
  --summary-csv "${ROLLOUT_DIR}/summary.csv" \
  --trajectories-dir "${ROLLOUT_DIR}/trajectories" \
  --output "${SFT_DATASET}" \
  --manifest-output "${SFT_MANIFEST}"

run_cmd env \
  DATASET="${SFT_DATASET}" \
  OUTPUT_DIR="${ADAPTER_DIR}" \
  MODEL_DIR="${MODEL_DIR}" \
  LORA_R="${LORA_R}" \
  bash scripts/train/launch_sql_rejection_sft.sh

if [ "${REUSE_ROLLOUT_FOR_BASE_TRAIN_EVAL}" = "1" ]; then
  BASE_TRAIN_DIR="${ROLLOUT_DIR}"
else
  run_eval "${TRAIN_DATASET}" "${BASE_TRAIN_DIR}"
fi
run_eval "${TEST_DATASET}" "${BASE_TEST_DIR}"

run_eval "${TRAIN_DATASET}" "${RS_TRAIN_DIR}" \
  --vllm-lora-module "${LORA_ID}=${ADAPTER_DIR}" \
  --vllm-lora-max-rank "${LORA_R}"
run_eval "${TEST_DATASET}" "${RS_TEST_DIR}" \
  --vllm-lora-module "${LORA_ID}=${ADAPTER_DIR}" \
  --vllm-lora-max-rank "${LORA_R}"

run_cmd python3 scripts/eval/summarize_sql_rejection_sft_results.py \
  --base-train-dir "${BASE_TRAIN_DIR}" \
  --base-test-dir "${BASE_TEST_DIR}" \
  --rs-train-dir "${RS_TRAIN_DIR}" \
  --rs-test-dir "${RS_TEST_DIR}" \
  --dataset-manifest "${SFT_MANIFEST}" \
  --adapter-path "${ADAPTER_DIR}" \
  --output-csv "${SUMMARY_CSV}" \
  --output-json "${SUMMARY_JSON}"
