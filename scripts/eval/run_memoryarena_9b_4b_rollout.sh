#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

_default_hf_hub_cache() {
  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    printf '%s' "${HF_HUB_CACHE}"
  elif [[ -n "${HF_HOME:-}" ]]; then
    printf '%s' "${HF_HOME}/hub"
  else
    printf '%s' "${HOME}/.cache/huggingface/hub"
  fi
}

_resolve_hub_snapshot_dir() {
  local hub_rel="$1"
  local hub="$(_default_hf_hub_cache)"
  local snaps="${hub}/${hub_rel}/snapshots"
  local best="" mt=-1 d m
  if [[ ! -d "${snaps}" ]]; then
    return 1
  fi
  shopt -s nullglob
  for d in "${snaps}"/*; do
    [[ -d "${d}" ]] || continue
    [[ -f "${d}/config.json" ]] || continue
    m="$(stat -c '%Y' "${d}" 2>/dev/null || echo 0)"
    if (( m > mt )); then
      mt=${m}
      best="${d}"
    fi
  done
  shopt -u nullglob
  [[ -n "${best}" ]] || return 1
  printf '%s' "${best}"
}

MODEL_9B="${MODEL_9B:-qwen/qwen3.5-9b}"
MODEL_4B="${MODEL_4B:-qwen/qwen3-4b}"

VLLM_MODEL_DIR_9B="${VLLM_MODEL_DIR_9B:-}"
VLLM_MODEL_DIR_4B="${VLLM_MODEL_DIR_4B:-}"

if [[ -z "${VLLM_MODEL_DIR_9B}" ]]; then
  if ! VLLM_MODEL_DIR_9B="$(_resolve_hub_snapshot_dir "models--Qwen--Qwen3.5-9B")"; then
    echo "Set VLLM_MODEL_DIR_9B to a local Qwen3.5-9B snapshot (HF hub cache missing)." >&2
    exit 1
  fi
  echo "Resolved VLLM_MODEL_DIR_9B=${VLLM_MODEL_DIR_9B}"
fi

if [[ -z "${VLLM_MODEL_DIR_4B}" ]]; then
  if ! VLLM_MODEL_DIR_4B="$(_resolve_hub_snapshot_dir "models--Qwen--Qwen3-4B")"; then
    echo "Set VLLM_MODEL_DIR_4B to a local Qwen3-4B snapshot (HF hub cache missing)." >&2
    exit 1
  fi
  echo "Resolved VLLM_MODEL_DIR_4B=${VLLM_MODEL_DIR_4B}"
fi

export HOST="${HOST:-127.0.0.1}"
export BASE_PORT="${BASE_PORT:-8000}"
export GPU_COUNT_9B="${GPU_COUNT_9B:-4}"
export VLLM_SHARD_HOST="${VLLM_SHARD_HOST:-${HOST}}"
export VLLM_SHARD_BASE_PORT="${VLLM_SHARD_BASE_PORT:-${BASE_PORT}}"
export VLLM_SHARD_9B_COUNT="${VLLM_SHARD_9B_COUNT:-${GPU_COUNT_9B}}"
export VLLM_SHARD_ROUTING=1

MATRIX=(uv run python "${ROOT}/scripts/eval/run_memoryarena_rollout_matrix.py" --vllm-shard-routing)
if [[ "${NO_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  MATRIX+=(--no-continue-on-error)
else
  MATRIX+=(--continue-on-error)
fi

if [[ "${EVALUATE:-0}" == "1" ]]; then
  if [[ -z "${JUDGE_OPENAI_BASE_URL:-}" ]]; then
    echo "EVALUATE=1 requires JUDGE_OPENAI_BASE_URL (OpenAI-compatible judge /v1 URL)." >&2
    exit 1
  fi
  MATRIX+=(--evaluate --judge-openai-base-url "${JUDGE_OPENAI_BASE_URL}")
  [[ -n "${JUDGE_MODEL:-}" ]] && MATRIX+=(--judge-model "${JUDGE_MODEL}")
  [[ -n "${JUDGE_MAX_CONCURRENCY:-}" ]] && MATRIX+=(--judge-max-concurrency "${JUDGE_MAX_CONCURRENCY}")
else
  MATRIX+=(--no-evaluate)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  MATRIX+=(--dry-run)
fi

EXTRA=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA=("$@")
elif [[ $# -gt 0 ]]; then
  echo "Pass extra bench/matrix flags after --  Example: $0 -- --samples 2" >&2
  exit 2
fi

run_block() {
  local model_key="$1"
  local tok_dir="$2"
  echo "========== MemoryArena matrix: model=${model_key} tokenizer_dir=${tok_dir} =========="
  "${MATRIX[@]}" --model "${model_key}" --vllm-model-dir "${tok_dir}" "${EXTRA[@]}"
}

STATUS=0
if ! run_block "${MODEL_9B}" "${VLLM_MODEL_DIR_9B}"; then
  STATUS=1
  if [[ "${NO_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    exit "${STATUS}"
  fi
fi

if ! run_block "${MODEL_4B}" "${VLLM_MODEL_DIR_4B}"; then
  STATUS=1
fi

exit "${STATUS}"
