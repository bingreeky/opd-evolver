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
  if VLLM_MODEL_DIR_9B="$(_resolve_hub_snapshot_dir "models--Qwen--Qwen3.5-9B")"; then
    echo "Resolved VLLM_MODEL_DIR_9B=${VLLM_MODEL_DIR_9B}"
  else
    echo "Note: no local HF snapshot for Qwen3.5-9B; bench will auto-resolve from HF_HOME or use char estimate." >&2
    VLLM_MODEL_DIR_9B=""
  fi
fi

if [[ -z "${VLLM_MODEL_DIR_4B}" ]]; then
  if VLLM_MODEL_DIR_4B="$(_resolve_hub_snapshot_dir "models--Qwen--Qwen3-4B")"; then
    echo "Resolved VLLM_MODEL_DIR_4B=${VLLM_MODEL_DIR_4B}"
  else
    echo "Note: no local HF snapshot for Qwen3-4B; bench will auto-resolve from HF_HOME or use char estimate." >&2
    VLLM_MODEL_DIR_4B=""
  fi
fi

URL_MAP="${AMA_GEN_URL_MAP:-${ROOT}/configs/ama_gen_urls.json}"
if [[ ! -f "${URL_MAP}" ]]; then
  echo "Missing URL map: ${URL_MAP}" >&2
  echo "Copy configs/ama_gen_urls.example.json and edit ports for your GPUs." >&2
  exit 1
fi

MATRIX=(
  uv run python "${ROOT}/scripts/eval/run_ama_rollout_matrix.py"
  --preset sixteen-grid
  --generation-url-map "${URL_MAP}"
  --models "${MODEL_9B},${MODEL_4B}"
)
if [[ "${NO_CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  MATRIX+=(--no-continue-on-error)
else
  MATRIX+=(--continue-on-error)
fi
[[ -n "${VLLM_MODEL_DIR_9B}" ]] && MATRIX+=(--vllm-model-dir-9b "${VLLM_MODEL_DIR_9B}")
[[ -n "${VLLM_MODEL_DIR_4B}" ]] && MATRIX+=(--vllm-model-dir-4b "${VLLM_MODEL_DIR_4B}")
[[ -n "${AMA_TEST_FILE:-}" ]] && MATRIX+=(--test-file "${AMA_TEST_FILE}")

if [[ "${EVALUATE:-0}" == "1" ]]; then
  MATRIX+=(--evaluate)
else
  MATRIX+=(--no-evaluate)
fi

if [[ -n "${AMA_ENDPOINT_MAX_CONCURRENT:-}" ]]; then
  MATRIX+=(--endpoint-max-concurrent "${AMA_ENDPOINT_MAX_CONCURRENT}")
fi

AMA_VLLM_MAX_MODEL_LEN="${AMA_VLLM_MAX_MODEL_LEN:-131072}"
AMA_VLLM_MAX_MODEL_LEN_9B="${AMA_VLLM_MAX_MODEL_LEN_9B:-${AMA_VLLM_MAX_MODEL_LEN}}"
AMA_VLLM_MAX_MODEL_LEN_4B="${AMA_VLLM_MAX_MODEL_LEN_4B:-${AMA_VLLM_MAX_MODEL_LEN}}"
AMA_LLM_MAX_COMPLETION_TOKENS_9B="${AMA_LLM_MAX_COMPLETION_TOKENS_9B:-4096}"
AMA_LLM_MAX_COMPLETION_TOKENS_4B="${AMA_LLM_MAX_COMPLETION_TOKENS_4B:-2048}"
MATRIX+=(
  --vllm-max-model-len-9b "${AMA_VLLM_MAX_MODEL_LEN_9B}"
  --vllm-max-model-len-4b "${AMA_VLLM_MAX_MODEL_LEN_4B}"
  --llm-max-completion-tokens-9b "${AMA_LLM_MAX_COMPLETION_TOKENS_9B}"
  --llm-max-completion-tokens-4b "${AMA_LLM_MAX_COMPLETION_TOKENS_4B}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  MATRIX+=(--dry-run)
fi

EXTRA=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA=("$@")
elif [[ $# -gt 0 ]]; then
  echo "Pass extra matrix/bench flags after --  Example: $0 -- --samples 2" >&2
  exit 2
fi

echo "========== AMA-Bench 16-job matrix (models=${MODEL_9B} + ${MODEL_4B}) =========="
"${MATRIX[@]}" "${EXTRA[@]}"
