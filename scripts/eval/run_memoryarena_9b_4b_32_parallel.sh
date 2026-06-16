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

URL_MAP="${MEMORYARENA_GEN_URL_MAP:-${ROOT}/configs/memoryarena_gen_urls.json}"
if [[ ! -f "${URL_MAP}" ]]; then
  echo "Missing URL map: ${URL_MAP}" >&2
  echo "Copy configs/memoryarena_gen_urls.example.json and edit ports for your GPUs." >&2
  exit 1
fi

MATRIX=(
  uv run python "${ROOT}/scripts/eval/run_memoryarena_rollout_matrix.py"
  --preset thirty-two-grid
  --no-vllm-shard-routing
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

if [[ -n "${MEMORYARENA_ENDPOINT_MAX_CONCURRENT:-}" ]]; then
  MATRIX+=(--endpoint-max-concurrent "${MEMORYARENA_ENDPOINT_MAX_CONCURRENT}")
fi

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

echo "========== MemoryArena 32-job matrix (models=${MODEL_9B} + ${MODEL_4B}) =========="
"${MATRIX[@]}" "${EXTRA[@]}"
