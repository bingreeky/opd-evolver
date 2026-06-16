resolve_qwen35_9b_model_dir() {
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

  if [ -n "${MODEL_DIR:-}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    export MODEL_PATH="${MODEL_PATH:-$MODEL_DIR}"
    return 0
  fi
  if [ -n "${MODEL_PATH:-}" ] && [ -f "${MODEL_PATH}/config.json" ]; then
    MODEL_DIR="${MODEL_PATH}"
    export MODEL_DIR
    return 0
  fi

  local _py="${RESOLVE_PYTHON:-.venv/bin/python}"
  local _resolved
  _resolved="$(
    MODEL_DIR="${MODEL_DIR:-}" MODEL_PATH="${MODEL_PATH:-}" "${_py}" -c \
      'from opd_evolver.base.hf_snapshot import resolve_hf_snapshot
import os
hint = os.environ.get("MODEL_DIR") or os.environ.get("MODEL_PATH")
p = resolve_hf_snapshot("Qwen/Qwen3.5-9B", hint=hint)
print(p or "")' 2>/dev/null || true
  )"

  if [ -n "${_resolved}" ]; then
    MODEL_DIR="${_resolved}"
    export MODEL_DIR
    export MODEL_PATH="${MODEL_PATH:-$MODEL_DIR}"
    return 0
  fi

  echo "ERROR: Qwen3.5-9B snapshot not found under HF_HOME=${HF_HOME}." >&2
  echo "       Set MODEL_DIR, MODEL_PATH, or download the model into the hub cache." >&2
  return 1
}
