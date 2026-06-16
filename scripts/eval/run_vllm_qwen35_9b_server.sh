#!/usr/bin/env bash
set -euo pipefail

HF_HOME="/path/to/hf"

export HF_HOME=${HF_HOME}
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets

if [ -x ".venv/bin/python" ]; then

  . ".venv/bin/activate"
fi

export CUDA_VISIBLE_DEVICES=0,1

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

MODEL_DIR="${MODEL_DIR:-/path/to/models/Qwen3.5-9B}"
HOST="127.0.0.1"
PORT="8000"

MAX_MODEL_LEN="262144"

python - <<'PY'
import sys
from packaging.version import Version
import transformers

min_v = Version("5.2.0")
cur_v = Version(transformers.__version__)
if cur_v < min_v:
    sys.stderr.write(
        f"Error: transformers {transformers.__version__} is too old for Qwen3.5 (need >= {min_v}).\n"
        "Fix: python -m pip install -U 'transformers>=5.2.0'\n"
    )
    raise SystemExit(2)
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec python -m vllm.entrypoints.openai.api_server --model "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "qwen/qwen3.5-9b" \
  --tensor-parallel-size 2 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code \
  --reasoning-parser qwen3
