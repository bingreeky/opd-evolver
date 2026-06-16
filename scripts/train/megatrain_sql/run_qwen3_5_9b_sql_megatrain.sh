#!/usr/bin/env bash

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this script with bash: bash $0" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MEGATRAIN_ROOT="$PROJECT_ROOT/reference/MegaTrain"
VERL_ROOT="$MEGATRAIN_ROOT/verl"
WORKDIR="$PROJECT_ROOT/scripts/train/megatrain_sql"
DATA_DIR="$WORKDIR/data"
SUCCESS_DATA_DIR="$WORKDIR/data_success"
LOG_DIR_DEFAULT="$WORKDIR/logs"

TRAIN_JSON_DEFAULT="$PROJECT_ROOT/data/sql/merged/ic_sql_merged_train_split.json"
TEST_JSON_DEFAULT="$PROJECT_ROOT/data/sql/merged/ic_sql_merged_test_split.json"
MODEL_PATH_DEFAULT="/path/to/models/Qwen3.5-9B"
INTERACTION_CONFIG_DEFAULT="$WORKDIR/sql_interaction_config.yaml"

cd "$PROJECT_ROOT"

if [ -f .venv/bin/activate ]; then

  . .venv/bin/activate
fi

export PATH="$PROJECT_ROOT/.venv/bin:${PATH:-}"
export PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/path/to/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib64" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$CUDA_HOME/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

export TMPDIR="${TMPDIR:-/tmp/opd_evolver_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_opd_evolver}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
TRAIN_JSON="${TRAIN_JSON:-$TRAIN_JSON_DEFAULT}"
TEST_JSON="${TEST_JSON:-$TEST_JSON_DEFAULT}"
PROJECT_NAME="${PROJECT_NAME:-GRPO-Qwen3_5-9B-InterCodeSQL-MegaTrain}"
EXP_NAME="${EXP_NAME:-grpo-qwen3_5-9b-sql-multiturn}"

RESUME_MODE="${RESUME_MODE:-disable}"
LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
INTERACTION_CONFIG_PATH="${INTERACTION_CONFIG_PATH:-$INTERACTION_CONFIG_DEFAULT}"
SQL_SERVICE_MODE="${SQL_SERVICE_MODE:-docker}"
SQL_HOST="${SQL_HOST:-127.0.0.1}"
SQL_PORT="${SQL_PORT:-3307}"
SQL_USER="${SQL_USER:-admin}"
SQL_PASSWORD="${SQL_PASSWORD:-admin}"
if [ "$SQL_SERVICE_MODE" != "docker" ] && [ "$SQL_SERVICE_MODE" != "local" ]; then
  echo "error: SQL_SERVICE_MODE must be docker or local, got: $SQL_SERVICE_MODE" >&2
  exit 1
fi
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-0}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
SUCCESS_ONLY="${SUCCESS_ONLY:-0}"
SUCCESS_TRAIN_SUMMARY_CSV="${SUCCESS_TRAIN_SUMMARY_CSV:-}"
SUCCESS_TRAIN_TRAJ_DIR="${SUCCESS_TRAIN_TRAJ_DIR:-}"

MEGATRAIN_OUTPUT_ROOT="${MEGATRAIN_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/megatrain}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MEGATRAIN_OUTPUT_ROOT/$PROJECT_NAME/${EXP_NAME}_${RUN_TS}}"
mkdir -p "$DEFAULT_LOCAL_DIR"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"

MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-65536}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-2}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-1}"
USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-196608}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-65536}"

MEGATRAIN_MAX_SEQ="${MEGATRAIN_MAX_SEQ:-262144}"

ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$MEGATRAIN_MAX_SEQ}"
ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-1}"
MULTI_TURN_ENABLE="${MULTI_TURN_ENABLE:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-100}"

VAL_END_ONLY="${VAL_END_ONLY:-0}"
if [ "$VAL_END_ONLY" = "1" ]; then
  TEST_FREQ=2147483647
fi

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
USE_WANDB="${USE_WANDB:-1}"

if [ "$PPO_MINI_BATCH" -gt "$TRAIN_BATCH_SIZE" ]; then
  echo "warning: PPO_MINI_BATCH ($PPO_MINI_BATCH) > TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE), capping to TRAIN_BATCH_SIZE" >&2
  PPO_MINI_BATCH="$TRAIN_BATCH_SIZE"
fi

MIN_REQUIRED_SEQ=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
if [ "$MEGATRAIN_MAX_SEQ" -lt "$MIN_REQUIRED_SEQ" ]; then
  echo "error: MEGATRAIN_MAX_SEQ=$MEGATRAIN_MAX_SEQ is smaller than MAX_PROMPT_LEN+MAX_RESPONSE_LEN=$MIN_REQUIRED_SEQ" >&2
  echo "hint: lower MAX_PROMPT_LEN/MAX_RESPONSE_LEN or raise MEGATRAIN_MAX_SEQ" >&2
  exit 1
fi
if [ "$ROLLOUT_MAX_MODEL_LEN" -lt "$MIN_REQUIRED_SEQ" ]; then
  echo "error: ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MAX_MODEL_LEN is smaller than MAX_PROMPT_LEN+MAX_RESPONSE_LEN=$MIN_REQUIRED_SEQ" >&2
  echo "hint: set ROLLOUT_MAX_MODEL_LEN>=${MIN_REQUIRED_SEQ} (or lower MAX_PROMPT_LEN/MAX_RESPONSE_LEN)" >&2
  exit 1
fi

if [ "$USE_WANDB" = "1" ]; then
  TRAINER_LOGGER='trainer.logger=[console,wandb]'
else
  TRAINER_LOGGER='trainer.logger=[console]'
fi

mkdir -p "$DATA_DIR" "$LOG_DIR"
mkdir -p "$SUCCESS_DATA_DIR"

if [ "$SUCCESS_ONLY" = "1" ]; then
  : "${TRAIN_FILE:=$SUCCESS_DATA_DIR/train.parquet}"
  : "${TEST_FILE:=$SUCCESS_DATA_DIR/test.parquet}"
else
  : "${TRAIN_FILE:=$DATA_DIR/train.parquet}"
  : "${TEST_FILE:=$DATA_DIR/test.parquet}"
fi

ensure_editable_verl() {
  if "$PYTHON_BIN" -c 'from verl.interactions.base import BaseInteraction' >/dev/null 2>&1; then
    return 0
  fi
  if [ -z "$UV_BIN" ]; then
    echo "uv not found; cannot install editable verl package" >&2
    exit 1
  fi
  "$UV_BIN" pip install --python "$PYTHON_BIN" -e "$VERL_ROOT"
}

vllm_pytorch_preflight() {
  if ! "$PYTHON_BIN" - <<'PY'
import importlib
import importlib.util
import sys

importlib.import_module("vllm")
print("ok: vllm")

from transformers import __version__ as tv
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "qwen3_5" not in CONFIG_MAPPING:
    print(
        f"error: transformers {tv!r} has no model_type 'qwen3_5' on this node",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"ok: transformers {tv!r} has qwen3_5")

if importlib.util.find_spec("causal_conv1d") is not None:
    try:
        import causal_conv1d
        print("ok: causal_conv1d import")
    except Exception as exc:
        print(
            "warning: causal_conv1d is installed but broken; consider: pip uninstall -y causal-conv1d"
            f" ({exc!r})",
            file=sys.stderr,
        )
else:
    print("skip: causal_conv1d not installed")
PY
  then
    echo "vLLM/transformers preflight failed; fix environment and retry." >&2
    return 1
  fi
}

start_mysql() {
  "$PYTHON_BIN" - <<'PY'
from scripts.eval.bench_simple_intercode import start_mysql_container

container = start_mysql_container(
    image_name="docker-env-sql:latest",
    container_name="docker-env-sql_ic_ctr",
    host_port=3307,
    container_port=3306,
    timeout=60,
)
if container is None:
    raise SystemExit(1)
PY
}

stop_mysql() {
  "$PYTHON_BIN" - <<'PY'
from scripts.eval.bench_simple_intercode import stop_mysql_container
stop_mysql_container(None, container_name="docker-env-sql_ic_ctr")
PY
}

if [ "$USE_WANDB" = "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "warning: USE_WANDB=1 but WANDB_API_KEY is unset." >&2
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "PYTHON=$PYTHON"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "TRAIN_FILE=$TRAIN_FILE"
  echo "TEST_FILE=$TEST_FILE"
  echo "INTERACTION_CONFIG_PATH=$INTERACTION_CONFIG_PATH"
  echo "SQL_SERVICE_MODE=$SQL_SERVICE_MODE"
  echo "SQL_HOST=$SQL_HOST"
  echo "SQL_PORT=$SQL_PORT"
  echo "SQL_USER=$SQL_USER"
  echo "LOG_DIR=$LOG_DIR"
fi

if [ "$SQL_SERVICE_MODE" = "local" ]; then
  echo "SQL_SERVICE_MODE=local: using existing MySQL at ${SQL_USER}@${SQL_HOST}:${SQL_PORT}" >&2
elif [ "${SKIP_MYSQL:-0}" = "1" ]; then
  echo "SKIP_MYSQL=1: skipping local MySQL container startup" >&2
else
  trap stop_mysql EXIT
  start_mysql
fi

if [ "$SUCCESS_ONLY" = "1" ]; then
  if [ -z "$SUCCESS_TRAIN_SUMMARY_CSV" ]; then
    echo "error: SUCCESS_ONLY=1 requires SUCCESS_TRAIN_SUMMARY_CSV=/path/to/summary.csv" >&2
    exit 1
  fi
  if [ "$REBUILD_DATASET" = "1" ] || [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    BUILD_CMD=(
      "$PYTHON_BIN" "$WORKDIR/build_sql_grpo_dataset_success_only.py"
      --train-json "$TRAIN_JSON"
      --test-json "$TEST_JSON"
      --output-dir "$SUCCESS_DATA_DIR"
      --train-success-summary-csv "$SUCCESS_TRAIN_SUMMARY_CSV"
      --sql-service-mode "$SQL_SERVICE_MODE"
      --sql-host "$SQL_HOST"
      --sql-port "$SQL_PORT"
      --sql-user "$SQL_USER"
      --sql-password "$SQL_PASSWORD"
    )
    if [ -n "$SUCCESS_TRAIN_TRAJ_DIR" ]; then
      BUILD_CMD+=(--train-trajectories-dir "$SUCCESS_TRAIN_TRAJ_DIR")
    fi
    if [ "$MAX_TRAIN_ROWS" != "0" ]; then
      BUILD_CMD+=(--max-train-rows "$MAX_TRAIN_ROWS")
    fi
    if [ "$MAX_TEST_ROWS" != "0" ]; then
      BUILD_CMD+=(--max-test-rows "$MAX_TEST_ROWS")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
      printf 'BUILD_CMD:'
      printf ' %q' "${BUILD_CMD[@]}"
      printf '\n'
    else
      "${BUILD_CMD[@]}"
    fi
  fi
else
  if [ "$REBUILD_DATASET" = "1" ] || [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    BUILD_CMD=(
      "$PYTHON_BIN" "$WORKDIR/build_sql_grpo_dataset.py"
      --train-json "$TRAIN_JSON"
      --test-json "$TEST_JSON"
      --output-dir "$DATA_DIR"
      --sql-service-mode "$SQL_SERVICE_MODE"
      --sql-host "$SQL_HOST"
      --sql-port "$SQL_PORT"
      --sql-user "$SQL_USER"
      --sql-password "$SQL_PASSWORD"
    )
    if [ "$MAX_TRAIN_ROWS" != "0" ]; then
      BUILD_CMD+=(--max-train-rows "$MAX_TRAIN_ROWS")
    fi
    if [ "$MAX_TEST_ROWS" != "0" ]; then
      BUILD_CMD+=(--max-test-rows "$MAX_TEST_ROWS")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
      printf 'BUILD_CMD:'
      printf ' %q' "${BUILD_CMD[@]}"
      printf '\n'
    else
      "${BUILD_CMD[@]}"
    fi
  fi
fi

if [ -d "$MEGATRAIN_ROOT" ]; then
  cd "$MEGATRAIN_ROOT"
elif [ "${DRY_RUN:-0}" = "1" ]; then
  echo "warning: MEGATRAIN_ROOT not found; DRY_RUN will only print commands: $MEGATRAIN_ROOT" >&2
else
  echo "error: MEGATRAIN_ROOT not found: $MEGATRAIN_ROOT" >&2
  exit 1
fi

ACTOR_GRAD_CKPT="actor_rollout_ref.model.enable_gradient_checkpointing=False"
if [ "$USE_GRAD_CHECKPOINT" = "1" ]; then
  ACTOR_GRAD_CKPT="actor_rollout_ref.model.enable_gradient_checkpointing=True"
fi

CMD=(
  bash "$MEGATRAIN_ROOT/examples/rl/run_qwen3_5_27b_megatrain.sh"
  data.return_raw_chat=True
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LEN"
  data.max_response_length="$MAX_RESPONSE_LEN"
  data.val_batch_size="$VAL_BATCH_SIZE"

  actor_rollout_ref.model.path="$MODEL_PATH"
  "$ACTOR_GRAD_CKPT"
  actor_rollout_ref.actor.megatrain.max_seq_len="$MEGATRAIN_MAX_SEQ"
  actor_rollout_ref.actor.megatrain.attn_implementation=flash_attention_2
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_PER_GPU"

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.quantization=null
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
  actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS"
  actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_WORKERS"
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
  actor_rollout_ref.rollout.nnodes=1
  actor_rollout_ref.rollout.n_gpus_per_node=1

  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.total_epochs=1
  trainer.val_before_train=False
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.default_local_dir="$DEFAULT_LOCAL_DIR"
  trainer.resume_mode="$RESUME_MODE"
  "$TRAINER_LOGGER"
)

if [ "$MULTI_TURN_ENABLE" = "1" ]; then
  CMD+=(
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG_PATH"
  )
else
  CMD+=(
    actor_rollout_ref.rollout.multi_turn.enable=False
  )
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'CMD:'
  printf ' %q' MODEL_PATH="$MODEL_PATH" TRAIN_FILE="$TRAIN_FILE" TEST_FILE="$TEST_FILE" PROJECT_NAME="$PROJECT_NAME" EXP_NAME="$EXP_NAME" LOG_DIR="$LOG_DIR" PYTHON="$PYTHON"
  printf ' %q' "${CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

MODEL_PATH="$MODEL_PATH" \
TRAIN_FILE="$TRAIN_FILE" \
TEST_FILE="$TEST_FILE" \
PROJECT_NAME="$PROJECT_NAME" \
EXP_NAME="$EXP_NAME" \
LOG_DIR="$LOG_DIR" \
PYTHON="$PYTHON" \
SQL_SERVICE_MODE="$SQL_SERVICE_MODE" \
SQL_HOST="$SQL_HOST" \
SQL_PORT="$SQL_PORT" \
SQL_USER="$SQL_USER" \
SQL_PASSWORD="$SQL_PASSWORD" \
"${CMD[@]}" "$@"
