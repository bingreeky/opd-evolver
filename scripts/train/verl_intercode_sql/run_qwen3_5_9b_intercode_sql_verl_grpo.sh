#!/usr/bin/env bash

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "Run this script with bash: bash $0" >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WORKDIR="$PROJECT_ROOT/scripts/train/verl_intercode_sql"
DATA_DIR="${DATA_DIR:-$WORKDIR/data}"
LOG_DIR_DEFAULT="$WORKDIR/logs"
AGENT_LOOP_CONFIG_DEFAULT="$WORKDIR/intercode_sql_agent_loop_config.yaml"
REWARD_FUNCTION_DEFAULT="$WORKDIR/intercode_sql_reward.py"

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
if [ ! -x "$PYTHON" ]; then
  export PYTHON="$PYTHON_BIN"
fi

export PYTHONPATH="$WORKDIR:$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib64" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$CUDA_HOME/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

source "$PROJECT_ROOT/scripts/train/common/prepend_python_nvjitlink_ld_path.sh"
_prepend_python_nvjitlink_to_ld_path

export TMPDIR="${TMPDIR:-/tmp/opd_evolver_tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_opd_evolver}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

source "$PROJECT_ROOT/scripts/train/common/resolve_hf_model_dir.sh"
if [ "${DRY_RUN:-0}" = "1" ]; then
  if ! resolve_qwen35_9b_model_dir 2>/dev/null; then
    MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3.5-9B}"
    export MODEL_DIR
    export MODEL_PATH="${MODEL_PATH:-$MODEL_DIR}"
    echo "warning: Qwen3.5-9B snapshot not found; DRY_RUN using MODEL_PATH=$MODEL_PATH" >&2
  fi
else
  resolve_qwen35_9b_model_dir
fi

MODEL_PATH="${MODEL_PATH:-$MODEL_DIR}"
PROJECT_NAME="${PROJECT_NAME:-GRPO-Qwen3_5-9B-InterCodeSQL-VERL}"
EXP_NAME="${EXP_NAME:-grpo-qwen3_5-9b-intercode-sql-fsdp-vllm}"
RESUME_MODE="${RESUME_MODE:-disable}"
LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-$AGENT_LOOP_CONFIG_DEFAULT}"
REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-$REWARD_FUNCTION_DEFAULT}"
TRAIN_JSON="${TRAIN_JSON:-$PROJECT_ROOT/data/sql/merged/ic_sql_merged_train_split.json}"
TEST_JSON="${TEST_JSON:-$PROJECT_ROOT/data/sql/merged/ic_sql_merged_test_split.json}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-0}"
MAX_TEST_ROWS="${MAX_TEST_ROWS:-0}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"
QWEN_ENABLE_THINKING="${QWEN_ENABLE_THINKING:-0}"

SQL_SERVICE_MODE="${SQL_SERVICE_MODE:-docker}"
SQL_HOST="${SQL_HOST:-127.0.0.1}"
SQL_PORT="${SQL_PORT:-3307}"
SQL_USER="${SQL_USER:-admin}"
SQL_PASSWORD="${SQL_PASSWORD:-admin}"
if [ "$SQL_SERVICE_MODE" != "docker" ] && [ "$SQL_SERVICE_MODE" != "local" ]; then
  echo "error: SQL_SERVICE_MODE must be docker or local, got: $SQL_SERVICE_MODE" >&2
  exit 1
fi
export SQL_SERVICE_MODE SQL_HOST SQL_PORT SQL_USER SQL_PASSWORD

VERL_OUTPUT_ROOT="${VERL_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/verl}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$VERL_OUTPUT_ROOT/$PROJECT_NAME/${EXP_NAME}_${RUN_TS}}"
mkdir -p "$DEFAULT_LOCAL_DIR" "$DATA_DIR" "$LOG_DIR"

TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-2}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-12288}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-16384}"
ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-4}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
INTERCODE_SQL_PER_TURN_MAX_TOKENS="${INTERCODE_SQL_PER_TURN_MAX_TOKENS:-512}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-1}"
ACTOR_OPTIM_OFFLOAD="${ACTOR_OPTIM_OFFLOAD:-1}"
if [ "$ACTOR_PARAM_OFFLOAD" = "1" ]; then ACTOR_PARAM_OFFLOAD=True; else ACTOR_PARAM_OFFLOAD=False; fi
if [ "$ACTOR_OPTIM_OFFLOAD" = "1" ]; then ACTOR_OPTIM_OFFLOAD=True; else ACTOR_OPTIM_OFFLOAD=False; fi
export INTERCODE_SQL_PER_TURN_MAX_TOKENS

MULTI_TURN_ENABLE="${MULTI_TURN_ENABLE:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-100}"
VAL_END_ONLY="${VAL_END_ONLY:-0}"
if [ "$VAL_END_ONLY" = "1" ]; then
  TEST_FREQ=2147483647
fi
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
USE_WANDB="${USE_WANDB:-1}"
BALANCE_BATCH="${BALANCE_BATCH:-1}"

_visible_gpu_count=0
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  _cvd_trim="${CUDA_VISIBLE_DEVICES// /}"
  if [ -n "${_cvd_trim}" ]; then
    IFS=',' read -ra _cvd_parts <<< "${_cvd_trim}"
    _visible_gpu_count="${#_cvd_parts[@]}"
  fi
fi
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-${_visible_gpu_count:-8}}"

if [ "$PPO_MINI_BATCH" -gt "$TRAIN_BATCH_SIZE" ]; then
  echo "warning: PPO_MINI_BATCH ($PPO_MINI_BATCH) > TRAIN_BATCH_SIZE ($TRAIN_BATCH_SIZE), capping to TRAIN_BATCH_SIZE" >&2
  PPO_MINI_BATCH="$TRAIN_BATCH_SIZE"
fi

MIN_REQUIRED_SEQ=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
if [ "$MAX_SEQ_LEN" -lt "$MIN_REQUIRED_SEQ" ]; then
  echo "error: MAX_SEQ_LEN=$MAX_SEQ_LEN is smaller than MAX_PROMPT_LEN+MAX_RESPONSE_LEN=$MIN_REQUIRED_SEQ" >&2
  exit 1
fi
if [ "$ROLLOUT_MAX_MODEL_LEN" -lt "$MIN_REQUIRED_SEQ" ]; then
  echo "error: ROLLOUT_MAX_MODEL_LEN=$ROLLOUT_MAX_MODEL_LEN is smaller than MAX_PROMPT_LEN+MAX_RESPONSE_LEN=$MIN_REQUIRED_SEQ" >&2
  exit 1
fi
if [ "$ROLLOUT_N" -lt 2 ]; then
  echo "warning: ROLLOUT_N=$ROLLOUT_N; GRPO works best with grouped rollouts, so use ROLLOUT_N>=2" >&2
fi

_rollout_samples=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
if [ "$TRAINER_N_GPUS_PER_NODE" -gt 0 ] && [ $((_rollout_samples % TRAINER_N_GPUS_PER_NODE)) -ne 0 ]; then
  echo "error: TRAIN_BATCH_SIZE*ROLLOUT_N=$_rollout_samples must divide TRAINER_N_GPUS_PER_NODE=$TRAINER_N_GPUS_PER_NODE" >&2
  exit 1
fi

if [ "$USE_WANDB" = "1" ]; then
  TRAINER_LOGGER='trainer.logger=["console","wandb"]'
else
  TRAINER_LOGGER='trainer.logger=["console"]'
fi
if [ "$BALANCE_BATCH" = "1" ]; then
  TRAINER_BALANCE_BATCH=True
else
  TRAINER_BALANCE_BATCH=False
fi

start_mysql() {
  "$PYTHON_BIN" - <<'PY'
import time
import os
import docker
import mysql.connector

image_name = "docker-env-sql:latest"
container_name = "docker-env-sql_ic_ctr"
host_port = int(os.environ.get("SQL_PORT", "3307"))
container_port = 3306
timeout = 60

client = docker.from_env()
try:
    existing = client.containers.get(container_name)
    existing.stop()
    existing.remove()
except docker.errors.NotFound:
    pass

container = client.containers.run(
    image_name,
    name=container_name,
    ports={f"{container_port}/tcp": host_port},
    detach=True,
    remove=False,
)

start = time.time()
while time.time() - start < timeout:
    try:
        cnx = mysql.connector.connect(host="127.0.0.1", port=host_port, user="admin", password="admin")
        cnx.close()
        print(f"MySQL ready: {container_name}")
        raise SystemExit(0)
    except mysql.connector.errors.Error:
        time.sleep(1)
container.stop()
container.remove()
raise SystemExit("MySQL did not become ready")
PY
}

stop_mysql() {
  "$PYTHON_BIN" - <<'PY'
import docker

container_name = "docker-env-sql_ic_ctr"
try:
    client = docker.from_env()
    container = client.containers.get(container_name)
    container.stop(timeout=5)
    container.remove()
    print(f"MySQL stopped: {container_name}")
except Exception as exc:
    print(f"MySQL cleanup skipped: {exc}")
PY
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "TRAIN_FILE=$TRAIN_FILE"
  echo "TEST_FILE=$TEST_FILE"
  echo "AGENT_LOOP_CONFIG_PATH=$AGENT_LOOP_CONFIG_PATH"
  echo "REWARD_FUNCTION_PATH=$REWARD_FUNCTION_PATH"
  echo "SQL_SERVICE_MODE=$SQL_SERVICE_MODE"
  echo "SQL_HOST=$SQL_HOST"
  echo "SQL_PORT=$SQL_PORT"
  echo "SQL_USER=$SQL_USER"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ "$SQL_SERVICE_MODE" = "local" ]; then
  echo "SQL_SERVICE_MODE=local: using existing MySQL at ${SQL_USER}@${SQL_HOST}:${SQL_PORT}" >&2
elif [ "${SKIP_MYSQL:-0}" = "1" ]; then
  echo "SKIP_MYSQL=1: skipping InterCode SQL docker startup" >&2
else
  trap stop_mysql EXIT
  if [ "${DRY_RUN:-0}" != "1" ]; then
    start_mysql
  fi
fi

if [ "$REBUILD_DATASET" = "1" ] || [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
  BUILD_CMD=(
    "$PYTHON_BIN" "$WORKDIR/build_intercode_sql_grpo_dataset.py"
    --train-json "$TRAIN_JSON"
    --test-json "$TEST_JSON"
    --output-dir "$DATA_DIR"
    --max-steps "$MAX_STEPS"
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

ACTOR_GRAD_CKPT="actor_rollout_ref.model.enable_gradient_checkpointing=False"
if [ "$USE_GRAD_CHECKPOINT" = "1" ]; then
  ACTOR_GRAD_CKPT="actor_rollout_ref.model.enable_gradient_checkpointing=True"
fi

CMD=(
  "$PYTHON_BIN" -m verl.trainer.main_ppo

  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  reward.custom_reward_function.path="$REWARD_FUNCTION_PATH"
  reward.custom_reward_function.name=compute_score

  data.train_files="$TRAIN_FILE"
  data.val_files="$TEST_FILE"
  data.return_raw_chat=True
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LEN"
  data.max_response_length="$MAX_RESPONSE_LEN"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.filter_overlong_prompts=True
  data.truncation=error

  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  "$ACTOR_GRAD_CKPT"

  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_PER_GPU"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIM_OFFLOAD"

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.quantization=null
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
  actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_WORKERS"
  actor_rollout_ref.rollout.agent.default_agent_loop=intercode_sql_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH"

  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  trainer.n_gpus_per_node="$TRAINER_N_GPUS_PER_NODE"
  trainer.nnodes=1
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.val_before_train=False
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.default_local_dir="$DEFAULT_LOCAL_DIR"
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$EXP_NAME"
  trainer.resume_mode="$RESUME_MODE"
  trainer.balance_batch="$TRAINER_BALANCE_BATCH"
  "$TRAINER_LOGGER"
)

if [ "$MAX_TRAIN_ROWS" != "0" ]; then
  CMD+=(data.train_max_samples="$MAX_TRAIN_ROWS")
fi
if [ "$MAX_TEST_ROWS" != "0" ]; then
  CMD+=(data.val_max_samples="$MAX_TEST_ROWS")
fi

if [ "$MULTI_TURN_ENABLE" = "1" ]; then
  CMD+=(
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_STEPS"
    actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_STEPS"
  )
else
  CMD+=(actor_rollout_ref.rollout.multi_turn.enable=False)
fi

if [ "$QWEN_ENABLE_THINKING" = "0" ]; then
  CMD+=(+data.apply_chat_template_kwargs.enable_thinking=False)
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'CMD:'
  printf ' %q' PYTHONPATH="$PYTHONPATH" SQL_SERVICE_MODE="$SQL_SERVICE_MODE" SQL_HOST="$SQL_HOST" SQL_PORT="$SQL_PORT" SQL_USER="$SQL_USER"
  printf ' %q' "${CMD[@]}" "$@"
  printf '\n'
  exit 0
fi

"${CMD[@]}" "$@"
