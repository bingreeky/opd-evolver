#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export JUDGE_VLLM_HOST="${JUDGE_VLLM_HOST:-127.0.0.1}"
export JUDGE_VLLM_PORT="${JUDGE_VLLM_PORT:-8006}"
export JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-8}"
export FILE_WORKERS="${FILE_WORKERS:-1}"

exec uv run python scripts/eval/batch_judge_ama.py "$@"
