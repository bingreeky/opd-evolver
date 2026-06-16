#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://127.0.0.1:8006/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3-32b}"
export JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-8}"
export FILE_WORKERS="${FILE_WORKERS:-1}"

exec uv run python scripts/eval/batch_judge_memoryarena.py "$@"
