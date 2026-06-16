#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import csv
import glob
import importlib.util
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger
try:
    from tqdm.asyncio import tqdm as tqdm_async
except Exception:
    class _TqdmAsyncFallback:
        @staticmethod
        async def gather(*aws: Any, **_: Any) -> list[Any]:
            return await asyncio.gather(*aws)
    tqdm_async = _TqdmAsyncFallback()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "MemoryArena"
DEFAULT_DATASETS = ("formal_reasoning_math", "formal_reasoning_phys")
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-32B"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
logger.remove()
logger.add(
    sys.stderr,
    level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>memoryarena</cyan> | "
        "<level>{message}</level>"
    ),
)
try:
    from opd_evolver.memory.evolvelab_adapter import (
        ALL_MEMORY_BACKENDS,
        EVOLVELAB_MEMORY_BACKENDS,
        EvolveLabMemoryProviderAdapter,
        OPD_HIERARCHICAL_BACKEND,
        REASONING_BANK_BACKEND,
        is_evolvelab_backend,
        is_provider_backend,
        is_reasoning_bank_backend,
    )
except Exception:
    EVOLVELAB_MEMORY_BACKENDS = (
        "lightweight_memory",
        "expel",
        "agent_workflow_memory",
        "dynamic_cheatsheet",
        "memp",
        "evolver",
    )
    OPD_HIERARCHICAL_BACKEND = "opd_hierarchical"
    REASONING_BANK_BACKEND = "reasoning_bank"
    ALL_MEMORY_BACKENDS = (OPD_HIERARCHICAL_BACKEND, *EVOLVELAB_MEMORY_BACKENDS, REASONING_BANK_BACKEND)
    EvolveLabMemoryProviderAdapter = None
    def is_evolvelab_backend(name: str | None) -> bool:
        return (name or "").strip() in EVOLVELAB_MEMORY_BACKENDS
    def is_reasoning_bank_backend(name: str | None) -> bool:
        return (name or "").strip() == REASONING_BANK_BACKEND
    def is_provider_backend(name: str | None) -> bool:
        return is_evolvelab_backend(name) or is_reasoning_bank_backend(name)
try:
    from opd_evolver.memory.reasoning_bank_adapter import ReasoningBankMemoryProviderAdapter
except Exception:
    ReasoningBankMemoryProviderAdapter = None
try:
    from opd_evolver.pipelines.memory_pipeline import MemoryAugmentedPipeline
except Exception:
    MemoryAugmentedPipeline = object
_INTERCODE_HELPERS: Any | None = None
@dataclass
class MemoryArenaSession:
    dataset: str
    row_id: str
    paper_name: str
    session_index: int
    background: str
    question: str
    gold_answer: str
    raw_row: dict[str, Any]
@dataclass
class MemoryArenaTaskRow:
    dataset: str
    row_id: str
    paper_name: str
    sessions: list[MemoryArenaSession]
    raw_row: dict[str, Any]
@dataclass
class PromptBuildResult:
    prompt: str
    context_for_trace: str
    truncated: bool
    original_context_tokens: int | None
    final_prompt_tokens: int | None
    original_context_chars: int
    final_prompt_chars: int
def _load_intercode_eval_helpers() -> Any:
    global _INTERCODE_HELPERS
    if _INTERCODE_HELPERS is not None:
        return _INTERCODE_HELPERS
    path = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_intercode.py"
    spec = importlib.util.spec_from_file_location("bench_simple_intercode_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import InterCode eval helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _INTERCODE_HELPERS = module
    return module
def _load_simple_scalar_yaml(config_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line or line.startswith("-"):
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        if not value:
            data[key] = None
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            data[key] = value[1:-1]
            continue
        lowered = value.lower()
        if lowered in {"null", "none", "~"}:
            data[key] = None
        elif lowered == "true":
            data[key] = True
        elif lowered == "false":
            data[key] = False
        else:
            try:
                data[key] = int(value)
            except ValueError:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
    return data
def load_task_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using CLI defaults")
        return {}
    if config_path.suffix == ".json":
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to load JSON config {config_path}: {exc}")
            return {}
    try:
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return _load_simple_scalar_yaml(config_path)
def _coalesce(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default
def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).replace(",", " ").split()
    return [str(part).strip() for part in items if str(part).strip()]
def _safe_model_label(model_name: str) -> str:
    return str(model_name).replace("/", "__").replace(":", "_")
def _safe_path_part(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "item"
def _method_file_suffix(args: argparse.Namespace) -> str:
    if args.method == "longcontext":
        return "longcontext"
    if args.method == "opd_evolver":
        return OPD_HIERARCHICAL_BACKEND
    return str(args.memory_backend)
def _output_method_dir(args: argparse.Namespace) -> str:
    if args.method == "opd_evolver":
        return "opd_evolver"
    return str(args.method)
def _dataset_label(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return datasets[0]
    return "+".join(datasets)
def _default_output_dir(args: argparse.Namespace, datasets: list[str]) -> Path:
    return (
        PROJECT_ROOT
        / "workspace"
        / "baselines"
        / "memoryarena"
        / _safe_model_label(args.model)
        / _output_method_dir(args)
        / _dataset_label(datasets)
    )
def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config).expanduser() if pre_args.config else None
    config = load_task_config(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=pre_args.config, help="YAML/JSON config path")
    ap.add_argument(
        "--datasets",
        default=_coalesce(config, "datasets", default=",".join(DEFAULT_DATASETS)),
        help="Comma/space separated MemoryArena dataset names.",
    )
    ap.add_argument("--data-root", default=_coalesce(config, "data_root", default=str(DEFAULT_DATA_ROOT)))
    ap.add_argument("--samples", type=int, default=_coalesce(config, "samples", default=None))
    ap.add_argument("--sample-seed", type=int, default=int(_coalesce(config, "sample_seed", default=0)))
    ap.add_argument("--task-ids", default=_coalesce(config, "task_ids", default=None))
    ap.add_argument("--output-dir", default=_coalesce(config, "output_dir", default=None))
    ap.add_argument("--answers-file", default=_coalesce(config, "answers_file", default=None))
    ap.add_argument("--dry-run-load", action="store_true", default=bool(_coalesce(config, "dry_run_load", default=False)))
    ap.add_argument(
        "--method",
        choices=["longcontext", "memory_provider", "opd_evolver"],
        default=_coalesce(config, "method", default="longcontext"),
    )
    ap.add_argument(
        "--memory-backend",
        choices=ALL_MEMORY_BACKENDS,
        default=_coalesce(config, "memory_backend", default=OPD_HIERARCHICAL_BACKEND),
        help="Used with --method memory_provider; opd_hierarchical dispatches to OPD Evolver.",
    )
    ap.add_argument("--memory-storage-dir", default=_coalesce(config, "memory_storage_dir", default=None))
    ap.add_argument("--memory-retrieval-top-k", type=int, default=int(_coalesce(config, "memory_retrieval_top_k", default=3)))
    ap.add_argument("--memory-min-similarity", type=float, default=float(_coalesce(config, "memory_min_similarity", default=0.3)))
    ap.add_argument("--memory-cold-start-threshold", type=int, default=int(_coalesce(config, "memory_cold_start_threshold", default=0)))
    ap.add_argument("--memory-tiers", default=_coalesce(config, "memory_tiers", default=None))
    ap.add_argument("--embedding-provider", default=_coalesce(config, "embedding_provider", default="local"))
    ap.add_argument("--embedding-model", default=_coalesce(config, "embedding_model", default="Qwen/Qwen3-Embedding-0.6B"))
    ap.add_argument("--model", default=_coalesce(config, "model", default="qwen/qwen3.5-9b"))
    ap.add_argument("--selector-model", default=_coalesce(config, "selector_model", default=None))
    ap.add_argument("--reflection-model", default=_coalesce(config, "reflection_model", default=None))
    ap.add_argument("--concurrency", type=int, default=int(_coalesce(config, "concurrency", default=1)))
    ap.add_argument("--llm-max-completion-tokens", type=int, default=int(_coalesce(config, "llm_max_completion_tokens", default=4096)))
    ap.add_argument("--prompt-safety-buffer", type=int, default=int(_coalesce(config, "prompt_safety_buffer", default=512)))
    ap.add_argument("--vllm", action=argparse.BooleanOptionalAction, default=bool(_coalesce(config, "vllm", default=False)))
    ap.add_argument(
        "--vllm-model-dir",
        default=_coalesce(config, "vllm_model_dir", default=None),
        help="HF tokenizer snapshot path or hub id; default: auto-resolve from HF_HOME/HF_HUB_CACHE for --model.",
    )
    ap.add_argument("--vllm-host", default=_coalesce(config, "vllm_host", default="127.0.0.1"))
    ap.add_argument("--vllm-port", type=int, default=int(_coalesce(config, "vllm_port", default=8000)))
    ap.add_argument("--vllm-tp", type=int, default=int(_coalesce(config, "vllm_tp", default=1)))
    ap.add_argument("--vllm-max-model-len", type=int, default=int(_coalesce(config, "vllm_max_model_len", default=32768)))
    ap.add_argument("--vllm-gpus", default=_coalesce(config, "vllm_gpus", default="0"))
    ap.add_argument("--vllm-gpu-mem", type=float, default=float(_coalesce(config, "vllm_gpu_mem", default=0.9)))
    ap.add_argument("--vllm-timeout", type=int, default=int(_coalesce(config, "vllm_timeout", default=300)))
    ap.add_argument("--vllm-lora-module", default=_coalesce(config, "vllm_lora_module", default=None))
    ap.add_argument("--vllm-lora-modules", action="append", default=list(_coalesce(config, "vllm_lora_modules", default=[]) or []))
    ap.add_argument("--vllm-task-lora-module", default=_coalesce(config, "vllm_task_lora_module", default=None))
    ap.add_argument("--vllm-selector-lora-module", default=_coalesce(config, "vllm_selector_lora_module", default=None))
    ap.add_argument("--vllm-reflection-lora-module", default=_coalesce(config, "vllm_reflection_lora_module", default=None))
    ap.add_argument("--vllm-lora-max-rank", type=int, default=int(_coalesce(config, "vllm_lora_max_rank", default=32)))
    ap.add_argument("--vllm-max-loras", type=int, default=_coalesce(config, "vllm_max_loras", default=None))
    ap.add_argument(
        "--openai-base-url",
        default=_coalesce(config, "openai_base_url", default=None),
        help="Existing OpenAI-compatible generation endpoint. Mutually exclusive with --vllm.",
    )
    ap.add_argument("--openai-api-key", default=_coalesce(config, "openai_api_key", default="EMPTY"))
    enable_thinking_raw = _coalesce(config, "enable_thinking", default=None)
    ap.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None if enable_thinking_raw is None else bool(enable_thinking_raw),
        help="Forward chat_template_kwargs.enable_thinking to vLLM/Qwen OpenAI endpoints.",
    )
    ap.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=bool(_coalesce(config, "evaluate", default=True)))
    ap.add_argument("--evaluate-only", action="store_true", default=bool(_coalesce(config, "evaluate_only", default=False)))
    ap.add_argument("--judge-model", default=_coalesce(config, "judge_model", default=DEFAULT_JUDGE_MODEL))
    ap.add_argument("--judge-max-concurrency", type=int, default=int(_coalesce(config, "judge_max_concurrency", default=3)))
    ap.add_argument("--judge-vllm", action=argparse.BooleanOptionalAction, default=bool(_coalesce(config, "judge_vllm", default=True)))
    ap.add_argument("--judge-vllm-model-dir", default=_coalesce(config, "judge_vllm_model_dir", default=None))
    ap.add_argument("--judge-vllm-host", default=_coalesce(config, "judge_vllm_host", default="127.0.0.1"))
    ap.add_argument("--judge-vllm-port", type=int, default=int(_coalesce(config, "judge_vllm_port", default=8002)))
    ap.add_argument("--judge-vllm-gpus", default=_coalesce(config, "judge_vllm_gpus", default="0,1"))
    ap.add_argument("--judge-vllm-tp", type=int, default=int(_coalesce(config, "judge_vllm_tp", default=2)))
    ap.add_argument("--judge-vllm-max-model-len", type=int, default=int(_coalesce(config, "judge_vllm_max_model_len", default=32000)))
    ap.add_argument("--judge-vllm-gpu-mem", type=float, default=float(_coalesce(config, "judge_vllm_gpu_mem", default=0.9)))
    ap.add_argument("--judge-vllm-timeout", type=int, default=int(_coalesce(config, "judge_vllm_timeout", default=300)))
    ap.add_argument(
        "--judge-openai-base-url",
        default=_coalesce(config, "judge_openai_base_url", default=None),
        help="Existing OpenAI-compatible judge endpoint; overrides judge host/port.",
    )
    ap.add_argument("--judge-openai-api-key", default=_coalesce(config, "judge_openai_api_key", default="EMPTY"))
    ns = ap.parse_args()
    if isinstance(ns.memory_tiers, str) and not ns.memory_tiers.strip():
        ns.memory_tiers = None
    _maybe_autoresolve_vllm_model_dir(ns)
    return ns
def _candidate_dataset_files(data_root: Path, dataset: str) -> list[Path]:
    raw = Path(dataset).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.extend([raw / "data.jsonl" if raw.is_dir() else raw, raw])
    candidates.extend(
        [
            data_root / dataset / "data.jsonl",
            data_root / f"{dataset}.jsonl",
            data_root / dataset,
        ]
    )
    out: list[Path] = []
    for path in candidates:
        if path not in out:
            out.append(path)
    return out
def _resolve_dataset_file(data_root: Path, dataset: str) -> Path:
    for path in _candidate_dataset_files(data_root, dataset):
        if path.is_file():
            return path
    searched = "\n".join(str(p) for p in _candidate_dataset_files(data_root, dataset))
    raise FileNotFoundError(f"Could not find MemoryArena dataset {dataset!r}. Searched:\n{searched}")
def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]
def load_memoryarena_rows(data_root: Path, dataset: str) -> list[MemoryArenaTaskRow]:
    path = _resolve_dataset_file(data_root, dataset)
    rows: list[MemoryArenaTaskRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            row_id = str(raw.get("id") or raw.get("uid") or f"{dataset}_{line_no - 1}")
            paper_name = str(raw.get("paper_name") or raw.get("paper") or raw.get("title") or "")
            backgrounds = _as_list(raw.get("backgrounds") if "backgrounds" in raw else raw.get("background"))
            questions = _as_list(raw.get("questions") if "questions" in raw else raw.get("question"))
            answers = _as_list(raw.get("answers") if "answers" in raw else raw.get("answer"))
            if not questions:
                raise ValueError(f"MemoryArena row has no questions at {path}:{line_no} (id={row_id})")
            sessions: list[MemoryArenaSession] = []
            for idx, question in enumerate(questions):
                bg = backgrounds[idx] if idx < len(backgrounds) else (backgrounds[-1] if backgrounds else "")
                ans = answers[idx] if idx < len(answers) else ""
                sessions.append(
                    MemoryArenaSession(
                        dataset=dataset,
                        row_id=row_id,
                        paper_name=paper_name,
                        session_index=idx,
                        background=str(bg or ""),
                        question=str(question or ""),
                        gold_answer=str(ans or ""),
                        raw_row=raw,
                    )
                )
            rows.append(
                MemoryArenaTaskRow(
                    dataset=dataset,
                    row_id=row_id,
                    paper_name=paper_name,
                    sessions=sessions,
                    raw_row=raw,
                )
            )
    return rows
def load_all_rows(args: argparse.Namespace) -> list[MemoryArenaTaskRow]:
    data_root = Path(args.data_root).expanduser()
    datasets = _split_csv(args.datasets) or list(DEFAULT_DATASETS)
    rows: list[MemoryArenaTaskRow] = []
    for dataset in datasets:
        rows.extend(load_memoryarena_rows(data_root, dataset))
    return select_rows(args, rows)
def select_rows(args: argparse.Namespace, rows: list[MemoryArenaTaskRow]) -> list[MemoryArenaTaskRow]:
    selected = list(rows)
    task_ids = set(_split_csv(args.task_ids))
    if task_ids:
        selected = [row for row in selected if row.row_id in task_ids or f"{row.dataset}:{row.row_id}" in task_ids]
    if args.samples is not None and args.samples < len(selected):
        rng = random.Random(args.sample_seed)
        selected = rng.sample(selected, args.samples)
    return selected
def _is_hf_snapshot_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "config.json").is_file()
def _pick_latest_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if _is_hf_snapshot_dir(cand):
            return cand
    return None
def _auto_resolve_hf_snapshot(model_id_or_path: str | None) -> str | None:
    if not model_id_or_path:
        return None
    candidate = Path(model_id_or_path).expanduser()
    if _is_hf_snapshot_dir(candidate):
        return str(candidate.resolve())
    latest = _pick_latest_snapshot(candidate)
    if latest is not None:
        return str(latest.resolve())
    if "/" in model_id_or_path and not str(candidate).startswith("/"):
        cache_candidates = [
            Path(os.environ.get("HF_HUB_CACHE", "")),
            Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else Path(""),
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("/root/.cache/huggingface/hub"),
            Path("/path/to/hf/hub"),
        ]
        for cache in cache_candidates:
            if not str(cache):
                continue
            snapshots = cache / f"models--{model_id_or_path.replace('/', '--')}" / "snapshots"
            latest = _pick_latest_snapshot(snapshots)
            if latest is not None:
                return str(latest.resolve())
    return None
def _hub_id_for_tokenizer(model: str) -> str:
    raw = str(model).strip()
    m = raw.lower().replace("_", "-")
    if "qwen3-4b" in m or "qwen3.5-4b" in m:
        return "Qwen/Qwen3-4B"
    if "qwen3.5-9b" in m or "qwen3-9b" in m:
        return "Qwen/Qwen3.5-9B"
    if "9b" in m and "4b" not in m and "qwen" in m:
        return "Qwen/Qwen3.5-9B"
    if "4b" in m and "qwen" in m:
        return "Qwen/Qwen3-4B"
    if "/" in raw and not raw.startswith("/") and not raw.startswith("."):
        return raw
    return "Qwen/Qwen3.5-9B"
def _maybe_autoresolve_vllm_model_dir(ns: argparse.Namespace) -> None:
    vdir = ns.vllm_model_dir
    if isinstance(vdir, str) and not vdir.strip():
        vdir = None
        ns.vllm_model_dir = None
    if vdir:
        return
    hub_id = _hub_id_for_tokenizer(ns.model)
    resolved = _auto_resolve_hf_snapshot(hub_id)
    if resolved:
        ns.vllm_model_dir = resolved
        logger.info(
            "Auto-resolved --vllm-model-dir for model {!r} (hub {!r}): {}",
            ns.model,
            hub_id,
            resolved,
        )
    else:
        logger.info(
            "No local HF snapshot for hub id {!r} (model {!r}); "
            "set --vllm-model-dir or download the model; using char estimate for prompt length.",
            hub_id,
            ns.model,
        )
def _resolve_qwen32b_snapshot(args: argparse.Namespace) -> str:
    candidates = [
        args.judge_vllm_model_dir,
        os.environ.get("QWEN32B_MODEL_DIR"),
        os.environ.get("QWEN3_32B_MODEL_DIR"),
        args.judge_model,
    ]
    for candidate in candidates:
        resolved = _auto_resolve_hf_snapshot(str(candidate)) if candidate else None
        if resolved:
            return resolved
    globs = [
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*",
        "/path/to/models/Qwen3-32B/snapshots/*",
    ]
    for pattern in globs:
        matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
        for match in matches:
            if _is_hf_snapshot_dir(Path(match)):
                return str(Path(match).resolve())
    return str(args.judge_vllm_model_dir or args.judge_model)
def _load_tokenizer(model_dir: str | None) -> Any | None:
    if not model_dir:
        return None
    resolved = _auto_resolve_hf_snapshot(str(model_dir)) or str(model_dir)
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    except Exception as exc:
        logger.warning(f"Failed to load tokenizer from {resolved}; using char estimate: {exc}")
        return None
def _encode(tokenizer: Any | None, text: str) -> list[int] | None:
    if tokenizer is None:
        return None
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return None
def _truncate_text(text: str, *, tokenizer: Any | None, max_tokens: int) -> tuple[str, bool, int | None]:
    token_ids = _encode(tokenizer, text)
    if token_ids is None:
        max_chars = max(400, max_tokens * 4)
        if len(text) <= max_chars:
            return text, False, None
        head = int(max_chars * 0.6)
        tail = max_chars - head
        return text[:head] + "\n\n... [TRUNCATED CONTEXT] ...\n\n" + text[-tail:], True, None
    original_tokens = len(token_ids)
    if original_tokens <= max_tokens:
        return text, False, original_tokens
    head_tokens = int(max_tokens * 0.6)
    tail_tokens = max_tokens - head_tokens
    truncated_ids = token_ids[:head_tokens] + token_ids[-tail_tokens:]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True), True, original_tokens
def build_prompt(
    session: MemoryArenaSession,
    *,
    tokenizer: Any | None,
    max_model_len: int,
    max_completion_tokens: int,
    safety_buffer: int,
    retrieved_memory: str = "",
) -> PromptBuildResult:
    memory_block = ""
    if retrieved_memory.strip():
        memory_block = (
            "## Retrieved Memory\n"
            "Use the following reusable problem-solving guidance when helpful. "
            "Ignore it if it conflicts with the current problem.\n\n"
            f"{retrieved_memory.strip()}\n\n"
        )
    context = (
        f"{memory_block}"
        f"## Paper / Source\n{session.paper_name or 'unknown'}\n\n"
        f"## Background\n{session.background}\n\n"
        f"## Question\n{session.question}\n"
    )
    suffix = (
        "\n\n## Instructions\n"
        "Solve the formal reasoning problem carefully. Provide a concise derivation if useful, "
        "then finish with exactly one final answer line.\n\n"
        "Required final line format:\n"
        "Final Answer: <your final answer>\n"
    )
    suffix_tokens = _encode(tokenizer, suffix)
    suffix_budget = len(suffix_tokens) if suffix_tokens is not None else max(1, len(suffix) // 4)
    target_context_tokens = max(100, max_model_len - max_completion_tokens - safety_buffer - suffix_budget)
    truncated_context, truncated, original_context_tokens = _truncate_text(
        context,
        tokenizer=tokenizer,
        max_tokens=target_context_tokens,
    )
    prompt = truncated_context + suffix
    final_ids = _encode(tokenizer, prompt)
    return PromptBuildResult(
        prompt=prompt,
        context_for_trace=truncated_context,
        truncated=truncated,
        original_context_tokens=original_context_tokens,
        final_prompt_tokens=len(final_ids) if final_ids is not None else None,
        original_context_chars=len(context),
        final_prompt_chars=len(prompt),
    )
def parse_final_answer(response: str) -> str:
    text = response or ""
    text = re.sub(r"(?is)<\s*think\s*>.*?</\s*think\s*>", "", text).strip()
    matches = list(re.finditer(r"(?im)^\s*Final\s+Answer\s*:\s*(.+?)\s*$", text))
    if matches:
        return matches[-1].group(1).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()
def task_description_for_session(session: MemoryArenaSession) -> str:
    return (
        "MemoryArena formal reasoning task.\n"
        f"Dataset: {session.dataset}\n"
        f"Paper: {session.paper_name or 'unknown'}\n"
        f"Session: {session.session_index}\n\n"
        f"Question:\n{session.question}"
    ).strip()
def memory_context_for_session(session: MemoryArenaSession) -> str:
    return f"Background:\n{session.background}\n\nQuestion:\n{session.question}"
def execution_trace_for_session(
    session: MemoryArenaSession,
    *,
    raw_response: str,
    prediction: str,
    selected_memory: str,
) -> list[dict[str, Any]]:
    return [
        {
            "action": {
                "action": "answer_memoryarena_formal_reasoning",
                "params": {
                    "dataset": session.dataset,
                    "row_id": session.row_id,
                    "session_index": session.session_index,
                },
            },
            "observation": {"output": raw_response},
            "reward": 1.0,
            "memoryarena": {
                "dataset": session.dataset,
                "row_id": session.row_id,
                "paper_name": session.paper_name,
                "session_index": session.session_index,
                "background": session.background,
                "question": session.question,
                "prediction": prediction,
                "gold_answer": session.gold_answer,
                "selected_memory": selected_memory,
            },
        }
    ]
MEMORYARENA_REFLECTION_PROMPT = """You are a memory writer for an OPD Evolver agent evaluated on MemoryArena formal reasoning tasks.

The agent solved a math or physics formal-reasoning question. Extract reusable reasoning guidance that can help future MemoryArena questions. Prefer general methods, traps, and verification procedures over facts specific to this exact paper.

## Task
{task_description}

## Dataset
{dataset}

## Background
{background}

## Question
{question}

## Predicted Answer
{prediction}

## Gold Answer
{gold_answer}

## Raw Response
{raw_response}

## Output Format
Respond with a JSON object:
```json
{{
  "new_skills": [
    {{
      "description": "Reusable formal reasoning method",
      "category": "memoryarena_formal_reasoning",
      "technique": "short technique name",
      "preconditions": "When this applies",
      "steps": ["Step 1", "Step 2", "Step 3"]
    }}
  ],
  "new_tips": [
    {{
      "content": "A specific reasoning heuristic or common trap",
      "category": "memoryarena_formal_reasoning",
      "severity": "info",
      "trigger": "When this situation appears"
    }}
  ],
  "new_tools": [],
  "key_learnings": ["Main takeaway"],
  "should_save_trajectory": true,
  "trajectory_outcome": "success"
}}
```

Rules:
- Keep memory concise and reusable.
- Do not invent facts beyond the task.
- Output JSON only, no other text:"""
def _truncate_chars_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + f"\n\n... [TRUNCATED: {len(text) - max_chars} chars omitted] ...\n\n" + text[-tail:]
class MemoryArenaMemoryPipeline(MemoryAugmentedPipeline):
    async def _reflect_on_execution(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
        success: bool,
        total_reward: float,
    ) -> tuple[Any, str | None]:
        from opd_evolver.pipelines.memory_prompts import parse_reflection_response
        from opd_evolver.pipelines.types import ReflectionResult
        ma = {}
        if execution_trace:
            ma = execution_trace[0].get("memoryarena", {}) or {}
        outcome = "SUCCESS" if success else ("PARTIAL" if total_reward > 0 else "FAILURE")
        prompt = MEMORYARENA_REFLECTION_PROMPT.format(
            task_description=task_description,
            dataset=ma.get("dataset", ""),
            background=_truncate_chars_middle(str(ma.get("background", "")), 20000),
            question=_truncate_chars_middle(str(ma.get("question", "")), 12000),
            prediction=_truncate_chars_middle(str(ma.get("prediction", "")), 12000),
            gold_answer=_truncate_chars_middle(str(ma.get("gold_answer", "")), 12000),
            raw_response=_truncate_chars_middle(str(ma.get("raw_response", "")), 20000),
        )
        reflect_llm = self._memory_writer_llm or self.llm
        try:
            response = await reflect_llm(prompt)
            if not response:
                raise ValueError("Empty LLM response")
            try:
                return parse_reflection_response(response, outcome), response
            except Exception as exc:
                logger.error(f"[MemoryArena] OPD writer reflection parse failed: {exc}")
                return (
                    ReflectionResult(
                        new_skills=[],
                        new_tips=[],
                        new_tools=[],
                        key_learnings=["MemoryArena formal reasoning task completed"],
                        should_save_trajectory=success,
                        trajectory_outcome=outcome.lower(),
                    ),
                    response,
                )
        except Exception as exc:
            logger.error(f"[MemoryArena] OPD writer reflection failed: {exc}")
            return (
                ReflectionResult(
                    new_skills=[],
                    new_tips=[],
                    new_tools=[],
                    key_learnings=["MemoryArena formal reasoning task completed"],
                    should_save_trajectory=success,
                    trajectory_outcome=outcome.lower(),
                ),
                None,
            )
def _selected_memory_count(filtered: Any | None) -> int:
    if filtered is None:
        return 0
    try:
        return sum(len(ids) for ids in filtered.get_all_selected_ids().values())
    except Exception:
        return 0
def _created_memory_count_from_reflection(reflection: Any) -> int:
    if reflection is None:
        return 0
    return (
        len(getattr(reflection, "new_skills", []) or [])
        + len(getattr(reflection, "new_tips", []) or [])
        + len(getattr(reflection, "new_tools", []) or [])
        + (1 if getattr(reflection, "should_save_trajectory", False) else 0)
    )
def _memory_storage_for(args: argparse.Namespace, output_dir: Path, dataset: str) -> Path:
    if args.memory_storage_dir:
        return Path(args.memory_storage_dir).expanduser() / dataset
    backend = OPD_HIERARCHICAL_BACKEND if args.method == "opd_evolver" else str(args.memory_backend)
    return output_dir / "memory" / backend / dataset
def _make_provider_adapter(args: argparse.Namespace, storage_dir: Path) -> Any:
    if is_reasoning_bank_backend(args.memory_backend):
        if ReasoningBankMemoryProviderAdapter is None:
            raise RuntimeError("ReasoningBankMemoryProviderAdapter is unavailable; install runtime deps.")
        return ReasoningBankMemoryProviderAdapter(
            storage_dir=storage_dir,
            model_name=args.model,
            max_completion_tokens=args.llm_max_completion_tokens,
            retrieval_top_k=args.memory_retrieval_top_k,
            min_similarity=args.memory_min_similarity,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
        )
    if EvolveLabMemoryProviderAdapter is None:
        raise RuntimeError("EvolveLabMemoryProviderAdapter is unavailable; install the OpenAI SDK/runtime deps.")
    return EvolveLabMemoryProviderAdapter(
        backend=args.memory_backend,
        storage_dir=storage_dir,
        model_name=args.model,
        max_completion_tokens=args.llm_max_completion_tokens,
    )
def _write_trajectory(output_dir: Path, answer_row: dict[str, Any]) -> None:
    traj_dir = output_dir / "trajectories" / str(answer_row["dataset"])
    traj_dir.mkdir(parents=True, exist_ok=True)
    path = traj_dir / f"{_safe_path_part(answer_row['row_id'])}_s{answer_row['session_index']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(answer_row, f, ensure_ascii=False, indent=2)
SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "dataset",
    "row_id",
    "paper_name",
    "session_index",
    "method",
    "memory_backend",
    "context_truncated",
    "original_context_tokens",
    "final_prompt_tokens",
    "original_context_chars",
    "final_prompt_chars",
    "selected_memory_count",
    "created_memory_count",
    "memory_storage_dir",
    "prediction_empty",
    "writer_error",
    "selector_error",
    "error",
)
def init_streaming_outputs(answers_path: Path, summary_path: Path) -> None:
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text("", encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDNAMES), extrasaction="ignore", restval="").writeheader()
def append_streaming_session(
    answers_path: Path,
    summary_path: Path,
    output_dir: Path,
    answer_row: dict[str, Any],
    summary_row: dict[str, Any],
) -> None:
    with answers_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(answer_row, ensure_ascii=False) + "\n")
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(SUMMARY_FIELDNAMES), extrasaction="ignore", restval="").writerow(summary_row)
    _write_trajectory(output_dir, answer_row)
async def run_session_longcontext(
    session: MemoryArenaSession,
    *,
    llm: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _base_summary(session, args, memory_storage_dir="")
    try:
        built = build_prompt(
            session,
            tokenizer=tokenizer,
            max_model_len=args.vllm_max_model_len,
            max_completion_tokens=args.llm_max_completion_tokens,
            safety_buffer=args.prompt_safety_buffer,
        )
        response = await llm(built.prompt, max_tokens=args.llm_max_completion_tokens)
        prediction = parse_final_answer(response or "")
        summary.update(_prompt_summary_fields(built))
        summary["prediction_empty"] = not bool(prediction.strip())
        return _answer_row(session, args, response or "", prediction, "", built), summary
    except Exception as exc:
        logger.exception(
            "MemoryArena session failed: {}/{}/{}: {}",
            session.dataset,
            session.row_id,
            session.session_index,
            exc,
        )
        summary["error"] = str(exc)
        return _answer_row(session, args, "", "", "", None), summary
async def run_task_row_memory_provider(
    row: MemoryArenaTaskRow,
    *,
    llm: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    output_dir: Path,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if not row.sessions:
        return results
    storage_dir = _memory_storage_for(args, output_dir, row.sessions[0].dataset)
    adapter = _make_provider_adapter(args, storage_dir)
    for session in row.sessions:
        summary = _base_summary(session, args, memory_storage_dir=str(storage_dir))
        selected_memory = ""
        task_id = f"memoryarena_{session.dataset}_{session.row_id}_s{session.session_index}"
        try:
            try:
                selected_memory = await adapter.provide_begin(
                    task_description=task_description_for_session(session),
                    context=memory_context_for_session(session),
                    task_id=task_id,
                )
                summary["selected_memory_count"] = 1 if selected_memory else 0
            except Exception as exc:
                summary["selector_error"] = str(exc)
                logger.warning(f"[MemoryArena] provider failed for {task_id}: {exc}", exc_info=True)
            built = build_prompt(
                session,
                tokenizer=tokenizer,
                max_model_len=args.vllm_max_model_len,
                max_completion_tokens=args.llm_max_completion_tokens,
                safety_buffer=args.prompt_safety_buffer,
                retrieved_memory=selected_memory,
            )
            response = await llm(built.prompt, max_tokens=args.llm_max_completion_tokens)
            prediction = parse_final_answer(response or "")
            summary.update(_prompt_summary_fields(built))
            summary["prediction_empty"] = not bool(prediction.strip())
            try:
                ok, msg = await adapter.take_in(
                    task_description=task_description_for_session(session),
                    trajectory=execution_trace_for_session(
                        session,
                        raw_response=response or "",
                        prediction=prediction,
                        selected_memory=selected_memory,
                    ),
                    success=True,
                    result={"success": True, "prediction": prediction, "gold_answer": session.gold_answer},
                    metadata={
                        "task_id": task_id,
                        "dataset": session.dataset,
                        "row_id": session.row_id,
                        "session_index": session.session_index,
                    },
                )
                summary["created_memory_count"] = 1 if ok else 0
                if not ok:
                    summary["writer_error"] = msg
            except Exception as exc:
                summary["writer_error"] = str(exc)
                logger.warning(f"[MemoryArena] provider writer failed for {task_id}: {exc}", exc_info=True)
            results.append((_answer_row(session, args, response or "", prediction, selected_memory, built), summary))
        except Exception as exc:
            logger.exception("MemoryArena provider session failed: {}: {}", task_id, exc)
            summary["error"] = str(exc)
            results.append((_answer_row(session, args, "", "", selected_memory, None), summary))
    return results
async def run_task_row_opd_evolver(
    row: MemoryArenaTaskRow,
    *,
    llm: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    output_dir: Path,
    pipelines: dict[str, MemoryArenaMemoryPipeline],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for session in row.sessions:
        pipeline = pipelines[session.dataset]
        storage_dir = _memory_storage_for(args, output_dir, session.dataset)
        summary = _base_summary(session, args, memory_storage_dir=str(storage_dir))
        selected_memory = ""
        task_id = f"memoryarena_{session.dataset}_{session.row_id}_s{session.session_index}"
        task_description = task_description_for_session(session)
        try:
            try:
                filtered = await pipeline.pre_execution(
                    task_id=task_id,
                    task_description=task_description,
                    additional_context=memory_context_for_session(session),
                    task_type=session.dataset,
                )
                if filtered is not None:
                    selected_memory = filtered.formatted_context or ""
                    summary["selected_memory_count"] = _selected_memory_count(filtered)
            except Exception as exc:
                summary["selector_error"] = str(exc)
                logger.opt(exception=True).warning("[MemoryArena] OPD selector failed for {}: {}", task_id, exc)
            built = build_prompt(
                session,
                tokenizer=tokenizer,
                max_model_len=args.vllm_max_model_len,
                max_completion_tokens=args.llm_max_completion_tokens,
                safety_buffer=args.prompt_safety_buffer,
                retrieved_memory=selected_memory,
            )
            response = await llm(built.prompt, max_tokens=args.llm_max_completion_tokens)
            prediction = parse_final_answer(response or "")
            summary.update(_prompt_summary_fields(built))
            summary["prediction_empty"] = not bool(prediction.strip())
            try:
                reflection = await pipeline.post_execution(
                    task_id=task_id,
                    task_description=task_description,
                    execution_trace=execution_trace_for_session(
                        session,
                        raw_response=response or "",
                        prediction=prediction,
                        selected_memory=selected_memory,
                    ),
                    success=True,
                    total_reward=1.0,
                    tags=["memoryarena", session.dataset, "formal_reasoning"],
                    task_type=session.dataset,
                )
                summary["created_memory_count"] = _created_memory_count_from_reflection(reflection)
            except Exception as exc:
                summary["writer_error"] = str(exc)
                logger.opt(exception=True).warning("[MemoryArena] OPD writer failed for {}: {}", task_id, exc)
            results.append((_answer_row(session, args, response or "", prediction, selected_memory, built), summary))
        except Exception as exc:
            logger.exception("MemoryArena OPD session failed: {}: {}", task_id, exc)
            summary["error"] = str(exc)
            results.append((_answer_row(session, args, "", "", selected_memory, None), summary))
    return results
def _base_summary(session: MemoryArenaSession, args: argparse.Namespace, *, memory_storage_dir: str) -> dict[str, Any]:
    return {
        "dataset": session.dataset,
        "row_id": session.row_id,
        "paper_name": session.paper_name,
        "session_index": session.session_index,
        "method": args.method,
        "memory_backend": OPD_HIERARCHICAL_BACKEND if args.method == "opd_evolver" else args.memory_backend,
        "context_truncated": False,
        "original_context_tokens": "",
        "final_prompt_tokens": "",
        "original_context_chars": "",
        "final_prompt_chars": "",
        "selected_memory_count": 0,
        "created_memory_count": 0,
        "memory_storage_dir": memory_storage_dir,
        "prediction_empty": False,
        "writer_error": "",
        "selector_error": "",
        "error": None,
    }
def _prompt_summary_fields(built: PromptBuildResult) -> dict[str, Any]:
    return {
        "context_truncated": built.truncated,
        "original_context_tokens": built.original_context_tokens or "",
        "final_prompt_tokens": built.final_prompt_tokens or "",
        "original_context_chars": built.original_context_chars,
        "final_prompt_chars": built.final_prompt_chars,
    }
def _answer_row(
    session: MemoryArenaSession,
    args: argparse.Namespace,
    raw_response: str,
    prediction: str,
    selected_memory: str,
    built: PromptBuildResult | None,
) -> dict[str, Any]:
    return {
        "dataset": session.dataset,
        "row_id": session.row_id,
        "paper_name": session.paper_name,
        "session_index": session.session_index,
        "method": args.method,
        "memory_backend": OPD_HIERARCHICAL_BACKEND if args.method == "opd_evolver" else args.memory_backend,
        "background": session.background,
        "question": session.question,
        "gold_answer": session.gold_answer,
        "prediction": prediction,
        "raw_response": raw_response,
        "selected_memory": selected_memory,
        "prompt_context": built.context_for_trace if built is not None else "",
    }
async def _run_row_streaming(
    row: MemoryArenaTaskRow,
    *,
    llm: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    output_dir: Path,
    pipelines: dict[str, MemoryArenaMemoryPipeline],
    semaphore: asyncio.Semaphore,
    io_lock: asyncio.Lock,
    answers_path: Path,
    summary_path: Path,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    async with semaphore:
        if args.method == "longcontext":
            pairs = [
                await run_session_longcontext(
                    session,
                    llm=llm,
                    args=args,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                )
                for session in row.sessions
            ]
        elif args.method == "opd_evolver" or (
            args.method == "memory_provider" and args.memory_backend == OPD_HIERARCHICAL_BACKEND
        ):
            pairs = await run_task_row_opd_evolver(
                row,
                llm=llm,
                args=args,
                tokenizer=tokenizer,
                output_dir=output_dir,
                pipelines=pipelines,
            )
        else:
            pairs = await run_task_row_memory_provider(
                row,
                llm=llm,
                args=args,
                tokenizer=tokenizer,
                output_dir=output_dir,
            )
        async with io_lock:
            for answer_row, summary_row in pairs:
                append_streaming_session(answers_path, summary_path, output_dir, answer_row, summary_row)
        return pairs
JUDGE_SYSTEM_PROMPT = "You are a strict but fair grader for formal math and physics reasoning answers."
JUDGE_USER_PROMPT = """Grade whether the predicted answer is mathematically/physically equivalent to the gold answer.

Return JSON only with this schema:
{{"score": 0 or 1, "reason": "short reason", "normalized_gold": "...", "normalized_prediction": "..."}}

Use score 1 only if the final prediction is equivalent to the gold answer. Accept algebraic rearrangements, equivalent units, and equivalent numeric forms. Use score 0 for missing, contradictory, or materially incomplete answers.

## Dataset
{dataset}

## Background
{background}

## Question
{question}

## Gold Answer
{gold_answer}

## Predicted Answer
{prediction}
"""
def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
async def _judge_one(
    row: dict[str, Any],
    *,
    client: AsyncOpenAI,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        prompt = JUDGE_USER_PROMPT.format(
            dataset=row.get("dataset", ""),
            background=row.get("background", ""),
            question=row.get("question", ""),
            gold_answer=row.get("gold_answer", ""),
            prediction=row.get("prediction", ""),
        )
        result = {
            "dataset": row.get("dataset", ""),
            "row_id": row.get("row_id", ""),
            "paper_name": row.get("paper_name", ""),
            "session_index": row.get("session_index", ""),
            "question": row.get("question", ""),
            "gold_answer": row.get("gold_answer", ""),
            "prediction": row.get("prediction", ""),
            "score": 0,
            "reason": "",
            "normalized_gold": "",
            "normalized_prediction": "",
            "judge_error": "",
        }
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
            )
            content = ""
            if response.choices:
                msg = response.choices[0].message
                content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
                if not content and hasattr(msg, "model_dump"):
                    dumped = msg.model_dump()
                    content = dumped.get("reasoning_content") or dumped.get("reasoning") or ""
            parsed = _extract_json_object(content)
            score = parsed.get("score", 0)
            result.update(
                {
                    "score": 1 if str(score).strip() in {"1", "1.0", "true", "True"} else 0,
                    "reason": str(parsed.get("reason", "")),
                    "normalized_gold": str(parsed.get("normalized_gold", "")),
                    "normalized_prediction": str(parsed.get("normalized_prediction", "")),
                    "judge_raw_response": content,
                }
            )
        except Exception as exc:
            result["judge_error"] = str(exc)
        return result
def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(int(row.get("score") or 0))
    return {
        name: {
            "count": len(scores),
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
        }
        for name, scores in sorted(grouped.items())
    }
def _task_row_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        key = f"{row.get('dataset', 'unknown')}:{row.get('row_id', 'unknown')}"
        grouped.setdefault(key, []).append(int(row.get("score") or 0))
    return {
        name: {
            "count": len(scores),
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
            "all_correct": 1 if scores and all(score == 1 for score in scores) else 0,
        }
        for name, scores in sorted(grouped.items())
    }
async def evaluate_answers(
    *,
    answers_file: Path,
    output_file: Path,
    judge_model: str,
    base_url: str,
    api_key: str,
    judge_max_concurrency: int,
) -> dict[str, Any]:
    from openai import AsyncOpenAI
    answer_rows = []
    with answers_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                answer_rows.append(json.loads(line))
    client = AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url)
    sem = asyncio.Semaphore(max(1, judge_max_concurrency))
    evaluated = await tqdm_async.gather(
        *[_judge_one(row, client=client, judge_model=judge_model, semaphore=sem) for row in answer_rows],
        desc="MemoryArena judge",
        unit="answer",
        total=len(answer_rows),
    )
    total = len(evaluated)
    correct = sum(int(row.get("score") or 0) for row in evaluated)
    summary = {
        "config": {
            "answers_file": str(answers_file),
            "judge_model": judge_model,
            "judge_base_url": base_url,
        },
        "overall": {
            "total_questions": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        },
        "by_dataset": _stats(evaluated, "dataset"),
        "by_task_row": _task_row_stats(evaluated),
        "results": evaluated,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to: {output_file}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return summary
def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url
def _bind_llm_configs(
    *,
    model_names: list[str | None],
    served_model_name: str,
    base_url: str,
    api_key: str,
    enable_thinking: bool | None = None,
) -> None:
    from opd_evolver.base.engine.async_llm import LLMsConfig
    lc = LLMsConfig.default()
    for name in model_names:
        if not name:
            continue
        existing = lc.configs.get(name)
        cfg: dict[str, Any] = {
            "model": served_model_name if name == served_model_name else name,
            "api_key": api_key or "EMPTY",
            "base_url": base_url,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if enable_thinking is not None:
            cfg["enable_thinking"] = enable_thinking
        if existing is None:
            lc.add_config(name, cfg)
        else:
            existing.update(cfg)
            if name == served_model_name:
                existing["model"] = served_model_name
def _make_opd_pipelines(
    *,
    args: argparse.Namespace,
    datasets: list[str],
    output_dir: Path,
    llm: Any,
) -> dict[str, MemoryArenaMemoryPipeline]:
    from opd_evolver.base.engine.async_llm import LLMsConfig, create_llm_instance
    from opd_evolver.memory.memory_manager import MemoryConfig
    selector_llm = None
    if args.selector_model:
        selector_llm = create_llm_instance(
            LLMsConfig.default().get(args.selector_model),
            max_completion_tokens=args.llm_max_completion_tokens,
        )
    writer_llm = None
    if args.reflection_model:
        writer_llm = create_llm_instance(
            LLMsConfig.default().get(args.reflection_model),
            max_completion_tokens=args.llm_max_completion_tokens,
        )
    pipelines: dict[str, MemoryArenaMemoryPipeline] = {}
    for dataset in datasets:
        storage_dir = _memory_storage_for(args, output_dir, dataset)
        memory_config = MemoryConfig(
            storage_dir=str(storage_dir),
            cold_start_threshold=args.memory_cold_start_threshold,
            retrieval_top_k=args.memory_retrieval_top_k,
            min_similarity=args.memory_min_similarity,
            writer_dataset_path=None,
            selector_dataset_path=None,
            memory_tiers=args.memory_tiers,
        )
        pipelines[dataset] = MemoryArenaMemoryPipeline(
            llm=llm,
            config=memory_config,
            enabled=True,
            memory_writer_llm=writer_llm,
            memory_selector_llm=selector_llm,
        )
    return pipelines
async def _evaluate_phase(args: argparse.Namespace, answers_path: Path, output_dir: Path) -> int:
    helpers = _load_intercode_eval_helpers()
    judge_proc = None
    judge_base_url = (
        _normalize_base_url(args.judge_openai_base_url)
        if args.judge_openai_base_url
        else _normalize_base_url(f"http://{args.judge_vllm_host}:{args.judge_vllm_port}")
    )
    try:
        if args.judge_vllm and not args.judge_openai_base_url:
            model_dir = _resolve_qwen32b_snapshot(args)
            logger.info(
                "[MemoryArena] Starting judge vLLM: "
                f"model={args.judge_model} endpoint=http://{args.judge_vllm_host}:{args.judge_vllm_port}"
            )
            judge_proc = helpers.start_vllm_server(
                model_dir=model_dir,
                served_model_name=args.judge_model,
                host=args.judge_vllm_host,
                port=args.judge_vllm_port,
                tensor_parallel_size=args.judge_vllm_tp,
                max_model_len=args.judge_vllm_max_model_len,
                cuda_visible_devices=args.judge_vllm_gpus,
                gpu_memory_utilization=args.judge_vllm_gpu_mem,
                timeout=args.judge_vllm_timeout,
                lora_modules=None,
                max_lora_rank=16,
                max_loras=None,
            )
            if judge_proc is None:
                return 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"results_{answers_path.stem}_{stamp}.json"
        await evaluate_answers(
            answers_file=answers_path,
            output_file=output_file,
            judge_model=args.judge_model,
            base_url=judge_base_url,
            api_key=args.judge_openai_api_key,
            judge_max_concurrency=args.judge_max_concurrency,
        )
        return 0
    finally:
        if judge_proc is not None:
            helpers.stop_vllm_server(judge_proc)
async def main() -> int:
    args = parse_args()
    datasets = _split_csv(args.datasets) or list(DEFAULT_DATASETS)
    if args.method == "memory_provider" and args.memory_backend not in ALL_MEMORY_BACKENDS:
        raise SystemExit("--memory-backend must be one of: " + ", ".join(ALL_MEMORY_BACKENDS))
    if args.method == "memory_provider" and args.memory_backend != OPD_HIERARCHICAL_BACKEND and not is_provider_backend(args.memory_backend):
        raise SystemExit("--method memory_provider requires a provider backend or opd_hierarchical")
    answers_path = (
        Path(args.answers_file).expanduser()
        if args.answers_file
        else None
    )
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    elif args.evaluate_only and answers_path is not None:
        output_dir = answers_path.parent
    else:
        output_dir = _default_output_dir(args, datasets)
    if answers_path is None:
        answers_path = output_dir / f"answers_{args.method}_{_method_file_suffix(args)}.jsonl"
    summary_path = output_dir / "summary.csv"
    if args.evaluate_only:
        if not answers_path.exists():
            raise SystemExit(f"answers file not found: {answers_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return await _evaluate_phase(args, answers_path, output_dir)
    rows = load_all_rows(args)
    if args.dry_run_load:
        total_sessions = sum(len(row.sessions) for row in rows)
        by_dataset: dict[str, int] = {}
        for row in rows:
            by_dataset[row.dataset] = by_dataset.get(row.dataset, 0) + len(row.sessions)
        print(
            json.dumps(
                {
                    "rows": len(rows),
                    "sessions": total_sessions,
                    "by_dataset_sessions": by_dataset,
                    "first_row": rows[0].raw_row if rows else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    helpers = _load_intercode_eval_helpers()
    if helpers._apply_vllm_lora_cli(args):
        return 1
    if args.vllm and args.openai_base_url:
        raise SystemExit("Choose either --vllm or --openai-base-url, not both.")
    output_dir.mkdir(parents=True, exist_ok=True)
    init_streaming_outputs(answers_path, summary_path)
    vllm_proc = None
    generation_base_url = _normalize_base_url(args.openai_base_url) if args.openai_base_url else None
    served_model_name = getattr(args, "_vllm_base_served_model_name", None) or args.model
    if args.vllm:
        model_dir = _auto_resolve_hf_snapshot(args.vllm_model_dir) or args.vllm_model_dir
        vllm_proc = helpers.start_vllm_server(
            model_dir=model_dir,
            served_model_name=served_model_name,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel_size=args.vllm_tp,
            max_model_len=args.vllm_max_model_len,
            cuda_visible_devices=args.vllm_gpus,
            gpu_memory_utilization=args.vllm_gpu_mem,
            timeout=args.vllm_timeout,
            lora_modules=getattr(args, "_vllm_lora_modules", None),
            max_lora_rank=args.vllm_lora_max_rank,
            max_loras=args.vllm_max_loras,
        )
        if vllm_proc is None:
            return 1
        generation_base_url = _normalize_base_url(f"http://{args.vllm_host}:{args.vllm_port}")
    if generation_base_url:
        _bind_llm_configs(
            model_names=[args.model, args.selector_model, args.reflection_model, served_model_name],
            served_model_name=served_model_name,
            base_url=generation_base_url,
            api_key=args.openai_api_key,
            enable_thinking=args.enable_thinking,
        )
    try:
        from opd_evolver.base.engine.async_llm import LLMsConfig, create_llm_instance
        tokenizer = _load_tokenizer(args.vllm_model_dir)
        llm = create_llm_instance(
            LLMsConfig.default().get(args.model),
            max_completion_tokens=args.llm_max_completion_tokens,
        )
        pipelines: dict[str, MemoryArenaMemoryPipeline] = {}
        if args.method == "opd_evolver" or (
            args.method == "memory_provider" and args.memory_backend == OPD_HIERARCHICAL_BACKEND
        ):
            pipelines = _make_opd_pipelines(args=args, datasets=datasets, output_dir=output_dir, llm=llm)
        io_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        logger.info(
            f"[MemoryArena] Running rows={len(rows)} sessions={sum(len(row.sessions) for row in rows)} "
            f"model={args.model} method={args.method} backend={_method_file_suffix(args)}"
        )
        row_pairs = await tqdm_async.gather(
            *[
                _run_row_streaming(
                    row,
                    llm=llm,
                    args=args,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    pipelines=pipelines,
                    semaphore=semaphore,
                    io_lock=io_lock,
                    answers_path=answers_path,
                    summary_path=summary_path,
                )
                for row in rows
            ],
            desc=f"MemoryArena ({args.method}:{_method_file_suffix(args)})",
            unit="row",
            total=len(rows),
        )
        flat_pairs = [pair for row in row_pairs for pair in row]
        summary_rows = [pair[1] for pair in flat_pairs]
        print("\n" + "=" * 64)
        print("MEMORYARENA GENERATION RESULTS")
        print("=" * 64)
        print(f"Model:       {args.model}")
        print(f"Method:      {args.method}")
        print(f"Backend:     {_method_file_suffix(args)}")
        print(f"Rows:        {len(rows)}")
        print(f"Sessions:    {len(flat_pairs)}")
        print(f"Answers:     {answers_path}")
        print(f"Summary:     {summary_path}")
        print(f"Traj dir:    {output_dir / 'trajectories'}")
        print(f"Truncated:   {sum(1 for row in summary_rows if row.get('context_truncated'))}")
        print(f"Empty preds: {sum(1 for row in summary_rows if row.get('prediction_empty'))}")
        if args.method != "longcontext":
            print(f"Selected:    {sum(int(row.get('selected_memory_count') or 0) for row in summary_rows)}")
            print(f"Created:     {sum(int(row.get('created_memory_count') or 0) for row in summary_rows)}")
            print(f"Memory:      {output_dir / 'memory'}")
        print("=" * 64)
        if args.evaluate:
            return await _evaluate_phase(args, answers_path, output_dir)
        return 0
    finally:
        if vllm_proc is not None:
            helpers.stop_vllm_server(vllm_proc)
if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
