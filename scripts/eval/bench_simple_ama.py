#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import csv
import importlib.util
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm.asyncio import tqdm as tqdm_async
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMA_ROOT = PROJECT_ROOT / "reference" / "AMA-Bench"
CONFIG_ROOT = PROJECT_ROOT / "config" / "ama_bench"
DEFAULT_TEST_FILE = str(PROJECT_ROOT / "data" / "ama" / "open_end_qa_set.jsonl")
DEFAULT_JUDGE_CONFIG = AMA_ROOT / "configs" / "llm_judge.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from opd_evolver.pipelines.memory_pipeline import MemoryAugmentedPipeline
except Exception:
    MemoryAugmentedPipeline = object
from opd_evolver.memory.evolvelab_adapter import (
    ALL_MEMORY_BACKENDS,
    EvolveLabMemoryProviderAdapter,
    OPD_HIERARCHICAL_BACKEND,
    default_baseline_output_dir,
    is_provider_backend,
    is_reasoning_bank_backend,
)
from opd_evolver.memory.reasoning_bank_adapter import ReasoningBankMemoryProviderAdapter
logger = logging.getLogger("bench_simple_ama")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
_VLLM_DEFAULT_MODEL_DIR = (
    "/path/to/models/Qwen3.5-9B"
    "/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
_INTERCODE_HELPERS: Any | None = None
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
@dataclass
class PromptBuildResult:
    prompt: str
    reasoning_trace: str
    truncated: bool
    original_context_tokens: int | None
    final_prompt_tokens: int | None
    original_context_chars: int
    final_prompt_chars: int
def load_task_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using CLI defaults")
        return {}
    try:
        import yaml
    except Exception as exc:
        logger.warning(f"PyYAML not available; using simple scalar YAML parser for {config_path}: {exc}")
        return _load_simple_scalar_yaml(config_path)
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Config must be a mapping, got {type(data)} at: {config_path}")
            return {}
        return data
    except Exception as exc:
        logger.error(f"Failed to load config file {config_path}: {exc}")
        return {}
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
def _coalesce(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default
def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace(",", " ").split()
    return [str(part).strip() for part in items if str(part).strip()]
def _safe_model_label(model_name: str) -> str:
    return str(model_name).replace("/", "__").replace(":", "_")
def _method_file_suffix(args: argparse.Namespace) -> str:
    if args.method == "longcontext":
        return "longcontext"
    if args.method == "opd_evolver":
        return OPD_HIERARCHICAL_BACKEND
    return str(args.memory_backend)
def default_matrix_output_dir(
    model: str,
    method: str,
    memory_backend: str | None,
    subset: str,
) -> Path:
    base = PROJECT_ROOT / "workspace" / "baselines" / "ama_bench" / _safe_model_label(model)
    if method == "memory_provider":
        return base / "memory_provider" / subset / (memory_backend or "unknown")
    return base / method / subset
def _parse_indices(spec: str, total: int) -> list[int]:
    indices: list[int] = []
    for part in _split_csv(spec):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    bad = [idx for idx in indices if idx < 0 or idx >= total]
    if bad:
        raise ValueError(f"Episode index out of range: {bad[:10]} (total={total})")
    return sorted(set(indices))
def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config).expanduser() if pre_args.config else None
    config = load_task_config(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=pre_args.config, help="YAML config path")
    ap.add_argument(
        "--method",
        choices=["longcontext", "opd_evolver", "memory_provider"],
        default=_coalesce(config, "method", default="longcontext"),
    )
    ap.add_argument("--test-file", default=_coalesce(config, "test_file", default=DEFAULT_TEST_FILE))
    ap.add_argument("--subset", choices=["openend", "mcq"], default=_coalesce(config, "subset", default="openend"))
    ap.add_argument("--samples", type=int, default=_coalesce(config, "samples", default=None))
    ap.add_argument("--sample-seed", type=int, default=int(_coalesce(config, "sample_seed", default=0)))
    ap.add_argument("--domains", default=_coalesce(config, "domains", default=None))
    ap.add_argument("--episode", type=int, default=_coalesce(config, "episode", default=None))
    ap.add_argument("--episodes", default=_coalesce(config, "episodes", default=None), help="Episode indices, e.g. 0,3 or 0-9.")
    ap.add_argument("--output-dir", default=_coalesce(config, "output_dir", default=None))
    ap.add_argument("--answers-file", default=_coalesce(config, "answers_file", default=None))
    ap.add_argument("--evaluate-only", action="store_true", default=bool(_coalesce(config, "evaluate_only", default=False)))
    ap.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=bool(_coalesce(config, "evaluate", default=True)))
    ap.add_argument("--judge-config", default=_coalesce(config, "judge_config", default=str(DEFAULT_JUDGE_CONFIG)))
    ap.add_argument("--judge-server", choices=["api", "vllm"], default=_coalesce(config, "judge_server", default="vllm"))
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
    ap.add_argument("--model", default=_coalesce(config, "model", default="qwen/qwen3.5-9b"))
    ap.add_argument(
        "--llm-max-completion-tokens",
        type=int,
        default=int(_coalesce(config, "llm_max_completion_tokens", default=16384)),
        help="Max tokens for answer generation per episode.",
    )
    ap.add_argument("--concurrency", type=int, default=int(_coalesce(config, "concurrency", default=1)))
    ap.add_argument("--prompt-safety-buffer", type=int, default=int(_coalesce(config, "prompt_safety_buffer", default=512)))
    ap.add_argument("--memory-storage-dir", default=_coalesce(config, "memory_storage_dir", default=None))
    ap.add_argument(
        "--memory-backend",
        choices=ALL_MEMORY_BACKENDS,
        default=_coalesce(config, "memory_backend", default=OPD_HIERARCHICAL_BACKEND),
        help="Used with --method memory_provider.",
    )
    ap.add_argument("--memory-retrieval-top-k", type=int, default=int(_coalesce(config, "memory_retrieval_top_k", default=3)))
    ap.add_argument("--memory-min-similarity", type=float, default=float(_coalesce(config, "memory_min_similarity", default=0.3)))
    ap.add_argument("--memory-cold-start-threshold", type=int, default=int(_coalesce(config, "memory_cold_start_threshold", default=0)))
    ap.add_argument("--embedding-provider", default=_coalesce(config, "embedding_provider", default="local"))
    ap.add_argument("--embedding-model", default=_coalesce(config, "embedding_model", default="Qwen/Qwen3-Embedding-0.6B"))
    ap.add_argument(
        "--memory-tiers",
        default=_coalesce(config, "memory_tiers", default=None),
        help="Subset of skill,tip,tool,trajectory (comma-separated) or YAML list; default all tiers.",
    )
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
    ap.add_argument("--vllm", action=argparse.BooleanOptionalAction, default=bool(_coalesce(config, "vllm", default=False)))
    ap.add_argument("--vllm-model-dir", default=_coalesce(config, "vllm_model_dir", default=None))
    ap.add_argument("--vllm-host", default=_coalesce(config, "vllm_host", default="127.0.0.1"))
    ap.add_argument("--vllm-port", type=int, default=int(_coalesce(config, "vllm_port", default=8000)))
    ap.add_argument("--vllm-tp", type=int, default=int(_coalesce(config, "vllm_tp", default=2)))
    ap.add_argument("--vllm-max-model-len", type=int, default=int(_coalesce(config, "vllm_max_model_len", default=262144)))
    ap.add_argument("--vllm-gpus", default=_coalesce(config, "vllm_gpus", default="0,1"))
    ap.add_argument("--vllm-gpu-mem", type=float, default=float(_coalesce(config, "vllm_gpu_mem", default=0.9)))
    ap.add_argument("--vllm-timeout", type=int, default=int(_coalesce(config, "vllm_timeout", default=300)))
    ap.add_argument("--vllm-lora-module", default=_coalesce(config, "vllm_lora_module", default=None))
    ap.add_argument(
        "--vllm-lora-modules",
        action="append",
        default=list(_coalesce(config, "vllm_lora_modules", default=[]) or []),
    )
    ap.add_argument("--vllm-task-lora-module", default=_coalesce(config, "vllm_task_lora_module", default=None))
    ap.add_argument("--vllm-selector-lora-module", default=_coalesce(config, "vllm_selector_lora_module", default=None))
    ap.add_argument("--vllm-reflection-lora-module", default=_coalesce(config, "vllm_reflection_lora_module", default=None))
    ap.add_argument("--selector-model", default=_coalesce(config, "selector_model", default=None))
    ap.add_argument("--reflection-model", default=_coalesce(config, "reflection_model", default=None))
    ap.add_argument("--vllm-lora-max-rank", type=int, default=int(_coalesce(config, "vllm_lora_max_rank", default=32)))
    ap.add_argument("--vllm-max-loras", type=int, default=_coalesce(config, "vllm_max_loras", default=None))
    ns = ap.parse_args()
    if isinstance(ns.memory_tiers, str) and not ns.memory_tiers.strip():
        ns.memory_tiers = None
    if ns.memory_tiers is not None:
        from opd_evolver.memory.memory_manager import MemoryConfig
        try:
            MemoryConfig(memory_tiers=ns.memory_tiers, storage_dir=".").enabled_memory_tiers()
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid memory_tiers: {exc}") from exc
    if getattr(ns, "openai_base_url", None) == "":
        ns.openai_base_url = None
    _maybe_autoresolve_vllm_model_dir(ns)
    return ns
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
def _bind_llm_configs_to_vllm(args: argparse.Namespace, served_model_name: str) -> None:
    _bind_llm_configs(
        model_names=[
            served_model_name,
            getattr(args, "model", None),
            getattr(args, "selector_model", None),
            getattr(args, "reflection_model", None),
        ],
        served_model_name=served_model_name,
        base_url=f"http://{args.vllm_host}:{args.vllm_port}/v1",
        api_key="EMPTY",
    )
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
def _auto_resolve_qwen35_9b_snapshot(model_dir_hint: str | None) -> str | None:
    hints = []
    if model_dir_hint:
        hints.append(Path(model_dir_hint).expanduser())
    hints.extend(
        [
            Path(_VLLM_DEFAULT_MODEL_DIR),
            Path("/path/to/models/Qwen3.5-9B/snapshots"),
            Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots",
        ]
    )
    for hint in hints:
        if _is_hf_snapshot_dir(hint):
            return str(hint.resolve())
        latest = _pick_latest_snapshot(hint)
        if latest is not None:
            return str(latest.resolve())
    return None
def _hf_hub_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HF_HUB_CACHE"),
        str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
        str(Path.home() / ".cache" / "huggingface" / "hub"),
        "/root/.cache/huggingface/hub",
        "/path/to/hf/hub",
    ):
        if raw:
            roots.append(Path(raw))
    return roots
def _hf_snapshot_cache_dir(model_id: str, hub_root: Path) -> Path:
    return hub_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
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
        for hub_root in _hf_hub_cache_roots():
            latest = _pick_latest_snapshot(_hf_snapshot_cache_dir(model_id_or_path, hub_root))
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
    if _looks_like_placeholder(vdir):
        vdir = None
        ns.vllm_model_dir = None
    elif isinstance(vdir, str) and not vdir.strip():
        vdir = None
        ns.vllm_model_dir = None
    if vdir:
        resolved = _auto_resolve_hf_snapshot(str(vdir))
        if resolved:
            ns.vllm_model_dir = resolved
        return
    hub_id = _hub_id_for_tokenizer(ns.model)
    resolved = _auto_resolve_hf_snapshot(hub_id)
    if resolved:
        ns.vllm_model_dir = resolved
        logger.info(
            "Auto-resolved --vllm-model-dir for model %r (hub %r): %s",
            ns.model,
            hub_id,
            resolved,
        )
    else:
        logger.info(
            "No local HF snapshot for hub id %r (model %r); "
            "set --vllm-model-dir or download the model; using char estimate for prompt length.",
            hub_id,
            ns.model,
        )
def _looks_like_placeholder(path: str | None) -> bool:
    if not path:
        return True
    text = str(path)
    return "PATH/TO/YOUR/MODEL" in text or text.strip() in {"", "null", "None"}
def _load_tokenizer(model_dir: str | None) -> Any | None:
    if not model_dir or _looks_like_placeholder(model_dir):
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as exc:
        logger.warning(f"Failed to load tokenizer from {model_dir}; using char estimate: {exc}")
        return None
def _load_mapping_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to load JSON config {path}: {exc}")
            return {}
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return _load_simple_scalar_yaml(path)
def _resolve_judge_vllm_settings(args: argparse.Namespace) -> dict[str, Any]:
    judge_config = _load_mapping_file(Path(args.judge_config).expanduser())
    launch = judge_config.get("vllm_launch") if isinstance(judge_config.get("vllm_launch"), dict) else {}
    served_model_name = str(judge_config.get("model") or "Qwen/Qwen3-32B")
    requested_model = args.judge_vllm_model_dir or served_model_name
    resolved_model = _auto_resolve_hf_snapshot(str(requested_model)) or str(requested_model)
    return {
        "model_dir": resolved_model,
        "served_model_name": served_model_name,
        "host": args.judge_vllm_host or judge_config.get("vllm_host", "127.0.0.1"),
        "port": int(args.judge_vllm_port or judge_config.get("vllm_port", 8002)),
        "gpus": args.judge_vllm_gpus or launch.get("gpus", "0,1"),
        "tp": int(args.judge_vllm_tp or launch.get("tensor_parallel_size", 2)),
        "max_model_len": int(args.judge_vllm_max_model_len or launch.get("max_model_len", 32000)),
        "gpu_mem": float(args.judge_vllm_gpu_mem),
        "timeout": int(args.judge_vllm_timeout),
    }
def _encode(tokenizer: Any | None, text: str) -> list[int] | None:
    if tokenizer is None:
        return None
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return None
def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return episodes
def select_episodes(args: argparse.Namespace, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = list(episodes)
    domains = set(_split_csv(args.domains))
    if domains:
        selected = [ep for ep in selected if str(ep.get("domain", "")) in domains]
    total_after_domain = len(selected)
    if args.episode is not None:
        selected = [selected[args.episode]]
    elif args.episodes:
        selected = [selected[idx] for idx in _parse_indices(args.episodes, total_after_domain)]
    elif args.samples is not None:
        if args.samples < len(selected):
            rng = random.Random(args.sample_seed)
            selected = rng.sample(selected, args.samples)
    return selected
def trajectory_to_text(trajectory: list[dict[str, Any]]) -> str:
    parts = []
    for step in trajectory:
        turn_idx = step.get("turn_idx", 0)
        action = step.get("action", "")
        observation = step.get("observation", "")
        parts.append(f"Step {turn_idx}:")
        parts.append(f"Action: {action}")
        parts.append(f"Observation: {observation}")
        parts.append("")
    return "\n".join(parts)
def _truncate_text(
    text: str,
    *,
    tokenizer: Any | None,
    max_tokens: int,
) -> tuple[str, bool, int | None]:
    token_ids = _encode(tokenizer, text)
    if token_ids is None:
        max_chars = max(400, max_tokens * 2)
        if len(text) <= max_chars:
            return text, False, None
        head = int(max_chars * 0.6)
        tail = max_chars - head
        truncated = text[:head] + "\n\n... [TRUNCATED CONTEXT] ...\n\n" + text[-tail:]
        return truncated, True, None
    original_tokens = len(token_ids)
    if original_tokens <= max_tokens:
        return text, False, original_tokens
    head_tokens = int(max_tokens * 0.6)
    tail_tokens = max_tokens - head_tokens
    truncated_ids = token_ids[:head_tokens] + token_ids[-tail_tokens:]
    try:
        truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    except Exception:
        truncated = tokenizer.decode(truncated_ids)
    return truncated, True, original_tokens
def build_prompt(
    episode: dict[str, Any],
    *,
    subset: str,
    tokenizer: Any | None,
    max_model_len: int,
    max_completion_tokens: int,
    safety_buffer: int,
    retrieved_memory: str = "",
) -> PromptBuildResult:
    task = str(episode.get("task", ""))
    memory_block = ""
    if retrieved_memory.strip():
        memory_block = (
            "## Retrieved OPD Memory\n"
            "Use the following reusable AMA-Bench QA skills, tips, tools, or past trajectory lessons when helpful. "
            "Do not cite memory if it conflicts with the current trajectory.\n\n"
            f"{retrieved_memory.strip()}\n\n"
        )
    context = (
        f"{memory_block}"
        f"## Task Description\n{task}\n\n"
        f"## Agent Trajectory\n"
        f"The following is a step-by-step trajectory of the agent's actions and observations.\n\n"
        f"{trajectory_to_text(episode.get('trajectory', []))}"
    )
    qa_pairs = episode.get("qa_pairs", [])
    questions = [str(qa.get("question", "")) for qa in qa_pairs]
    questions_block = "\n".join(f"Question {i}: {q}\n" for i, q in enumerate(questions, 1))
    n_q = len(questions)
    if subset == "mcq":
        section_intro = "Please answer the following multiple-choice questions based on the task description and agent trajectory above."
        instructions = (
            "For each question, select all correct options using (A), (B), (C), or (D). "
            "If multiple options are correct, combine them like (A)(B).\n"
            f"- Your FIRST line must be Answer[1]: … — no analysis before it.\n"
            f"- Complete Answer[1]: through Answer[{n_q}]: in order; keep each line short.\n"
            "- Do not repeat the same explanation in loops."
        )
        answer_slots = "\n".join(
            f"Answer[{i}]: [(A)/(B)/(C)/(D) or combination such as (A)(B)]"
            for i in range(1, n_q + 1)
        )
    else:
        section_intro = (
            "Please answer the following questions based on the task description and agent trajectory above. "
            "For each question, provide a direct and concise answer."
        )
        instructions = (
            "Required format:\n"
            f"- Your FIRST line must be Answer[1]: … — no preamble, headings, or chain-of-thought before it.\n"
            f"- Fill Answer[1]: through Answer[{n_q}]: in order; one concise answer per slot (one or two sentences).\n"
            "- Do not repeat the same paragraph; finish all questions without looping.\n"
            "- Use exactly these prefixes with square brackets: Answer[1]:, Answer[2]:, …"
        )
        answer_slots = "\n".join(f"Answer[{i}]: [your answer here]" for i in range(1, n_q + 1))
    suffix = (
        f"\n\n## Questions\n{section_intro}\n\n"
        f"{questions_block}\n"
        f"## Instructions\n{instructions}\n\n"
        f"{answer_slots}\n\n"
        "(Reminder: begin your reply immediately with Answer[1]: — nothing above that line.)"
    )
    suffix_tokens = _encode(tokenizer, suffix)
    suffix_budget = len(suffix_tokens) if suffix_tokens is not None else max(1, len(suffix) // 4)
    target_context_tokens = max(
        100,
        max_model_len - max_completion_tokens - safety_buffer - suffix_budget,
    )
    truncated_context, truncated, original_context_tokens = _truncate_text(
        context,
        tokenizer=tokenizer,
        max_tokens=target_context_tokens,
    )
    prompt = truncated_context + suffix
    final_ids = _encode(tokenizer, prompt)
    final_prompt_tokens = len(final_ids) if final_ids is not None else None
    return PromptBuildResult(
        prompt=prompt,
        reasoning_trace=truncated_context,
        truncated=truncated,
        original_context_tokens=original_context_tokens,
        final_prompt_tokens=final_prompt_tokens,
        original_context_chars=len(context),
        final_prompt_chars=len(prompt),
    )
AMA_QA_REFLECTION_PROMPT = """You are a memory writer for an OPD Evolver agent evaluated on AMA-Bench.

AMA-Bench tasks provide a completed agent trajectory and questions about that trajectory. Your job is to extract reusable memory that helps future executor models answer AMA-style trajectory QA more accurately.

## AMA Task
{task_description}

## Domain Metadata
domain: {domain}
task_type: {task_type}

## Questions
{questions}

## Parsed Executor Answers
{answers}

## Raw Executor Response
{raw_response}

## Full Trajectory
{trajectory}

## What to Extract
Focus on reusable abilities for AMA-Bench QA, not facts that only identify this exact episode:
- Skills for reading long trajectories and compressing them into useful state.
- Skills for locating key steps, actions, observations, object state changes, inventory changes, command outputs, or causal transitions.
- Tips about common QA traps, such as step indexing, before/after wording, multi-step dependencies, disappeared objects, repeated actions, and final-state questions.
- Tools only when a reusable algorithm or code snippet would help search/count/compare trajectory events.

## Output Format
Respond with a JSON object:
```json
{{
  "new_skills": [
    {{
      "description": "How to answer a recurring AMA trajectory QA pattern",
      "category": "ama_trajectory_qa",
      "technique": "short technique name",
      "preconditions": "When this applies",
      "steps": ["Step 1", "Step 2", "Step 3"]
    }}
  ],
  "new_tips": [
    {{
      "content": "A specific QA heuristic or gotcha",
      "category": "ama_trajectory_qa",
      "severity": "info",
      "trigger": "When this situation appears"
    }}
  ],
  "new_tools": [
    {{
      "name": "tool_name",
      "description": "What reusable trajectory analysis it performs",
      "language": "python",
      "code": "self-contained code if genuinely useful",
      "input_description": "expected inputs",
      "output_description": "expected outputs"
    }}
  ],
  "key_learnings": [
    "Main takeaway 1",
    "Main takeaway 2"
  ],
  "should_save_trajectory": true,
  "trajectory_outcome": "success"
}}
```

Rules:
- Prefer general QA procedures over episode-specific facts.
- Keep memory concise and useful for future AMA episodes.
- Do not invent trajectory facts.
- Output JSON only, no other text:"""
def _truncate_chars_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + f"\n\n... [TRUNCATED: {len(text) - max_chars} chars omitted] ...\n\n" + text[-tail:]
def _questions_for_memory(episode: dict[str, Any]) -> str:
    qa_pairs = episode.get("qa_pairs", [])
    return "\n".join(
        f"Question {idx}: {qa.get('question', '')}"
        for idx, qa in enumerate(qa_pairs, 1)
    )
def _answers_for_memory(answer_list: list[str]) -> str:
    return "\n".join(
        f"Answer[{idx}]: {answer}"
        for idx, answer in enumerate(answer_list, 1)
    )
def _selector_query_for_episode(episode: dict[str, Any]) -> str:
    return (
        f"AMA-Bench trajectory QA task.\n"
        f"Task: {episode.get('task', '')}\n"
        f"Domain: {episode.get('domain', 'unknown')}\n"
        f"Task type: {episode.get('task_type', 'unknown')}\n\n"
        f"Questions:\n{_questions_for_memory(episode)}"
    ).strip()
def _selected_memory_count(filtered: Any | None) -> int:
    if filtered is None:
        return 0
    try:
        return sum(len(ids) for ids in filtered.get_all_selected_ids().values())
    except Exception:
        return 0
def _memory_total_items(stats: dict[str, Any] | None) -> int:
    if not stats:
        return 0
    return int(stats.get("total_items") or 0)
def _created_memory_count_from_reflection(reflection: Any) -> int:
    if reflection is None:
        return 0
    return (
        len(getattr(reflection, "new_skills", []) or [])
        + len(getattr(reflection, "new_tips", []) or [])
        + len(getattr(reflection, "new_tools", []) or [])
        + (1 if getattr(reflection, "should_save_trajectory", False) else 0)
    )
def _ama_execution_trace(
    episode: dict[str, Any],
    *,
    raw_response: str,
    answer_list: list[str],
    selected_memory: str,
) -> list[dict[str, Any]]:
    return [
        {
            "action": {
                "action": "answer_ama_questions",
                "params": {
                    "num_questions": len(episode.get("qa_pairs", [])),
                    "domain": episode.get("domain", "unknown"),
                    "task_type": episode.get("task_type", "unknown"),
                },
            },
            "observation": {
                "output": raw_response,
            },
            "reward": 1.0,
            "ama": {
                "task": episode.get("task", ""),
                "domain": episode.get("domain", "unknown"),
                "task_type": episode.get("task_type", "unknown"),
                "trajectory": trajectory_to_text(episode.get("trajectory", [])),
                "questions": _questions_for_memory(episode),
                "answers": _answers_for_memory(answer_list),
                "raw_response": raw_response,
                "selected_memory": selected_memory,
            },
        }
    ]
def _make_provider_adapter(args: argparse.Namespace, storage_dir: str | Path) -> Any:
    if is_reasoning_bank_backend(args.memory_backend):
        rb_dir = Path(storage_dir).expanduser()
        if rb_dir.name != "reasoning_bank":
            rb_dir = rb_dir / "reasoning_bank"
        return ReasoningBankMemoryProviderAdapter(
            storage_dir=rb_dir,
            model_name=args.model,
            max_completion_tokens=args.llm_max_completion_tokens,
            retrieval_top_k=args.memory_retrieval_top_k,
            min_similarity=args.memory_min_similarity,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
        )
    return EvolveLabMemoryProviderAdapter(
        backend=args.memory_backend,
        storage_dir=storage_dir,
        model_name=args.model,
        max_completion_tokens=args.llm_max_completion_tokens,
    )
class AMAQAMemoryPipeline(MemoryAugmentedPipeline):
    async def _reflect_on_execution(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
        success: bool,
        total_reward: float,
    ) -> tuple[Any, str | None]:
        from opd_evolver.pipelines.memory_prompts import parse_reflection_response
        from opd_evolver.pipelines.types import ReflectionResult
        ama = {}
        if execution_trace:
            ama = execution_trace[0].get("ama", {}) or {}
        outcome = "SUCCESS" if success else ("PARTIAL" if total_reward > 0 else "FAILURE")
        prompt = AMA_QA_REFLECTION_PROMPT.format(
            task_description=task_description,
            domain=ama.get("domain", ""),
            task_type=ama.get("task_type", ""),
            questions=_truncate_chars_middle(str(ama.get("questions", "")), 12000),
            answers=_truncate_chars_middle(str(ama.get("answers", "")), 12000),
            raw_response=_truncate_chars_middle(str(ama.get("raw_response", "")), 12000),
            trajectory=_truncate_chars_middle(str(ama.get("trajectory", "")), 50000),
        )
        reflect_llm = self._memory_writer_llm or self.llm
        try:
            response = await reflect_llm(prompt)
            if not response:
                raise ValueError("Empty LLM response")
            try:
                return parse_reflection_response(response, outcome), response
            except Exception as exc:
                logger.error(f"[AMA-Bench] AMA writer reflection parse failed: {exc}")
                return (
                    ReflectionResult(
                        new_skills=[],
                        new_tips=[],
                        new_tools=[],
                        key_learnings=["AMA-Bench trajectory QA episode completed"],
                        should_save_trajectory=success,
                        trajectory_outcome=outcome.lower(),
                    ),
                    response,
                )
        except Exception as exc:
            logger.error(f"[AMA-Bench] AMA writer reflection failed: {exc}")
            return (
                ReflectionResult(
                    new_skills=[],
                    new_tips=[],
                    new_tools=[],
                    key_learnings=["AMA-Bench trajectory QA episode completed"],
                    should_save_trajectory=success,
                    trajectory_outcome=outcome.lower(),
                ),
                None,
            )
def _strip_hidden_think_tags(text: str) -> str:
    patterns = (
        r"<\s*think\s*>[\s\S]*?</\s*think\s*>",
        r"<\s*reasoning\s*>[\s\S]*?</\s*reasoning\s*>",
    )
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()
def _parse_slots_answer_brackets(response: str, expected: int) -> list[str]:
    answers: list[str] = []
    for idx in range(1, expected + 1):
        nxt = idx + 1
        pat = rf"(?is)Answer\s*\[\s*{idx}\s*\]\s*:\s*(.*?)(?=Answer\s*\[\s*{nxt}\s*\]\s*:|$)"
        match = re.search(pat, response)
        if not match:
            answers.append("")
            continue
        text = match.group(1).strip()
        text = re.sub(r"^\[?(your answer here|answer)\]?:?\s*", "", text, flags=re.IGNORECASE).strip()
        answers.append(text)
    return answers
def _parse_slots_markdown_q(response: str, expected: int) -> list[str] | None:
    pat = re.compile(r"(?ms)^\*\*Q(\d+)\s*:\s*", re.MULTILINE)
    matches = list(pat.finditer(response))
    if not matches:
        return None
    slot_text: dict[int, str] = {}
    for i, m in enumerate(matches):
        qn = int(m.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        chunk = response[m.end() : end].strip()
        chunk = re.sub(r"\*\*\s*$", "", chunk).strip()
        if 1 <= qn <= expected:
            prev = slot_text.get(qn, "")
            if len(chunk) > len(prev):
                slot_text[qn] = chunk
    merged = [slot_text.get(i, "").strip() for i in range(1, expected + 1)]
    if not any(merged):
        return None
    return merged
def parse_answers(response: str, expected: int) -> tuple[list[str], int]:
    response = _strip_hidden_think_tags(response or "")
    if expected <= 0:
        return [], 0
    bracket = _parse_slots_answer_brackets(response, expected)
    md_q = _parse_slots_markdown_q(response, expected)
    def filled(row: list[str]) -> int:
        return sum(1 for a in row if a.strip())
    if md_q is None:
        answers = bracket
    elif filled(md_q) > filled(bracket):
        answers = md_q
    else:
        answers = [
            bracket[i] if bracket[i].strip() else md_q[i]
            for i in range(expected)
        ]
    missing = sum(1 for a in answers if not a.strip())
    if missing == expected and response.strip():
        merged = [response.strip()] + [""] * (expected - 1)
        answers = merged
        missing = sum(1 for a in answers if not a.strip())
    return answers, missing
AMA_SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "episode_id",
    "task_type",
    "domain",
    "num_questions",
    "num_turns",
    "total_tokens",
    "context_truncated",
    "original_context_tokens",
    "final_prompt_tokens",
    "original_context_chars",
    "final_prompt_chars",
    "missing_answers",
    "selected_memory_count",
    "created_memory_count",
    "memory_storage_dir",
    "writer_error",
    "selector_error",
    "error",
)
def write_answers(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(AMA_SUMMARY_FIELDNAMES),
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)
def init_ama_streaming_outputs(answers_path: Path, summary_path: Path) -> None:
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text("", encoding="utf-8")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(
            f,
            fieldnames=list(AMA_SUMMARY_FIELDNAMES),
            extrasaction="ignore",
            restval="",
        ).writeheader()
def append_ama_streaming_episode(
    answers_path: Path,
    summary_path: Path,
    answer_row: dict[str, Any],
    summary_row: dict[str, Any],
) -> None:
    with answers_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(answer_row, ensure_ascii=False) + "\n")
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(
            f,
            fieldnames=list(AMA_SUMMARY_FIELDNAMES),
            extrasaction="ignore",
            restval="",
        ).writerow(summary_row)
async def _run_longcontext_episode_streaming(
    episode: dict[str, Any],
    *,
    llm: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    semaphore: asyncio.Semaphore,
    io_lock: asyncio.Lock,
    answers_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair = await run_episode(
        episode,
        llm=llm,
        args=args,
        tokenizer=tokenizer,
        semaphore=semaphore,
    )
    async with io_lock:
        append_ama_streaming_episode(answers_path, summary_path, pair[0], pair[1])
    return pair
async def _run_opd_episode_streaming(
    episode: dict[str, Any],
    *,
    llm: Any,
    memory_pipeline: AMAQAMemoryPipeline,
    args: argparse.Namespace,
    tokenizer: Any | None,
    semaphore: asyncio.Semaphore,
    io_lock: asyncio.Lock,
    answers_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair = await run_episode_opd_evolver(
        episode,
        llm=llm,
        memory_pipeline=memory_pipeline,
        args=args,
        tokenizer=tokenizer,
        semaphore=semaphore,
    )
    async with io_lock:
        append_ama_streaming_episode(answers_path, summary_path, pair[0], pair[1])
    return pair
async def _run_memory_provider_episode_streaming(
    episode: dict[str, Any],
    *,
    llm: Any,
    adapter: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    semaphore: asyncio.Semaphore,
    io_lock: asyncio.Lock,
    answers_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair = await run_episode_memory_provider(
        episode,
        llm=llm,
        adapter=adapter,
        args=args,
        tokenizer=tokenizer,
        semaphore=semaphore,
    )
    async with io_lock:
        append_ama_streaming_episode(answers_path, summary_path, pair[0], pair[1])
    return pair
async def run_episode(
    episode: dict[str, Any],
    *,
    llm: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with semaphore:
        episode_id = episode.get("episode_id")
        qa_pairs = episode.get("qa_pairs", [])
        summary = {
            "episode_id": episode_id,
            "task_type": episode.get("task_type", "unknown"),
            "domain": episode.get("domain", "unknown"),
            "num_questions": len(qa_pairs),
            "num_turns": episode.get("num_turns", len(episode.get("trajectory", []))),
            "total_tokens": episode.get("total_tokens", ""),
            "context_truncated": False,
            "original_context_tokens": "",
            "final_prompt_tokens": "",
            "original_context_chars": "",
            "final_prompt_chars": "",
            "missing_answers": 0,
            "selected_memory_count": 0,
            "created_memory_count": 0,
            "memory_storage_dir": "",
            "writer_error": "",
            "selector_error": "",
            "error": None,
        }
        try:
            built = build_prompt(
                episode,
                subset=args.subset,
                tokenizer=tokenizer,
                max_model_len=args.vllm_max_model_len,
                max_completion_tokens=args.llm_max_completion_tokens,
                safety_buffer=args.prompt_safety_buffer,
            )
            response = await llm(built.prompt, max_tokens=args.llm_max_completion_tokens)
            answer_list, missing = parse_answers(response or "", len(qa_pairs))
            summary.update(
                {
                    "context_truncated": built.truncated,
                    "original_context_tokens": built.original_context_tokens or "",
                    "final_prompt_tokens": built.final_prompt_tokens or "",
                    "original_context_chars": built.original_context_chars,
                    "final_prompt_chars": built.final_prompt_chars,
                    "missing_answers": missing,
                }
            )
            return (
                {
                    "episode_id": episode_id,
                    "answer_list": answer_list,
                    "reasoning_trace": built.reasoning_trace,
                },
                summary,
            )
        except Exception as exc:
            logger.error(f"AMA episode failed: {episode_id} {exc}", exc_info=True)
            summary["error"] = str(exc)
            return (
                {
                    "episode_id": episode_id,
                    "answer_list": ["" for _ in qa_pairs],
                    "reasoning_trace": "",
                },
                summary,
            )
async def run_episode_memory_provider(
    episode: dict[str, Any],
    *,
    llm: Any,
    adapter: Any,
    args: argparse.Namespace,
    tokenizer: Any | None,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with semaphore:
        episode_id = episode.get("episode_id")
        qa_pairs = episode.get("qa_pairs", [])
        task_id = f"ama_episode_{episode_id}"
        memory_storage_dir = str(Path(args.memory_storage_dir).expanduser())
        summary = {
            "episode_id": episode_id,
            "task_type": episode.get("task_type", "unknown"),
            "domain": episode.get("domain", "unknown"),
            "num_questions": len(qa_pairs),
            "num_turns": episode.get("num_turns", len(episode.get("trajectory", []))),
            "total_tokens": episode.get("total_tokens", ""),
            "context_truncated": False,
            "original_context_tokens": "",
            "final_prompt_tokens": "",
            "original_context_chars": "",
            "final_prompt_chars": "",
            "missing_answers": 0,
            "selected_memory_count": 0,
            "created_memory_count": 0,
            "memory_storage_dir": memory_storage_dir,
            "writer_error": "",
            "selector_error": "",
            "error": None,
        }
        selected_memory = ""
        task_description = _selector_query_for_episode(episode)
        try:
            try:
                selected_memory = await adapter.provide_begin(
                    task_description=task_description,
                    context="",
                    task_id=task_id,
                )
                summary["selected_memory_count"] = 1 if selected_memory else 0
            except Exception as exc:
                summary["selector_error"] = str(exc)
                logger.warning(
                    f"[AMA-Bench] provider failed for episode {episode_id}: {exc}",
                    exc_info=True,
                )
            built = build_prompt(
                episode,
                subset=args.subset,
                tokenizer=tokenizer,
                max_model_len=args.vllm_max_model_len,
                max_completion_tokens=args.llm_max_completion_tokens,
                safety_buffer=args.prompt_safety_buffer,
                retrieved_memory=selected_memory,
            )
            response = await llm(built.prompt, max_tokens=args.llm_max_completion_tokens)
            answer_list, missing = parse_answers(response or "", len(qa_pairs))
            summary.update(
                {
                    "context_truncated": built.truncated,
                    "original_context_tokens": built.original_context_tokens or "",
                    "final_prompt_tokens": built.final_prompt_tokens or "",
                    "original_context_chars": built.original_context_chars,
                    "final_prompt_chars": built.final_prompt_chars,
                    "missing_answers": missing,
                }
            )
            try:
                ok, msg = await adapter.take_in(
                    task_description=task_description,
                    trajectory=_ama_execution_trace(
                        episode,
                        raw_response=response or "",
                        answer_list=answer_list,
                        selected_memory=selected_memory,
                    ),
                    success=True,
                    result={"success": True, "episode_id": episode_id},
                    metadata={
                        "task_id": task_id,
                        "task_type": str(episode.get("task_type", "unknown")),
                        "domain": str(episode.get("domain", "unknown")),
                    },
                )
                summary["created_memory_count"] = 1 if ok else 0
                if not ok:
                    summary["writer_error"] = msg
            except Exception as exc:
                summary["writer_error"] = str(exc)
                logger.warning(
                    f"[AMA-Bench] provider writer failed for episode {episode_id}: {exc}",
                    exc_info=True,
                )
            reasoning_parts = []
            if selected_memory:
                reasoning_parts.append(f"## Retrieved {args.memory_backend} Memory\n" + selected_memory)
            reasoning_parts.append("## Current Episode Context\n" + built.reasoning_trace)
            return (
                {
                    "episode_id": episode_id,
                    "answer_list": answer_list,
                    "reasoning_trace": "\n\n".join(reasoning_parts),
                },
                summary,
            )
        except Exception as exc:
            logger.error(f"AMA memory-provider episode failed: {episode_id} {exc}", exc_info=True)
            summary["error"] = str(exc)
            return (
                {
                    "episode_id": episode_id,
                    "answer_list": ["" for _ in qa_pairs],
                    "reasoning_trace": selected_memory,
                },
                summary,
            )
async def run_episode_opd_evolver(
    episode: dict[str, Any],
    *,
    llm: Any,
    memory_pipeline: AMAQAMemoryPipeline,
    args: argparse.Namespace,
    tokenizer: Any | None,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any]]:
    async with semaphore:
        episode_id = episode.get("episode_id")
        qa_pairs = episode.get("qa_pairs", [])
        task_id = f"ama_episode_{episode_id}"
        memory_storage_dir = str(Path(args.memory_storage_dir).expanduser())
        summary = {
            "episode_id": episode_id,
            "task_type": episode.get("task_type", "unknown"),
            "domain": episode.get("domain", "unknown"),
            "num_questions": len(qa_pairs),
            "num_turns": episode.get("num_turns", len(episode.get("trajectory", []))),
            "total_tokens": episode.get("total_tokens", ""),
            "context_truncated": False,
            "original_context_tokens": "",
            "final_prompt_tokens": "",
            "original_context_chars": "",
            "final_prompt_chars": "",
            "missing_answers": 0,
            "selected_memory_count": 0,
            "created_memory_count": 0,
            "memory_storage_dir": memory_storage_dir,
            "writer_error": "",
            "selector_error": "",
            "error": None,
        }
        selected_memory = ""
        filtered = None
        task_description = _selector_query_for_episode(episode)
        try:
            try:
                filtered = await memory_pipeline.pre_execution(
                    task_id=task_id,
                    task_description=task_description,
                    additional_context="",
                    task_type=str(episode.get("domain") or episode.get("task_type") or "ama"),
                )
                if filtered is not None:
                    selected_memory = filtered.formatted_context or ""
                    summary["selected_memory_count"] = _selected_memory_count(filtered)
            except Exception as exc:
                summary["selector_error"] = str(exc)
                logger.warning(f"[AMA-Bench] OPD selector failed for episode {episode_id}: {exc}")
            built = build_prompt(
                episode,
                subset=args.subset,
                tokenizer=tokenizer,
                max_model_len=args.vllm_max_model_len,
                max_completion_tokens=args.llm_max_completion_tokens,
                safety_buffer=args.prompt_safety_buffer,
                retrieved_memory=selected_memory,
            )
            response = await llm(built.prompt, max_tokens=args.llm_max_completion_tokens)
            answer_list, missing = parse_answers(response or "", len(qa_pairs))
            summary.update(
                {
                    "context_truncated": built.truncated,
                    "original_context_tokens": built.original_context_tokens or "",
                    "final_prompt_tokens": built.final_prompt_tokens or "",
                    "original_context_chars": built.original_context_chars,
                    "final_prompt_chars": built.final_prompt_chars,
                    "missing_answers": missing,
                }
            )
            try:
                execution_trace = _ama_execution_trace(
                    episode,
                    raw_response=response or "",
                    answer_list=answer_list,
                    selected_memory=selected_memory,
                )
                reflection = await memory_pipeline.post_execution(
                    task_id=task_id,
                    task_description=task_description,
                    execution_trace=execution_trace,
                    success=True,
                    total_reward=1.0,
                    tags=[
                        "ama_bench",
                        str(episode.get("domain", "unknown")),
                        str(episode.get("task_type", "unknown")),
                    ],
                    task_type=str(episode.get("domain") or episode.get("task_type") or "ama"),
                )
                summary["created_memory_count"] = _created_memory_count_from_reflection(reflection)
            except Exception as exc:
                summary["writer_error"] = str(exc)
                logger.warning(f"[AMA-Bench] OPD writer failed for episode {episode_id}: {exc}", exc_info=True)
            reasoning_parts = []
            if selected_memory:
                reasoning_parts.append("## Retrieved OPD Memory\n" + selected_memory)
            reasoning_parts.append("## Current Episode Context\n" + built.reasoning_trace)
            return (
                {
                    "episode_id": episode_id,
                    "answer_list": answer_list,
                    "reasoning_trace": "\n\n".join(reasoning_parts),
                },
                summary,
            )
        except Exception as exc:
            logger.error(f"AMA OPD episode failed: {episode_id} {exc}", exc_info=True)
            summary["error"] = str(exc)
            return (
                {
                    "episode_id": episode_id,
                    "answer_list": ["" for _ in qa_pairs],
                    "reasoning_trace": selected_memory,
                },
                summary,
            )
def evaluate_answers(
    *,
    answers_file: Path,
    test_file: Path,
    judge_config: Path,
    judge_server: str,
    output_file: Path,
    judge_max_concurrency: int,
    judge_vllm_host: str | None = None,
    judge_vllm_port: int | None = None,
) -> dict[str, Any]:
    if str(AMA_ROOT) not in sys.path:
        sys.path.insert(0, str(AMA_ROOT))
    from src.evaluate import evaluate_batch, print_evaluation_summary
    from src.model_client import ModelClient
    judge_client = ModelClient(config_path=str(judge_config), server_type=judge_server)
    if judge_server == "vllm" and (judge_vllm_host or judge_vllm_port):
        judge_client.config["vllm_host"] = judge_vllm_host or judge_client.config.get("vllm_host", "localhost")
        judge_client.config["vllm_port"] = judge_vllm_port or judge_client.config.get("vllm_port", 8000)
        judge_client.config["base_url"] = f"http://{judge_client.config['vllm_host']}:{judge_client.config['vllm_port']}/v1"
        judge_client.client = judge_client._initialize_client()
    print(f"Initialized judge client: {judge_client.provider}/{judge_client.model}")
    original_episodes = {}
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            original_episodes[row.get("episode_id")] = row
    episode_results = []
    with answers_file.open("r", encoding="utf-8") as f:
        for line in f:
            episode_results.append(json.loads(line))
    qa_results = []
    for episode in episode_results:
        episode_id = episode["episode_id"]
        answer_list = episode.get("answer_list", [])
        original = original_episodes.get(episode_id, {})
        qa_pairs = original.get("qa_pairs", [])
        for predicted_answer, qa_pair in zip(answer_list, qa_pairs):
            qa_results.append(
                {
                    "episode_id": episode_id,
                    "task_type": original.get("task_type", "unknown"),
                    "domain": original.get("domain", "unknown"),
                    "task_description": original.get("task", ""),
                    "question": qa_pair.get("question", ""),
                    "golden_answer": qa_pair.get("answer", ""),
                    "predicted_answer": predicted_answer,
                    "qa_type": qa_pair.get("type") or "unknown",
                }
            )
    evaluated = evaluate_batch(
        qa_results=qa_results,
        judge_client=judge_client,
        max_workers=max(1, judge_max_concurrency),
    )
    def _stats(key: str) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[float]] = {}
        for row in evaluated:
            grouped.setdefault(str(row.get(key, "unknown")), []).append(float(row["score"]))
        return {
            name: {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "accuracy": sum(1 for score in scores if score == 1.0) / len(scores) if scores else 0.0,
            }
            for name, scores in grouped.items()
        }
    summary = {
        "config": {
            "judge_provider": judge_client.provider,
            "judge_model": judge_client.model,
            "answers_file": str(answers_file),
            "test_file": str(test_file),
        },
        "overall": {
            "total_questions": len(evaluated),
            "avg_score": sum(float(row["score"]) for row in evaluated) / len(evaluated) if evaluated else 0.0,
            "accuracy": sum(1 for row in evaluated if float(row["score"]) == 1.0) / len(evaluated) if evaluated else 0.0,
        },
        "by_task_type": _stats("task_type"),
        "by_domain": _stats("domain"),
        "by_qa_type": _stats("qa_type"),
        "results": evaluated,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to: {output_file}")
    print_evaluation_summary(summary)
    return summary
async def main() -> int:
    args = parse_args()
    if args.evaluate_only:
        if not args.answers_file:
            raise SystemExit("--evaluate-only requires --answers-file")
        answers_file = Path(args.answers_file).expanduser()
        if not answers_file.exists():
            raise SystemExit(f"answers file not found: {answers_file}")
        test_file = Path(args.test_file).expanduser()
        if not test_file.exists():
            raise SystemExit(f"test file not found: {test_file}")
        judge_config = Path(args.judge_config).expanduser()
        if not judge_config.exists():
            raise SystemExit(f"judge config not found: {judge_config}")
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else answers_file.parent
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"results_{answers_file.stem}_{stamp}.json"
        judge_proc = None
        judge_host = args.judge_vllm_host
        judge_port = args.judge_vllm_port
        try:
            if args.judge_server == "vllm" and args.judge_vllm:
                helpers = _load_intercode_eval_helpers()
                settings = _resolve_judge_vllm_settings(args)
                judge_host = settings["host"]
                judge_port = settings["port"]
                logger.info(
                    "[AMA-Bench] Starting judge vLLM: "
                    f"model={settings['served_model_name']} endpoint=http://{judge_host}:{judge_port}"
                )
                judge_proc = helpers.start_vllm_server(
                    model_dir=settings["model_dir"],
                    served_model_name=settings["served_model_name"],
                    host=judge_host,
                    port=judge_port,
                    tensor_parallel_size=settings["tp"],
                    max_model_len=settings["max_model_len"],
                    cuda_visible_devices=settings["gpus"],
                    gpu_memory_utilization=settings["gpu_mem"],
                    timeout=settings["timeout"],
                    lora_modules=None,
                    max_lora_rank=16,
                    max_loras=None,
                )
                if judge_proc is None:
                    return 1
            evaluate_answers(
                answers_file=answers_file,
                test_file=test_file,
                judge_config=judge_config,
                judge_server=args.judge_server,
                output_file=output_file,
                judge_max_concurrency=args.judge_max_concurrency,
                judge_vllm_host=judge_host if args.judge_server == "vllm" else None,
                judge_vllm_port=judge_port if args.judge_server == "vllm" else None,
            )
            return 0
        finally:
            if judge_proc is not None:
                _load_intercode_eval_helpers().stop_vllm_server(judge_proc)
    helpers = _load_intercode_eval_helpers()
    if helpers._apply_vllm_lora_cli(args):
        return 1
    if args.method == "memory_provider" and not is_provider_backend(args.memory_backend):
        raise SystemExit(
            "--method memory_provider requires --memory-backend to be one of: "
            + ", ".join(ALL_MEMORY_BACKENDS[1:])
        )
    if args.vllm and args.openai_base_url:
        raise SystemExit("Choose either --vllm or --openai-base-url, not both.")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    elif args.method == "memory_provider":
        output_dir = default_baseline_output_dir(
            "ama_bench", args.model, args.memory_backend, args.subset
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "workspace" / "logs" / "ama_bench" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.memory_storage_dir is None:
        if args.method in {"opd_evolver", "memory_provider"}:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.memory_storage_dir = str(output_dir / f"memory_{stamp}")
        else:
            args.memory_storage_dir = str(output_dir / "memory")
    answers_path = (
        Path(args.answers_file).expanduser()
        if args.answers_file
        else output_dir / f"answers_{args.method}_{_method_file_suffix(args)}.jsonl"
    )
    summary_path = output_dir / "summary.csv"
    vllm_proc = None
    served_model_name = getattr(args, "_vllm_base_served_model_name", None) or args.model
    if args.vllm:
        base_served = served_model_name
        vllm_proc = helpers.start_vllm_server(
            model_dir=args.vllm_model_dir,
            served_model_name=base_served,
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
        _bind_llm_configs_to_vllm(args, served_model_name=base_served)
    elif args.openai_base_url:
        _bind_llm_configs(
            model_names=[args.model, args.selector_model, args.reflection_model, served_model_name],
            served_model_name=served_model_name,
            base_url=_normalize_base_url(args.openai_base_url),
            api_key=args.openai_api_key,
            enable_thinking=args.enable_thinking,
        )
    try:
        from opd_evolver.base.engine.async_llm import LLMsConfig, create_llm_instance
        test_file = Path(args.test_file).expanduser()
        episodes = select_episodes(args, load_episodes(test_file))
        tokenizer = _load_tokenizer(args.vllm_model_dir)
        llm = create_llm_instance(
            LLMsConfig.default().get(args.model),
            max_completion_tokens=args.llm_max_completion_tokens,
        )
        memory_pipeline = None
        if args.method == "opd_evolver":
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
            memory_config = MemoryConfig(
                storage_dir=str(Path(args.memory_storage_dir).expanduser()),
                cold_start_threshold=args.memory_cold_start_threshold,
                retrieval_top_k=args.memory_retrieval_top_k,
                min_similarity=args.memory_min_similarity,
                writer_dataset_path=None,
                selector_dataset_path=None,
                memory_tiers=args.memory_tiers,
            )
            memory_pipeline = AMAQAMemoryPipeline(
                llm=llm,
                config=memory_config,
                enabled=True,
                memory_writer_llm=writer_llm,
                memory_selector_llm=selector_llm,
            )
        logger.info(f"[AMA-Bench] Running {len(episodes)} episodes with model={args.model} method={args.method}")
        init_ama_streaming_outputs(answers_path, summary_path)
        io_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, args.concurrency))
        _pbar_kw: dict[str, Any] = {
            "desc": f"AMA-Bench ({args.method})",
            "unit": "episode",
            "total": len(episodes),
        }
        if args.method == "opd_evolver":
            assert memory_pipeline is not None
            pairs = await tqdm_async.gather(
                *[
                    _run_opd_episode_streaming(
                        episode,
                        llm=llm,
                        memory_pipeline=memory_pipeline,
                        args=args,
                        tokenizer=tokenizer,
                        semaphore=semaphore,
                        io_lock=io_lock,
                        answers_path=answers_path,
                        summary_path=summary_path,
                    )
                    for episode in episodes
                ],
                **_pbar_kw,
            )
        elif args.method == "memory_provider":
            provider_adapter = _make_provider_adapter(args, args.memory_storage_dir)
            pairs = await tqdm_async.gather(
                *[
                    _run_memory_provider_episode_streaming(
                        episode,
                        llm=llm,
                        adapter=provider_adapter,
                        args=args,
                        tokenizer=tokenizer,
                        semaphore=semaphore,
                        io_lock=io_lock,
                        answers_path=answers_path,
                        summary_path=summary_path,
                    )
                    for episode in episodes
                ],
                **_pbar_kw,
            )
        else:
            pairs = await tqdm_async.gather(
                *[
                    _run_longcontext_episode_streaming(
                        episode,
                        llm=llm,
                        args=args,
                        tokenizer=tokenizer,
                        semaphore=semaphore,
                        io_lock=io_lock,
                        answers_path=answers_path,
                        summary_path=summary_path,
                    )
                    for episode in episodes
                ],
                **_pbar_kw,
            )
        answer_rows = [pair[0] for pair in pairs]
        summary_rows = [pair[1] for pair in pairs]
        print("\n" + "=" * 60)
        print(f"AMA-BENCH {args.method.upper()} GENERATION RESULTS")
        print("=" * 60)
        print(f"Model:      {args.model}")
        print(f"Episodes:   {len(answer_rows)}")
        print(f"Answers:    {answers_path}")
        print(f"Summary:    {summary_path}")
        if args.method in {"opd_evolver", "memory_provider"}:
            print(f"Memory:     {Path(args.memory_storage_dir).expanduser()}")
            print(f"Selected:   {sum(int(row.get('selected_memory_count') or 0) for row in summary_rows)}")
            print(f"Created:    {sum(int(row.get('created_memory_count') or 0) for row in summary_rows)}")
        print(f"Truncated:  {sum(1 for row in summary_rows if row['context_truncated'])}")
        print(f"Parse gaps: {sum(int(row['missing_answers']) for row in summary_rows)}")
        print("=" * 60)
        if args.evaluate:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = output_dir / f"results_{answers_path.stem}_{stamp}.json"
            evaluate_answers(
                answers_file=answers_path,
                test_file=test_file,
                judge_config=Path(args.judge_config).expanduser(),
                judge_server=args.judge_server,
                output_file=results_path,
                judge_max_concurrency=args.judge_max_concurrency,
            )
        return 0
    finally:
        if vllm_proc is not None:
            _load_intercode_eval_helpers().stop_vllm_server(vllm_proc)
if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
