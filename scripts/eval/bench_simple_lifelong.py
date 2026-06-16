#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import ast
import csv
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from opd_evolver.base.engine.logs import logger
from opd_evolver.base.engine.async_llm import LLMsConfig
from opd_evolver.benchmark.bench_lifelong_agent import (
    DEFAULT_DATA_ROOT,
    TASK_TYPES,
    LifelongAgentBenchmark,
)
from opd_evolver.memory.evolvelab_adapter import (
    ALL_MEMORY_BACKENDS,
    OPD_HIERARCHICAL_BACKEND,
    default_baseline_output_dir,
    is_provider_backend,
)
from opd_evolver.runners.task_runner import (
    MemoryAugmentedTaskRunner,
    SimpleTaskRunner,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "lifelong_task"
def _load_intercode_eval_helpers() -> Any:
    path = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_intercode.py"
    spec = importlib.util.spec_from_file_location("bench_simple_intercode_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import InterCode eval helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
_INTERCODE_HELPERS = _load_intercode_eval_helpers()
_VLLM_DEFAULT_MODEL_DIR = _INTERCODE_HELPERS._VLLM_DEFAULT_MODEL_DIR
_apply_vllm_lora_cli = _INTERCODE_HELPERS._apply_vllm_lora_cli
parse_task_indices = _INTERCODE_HELPERS.parse_task_indices
start_vllm_server = _INTERCODE_HELPERS.start_vllm_server
stop_vllm_server = _INTERCODE_HELPERS.stop_vllm_server
def _config_bench_task_types_default(config: dict[str, Any]) -> str | None:
    raw = config.get("task_types")
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw if str(x).strip()]
        return ",".join(parts) if parts else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None
def load_task_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        logger.warning(f"PyYAML not available; ignoring config at {config_path}: {exc}")
        return {}
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using built-in defaults")
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"Config must be a mapping, got {type(data)} at: {config_path}")
            return {}
        return data
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return {}
def resolve_config_path(task_type: str | None, cli_config: str | None) -> Path | None:
    if cli_config:
        return Path(cli_config)
    if task_type:
        name = task_type if task_type in {*TASK_TYPES, "all"} else "all"
        return CONFIG_ROOT / f"{name}.yaml"
    return CONFIG_ROOT / "all.yaml"
def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task-type", choices=[*TASK_TYPES, "all"], default=None)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = resolve_config_path(pre_args.task_type, pre_args.config)
    config = load_task_config(config_path) if config_path else {}
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=str,
        default=str(config_path) if config_path else None,
        help="YAML config path. Default: config/lifelong_task/{task-type}.yaml",
    )
    ap.add_argument(
        "--task-type",
        choices=[*TASK_TYPES, "all"],
        default=config.get("task_type", pre_args.task_type or "all"),
    )
    ap.add_argument(
        "--task-types",
        type=str,
        default=_config_bench_task_types_default(config),
        dest="bench_task_types",
        help=(
            "Comma-separated Lifelong task types (db,os,kg). When set, overrides --task-type "
            "for which environments run. YAML key: task_types."
        ),
    )
    ap.add_argument("--split", choices=["train", "test", "all"], default=config.get("split", "test"))
    ap.add_argument("--data-dir", default=config.get("data_dir", str(DEFAULT_DATA_ROOT)))
    ap.add_argument("--task", type=int, default=config.get("task"))
    ap.add_argument(
        "--tasks",
        default=config.get("tasks"),
        help="Task indices, e.g. 0,3,8 or 0-99",
    )
    ap.add_argument(
        "--task-ids",
        default=config.get("task_ids", config.get("task-ids")),
        help=(
            "Select tasks by task_id instead of indices. Accepts a comma/space-separated list, "
            'e.g. "db_0010,os_0084" or "db_0010 os_0084". When set, overrides --task/--tasks/--max-tasks.'
        ),
    )
    ap.add_argument("--max-tasks", type=int, default=config.get("max_tasks"))
    ap.add_argument("--model", default=config.get("model", "qwen/qwen3.5-9b"))
    ap.add_argument(
        "--gold",
        action="store_true",
        default=bool(config.get("gold", False)),
        help="Run a gold (no-LLM) smoke path. For DB tasks this executes the ground-truth SQL then submits.",
    )
    ap.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("memory", False)),
    )
    ap.add_argument(
        "--memory-backend",
        choices=ALL_MEMORY_BACKENDS,
        default=config.get("memory_backend", OPD_HIERARCHICAL_BACKEND),
        help="Memory backend. Default preserves existing OPD hierarchical memory.",
    )
    ap.add_argument("--cold-start-threshold", type=int, default=int(config.get("cold_start_threshold", 20)))
    ap.add_argument("--retrieval-top-k", type=int, default=int(config.get("retrieval_top_k", 3)))
    ap.add_argument("--memory-min-similarity", type=float, default=float(config.get("memory_min_similarity", 0.0)))
    ap.add_argument("--embedding-provider", default=config.get("embedding_provider", "local"))
    ap.add_argument("--embedding-model", default=config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"))
    ap.add_argument("--memory-storage-dir", default=config.get("memory_storage_dir"))
    ap.add_argument("--reflection-model", default=config.get("reflection_model"))
    ap.add_argument("--selector-model", default=config.get("selector_model"))
    ap.add_argument(
        "--writer-dataset",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("writer_dataset", False)),
    )
    ap.add_argument("--writer-dataset-path", default=config.get("writer_dataset_path"))
    ap.add_argument(
        "--selector-dataset",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("selector_dataset", False)),
    )
    ap.add_argument("--selector-dataset-path", default=config.get("selector_dataset_path"))
    ap.add_argument("--max-steps", type=int, default=config.get("max_steps"))
    ap.add_argument("--step-timeout", type=float, default=float(config.get("step_timeout", 180.0)))
    ap.add_argument("--concurrency", type=int, default=int(config.get("concurrency", 1)))
    ap.add_argument("--llm-max-completion-tokens", type=int, default=config.get("llm_max_completion_tokens"))
    mix_task_types_default = config.get("mix_task_types", config.get("mix-task-types", "sequential"))
    mix_seed_default = config.get("mix_seed", config.get("mix-seed"))
    ap.add_argument(
        "--mix-task-types",
        choices=["sequential", "interleave", "shuffle"],
        default=str(mix_task_types_default),
        help=(
            "How to order tasks when multiple task types are evaluated. "
            "sequential: run all of one type then next (default); "
            "interleave: round-robin across types by index; "
            "shuffle: shuffle all (task_type, idx) pairs."
        ),
    )
    ap.add_argument(
        "--mix-seed",
        type=int,
        default=mix_seed_default,
        help="Random seed used when --mix-task-types=shuffle (optional).",
    )
    ap.add_argument("--output-dir", default=config.get("output_dir"))
    ap.add_argument("--os-timeout", type=int, default=int(config.get("os_timeout", 20)))
    ap.add_argument("--sparql-url", default=config.get("sparql_url", "http://127.0.0.1:3001/sparql"))
    ap.add_argument(
        "--ontology-dir",
        default=config.get("ontology_dir"),
        help="KG ontology dir containing vocab.json/fb_roles. Defaults to reference data path.",
    )
    ap.add_argument(
        "--vllm",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("vllm", False)),
    )
    ap.add_argument("--vllm-model-dir", default=config.get("vllm_model_dir", _VLLM_DEFAULT_MODEL_DIR))
    ap.add_argument("--vllm-host", default=config.get("vllm_host", "127.0.0.1"))
    ap.add_argument("--vllm-port", type=int, default=int(config.get("vllm_port", 8000)))
    ap.add_argument("--vllm-tp", type=int, default=int(config.get("vllm_tp", 2)))
    ap.add_argument("--vllm-max-model-len", type=int, default=int(config.get("vllm_max_model_len", 262144)))
    ap.add_argument("--vllm-gpus", default=config.get("vllm_gpus", "0,1"))
    ap.add_argument("--vllm-gpu-mem", type=float, default=float(config.get("vllm_gpu_mem", 0.9)))
    ap.add_argument("--vllm-timeout", type=int, default=int(config.get("vllm_timeout", 300)))
    ap.add_argument("--vllm-lora-module", default=config.get("vllm_lora_module"))
    ap.add_argument(
        "--vllm-lora-modules",
        action="append",
        default=list(config.get("vllm_lora_modules", []) or []),
    )
    ap.add_argument("--vllm-task-lora-module", default=config.get("vllm_task_lora_module"))
    ap.add_argument("--vllm-selector-lora-module", default=config.get("vllm_selector_lora_module"))
    ap.add_argument("--vllm-reflection-lora-module", default=config.get("vllm_reflection_lora_module"))
    ap.add_argument("--vllm-lora-max-rank", type=int, default=int(config.get("vllm_lora_max_rank", 32)))
    ap.add_argument("--vllm-max-loras", type=int, default=config.get("vllm_max_loras"))
    ap.add_argument(
        "--openai-base-url",
        type=str,
        default=(config.get("openai_base_url") or None),
        help=(
            "OpenAI-compatible API base URL for a server you started yourself "
            "(e.g. vLLM). Mutually exclusive with --vllm."
        ),
    )
    ap.add_argument(
        "--openai-api-key",
        type=str,
        default=config.get("openai_api_key", "EMPTY"),
        help="API key for --openai-base-url (local vLLM often uses EMPTY).",
    )
    args = ap.parse_args()
    if getattr(args, "openai_base_url", None) == "":
        args.openai_base_url = None
    return args
def _effective_task_types(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "bench_task_types", None)
    if raw:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        bad = [p for p in parts if p not in TASK_TYPES]
        if bad:
            raise SystemExit(f"Invalid task type(s) in --task-types: {bad} (allowed: {TASK_TYPES})")
        return parts
    if args.task_type == "all":
        return list(TASK_TYPES)
    return [args.task_type]
def _selected_indices(args: argparse.Namespace, total: int) -> list[int]:
    if args.task is not None:
        return [args.task]
    if args.tasks:
        return parse_task_indices(args.tasks, total)
    if args.max_tasks:
        return list(range(min(args.max_tasks, total)))
    return list(range(total))
def _parse_task_ids(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        parts = [str(x) for x in raw]
    else:
        text = str(raw).strip()
        if not text:
            return set()
        parts = [p for p in text.replace(",", " ").split() if p]
    return {p.strip() for p in parts if str(p).strip()}
def _selected_indices_by_task_ids(
    benchmark: LifelongAgentBenchmark,
    want_ids: set[str],
) -> list[int] | None:
    if not want_ids:
        return None
    prefix = f"{benchmark.task_type}_"
    scoped = {tid for tid in want_ids if tid.startswith(prefix)}
    if not scoped:
        return None
    id_to_idx: dict[str, int] = {}
    for idx, level in enumerate(benchmark.list_levels()):
        tid = str(level.get("id", ""))
        if tid:
            id_to_idx[tid] = idx
    missing = sorted(scoped - set(id_to_idx))
    if missing:
        logger.warning(
            f"{benchmark.task_type}: {len(missing)} task_ids not found in split; first few: {missing[:10]}"
        )
    chosen = [id_to_idx[tid] for tid in sorted(scoped) if tid in id_to_idx]
    return sorted(set(chosen))
def _memory_config(
    args: argparse.Namespace, task_type: str, *, per_type_subdirectory: bool
) -> dict[str, Any]:
    storage_dir = args.memory_storage_dir
    if storage_dir and per_type_subdirectory:
        storage_dir = str(Path(storage_dir) / task_type)
    return {
        "env_type": task_type,
        "memory_backend": args.memory_backend,
        "cold_start_threshold": args.cold_start_threshold,
        "retrieval_top_k": args.retrieval_top_k,
        "min_similarity": args.memory_min_similarity,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "storage_dir": storage_dir,
        "reflection_model": args.reflection_model,
        "writer_dataset_path": args.writer_dataset_path,
        "writer_dataset": args.writer_dataset,
        "selector_model": args.selector_model,
        "selector_dataset_path": args.selector_dataset_path,
        "selector_dataset": args.selector_dataset,
        "llm_max_completion_tokens": args.llm_max_completion_tokens,
    }
def _make_runner(
    args: argparse.Namespace,
    task_type: str,
    trajectory_dir: Path,
    csv_path: Path,
    *,
    per_type_memory_subdir: bool,
) -> SimpleTaskRunner:
    if args.gold:
        raise ValueError("--gold is not compatible with LLM runners; it bypasses the runner loop.")
    kwargs = {
        "model": args.model,
        "env_type": task_type,
        "max_steps": args.max_steps or {"db": 6, "os": 8, "kg": 18}[task_type],
        "step_timeout": args.step_timeout,
        "trajectory_dir": trajectory_dir,
        "csv_summary_path": csv_path,
        "llm_max_completion_tokens": args.llm_max_completion_tokens,
    }
    if args.memory:
        return MemoryAugmentedTaskRunner(
            memory_config=_memory_config(
                args, task_type, per_type_subdirectory=per_type_memory_subdir
            ),
            **kwargs,
        )
    return SimpleTaskRunner(**kwargs)
def _maybe_parse_obj(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value
def _bind_llm_configs_to_vllm(args: argparse.Namespace, served_model_name: str) -> None:
    lc = LLMsConfig.default()
    base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"
    def _ensure(name: str | None) -> None:
        if not name:
            return
        existing = lc.configs.get(name)
        if existing is None:
            lc.add_config(
                name,
                {
                    "model": name,
                    "api_key": "EMPTY",
                    "base_url": base_url,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            )
            return
        existing["base_url"] = base_url
        existing.setdefault("api_key", "EMPTY")
        if name == served_model_name:
            existing["model"] = served_model_name
    _ensure(served_model_name)
    _ensure(getattr(args, "model", None))
    _ensure(getattr(args, "selector_model", None))
    _ensure(getattr(args, "reflection_model", None))
def _looks_like_placeholder(path: str | None) -> bool:
    if not path:
        return True
    text = str(path)
    return "PATH/TO/YOUR/MODEL" in text or text.strip() in {"", "null", "None"}
def _is_hf_snapshot_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "config.json").is_file()
def _pick_latest_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if _is_hf_snapshot_dir(cand):
            return cand
    return None
def _auto_resolve_qwen35_9b_snapshot(model_dir_hint: str | None) -> str | None:
    hints: list[Path] = []
    if model_dir_hint:
        hints.append(Path(model_dir_hint).expanduser())
    hints.append(Path(_VLLM_DEFAULT_MODEL_DIR))
    hints.extend(
        [
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
async def _run_one_gold_db(
    benchmark: LifelongAgentBenchmark,
    task_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        level = benchmark.list_levels()[task_idx]
        env = benchmark.make_env(level)
        env_any: Any = env
        try:
            entry = benchmark.rows[task_idx]
            answer_info = _maybe_parse_obj(entry.get("answer_info", {}))
            if not isinstance(answer_info, dict):
                raise TypeError(f"answer_info must be dict, got {type(answer_info)}")
            sql = str(answer_info.get("sql") or "").strip()
            if not sql:
                raise ValueError("Missing ground-truth SQL in answer_info.sql")
            await env_any.reset()
            await env_any.step({"action": "execute", "params": {"command": sql}})
            _, reward, _, info = await env_any.step({"action": "submit", "params": {}})
            success = bool(reward == 1.0 or info.get("submitted"))
            return {
                "task_id": level["id"],
                "task_type": benchmark.task_type,
                "task_idx": task_idx,
                "success": bool(reward == 1.0),
                "reward": float(reward),
                "steps": 2,
                "cost": 0.0,
                "error": None,
            }
        except Exception as exc:
            logger.error(f"Lifelong GOLD DB task failed: {level['id']} {exc}", exc_info=True)
            return {
                "task_id": level["id"],
                "task_type": benchmark.task_type,
                "task_idx": task_idx,
                "success": False,
                "reward": 0.0,
                "steps": 0,
                "cost": 0.0,
                "error": str(exc),
            }
        finally:
            try:
                await env_any.close()
            except Exception:
                pass
async def _run_one(
    runner: SimpleTaskRunner,
    benchmark: LifelongAgentBenchmark,
    task_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        level = benchmark.list_levels()[task_idx]
        env = benchmark.make_env(level)
        try:
            result = await runner.run(agent=None, env=env)
            return {
                "task_id": level["id"],
                "task_type": benchmark.task_type,
                "task_idx": task_idx,
                "success": result.success,
                "reward": result.total_reward,
                "steps": result.steps,
                "cost": result.cost,
                "error": None,
            }
        except Exception as exc:
            logger.error(f"Lifelong task failed: {level['id']} {exc}", exc_info=True)
            return {
                "task_id": level["id"],
                "task_type": benchmark.task_type,
                "task_idx": task_idx,
                "success": False,
                "reward": 0.0,
                "steps": 0,
                "cost": 0.0,
                "error": str(exc),
            }
def _write_combined_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_id", "task_type", "task_idx", "success", "reward", "steps", "cost", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def _schedule_jobs(
    task_types: list[str],
    indices_by_type: dict[str, list[int]],
    *,
    mode: str,
    seed: int | None,
) -> list[tuple[str, int]]:
    if mode == "sequential":
        ordered: list[tuple[str, int]] = []
        for t in task_types:
            for i in indices_by_type.get(t, []):
                ordered.append((t, i))
        return ordered
    if mode == "interleave":
        queues = {t: list(indices_by_type.get(t, [])) for t in task_types}
        ordered = []
        remaining = sum(len(v) for v in queues.values())
        while remaining:
            progressed = False
            for t in task_types:
                q = queues.get(t) or []
                if q:
                    ordered.append((t, q.pop(0)))
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return ordered
    if mode == "shuffle":
        ordered = []
        for t in task_types:
            for i in indices_by_type.get(t, []):
                ordered.append((t, i))
        rng = random.Random(seed)
        rng.shuffle(ordered)
        return ordered
    raise ValueError(f"Unknown mix mode: {mode}")
async def main() -> int:
    args = parse_args()
    if _apply_vllm_lora_cli(args):
        return 1
    if args.vllm and getattr(args, "openai_base_url", None):
        logger.error(
            "Choose either --vllm (auto-start server) or --openai-base-url (existing server), not both."
        )
        return 1
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.memory and is_provider_backend(args.memory_backend):
        selected = _effective_task_types(args)
        env_label = "_".join(selected)
        output_dir = default_baseline_output_dir(
            "lifelong_agent_bench", args.model, args.memory_backend, env_label
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "workspace" / "logs" / "lifelong_agent_bench" / args.split / stamp
    trajectory_dir = output_dir / "trajectories"
    csv_path = output_dir / "summary.csv"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    if args.memory and is_provider_backend(args.memory_backend) and not args.memory_storage_dir:
        args.memory_storage_dir = str(output_dir / "memory")
    task_types = _effective_task_types(args)
    per_type_memory_subdir = len(task_types) > 1
    vllm_proc = None
    if args.vllm:
        if _looks_like_placeholder(args.vllm_model_dir):
            resolved = _auto_resolve_qwen35_9b_snapshot(None)
            if resolved:
                logger.info(f"Auto-resolved vLLM model dir: {resolved}")
                args.vllm_model_dir = resolved
        base_served = getattr(args, "_vllm_base_served_model_name", None) or args.model
        vllm_proc = start_vllm_server(
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
    elif getattr(args, "openai_base_url", None):
        base_served = getattr(args, "_vllm_base_served_model_name", None) or args.model
        _INTERCODE_HELPERS._bind_llm_configs_to_openai_endpoint(
            args,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key or "EMPTY",
            served_model_name=base_served,
        )
        logger.info(
            "Using OpenAI-compatible endpoint: "
            f"{_INTERCODE_HELPERS._normalize_openai_base_url(args.openai_base_url)}"
        )
    try:
        want_task_ids = _parse_task_ids(getattr(args, "task_ids", None))
        semaphore = asyncio.Semaphore(args.concurrency)
        jobs = []
        if args.gold and any(t != "db" for t in task_types):
            raise ValueError("--gold currently supports only DB tasks; set --task-types=db")
        benchmarks: dict[str, LifelongAgentBenchmark] = {}
        runners: dict[str, SimpleTaskRunner] = {}
        indices_by_type: dict[str, list[int]] = {}
        for task_type in task_types:
            benchmark = LifelongAgentBenchmark(
                task_type=task_type,
                split=args.split,
                data_root=args.data_dir,
                max_steps=args.max_steps,
                max_tasks=args.max_tasks if not args.tasks and args.task is None else None,
                os_timeout=args.os_timeout,
                sparql_url=args.sparql_url,
                ontology_dir=args.ontology_dir,
            )
            levels = benchmark.list_levels()
            benchmarks[task_type] = benchmark
            indices = _selected_indices_by_task_ids(benchmark, want_task_ids)
            if indices is None:
                indices = _selected_indices(args, len(levels))
            indices_by_type[task_type] = indices
            if not args.gold:
                runners[task_type] = _make_runner(
                    args,
                    task_type,
                    trajectory_dir,
                    csv_path,
                    per_type_memory_subdir=per_type_memory_subdir,
                )
        schedule = _schedule_jobs(
            task_types,
            indices_by_type,
            mode=args.mix_task_types,
            seed=args.mix_seed,
        )
        for task_type, idx in schedule:
            benchmark = benchmarks[task_type]
            if args.gold:
                jobs.append(_run_one_gold_db(benchmark, idx, semaphore))
            else:
                runner = runners[task_type]
                jobs.append(_run_one(runner, benchmark, idx, semaphore))
        rows = await asyncio.gather(*jobs)
        _write_combined_summary(csv_path, rows)
        successes = sum(1 for r in rows if r["success"])
        total = len(rows)
        print("\n" + "=" * 72)
        print("LIFELONG AGENT BENCH RESULTS")
        print("=" * 72)
        print(f"Split:        {args.split}")
        print(f"Task types:   {', '.join(task_types)}")
        print(f"Model:        {args.model}")
        print(f"Memory:       {'enabled' if args.memory else 'disabled'}")
        if args.memory:
            print(f"Memory backend: {args.memory_backend}")
        print(f"Tasks:        {total}")
        print(f"Successes:    {successes}")
        print(f"Success Rate: {(successes / total * 100) if total else 0:.1f}%")
        print(f"Output:       {output_dir}")
        print("=" * 72)
        return 0 if successes == total else 1
    finally:
        stop_vllm_server(vllm_proc)
if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
