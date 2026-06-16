#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import urlparse
try:
    from tqdm.asyncio import tqdm as tqdm_async
except ImportError:
    tqdm_async = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENENV_ROOT = PROJECT_ROOT / "reference" / "OpenEnv-main"
DEFAULT_AWM_CONFIG = PROJECT_ROOT / "config" / "awm_bench" / "default.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
def _load_intercode_eval_helpers():
    path = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_intercode.py"
    spec = importlib.util.spec_from_file_location("bench_simple_intercode_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load InterCode eval helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
_INTERCODE = _load_intercode_eval_helpers()
_VLLM_DEFAULT_MODEL_DIR = _INTERCODE._VLLM_DEFAULT_MODEL_DIR
start_vllm_server = _INTERCODE.start_vllm_server
stop_vllm_server = _INTERCODE.stop_vllm_server
_kill_listen_port_guess_host = _INTERCODE._kill_existing_vllm
from opd_evolver.base.engine.async_llm import LLMsConfig
from opd_evolver.base.engine.logs import logger
from opd_evolver.benchmark.bench_awm import AWMEnvironment
from opd_evolver.runners.task_runner import SimpleTaskRunner
AWM_SUMMARY_FIELDNAMES = (
    "task_id",
    "model",
    "memory_group_id",
    "memory_storage_dir",
    "success",
    "reward",
    "steps",
    "cost",
    "timestamp",
    "error",
)
@dataclass(frozen=True)
class AWMMemoryRuntime:
    index: int
    group_id: str
    start_job_index: int
    end_job_index: int
    storage_dir: Path
    writer_dataset_path: str | None
    selector_dataset_path: str | None
    pipeline: Any
def _parse_tasks(value: str) -> List[int]:
    tasks: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid task range: {part}")
            tasks.extend(range(start, end + 1))
        else:
            tasks.append(int(part))
    if not tasks:
        raise argparse.ArgumentTypeError("Task list cannot be empty")
    return tasks
def load_task_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to load config {config_path}: {exc} — using CLI defaults.")
        return {}
def _coalesce(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default
def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
def _tasks_from_config(config: dict[str, Any]) -> List[int] | None:
    raw = _coalesce(config, "tasks")
    if raw is None:
        ti = config.get("task_idx")
        if isinstance(ti, str):
            s = ti.strip()
            if s and ("-" in s or "," in s):
                return _parse_tasks(s)
        return None
    if isinstance(raw, list):
        out: List[int] = []
        for x in raw:
            out.append(int(x))
        return out
    return _parse_tasks(str(raw))
def _scalar_task_idx_from_config(cfg: dict[str, Any]) -> int:
    raw = _coalesce(cfg, "task_idx", default=0)
    if isinstance(raw, str):
        s = raw.strip()
        if s and ("-" in s or "," in s):
            return 0
        return int(s, 10)
    return int(raw)
def _openenv_default(cfg: dict[str, Any]) -> Path | None:
    raw = _coalesce(cfg, "openenv_root")
    if raw is None:
        return DEFAULT_OPENENV_ROOT if DEFAULT_OPENENV_ROOT.is_dir() else None
    expanded = os.path.expanduser(os.path.expandvars(str(raw))).strip()
    if not expanded or expanded.lower() in {"null", "none"}:
        return DEFAULT_OPENENV_ROOT if DEFAULT_OPENENV_ROOT.is_dir() else None
    return Path(expanded)
def _parse_scenarios(args: argparse.Namespace) -> List[str]:
    raw = args.scenarios or args.scenario
    if not raw:
        raise SystemExit("Provide --scenario or --scenarios")
    scenarios = [item.strip() for item in raw.split(",") if item.strip()]
    if not scenarios:
        raise SystemExit("Scenario list cannot be empty")
    return scenarios
def _parse_base_url(raw: str) -> tuple[int, str]:
    normalized = raw.strip().rstrip("/")
    parsed = urlparse(normalized if "://" in normalized else f"http://{normalized}")
    if parsed.scheme not in {"http", "https"}:
        raise argparse.ArgumentTypeError(f"Unsupported URL scheme for --base-url: {parsed.scheme}")
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is None:
        port = 443 if parsed.scheme == "https" else 80
    else:
        port = parsed.port
    path = parsed.path or ""
    base = f"{parsed.scheme}://{host}:{port}{path}".rstrip("/")
    return port, base
def _prepend_openenv_client_sys_path(openenv_root: Path | None) -> None:
    root: Path | None = None
    if openenv_root is not None:
        cand = Path(openenv_root).resolve()
        if cand.is_dir():
            root = cand
    if root is None and DEFAULT_OPENENV_ROOT.is_dir():
        root = DEFAULT_OPENENV_ROOT.resolve()
    if root is None:
        logger.warning(
            f"Could not resolve an OpenEnv root for AWM client imports. "
            f"Set --openenv-root or ensure {DEFAULT_OPENENV_ROOT} exists."
        )
        return
    for sub in ("envs", "src"):
        p = root / sub
        if not p.is_dir():
            logger.warning(f"OpenEnv checkout missing expected directory {p}")
            continue
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
            logger.info(f"[AWM] Prepended client import path: {s}")
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
def _normalize_cli_memory_tiers(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    return raw
def _validate_memory_tiers_config(raw: Any) -> None:
    if raw is None:
        return
    from opd_evolver.memory.memory_manager import MemoryConfig
    MemoryConfig(memory_tiers=raw, storage_dir=".").enabled_memory_tiers()
def _memory_config(args: argparse.Namespace) -> dict[str, Any]:
    storage_dir = args.memory_storage_dir
    return {
        "env_type": "awm",
        "cold_start_threshold": args.cold_start_threshold,
        "retrieval_top_k": args.retrieval_top_k,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "storage_dir": storage_dir,
        "reflection_model": args.reflection_model,
        "writer_dataset_path": args.writer_dataset_path,
        "writer_dataset": args.writer_dataset,
        "selector_model": args.selector_model,
        "selector_dataset_path": args.selector_dataset_path,
        "selector_dataset": args.selector_dataset,
        "memory_tiers": args.memory_tiers,
    }
def _resolve_memory_dataset_paths(memory_config: dict[str, Any]) -> dict[str, Any]:
    out = dict(memory_config)
    storage_dir = out.get("storage_dir")
    if not storage_dir:
        storage_dir = str(Path("workspace") / "memory" / "awm")
    else:
        storage_dir = str(Path(os.path.expandvars(str(storage_dir))).expanduser())
    out["storage_dir"] = storage_dir
    if out.get("writer_dataset_path") is None and out.get("writer_dataset", False):
        out["writer_dataset_path"] = str(Path(storage_dir) / "memory_writer_dataset.jsonl")
    if out.get("selector_dataset_path") is None and out.get("selector_dataset", False):
        out["selector_dataset_path"] = str(Path(storage_dir) / "memory_selector_dataset.jsonl")
    if out.get("writer_dataset_path") is not None:
        out["writer_dataset_path"] = os.path.expandvars(str(out["writer_dataset_path"]))
    if out.get("selector_dataset_path") is not None:
        out["selector_dataset_path"] = os.path.expandvars(str(out["selector_dataset_path"]))
    return out
def _init_summary_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(AWM_SUMMARY_FIELDNAMES)).writeheader()
def _append_summary_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(
            f,
            fieldnames=list(AWM_SUMMARY_FIELDNAMES),
            extrasaction="ignore",
            restval="",
        ).writerow(row)
def _summary_row(
    task_id: str,
    result: Any | None,
    *,
    model: str,
    memory_group_id: str = "",
    memory_storage_dir: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    if result is None:
        return {
            "task_id": task_id,
            "model": model,
            "memory_group_id": memory_group_id,
            "memory_storage_dir": memory_storage_dir,
            "success": False,
            "reward": "0.0000",
            "steps": 0,
            "cost": "0.000000",
            "timestamp": time.time(),
            "error": error or "",
        }
    return {
        "task_id": task_id,
        "model": result.model,
        "memory_group_id": memory_group_id,
        "memory_storage_dir": memory_storage_dir,
        "success": result.success,
        "reward": f"{result.total_reward:.4f}",
        "steps": result.steps,
        "cost": f"{result.cost:.6f}",
        "timestamp": result.timestamp,
        "error": error or "",
    }
def _build_memory_role_llms(args: argparse.Namespace) -> tuple[Any, Any | None, Any | None]:
    from opd_evolver.base.engine.async_llm import create_llm_instance
    memory_llm = create_llm_instance(
        LLMsConfig.default().get(args.model),
        max_completion_tokens=args.llm_max_completion_tokens,
    )
    writer_llm = None
    if args.reflection_model:
        writer_llm = create_llm_instance(
            LLMsConfig.default().get(args.reflection_model),
            max_completion_tokens=args.llm_max_completion_tokens,
        )
    selector_llm = None
    if args.selector_model:
        selector_llm = create_llm_instance(
            LLMsConfig.default().get(args.selector_model),
            max_completion_tokens=args.llm_max_completion_tokens,
        )
    return memory_llm, writer_llm, selector_llm
def _memory_config_from_resolved(cfg: dict[str, Any]):
    from opd_evolver.memory.memory_manager import MemoryConfig
    return MemoryConfig(
        storage_dir=cfg["storage_dir"],
        embedding_provider=cfg.get("embedding_provider", "local"),
        embedding_model=cfg.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        cold_start_threshold=cfg.get("cold_start_threshold", 20),
        retrieval_top_k=cfg.get("retrieval_top_k", 3),
        writer_dataset_path=cfg.get("writer_dataset_path"),
        selector_dataset_path=cfg.get("selector_dataset_path"),
        memory_tiers=cfg.get("memory_tiers"),
    )
def _create_shared_embedding_provider(memory_config: Any) -> Any:
    from opd_evolver.memory.embeddings import (
        LocalHFEmbeddingProvider,
        OpenAIEmbeddingProvider,
        OpenRouterEmbeddingProvider,
        local_hf_embedding_settings_from_env,
    )
    cache_path = memory_config.get_cache_path() if memory_config.embedding_cache else None
    provider = (memory_config.embedding_provider or "").lower()
    if provider == "local":
        device, max_len, dtype = local_hf_embedding_settings_from_env()
        logger.info(
            f"[AWM] Using shared local HF embedding: {memory_config.embedding_model} "
            f"(device={device}, max_length={max_len}, dtype={dtype})"
        )
        return LocalHFEmbeddingProvider(
            model_id=memory_config.embedding_model,
            device=device,
            torch_dtype=dtype,
            max_length=max_len,
        )
    if provider == "openrouter":
        logger.info(f"[AWM] Using shared OpenRouter embedding: {memory_config.embedding_model}")
        return OpenRouterEmbeddingProvider(
            model=memory_config.embedding_model,
            cache_path=cache_path,
        )
    logger.info(f"[AWM] Using shared OpenAI embedding: {memory_config.embedding_model}")
    return OpenAIEmbeddingProvider(
        model=memory_config.embedding_model,
        cache_path=cache_path,
    )
def _build_memory_runtime(
    args: argparse.Namespace,
    *,
    index: int,
    group_id: str,
    start_job_index: int,
    end_job_index: int,
    cfg: dict[str, Any],
    role_llms: tuple[Any, Any | None, Any | None],
    embedding_provider: Any,
) -> AWMMemoryRuntime:
    from opd_evolver.memory.memory_manager import HierarchicalMemoryManager
    from opd_evolver.pipelines.memory_pipeline import MemoryAugmentedPipeline
    memory_config = _memory_config_from_resolved(cfg)
    memory_manager = HierarchicalMemoryManager(
        config=memory_config,
        embedding_provider=embedding_provider,
    )
    memory_llm, writer_llm, selector_llm = role_llms
    pipeline = MemoryAugmentedPipeline(
        llm=memory_llm,
        memory_manager=memory_manager,
        memory_writer_llm=writer_llm,
        memory_selector_llm=selector_llm,
        config=memory_config,
        enabled=True,
    )
    logger.info(
        f"[AWM] Memory runtime {group_id}: "
        f"storage_dir={memory_config.storage_dir}, "
        f"memory_tiers={sorted(memory_config.enabled_memory_tiers())}, "
        f"writer_dataset={memory_config.writer_dataset_path or 'off'}, "
        f"selector_dataset={memory_config.selector_dataset_path or 'off'}"
    )
    return AWMMemoryRuntime(
        index=index,
        group_id=group_id,
        start_job_index=start_job_index,
        end_job_index=end_job_index,
        storage_dir=Path(os.path.expandvars(str(memory_config.storage_dir))).expanduser(),
        writer_dataset_path=memory_config.resolved_writer_dataset_path(),
        selector_dataset_path=memory_config.resolved_selector_dataset_path(),
        pipeline=pipeline,
    )
def _write_memory_groups_manifest(
    base_storage_dir: Path,
    *,
    memory_group_size: int,
    jobs: list[tuple[str, int]],
    groups: list[AWMMemoryRuntime],
) -> Path:
    base_storage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_storage_dir / "memory_groups.json"
    payload = {
        "memory_group_size": memory_group_size,
        "total_jobs": len(jobs),
        "base_storage_dir": str(base_storage_dir),
        "groups": [],
    }
    for group in groups:
        first_job = jobs[group.start_job_index] if jobs else None
        last_job = jobs[group.end_job_index] if jobs else None
        payload["groups"].append(
            {
                "group_id": group.group_id,
                "index": group.index,
                "start_job_index": group.start_job_index,
                "end_job_index": group.end_job_index,
                "num_jobs": group.end_job_index - group.start_job_index + 1,
                "storage_dir": str(group.storage_dir),
                "writer_dataset_path": group.writer_dataset_path,
                "selector_dataset_path": group.selector_dataset_path,
                "first_job": (
                    {"scenario": first_job[0], "task_idx": first_job[1]}
                    if first_job is not None
                    else None
                ),
                "last_job": (
                    {"scenario": last_job[0], "task_idx": last_job[1]}
                    if last_job is not None
                    else None
                ),
            }
        )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    return manifest_path
async def _build_memory_runtimes(
    args: argparse.Namespace,
    jobs: list[tuple[str, int]],
) -> tuple[list[AWMMemoryRuntime | None], list[AWMMemoryRuntime]]:
    if not args.memory:
        return [None] * len(jobs), []
    group_size = int(getattr(args, "memory_group_size", 0) or 0)
    if group_size < 0:
        raise SystemExit("--memory-group-size must be >= 0")
    if group_size > 0:
        if args.writer_dataset_path is not None and str(args.writer_dataset_path).strip():
            raise SystemExit(
                "--memory-group-size cannot be used with a single --writer-dataset-path; "
                "each group writes its own memory_writer_dataset.jsonl"
            )
        if args.selector_dataset_path is not None and str(args.selector_dataset_path).strip():
            raise SystemExit(
                "--memory-group-size cannot be used with a single --selector-dataset-path; "
                "each group writes its own memory_selector_dataset.jsonl"
            )
    base_cfg = _resolve_memory_dataset_paths(_memory_config(args))
    base_storage_dir = Path(os.path.expandvars(str(base_cfg["storage_dir"]))).expanduser()
    role_llms = _build_memory_role_llms(args)
    embedding_cfg = dict(base_cfg)
    embedding_cfg["writer_dataset_path"] = None
    embedding_cfg["selector_dataset_path"] = None
    shared_embedding_provider = _create_shared_embedding_provider(
        _memory_config_from_resolved(embedding_cfg)
    )
    if group_size <= 0:
        runtime = _build_memory_runtime(
            args,
            index=0,
            group_id="global",
            start_job_index=0,
            end_job_index=max(len(jobs) - 1, 0),
            cfg=base_cfg,
            role_llms=role_llms,
            embedding_provider=shared_embedding_provider,
        )
        logger.info(
            f"[AWM] Memory enabled without grouping: storage_dir={runtime.storage_dir}"
        )
        return [runtime] * len(jobs), [runtime]
    logger.info(
        f"[AWM] Memory grouping enabled: group_size={group_size}, "
        f"base_storage_dir={base_storage_dir}"
    )
    groups: list[AWMMemoryRuntime] = []
    runtimes_by_job: list[AWMMemoryRuntime | None] = [None] * len(jobs)
    for group_index, start in enumerate(range(0, len(jobs), group_size)):
        end = min(start + group_size, len(jobs)) - 1
        group_id = f"group_{group_index:04d}"
        group_cfg_raw = _memory_config(args)
        group_cfg_raw["storage_dir"] = str(base_storage_dir / group_id)
        group_cfg_raw["writer_dataset_path"] = None
        group_cfg_raw["selector_dataset_path"] = None
        group_cfg = _resolve_memory_dataset_paths(group_cfg_raw)
        runtime = _build_memory_runtime(
            args,
            index=group_index,
            group_id=group_id,
            start_job_index=start,
            end_job_index=end,
            cfg=group_cfg,
            role_llms=role_llms,
            embedding_provider=shared_embedding_provider,
        )
        groups.append(runtime)
        for job_index in range(start, end + 1):
            runtimes_by_job[job_index] = runtime
    manifest_path = _write_memory_groups_manifest(
        base_storage_dir,
        memory_group_size=group_size,
        jobs=jobs,
        groups=groups,
    )
    logger.info(f"[AWM] Memory groups manifest: {manifest_path}")
    return runtimes_by_job, groups
def start_awm_server(
    openenv_root: Path,
    *,
    bind_host: str,
    port: int,
    timeout: int = 180,
) -> subprocess.Popen | None:
    root = openenv_root.resolve()
    app_py = root / "envs" / "agent_world_model_env" / "server" / "app.py"
    if not app_py.is_file():
        logger.error(
            "OpenEnv root does not look like a checkout with AWM server "
            f"(missing {app_py}). Set --openenv-root."
        )
        return None
    _kill_listen_port_guess_host("127.0.0.1", port)
    env = os.environ.copy()
    bundled_awm = PROJECT_ROOT / "data" / "agent_world_model"
    if "AWM_DATA_DIR" not in env and (bundled_awm / "gen_scenario.jsonl").is_file():
        env["AWM_DATA_DIR"] = str(bundled_awm.resolve())
        logger.info(f"[AWM] Using bundled dataset: AWM_DATA_DIR={env['AWM_DATA_DIR']}")
    prepend = os.pathsep.join(["src", "envs"])
    env["PYTHONPATH"] = (
        prepend + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else prepend
    )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "envs.agent_world_model_env.server.app:app",
        "--host",
        bind_host,
        "--port",
        str(port),
    ]
    logger.info(f"Launching AWM server: {sys.executable} -m uvicorn (current Python env)")
    logger.info(f"AWM OpenEnv cwd: {root}")
    logger.info(f"AWM listening on http://{bind_host}:{port} (poll health at http://127.0.0.1:{port}/stats)")
    proc = subprocess.Popen(cmd, cwd=str(root), env=env)
    stats_url = f"http://127.0.0.1:{port}/stats"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            logger.error(f"AWM server process exited early with code {proc.returncode}")
            return None
        try:
            with urllib.request.urlopen(stats_url, timeout=3) as resp:
                if resp.status == 200:
                    logger.info(f"AWM server ready after {time.time() - t0:.1f}s (pid={proc.pid})")
                    return proc
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    logger.error(f"AWM server did not become healthy within {timeout}s")
    proc.terminate()
    return None
def build_parser(cfg: dict[str, Any], *, effective_config_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OPD Evolver ReAct rollouts against OpenEnv Agent World Model tasks."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=effective_config_path,
        help="YAML benchmark config (CLI overrides YAML). Default ships under config/awm_bench/.",
    )
    parser.add_argument(
        "--model",
        default=_coalesce(cfg, "model"),
        help="Model key for OPD (and vLLM served name unless --vllm-served-model-name is set). YAML: model:",
    )
    parser.add_argument(
        "--base-url",
        default=str(_coalesce(cfg, "base_url", default="http://localhost:8899")),
        help="OpenEnv AWM server URL (used by the client; port must match the spawned server). YAML: base_url:",
    )
    parser.add_argument(
        "--start-awm",
        action=argparse.BooleanOptionalAction,
        default=bool(_coalesce(cfg, "start_awm", default=True)),
        help="Spawn the OpenEnv AWM uvicorn server before rollouts. YAML: start_awm:",
    )
    parser.add_argument(
        "--openenv-root",
        type=Path,
        default=_openenv_default(cfg),
        help=f"OpenEnv checkout containing src/ and envs/. YAML: openenv_root: (nullable). Fallback: {DEFAULT_OPENENV_ROOT}",
    )
    parser.add_argument(
        "--awm-bind-host",
        default=str(_coalesce(cfg, "awm_bind_host", default="0.0.0.0")),
        help="Host interface for uvicorn when --start-awm is on.",
    )
    parser.add_argument(
        "--awm-start-timeout",
        type=int,
        default=int(_coalesce(cfg, "awm_start_timeout", default=240)),
        help="Seconds to wait for AWM /stats before giving up.",
    )
    parser.add_argument(
        "--awm-connect-timeout",
        type=float,
        default=float(_coalesce(cfg, "awm_connect_timeout", default=10.0)),
        help=(
            "WebSocket opening-handshake timeout (seconds). Under high --concurrency the "
            "server may accept connections slowly; increase if you see handshake timeouts. "
            "YAML: awm_connect_timeout:"
        ),
    )
    parser.add_argument(
        "--awm-message-timeout",
        type=float,
        default=float(_coalesce(cfg, "awm_message_timeout", default=60.0)),
        help="WebSocket response timeout (seconds) per message. YAML: awm_message_timeout:",
    )
    parser.add_argument(
        "--start-vllm",
        action=argparse.BooleanOptionalAction,
        default=bool(_coalesce(cfg, "start_vllm", default=True)),
        help="Spawn local vLLM OpenAI-compatible server before rollouts.",
    )
    vm_dir = _coalesce(cfg, "vllm_model_dir", default=_VLLM_DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--vllm-model-dir",
        default=str(vm_dir),
        help="HF snapshot dir for weights (YAML may use $HF_HOME/... ). YAML: vllm_model_dir:",
    )
    parser.add_argument("--vllm-served-model-name", default=_coalesce(cfg, "vllm_served_model_name"))
    parser.add_argument("--vllm-host", default=str(_coalesce(cfg, "vllm_host", default="127.0.0.1")))
    parser.add_argument("--vllm-port", type=int, default=int(_coalesce(cfg, "vllm_port", default=8000)))
    parser.add_argument("--vllm-tp", type=int, default=int(_coalesce(cfg, "vllm_tp", default=2)))
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=int(_coalesce(cfg, "vllm_max_model_len", default=262144)),
    )
    parser.add_argument("--vllm-gpus", default=str(_coalesce(cfg, "vllm_gpus", default="0,1")))
    parser.add_argument(
        "--vllm-gpu-mem",
        type=float,
        default=float(_coalesce(cfg, "vllm_gpu_mem", default=0.9)),
    )
    parser.add_argument("--vllm-timeout", type=int, default=int(_coalesce(cfg, "vllm_timeout", default=300)))
    if "vllm_reasoning_parser" in cfg and cfg["vllm_reasoning_parser"] is not None:
        rp_cli_default = str(cfg["vllm_reasoning_parser"])
    elif "vllm_reasoning_parser" in cfg and cfg["vllm_reasoning_parser"] is None:
        rp_cli_default = ""
    else:
        rp_cli_default = "qwen3"
    parser.add_argument(
        "--vllm-reasoning-parser",
        default=rp_cli_default,
        help="vLLM --reasoning-parser; empty skips the flag.",
    )
    parser.add_argument(
        "--vllm-lora-module",
        default=_coalesce(cfg, "vllm_lora_module"),
        help="Static LoRA for vLLM: 'adapter_id=/path/to/peft'.",
    )
    parser.add_argument(
        "--vllm-lora-modules",
        action="append",
        default=_list_value(_coalesce(cfg, "vllm_lora_modules", default=[])),
        help="Repeatable static LoRA specs. Each item may be 'id=/path' or comma-separated specs.",
    )
    parser.add_argument(
        "--vllm-task-lora-module",
        default=_coalesce(cfg, "vllm_task_lora_module"),
        help="LoRA used by task execution; auto-sets --model to the served LoRA id.",
    )
    parser.add_argument(
        "--vllm-selector-lora-module",
        default=_coalesce(cfg, "vllm_selector_lora_module"),
        help="LoRA used by pre-task memory selection; auto-sets --selector-model when unset.",
    )
    parser.add_argument(
        "--vllm-reflection-lora-module",
        default=_coalesce(cfg, "vllm_reflection_lora_module"),
        help="LoRA used by post-task memory writing; auto-sets --reflection-model when unset.",
    )
    parser.add_argument(
        "--vllm-lora-max-rank",
        type=int,
        default=int(_coalesce(cfg, "vllm_lora_max_rank", default=32)),
    )
    parser.add_argument("--vllm-max-loras", type=int, default=_coalesce(cfg, "vllm_max_loras"))
    parser.add_argument("--scenario", default=_coalesce(cfg, "scenario"))
    parser.add_argument("--scenarios", default=_coalesce(cfg, "scenarios"))
    parser.add_argument(
        "--task-idx",
        type=int,
        default=_scalar_task_idx_from_config(cfg),
    )
    parser.add_argument(
        "--tasks",
        type=_parse_tasks,
        default=_tasks_from_config(cfg),
        help="Task indices or ranges, e.g. 0-9 or 0,2,5. YAML: tasks:",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(_coalesce(cfg, "max_steps", default=30)),
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=float(_coalesce(cfg, "step_timeout", default=120.0)),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_coalesce(cfg, "output_dir", default="workspace/awm_runs")),
    )
    vm = _coalesce(cfg, "verifier_mode", default="code")
    parser.add_argument("--verifier-mode", default=str(vm), choices=["code", "sql"])
    parser.add_argument(
        "--keep-session",
        action=argparse.BooleanOptionalAction,
        default=bool(_coalesce(cfg, "keep_session", default=False)),
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=bool(_coalesce(cfg, "memory", default=False)),
        help="Enable OPD hierarchical memory around AWM rollouts.",
    )
    parser.add_argument(
        "--memory-tiers",
        default=_coalesce(cfg, "memory_tiers"),
        help=(
            "Comma-separated subset of skill,tip,tool,trajectory; omit YAML key (or leave unset) "
            "for all tiers. YAML: memory_tiers: [skill, tip, ...]"
        ),
    )
    parser.add_argument(
        "--cold-start-threshold",
        type=int,
        default=int(_coalesce(cfg, "cold_start_threshold", default=20)),
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=int(_coalesce(cfg, "retrieval_top_k", default=3)),
    )
    parser.add_argument(
        "--embedding-provider",
        default=str(_coalesce(cfg, "embedding_provider", default="local")),
    )
    parser.add_argument(
        "--embedding-model",
        default=str(_coalesce(cfg, "embedding_model", default="Qwen/Qwen3-Embedding-0.6B")),
    )
    parser.add_argument(
        "--memory-storage-dir",
        default=_coalesce(cfg, "memory_storage_dir"),
        help="Directory for hierarchical memory files. Default: workspace/memory/awm.",
    )
    parser.add_argument(
        "--memory-group-size",
        type=int,
        default=int(_coalesce(cfg, "memory_group_size", default=0)),
        help=(
            "Split expanded AWM jobs into contiguous memory groups of this size. "
            "0 disables grouping and uses one memory_storage_dir. YAML: memory_group_size:"
        ),
    )
    parser.add_argument(
        "--reflection-model",
        default=_coalesce(cfg, "reflection_model"),
        help="Optional LLM config/model id for post-task memory writing.",
    )
    parser.add_argument(
        "--selector-model",
        default=_coalesce(cfg, "selector_model"),
        help="Optional LLM config/model id for pre-task memory selection.",
    )
    parser.add_argument(
        "--writer-dataset",
        action=argparse.BooleanOptionalAction,
        default=bool(_coalesce(cfg, "writer_dataset", default=False)),
        help="Append memory-writer JSONL rows under memory storage unless path is set.",
    )
    parser.add_argument("--writer-dataset-path", default=_coalesce(cfg, "writer_dataset_path"))
    parser.add_argument(
        "--selector-dataset",
        action=argparse.BooleanOptionalAction,
        default=bool(_coalesce(cfg, "selector_dataset", default=False)),
        help="Append memory-selector JSONL rows under memory storage unless path is set.",
    )
    parser.add_argument("--selector-dataset-path", default=_coalesce(cfg, "selector_dataset_path"))
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(_coalesce(cfg, "concurrency", default=1)),
    )
    lm_raw = _coalesce(cfg, "llm_max_completion_tokens")
    parser.add_argument(
        "--llm-max-completion-tokens",
        type=int,
        default=None if lm_raw is None else int(lm_raw),
    )
    return parser
async def run_rollouts(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    trajectory_dir = output_dir / "trajectories"
    csv_path = output_dir / "summary.csv"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    _init_summary_csv(csv_path)
    task_indices = args.tasks if args.tasks is not None else [args.task_idx]
    scenarios = _parse_scenarios(args)
    jobs: List[tuple[str, int]] = [
        (scenario, task_idx) for scenario in scenarios for task_idx in task_indices
    ]
    concurrency = max(1, args.concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    io_lock = asyncio.Lock()
    memory_runtimes_by_job, memory_groups = await _build_memory_runtimes(args, jobs)
    logger.info(f"[AWM] Running {len(jobs)} rollout(s) with concurrency={concurrency}")
    logger.info(
        "[AWM] High-concurrency mode: each rollout uses an independent runner/LLM; "
        "summary writes are serialized; memory file/state mutations are serialized per memory group."
    )
    if memory_groups:
        logger.info(f"[AWM] Memory runtime count: {len(memory_groups)}")
    def _steprecord_to_dict(step: Any) -> dict[str, Any]:
        return {
            "observation": step.observation,
            "observation_before": step.observation,
            "observation_after": step.observation_after,
            "action": step.action,
            "reward": step.reward,
            "done": step.done,
            "info": step.info,
            "raw_response": step.raw_response,
            "raw_input": step.raw_input,
            "act_prompt": getattr(step, "act_prompt", None),
        }
    async def _run_one(job_index: int, scenario: str, task_idx: int) -> Any | None:
        task_id = f"{scenario}:{task_idx}"
        memory_runtime = memory_runtimes_by_job[job_index]
        memory_pipeline = memory_runtime.pipeline if memory_runtime is not None else None
        memory_group_id = memory_runtime.group_id if memory_runtime is not None else ""
        memory_storage_dir = str(memory_runtime.storage_dir) if memory_runtime is not None else ""
        logger.info(
            f"[AWM] Running {scenario}:{task_idx}"
            + (f" memory_group={memory_group_id}" if memory_group_id else "")
        )
        env = AWMEnvironment(
            base_url=args.base_url,
            scenario=scenario,
            task_idx=task_idx,
            max_steps=args.max_steps,
            verifier_mode=args.verifier_mode,
            keep_session=args.keep_session,
            connect_timeout_s=args.awm_connect_timeout,
            message_timeout_s=args.awm_message_timeout,
        )
        runner = SimpleTaskRunner(
            model=args.model,
            env_type="awm",
            max_steps=args.max_steps,
            step_timeout=args.step_timeout,
            trajectory_dir=trajectory_dir,
            csv_summary_path=None,
            llm_max_completion_tokens=args.llm_max_completion_tokens,
        )
        memory_context = ""
        info = None
        try:
            async with semaphore:
                await env.prepare()
                info = env.get_basic_info()
                if memory_pipeline is not None:
                    try:
                        filtered = await memory_pipeline.pre_execution(
                            task_id=task_id,
                            task_description=info.instruction,
                            additional_context=json.dumps(info.meta_data or {}, ensure_ascii=False),
                            task_type="awm",
                        )
                        if filtered is not None and filtered.formatted_context:
                            memory_context = filtered.formatted_context
                            logger.info(
                                f"[AWM] Retrieved memory context for {task_id} "
                                f"({len(memory_context)} chars)"
                            )
                    except Exception as exc:
                        logger.warning(f"[AWM] Memory pre_execution failed for {task_id}: {exc}")
                result = await runner.run(agent=None, env=env, memory_context=memory_context)
                async with io_lock:
                    _append_summary_row(
                        csv_path,
                        _summary_row(
                            task_id,
                            result,
                            model=args.model,
                            memory_group_id=memory_group_id,
                            memory_storage_dir=memory_storage_dir,
                        ),
                    )
                logger.info(
                    f"[AWM] Finished {scenario}:{task_idx} "
                    f"success={result.success} reward={result.total_reward:.4f} steps={result.steps}"
                )
            if memory_pipeline is not None:
                try:
                    trace_dicts = [_steprecord_to_dict(step) for step in result.trace]
                    await memory_pipeline.post_execution(
                        task_id=task_id,
                        task_description=(info.instruction if info is not None else task_id),
                        execution_trace=trace_dicts,
                        success=result.success,
                        total_reward=result.total_reward,
                        tags=["awm", scenario],
                        task_type="awm",
                    )
                except Exception as exc:
                    logger.warning(f"[AWM] Memory post_execution failed for {task_id}: {exc}", exc_info=True)
            return result
        except Exception as exc:
            logger.error(f"[AWM] Rollout failed for {task_id}: {exc}", exc_info=True)
            async with io_lock:
                _append_summary_row(
                    csv_path,
                    _summary_row(
                        task_id,
                        None,
                        model=args.model,
                        memory_group_id=memory_group_id,
                        memory_storage_dir=memory_storage_dir,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
            return None
    rollouts = [_run_one(i, s, ti) for i, (s, ti) in enumerate(jobs)]
    _pbar_kw: dict[str, Any] = {
        "desc": "AWM-Bench",
        "unit": "rollout",
        "total": len(rollouts),
    }
    if not rollouts:
        results = []
    elif tqdm_async is not None:
        results = await tqdm_async.gather(*rollouts, **_pbar_kw)
    else:
        results = await asyncio.gather(*rollouts)
    total = len(results)
    succeeded = sum(1 for r in results if r is not None and r.success)
    logger.info(f"[AWM] Completed {total} rollout(s), success={succeeded}")
    logger.info(f"[AWM] Summary CSV: {csv_path}")
    logger.info(f"[AWM] Trajectories: {trajectory_dir}")
def main(argv: Iterable[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=str(DEFAULT_AWM_CONFIG))
    pre_ns, _ = pre.parse_known_args(argv_list)
    cfg_path = Path(pre_ns.config).expanduser()
    cfg = load_task_config(cfg_path)
    if not cfg_path.is_file():
        logger.warning(f"Bench YAML not found: {cfg_path} — using CLI-only defaults for missing YAML keys.")
    parser = build_parser(cfg, effective_config_path=str(cfg_path))
    args = parser.parse_args(argv_list)
    args.memory_tiers = _normalize_cli_memory_tiers(args.memory_tiers)
    try:
        _validate_memory_tiers_config(args.memory_tiers)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid memory_tiers: {exc}") from exc
    args.vllm_model_dir = os.path.expandvars(os.path.expanduser(str(args.vllm_model_dir)))
    if not args.model:
        raise SystemExit("--model missing: set YAML `model:` or pass --model on the CLI.")
    if _INTERCODE._apply_vllm_lora_cli(args):
        raise SystemExit(1)
    if args.openenv_root is not None:
        expanded = os.path.expandvars(os.path.expanduser(str(args.openenv_root))).strip()
        if not expanded or expanded.lower() in {"none", "null"}:
            args.openenv_root = None
        else:
            args.openenv_root = Path(expanded)
    _prepend_openenv_client_sys_path(args.openenv_root)
    url_port, base = _parse_base_url(args.base_url)
    setattr(args, "base_url", base)
    vllm_proc: subprocess.Popen | None = None
    awm_proc: subprocess.Popen | None = None
    if args.start_vllm:
        served = (
            args.vllm_served_model_name
            or getattr(args, "_vllm_base_served_model_name", None)
            or args.model
        )
        rp = (args.vllm_reasoning_parser or "").strip() or None
        vllm_proc = start_vllm_server(
            model_dir=args.vllm_model_dir,
            served_model_name=served,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel_size=args.vllm_tp,
            max_model_len=args.vllm_max_model_len,
            cuda_visible_devices=args.vllm_gpus,
            gpu_memory_utilization=args.vllm_gpu_mem,
            timeout=args.vllm_timeout,
            reasoning_parser=rp,
            lora_modules=getattr(args, "_vllm_lora_modules", None),
            max_lora_rank=args.vllm_lora_max_rank,
            max_loras=args.vllm_max_loras,
        )
        if vllm_proc is None:
            raise SystemExit(1)
        _bind_llm_configs_to_vllm(args, served_model_name=served)
    if args.start_awm:
        if args.openenv_root is None:
            raise SystemExit("--openenv-root is required when --start-awm is enabled (unset default missing).")
        awm_proc = start_awm_server(
            Path(args.openenv_root),
            bind_host=args.awm_bind_host,
            port=url_port,
            timeout=args.awm_start_timeout,
        )
        if awm_proc is None:
            if vllm_proc is not None:
                stop_vllm_server(vllm_proc)
            raise SystemExit(1)
    try:
        asyncio.run(run_rollouts(args))
    finally:
        if awm_proc is not None:
            logger.info(f"Stopping AWM server (pid={awm_proc.pid})...")
            awm_proc.terminate()
            try:
                awm_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logger.warning("AWM server did not exit after SIGTERM, sending SIGKILL")
                awm_proc.kill()
                awm_proc.wait()
            logger.info("AWM server stopped")
        if vllm_proc is not None:
            stop_vllm_server(vllm_proc)
if __name__ == "__main__":
    main()
