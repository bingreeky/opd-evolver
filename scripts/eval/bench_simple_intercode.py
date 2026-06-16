#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import docker
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from opd_evolver.base.engine.logs import logger
from opd_evolver.benchmark.bench_intercode import InterCodeBenchmark
from opd_evolver.base.engine.async_llm import LLMsConfig
from opd_evolver.memory.evolvelab_adapter import (
    ALL_MEMORY_BACKENDS,
    OPD_HIERARCHICAL_BACKEND,
    default_baseline_output_dir,
    is_provider_backend,
)
from opd_evolver.runners.task_runner import (
    SimpleTaskRunner,
    MemoryAugmentedTaskRunner,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "intercode_task"
SUPPORTED_ENVS = ["bash", "sql", "ctf"]
def load_task_config(config_path: Path) -> Dict[str, Any]:
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
def resolve_config_path(env_name: Optional[str], cli_config: Optional[str]) -> Optional[Path]:
    if cli_config:
        return Path(cli_config)
    if env_name:
        return CONFIG_ROOT / f"{env_name}.yaml"
    return None
def default_data_path_for_env(env_name: str) -> Optional[str]:
    if env_name == "ctf":
        return str(PROJECT_ROOT / "data" / "ctf" / "ic_ctf.json")
    if env_name == "bash":
        return None
    if env_name == "sql":
        return None
    return None
def resolve_data_path_for_env(env_name: str, raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return default_data_path_for_env(env_name)
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        return str(path)
    if env_name in {"bash", "sql"}:
        logger.warning(
            f"{env_name.upper()} data_path not found, using intercode-bench built-in data: {path}"
        )
        return None
    logger.warning(f"{env_name.upper()} data_path not found: {path}")
    return str(path)
def start_mysql_container(image_name: str = "docker-env-sql:latest",
                         container_name: str = "intercode_mysql_server",
                         host_port: int = 3307,
                         container_port: int = 3306,
                         timeout: int = 30) -> Optional[Any]:
    import time
    import mysql.connector
    try:
        client = docker.from_env()
        try:
            existing = client.containers.get(container_name)
            logger.warning(f"MySQL container '{container_name}' already exists, stopping it...")
            existing.stop()
            existing.remove()
        except docker.errors.NotFound:
            pass
        logger.info(f"Starting MySQL container: {container_name}")
        logger.info(f"  Image: {image_name}")
        logger.info(f"  Port mapping: {host_port}:{container_port}")
        container = client.containers.run(
            image_name,
            name=container_name,
            ports={f"{container_port}/tcp": host_port},
            detach=True,
            remove=False,
        )
        logger.info("Waiting for MySQL to be ready...")
        sql_config = {
            'host': '127.0.0.1',
            'port': host_port,
            'user': 'admin',
            'password': 'admin',
        }
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                cnx = mysql.connector.connect(**sql_config)
                cnx.close()
                elapsed = time.time() - start_time
                logger.info(f"✓ MySQL ready after {elapsed:.1f}s")
                logger.info(f"✓ MySQL container started: {container.id[:12]}")
                return container
            except mysql.connector.errors.Error as e:
                time.sleep(1)
                continue
        logger.error(f"MySQL failed to become ready within {timeout}s")
        container.stop()
        container.remove()
        return None
    except Exception as e:
        logger.error(f"Failed to start MySQL container: {e}")
        return None
def stop_mysql_container(container: Optional[Any], container_name: str = "intercode_mysql_server"):
    if container is None:
        try:
            client = docker.from_env()
            container = client.containers.get(container_name)
        except:
            logger.warning(f"MySQL container '{container_name}' not found, skipping cleanup")
            return
    try:
        logger.info(f"Stopping MySQL container: {container.name}")
        container.stop(timeout=5)
        container.remove()
        logger.info("✓ MySQL container stopped and removed")
    except Exception as e:
        logger.error(f"Failed to stop MySQL container: {e}")
_VLLM_DEFAULT_MODEL_DIR: str | None = None
_LEGACY_VLLM_MODEL_DIR = (
    "/path/to/models/Qwen3.5-9B"
    "/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
def _is_hf_snapshot_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()
def _pick_latest_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if _is_hf_snapshot_dir(cand):
            return cand
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
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique
def _hf_snapshot_cache_dir(model_id: str, hub_root: Path) -> Path:
    return hub_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
def _auto_resolve_qwen35_9b_snapshot(model_dir_hint: str | None = None) -> str | None:
    hints: list[Path] = []
    if model_dir_hint:
        hints.append(Path(model_dir_hint).expanduser())
    hints.append(Path(_LEGACY_VLLM_MODEL_DIR))
    for hub_root in _hf_hub_cache_roots():
        hints.append(_hf_snapshot_cache_dir("Qwen/Qwen3.5-9B", hub_root))
    for hint in hints:
        if _is_hf_snapshot_dir(hint):
            return str(hint.resolve())
        latest = _pick_latest_snapshot(hint)
        if latest is not None:
            return str(latest.resolve())
    return None
def resolve_vllm_model_dir(model_dir: str | None) -> str | None:
    hint: str | None = None
    if model_dir:
        text = os.path.expandvars(os.path.expanduser(str(model_dir).strip()))
        if text and text.lower() not in {"null", "none"} and "PATH/TO/YOUR/MODEL" not in text:
            hint = text
            path = Path(hint)
            if _is_hf_snapshot_dir(path):
                return str(path.resolve())
            latest = _pick_latest_snapshot(path)
            if latest is not None:
                return str(latest.resolve())
    return _auto_resolve_qwen35_9b_snapshot(hint)
def _kill_existing_vllm(host: str, port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid.isdigit():
                logger.warning(f"Killing existing process on port {port}: pid={pid}")
                os.kill(int(pid), signal.SIGKILL)
        if pids:
            time.sleep(2)
    except Exception as e:
        logger.warning(f"Could not check/kill existing vLLM process: {e}")
def start_vllm_server(
    model_dir: str = _VLLM_DEFAULT_MODEL_DIR,
    served_model_name: str = "qwen/qwen3.5-9b",
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel_size: int = 2,
    max_model_len: int = 262144,
    cuda_visible_devices: str = "0,1",
    gpu_memory_utilization: float = 0.9,
    timeout: int = 300,
    lora_modules: Optional[List[str]] = None,
    max_lora_rank: int = 16,
    max_loras: Optional[int] = None,
    reasoning_parser: Optional[str] = "qwen3",
) -> Optional[subprocess.Popen]:
    resolved_dir = resolve_vllm_model_dir(model_dir)
    if resolved_dir and resolved_dir != model_dir:
        logger.info(f"Resolved vLLM model dir: {resolved_dir}")
    model_dir = resolved_dir or model_dir
    model_path = Path(model_dir).expanduser()
    if not model_path.exists():
        logger.error(
            "vLLM model_dir does not exist on disk. "
            "Set --vllm-model-dir (or YAML vllm_model_dir) to a local HF snapshot directory. "
            f"Got: {model_dir}"
        )
        return None
    if not model_path.is_dir():
        logger.error(f"vLLM model_dir must be a directory, got: {model_dir}")
        return None
    _kill_existing_vllm(host, port)
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    env = os.environ.copy()
    env.update({
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE") or f"{hf_home}/hub",
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE") or f"{hf_home}/datasets",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_IB_DISABLE": "0",
    })
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir,
        "--host", host,
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    if lora_modules:
        cmd += [
            "--enable-lora",
            "--lora-modules",
            *lora_modules,
            "--max-lora-rank",
            str(max_lora_rank),
        ]
        if max_loras is not None:
            cmd += ["--max-loras", str(max_loras)]
    logger.info(f"Starting vLLM server: {served_model_name}")
    logger.info(f"  Model dir:  {model_dir}")
    logger.info(f"  Endpoint:   http://{host}:{port}")
    logger.info(f"  TP size:    {tensor_parallel_size}  GPUs: {cuda_visible_devices}")
    if lora_modules:
        logger.info(f"  LoRAs:      {', '.join(lora_modules)}")
        if max_loras is not None:
            logger.info(f"  Max LoRAs:  {max_loras}")
    proc = subprocess.Popen(cmd, env=env)
    health_url = f"http://{host}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            logger.error(f"vLLM process exited early with code {proc.returncode}")
            return None
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    logger.info(f"✓ vLLM server ready after {elapsed:.1f}s (pid={proc.pid})")
                    return proc
        except Exception:
            pass
        time.sleep(3)
    logger.error(f"vLLM server did not become healthy within {timeout}s")
    proc.terminate()
    return None
def stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        logger.info(f"vLLM server (pid={proc.pid}) already exited")
        return
    logger.info(f"Stopping vLLM server (pid={proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        logger.warning("vLLM did not exit after SIGTERM, sending SIGKILL")
        proc.kill()
        proc.wait()
    logger.info("✓ vLLM server stopped")
def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env", type=str, choices=SUPPORTED_ENVS)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_path = resolve_config_path(pre_args.env, pre_args.config)
    config = load_task_config(config_path) if config_path else {}
    env_default = pre_args.env
    if env_default is None:
        cfg_env = config.get("env")
        if cfg_env in SUPPORTED_ENVS:
            env_default = cfg_env
    parser = argparse.ArgumentParser(description="Simple InterCode Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=str(config_path) if config_path else None,
        help="YAML config path. Default: config/intercode_task/{env}.yaml",
    )
    parser.add_argument("--env", type=str, required=env_default is None,
                        default=env_default,
                        choices=SUPPORTED_ENVS,
                        help="Environment type: bash, sql, or ctf")
    parser.add_argument("--task", type=int, default=config.get("task"),
                        help="Run specific task index (0-based)")
    parser.add_argument("--tasks", type=str, default=config.get("tasks"),
                        help="Task indices: '0,1,2' or '0-10'")
    parser.add_argument("--max-tasks", type=int, default=config.get("max_tasks"),
                        help="Maximum number of tasks to run")
    parser.add_argument("--model", type=str, default=config.get("model", "qwen/qwen3.5-9b"),
                        help="LLM model to use")
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction,
                        default=config.get("memory", False),
                        help="Enable memory augmentation (4-tier hierarchical)")
    parser.add_argument("--memory-backend", choices=ALL_MEMORY_BACKENDS,
                        default=config.get("memory_backend", OPD_HIERARCHICAL_BACKEND),
                        help="Memory backend. Default preserves existing OPD hierarchical memory.")
    parser.add_argument("--cold-start-threshold", type=int, default=config.get("cold_start_threshold", 20),
                        help="Tasks before enabling memory retrieval")
    parser.add_argument("--retrieval-top-k", type=int, default=config.get("retrieval_top_k", 3),
                        help="Max items to retrieve per memory tier")
    parser.add_argument("--memory-min-similarity", type=float, default=config.get("memory_min_similarity", 0.0),
                        help="Minimum similarity for provider-style memory retrieval")
    parser.add_argument("--embedding-provider", type=str, default=config.get("embedding_provider", "local"),
                        help="Embedding provider for memory runner")
    parser.add_argument("--embedding-model", type=str, default=config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                        help="Embedding model for memory runner")
    parser.add_argument(
        "--memory-storage-dir",
        type=str,
        default=config.get("memory_storage_dir"),
        help="Directory for memory storage. Default: workspace/memory/{env}",
    )
    parser.add_argument("--memory-read-only", action=argparse.BooleanOptionalAction,
                        default=config.get("memory_read_only", False),
                        help="Retrieve provider memory but skip post-task updates.")
    parser.add_argument("--memrl-storage-dir", type=str, default=config.get("memrl_storage_dir"),
                        help="Storage directory for MemRL backend. Defaults to memory_storage_dir.")
    parser.add_argument("--memrl-mos-config-path", type=str, default=config.get("memrl_mos_config_path"),
                        help="Optional path to MemRL mos_config_final.json.")
    parser.add_argument("--memrl-user-id", type=str, default=config.get("memrl_user_id"),
                        help="MemRL user_id namespace.")
    parser.add_argument("--memrl-build-strategy", type=str, default=config.get("memrl_build_strategy", "proceduralization"))
    parser.add_argument("--memrl-retrieve-strategy", type=str, default=config.get("memrl_retrieve_strategy", "query"))
    parser.add_argument("--memrl-update-strategy", type=str, default=config.get("memrl_update_strategy", "adjustment"))
    parser.add_argument("--memrl-enable-value-driven", action=argparse.BooleanOptionalAction,
                        default=config.get("memrl_enable_value_driven", True),
                        help="Enable MemRL value-aware retrieval/update when supported.")
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=config.get("reflection_model"),
        help="Optional LLM for memory-writing reflection only (e.g. LoRA); "
        "task steps use --model; memory selection uses --model unless --selector-model is set",
    )
    parser.add_argument(
        "--writer-dataset-path",
        type=str,
        default=config.get("writer_dataset_path"),
        help="Append memory-writer JSONL here (trajectory→memory training rows). None disables.",
    )
    parser.add_argument(
        "--writer-dataset",
        action=argparse.BooleanOptionalAction,
        default=config.get("writer_dataset", False),
        help="If true and --writer-dataset-path unset, use <workspace/memory/{env}/memory_writer_dataset.jsonl>",
    )
    parser.add_argument(
        "--selector-model",
        type=str,
        default=config.get("selector_model"),
        help="Optional LLM for memory selection only (e.g. LoRA); "
        "task steps use --model; if unset, selection uses --model",
    )
    parser.add_argument(
        "--selector-dataset-path",
        type=str,
        default=config.get("selector_dataset_path"),
        help="Append memory-selector JSONL here. None disables.",
    )
    parser.add_argument(
        "--selector-dataset",
        action=argparse.BooleanOptionalAction,
        default=config.get("selector_dataset", False),
        help="If true and --selector-dataset-path unset, use <workspace/memory/{env}/memory_selector_dataset.jsonl>",
    )
    parser.add_argument("--max-steps", type=int, default=config.get("max_steps", 30),
                        help="Maximum steps per task")
    parser.add_argument("--step-timeout", type=float, default=config.get("step_timeout", 180.0),
                        help="Timeout per agent step (seconds)")
    parser.add_argument("--concurrency", type=int, default=config.get("concurrency", 1),
                        help="Number of tasks to run concurrently")
    parser.add_argument("--llm-max-completion-tokens", type=int,
                        default=config.get("llm_max_completion_tokens"),
                        help="Optional max tokens per LLM completion")
    parser.add_argument("--output-dir", type=str, default=config.get("output_dir"),
                        help="Output directory for trajectories and summaries")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=config.get("verbose", False),
                        help="Enable verbose logging")
    parser.add_argument("--data-path", type=str, default=config.get("data_path"),
                        help="Path to InterCode data (bash/sql)")
    parser.add_argument("--image-name", type=str, default=config.get("image_name"),
                        help="Docker image name (defaults: intercode-bash/sql/ctf)")
    parser.add_argument("--mysql-image-name", type=str, default=config.get("mysql_image_name", "docker-env-sql:latest"),
                        help="MySQL docker image name for sql env")
    parser.add_argument("--mysql-container-name", type=str, default=config.get("mysql_container_name", "intercode_mysql_server"),
                        help="MySQL container name for sql env")
    parser.add_argument("--mysql-host-port", type=int, default=config.get("mysql_host_port", 3307),
                        help="Host port for MySQL container")
    parser.add_argument("--mysql-container-port", type=int, default=config.get("mysql_container_port", 3306),
                        help="Container MySQL port")
    parser.add_argument("--mysql-timeout", type=int, default=config.get("mysql_timeout", 30),
                        help="Seconds to wait until MySQL is ready")
    parser.add_argument("--sql-service-mode", choices=["docker", "local"],
                        default=config.get("sql_service_mode", "docker"),
                        help="For sql env: 'docker' starts docker-env-sql; 'local' uses an existing MySQL service.")
    parser.add_argument("--sql-host", type=str, default=config.get("sql_host", "127.0.0.1"),
                        help="MySQL host when --sql-service-mode=local")
    parser.add_argument("--sql-port", type=int, default=config.get("sql_port", 3307),
                        help="MySQL port when --sql-service-mode=local")
    parser.add_argument("--sql-user", type=str, default=config.get("sql_user", "admin"),
                        help="MySQL user when --sql-service-mode=local")
    parser.add_argument("--sql-password", type=str, default=config.get("sql_password", "admin"),
                        help="MySQL password when --sql-service-mode=local")
    parser.add_argument("--vllm", action=argparse.BooleanOptionalAction,
                        default=config.get("vllm", False),
                        help="Auto-start and stop a local vLLM server around the benchmark run")
    parser.add_argument("--vllm-model-dir", type=str, default=config.get("vllm_model_dir", _VLLM_DEFAULT_MODEL_DIR),
                        help="Path to the vLLM model snapshot directory")
    parser.add_argument("--vllm-host", type=str, default=config.get("vllm_host", "127.0.0.1"),
                        help="vLLM server host")
    parser.add_argument("--vllm-port", type=int, default=config.get("vllm_port", 8000),
                        help="vLLM server port")
    parser.add_argument("--vllm-tp", type=int, default=config.get("vllm_tp", 2),
                        help="vLLM tensor-parallel-size (number of GPUs)")
    parser.add_argument("--vllm-max-model-len", type=int, default=config.get("vllm_max_model_len", 262144),
                        help="vLLM max-model-len")
    parser.add_argument("--vllm-gpus", type=str, default=config.get("vllm_gpus", "0,1"),
                        help="CUDA_VISIBLE_DEVICES for vLLM (e.g. '0,1')")
    parser.add_argument("--vllm-gpu-mem", type=float, default=config.get("vllm_gpu_mem", 0.9),
                        help="vLLM gpu-memory-utilization (0.0-1.0)")
    parser.add_argument("--vllm-timeout", type=int, default=config.get("vllm_timeout", 300),
                        help="Seconds to wait for vLLM server to become healthy")
    parser.add_argument(
        "--vllm-lora-module",
        type=str,
        default=config.get("vllm_lora_module"),
        help="Static LoRA for vLLM: 'adapter_id=/path/to/peft' (API model id = adapter_id).",
    )
    parser.add_argument(
        "--vllm-lora-modules",
        action="append",
        default=list(config.get("vllm_lora_modules", []) or []),
        help="Repeatable static LoRA specs for vLLM. Each entry may be "
        "'adapter_id=/path/to/peft' or a comma-separated list of such specs.",
    )
    parser.add_argument(
        "--vllm-task-lora-module",
        type=str,
        default=config.get("vllm_task_lora_module"),
        help="Convenience alias for the LoRA used by task execution. "
        "Auto-registers the served model id and sets --model to that id.",
    )
    parser.add_argument(
        "--vllm-selector-lora-module",
        type=str,
        default=config.get("vllm_selector_lora_module"),
        help="Convenience alias for the LoRA used by pre-task memory selection. "
        "If --selector-model is unset, it is auto-set to this served model id.",
    )
    parser.add_argument(
        "--vllm-reflection-lora-module",
        type=str,
        default=config.get("vllm_reflection_lora_module"),
        help="Convenience alias for the LoRA used by post-task memory writing. "
        "If --reflection-model is unset, it is auto-set to this served model id.",
    )
    parser.add_argument(
        "--vllm-lora-max-rank",
        type=int,
        default=config.get("vllm_lora_max_rank", 32),
        help="vLLM --max-lora-rank (>= LoRA r in adapter_config.json).",
    )
    parser.add_argument(
        "--vllm-max-loras",
        type=int,
        default=config.get("vllm_max_loras"),
        help="vLLM --max-loras. Defaults to the number of configured LoRAs when omitted.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default=(config.get("openai_base_url") or None),
        help=(
            "OpenAI-compatible API base URL for a server you started yourself "
            "(e.g. vLLM). Mutually exclusive with --vllm. "
            "Example: http://127.0.0.1:8000/v1"
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=config.get("openai_api_key", "EMPTY"),
        help="API key for --openai-base-url (vLLM locally often uses EMPTY).",
    )
    args = parser.parse_args()
    args.data_path = resolve_data_path_for_env(args.env, args.data_path)
    if getattr(args, "openai_base_url", None) == "":
        args.openai_base_url = None
    return args
def _expand_lora_specs(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw if x]
    else:
        items = [str(raw)]
    specs: List[str] = []
    for item in items:
        for part in item.split(","):
            spec = part.strip()
            if spec:
                specs.append(spec)
    return specs
def _normalize_lora_spec(raw: str, flag_name: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"{flag_name} must look like 'adapter_id=/path/to/adapter'")
    lora_id, lora_path = raw.split("=", 1)
    lora_id = lora_id.strip()
    lora_path = lora_path.strip()
    if not lora_id or not lora_path:
        raise ValueError(f"{flag_name}: empty name or path")
    abs_path = str(Path(lora_path).expanduser().resolve())
    return lora_id, abs_path
def _register_vllm_model_alias(lc: LLMsConfig, base_key: str, alias: str) -> None:
    base_cfg = lc.configs[base_key]
    alias_cfg = dict(base_cfg)
    alias_cfg["model"] = alias
    lc.add_config(alias, alias_cfg)
def _normalize_openai_base_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"
def _bind_llm_configs_to_openai_endpoint(
    args: argparse.Namespace,
    *,
    base_url: str,
    api_key: str,
    served_model_name: str,
) -> None:
    lc = LLMsConfig.default()
    endpoint = _normalize_openai_base_url(base_url)
    def _ensure(name: str | None) -> None:
        if not name:
            return
        existing = lc.configs.get(name)
        if existing is None:
            lc.add_config(
                name,
                {
                    "model": name,
                    "api_key": api_key,
                    "base_url": endpoint,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            )
            return
        existing["base_url"] = endpoint
        existing["api_key"] = api_key
        if name == served_model_name:
            existing["model"] = served_model_name
    _ensure(served_model_name)
    _ensure(getattr(args, "model", None))
    _ensure(getattr(args, "selector_model", None))
    _ensure(getattr(args, "reflection_model", None))
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
def _apply_vllm_lora_cli(args: argparse.Namespace) -> int:
    raw_specs: List[str] = []
    if getattr(args, "vllm_lora_module", None):
        raw_specs.extend(_expand_lora_specs(args.vllm_lora_module))
    raw_specs.extend(_expand_lora_specs(getattr(args, "vllm_lora_modules", None)))
    role_specs: Dict[str, Optional[str]] = {
        "task": getattr(args, "vllm_task_lora_module", None),
        "selector": getattr(args, "vllm_selector_lora_module", None),
        "reflection": getattr(args, "vllm_reflection_lora_module", None),
    }
    for raw in role_specs.values():
        raw_specs.extend(_expand_lora_specs(raw))
    if not raw_specs:
        setattr(args, "_vllm_lora_modules", [])
        return 0
    normalized_by_id: Dict[str, str] = {}
    normalized_list: List[str] = []
    try:
        for raw in raw_specs:
            lora_id, abs_path = _normalize_lora_spec(raw, "--vllm-lora-module")
            prev = normalized_by_id.get(lora_id)
            if prev and prev != abs_path:
                raise ValueError(
                    f"LoRA id {lora_id!r} is mapped to multiple paths: {prev} vs {abs_path}"
                )
            if not prev:
                normalized_by_id[lora_id] = abs_path
                normalized_list.append(f"{lora_id}={abs_path}")
    except ValueError as e:
        logger.error(str(e))
        return 1
    lc = LLMsConfig.default()
    base_key = args.model
    setattr(args, "_vllm_base_served_model_name", base_key)
    if base_key not in lc.configs:
        logger.error(
            f"Cannot register LoRA models: base {base_key!r} missing from model config."
        )
        return 1
    for lora_id in normalized_by_id:
        _register_vllm_model_alias(lc, base_key, lora_id)
    def _resolve_role_id(role_name: str, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        specs = _expand_lora_specs(raw)
        if len(specs) != 1:
            logger.error(
                f"--vllm-{role_name}-lora-module accepts exactly one LoRA spec, got: {raw!r}"
            )
            raise ValueError(role_name)
        lora_id, _ = _normalize_lora_spec(specs[0], f"--vllm-{role_name}-lora-module")
        return lora_id
    try:
        task_lora_id = _resolve_role_id("task", role_specs["task"])
        selector_lora_id = _resolve_role_id("selector", role_specs["selector"])
        reflection_lora_id = _resolve_role_id("reflection", role_specs["reflection"])
    except ValueError:
        return 1
    if task_lora_id is None and getattr(args, "vllm_lora_module", None) and len(normalized_by_id) == 1:
        task_lora_id = next(iter(normalized_by_id))
    if task_lora_id:
        args.model = task_lora_id
    if selector_lora_id and not args.selector_model:
        args.selector_model = selector_lora_id
    if reflection_lora_id and not args.reflection_model:
        args.reflection_model = reflection_lora_id
    setattr(args, "_vllm_lora_modules", normalized_list)
    if args.vllm_max_loras is None and normalized_list:
        args.vllm_max_loras = len(normalized_list)
    logger.info(
        "LoRA eval: served model ids registered on the local vLLM server: "
        + ", ".join(normalized_by_id.keys())
    )
    if task_lora_id:
        logger.info(f"  Task model -> {task_lora_id}")
    if selector_lora_id:
        logger.info(f"  Selector model -> {args.selector_model}")
    if reflection_lora_id:
        logger.info(f"  Reflection model -> {args.reflection_model}")
    return 0
def parse_task_indices(tasks_str: str, max_count: int = 1000) -> List[int]:
    indices = []
    for part in tasks_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < max_count]
def compute_sql_success_summary(output_dir: Path) -> Optional[Dict[str, Any]]:
    summary_csv = output_dir / "summary.csv"
    if not summary_csv.exists():
        logger.warning(
            f"SQL summary.csv not found, skipping CSV-based success rate: {summary_csv}"
        )
        return None
    module_path = PROJECT_ROOT / "scripts" / "calc_success_rate_summary_csv.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "calc_success_rate_summary_csv",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        logs_ic_sql_dir = output_dir / "trajectories" / "logs_ic_sql"
        corrupt_map, _ = module.load_corrupt_gold_from_ic_logs(logs_ic_sql_dir)
        stats = module.compute_one_csv(
            summary_csv,
            logs_ic_sql_dir=None,
            auto_logs_ic_sql=True,
            corrupt_cache={},
        )
        return {
            "total_rows": stats.total_rows,
            "ignored_corrupt_gold": stats.ignored_corrupt_gold,
            "counted": stats.counted,
            "success_true": stats.success_true,
            "success_rate": stats.success_rate,
            "read_errors": stats.read_errors,
            "missing_success": stats.missing_success,
            "missing_corrupt_lookup": stats.missing_corrupt_lookup,
            "ignored_task_ids": {
                str(task_id) for task_id, is_corrupt in corrupt_map.items() if is_corrupt
            },
        }
    except Exception as e:
        logger.warning(f"Failed to compute SQL success summary from CSV: {e}")
        return None
async def run_task(
    runner: SimpleTaskRunner,
    benchmark: InterCodeBenchmark,
    task_idx: int,
    env_type: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    async with semaphore:
        try:
            level = benchmark.list_levels()[task_idx]
        except IndexError:
            level = {"id": f"{env_type}_{task_idx}", "index": task_idx}
        env = benchmark.make_env(level)
        try:
            logger.info(f"Running task {task_idx}: {level['id']}")
            result = await runner.run(agent=None, env=env)
            logger.info(f"Task {task_idx} completed: success={result.success}, reward={result.total_reward}, steps={result.steps}, cost=${result.cost:.4f}")
            return {
                "task_id": level["id"],
                "task_idx": task_idx,
                "success": result.success,
                "reward": result.total_reward,
                "steps": result.steps,
                "cost": result.cost,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Task {task_idx} failed: {e}")
            return {
                "task_id": level["id"],
                "task_idx": task_idx,
                "success": False,
                "reward": 0.0,
                "steps": 0,
                "cost": 0.0,
                "error": str(e),
            }
async def main():
    args = parse_args()
    if _apply_vllm_lora_cli(args):
        return 1
    if args.vllm and getattr(args, "openai_base_url", None):
        logger.error("Choose either --vllm (auto-start server) or --openai-base-url (existing server), not both.")
        return 1
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.memory and is_provider_backend(args.memory_backend):
        output_dir = default_baseline_output_dir(
            "intercode", args.model, args.memory_backend, args.env
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "workspace" / "logs" / f"{args.env}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "summary.csv"
    logger.info(f"[Simple{args.env.upper()}] Output directory: {output_dir}")
    logger.info(f"[Simple{args.env.upper()}] Model: {args.model}")
    logger.info(f"[Simple{args.env.upper()}] Memory: {'enabled' if args.memory else 'disabled'}")
    if args.memory:
        logger.info(f"[Simple{args.env.upper()}] Memory backend: {args.memory_backend}")
    vllm_proc = None
    if args.vllm:
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
            logger.error("Failed to start vLLM server, aborting benchmark")
            return 1
        _bind_llm_configs_to_vllm(args, served_model_name=base_served)
    elif getattr(args, "openai_base_url", None):
        base_served = getattr(args, "_vllm_base_served_model_name", None) or args.model
        _bind_llm_configs_to_openai_endpoint(
            args,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key or "EMPTY",
            served_model_name=base_served,
        )
        logger.info(f"[Simple{args.env.upper()}] Using OpenAI-compatible endpoint: {_normalize_openai_base_url(args.openai_base_url)}")
    mysql_container = None
    if args.env == "sql" and args.sql_service_mode == "docker":
        mysql_container = start_mysql_container(
            image_name=args.mysql_image_name,
            container_name="docker-env-sql_ic_ctr",
            host_port=args.mysql_host_port,
            container_port=args.mysql_container_port,
            timeout=args.mysql_timeout,
        )
        if mysql_container is None:
            logger.error("Failed to start MySQL container, aborting SQL benchmark")
            return
    elif args.env == "sql":
        logger.info(
            f"Using local SQL service: {args.sql_user}@{args.sql_host}:{args.sql_port}"
        )
    data_path = args.data_path
    if data_path:
        logger.info(f"Using {args.env.upper()} data path: {data_path}")
    benchmark = InterCodeBenchmark(
        env_type=args.env,
        image_name=args.image_name,
        data_path=data_path,
        traj_dir=str(trajectory_dir),
        verbose=args.verbose,
        max_steps=args.max_steps,
        sql_service_mode=args.sql_service_mode,
        sql_host=args.sql_host,
        sql_port=args.sql_port,
        sql_user=args.sql_user,
        sql_password=args.sql_password,
    )
    all_levels = benchmark.list_levels()
    total_tasks = len(all_levels)
    if args.task is not None:
        task_indices = [args.task]
    elif args.tasks:
        task_indices = parse_task_indices(args.tasks, total_tasks)
    elif args.max_tasks:
        task_indices = list(range(min(args.max_tasks, total_tasks)))
    else:
        task_indices = list(range(total_tasks))
    logger.info(f"[Simple{args.env.upper()}] Running {len(task_indices)} tasks: {task_indices[:10]}{'...' if len(task_indices) > 10 else ''}")
    if args.memory:
        memory_config = {
            "env_type": args.env,
            "memory_backend": args.memory_backend,
            "cold_start_threshold": args.cold_start_threshold,
            "retrieval_top_k": args.retrieval_top_k,
            "min_similarity": args.memory_min_similarity,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "storage_dir": args.memory_storage_dir
            or (str(output_dir / "memory") if is_provider_backend(args.memory_backend) else None),
            "memory_read_only": args.memory_read_only,
            "memrl_storage_dir": args.memrl_storage_dir,
            "memrl_mos_config_path": args.memrl_mos_config_path,
            "memrl_user_id": args.memrl_user_id,
            "memrl_build_strategy": args.memrl_build_strategy,
            "memrl_retrieve_strategy": args.memrl_retrieve_strategy,
            "memrl_update_strategy": args.memrl_update_strategy,
            "memrl_enable_value_driven": args.memrl_enable_value_driven,
            "reflection_model": args.reflection_model,
            "writer_dataset_path": args.writer_dataset_path,
            "writer_dataset": args.writer_dataset,
            "selector_model": args.selector_model,
            "selector_dataset_path": args.selector_dataset_path,
            "selector_dataset": args.selector_dataset,
            "llm_max_completion_tokens": getattr(args, "llm_max_completion_tokens", None),
        }
        runner = MemoryAugmentedTaskRunner(
            model=args.model,
            env_type=args.env,
            memory_config=memory_config,
            max_steps=args.max_steps,
            step_timeout=args.step_timeout,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_path,
            llm_max_completion_tokens=args.llm_max_completion_tokens,
        )
        logger.info(f"[Simple{args.env.upper()}] Using MemoryAugmentedTaskRunner")
    else:
        runner = SimpleTaskRunner(
            model=args.model,
            env_type=args.env,
            max_steps=args.max_steps,
            step_timeout=args.step_timeout,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_path,
            llm_max_completion_tokens=args.llm_max_completion_tokens,
        )
        logger.info(f"[Simple{args.env.upper()}] Using SimpleTaskRunner")
    try:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            run_task(runner, benchmark, idx, args.env, semaphore)
            for idx in task_indices
        ]
        results_list = await asyncio.gather(*tasks)
        successes = sum(1 for r in results_list if r["success"])
        failures = len(results_list) - successes
        success_rate = (successes / len(results_list) * 100) if results_list else 0
        total_cost = sum(r["cost"] for r in results_list)
        sql_summary = None
        if args.env == "sql":
            sql_summary = compute_sql_success_summary(output_dir)
            if sql_summary is not None:
                successes = sql_summary["success_true"]
                failures = max(sql_summary["counted"] - successes, 0)
                success_rate = sql_summary["success_rate"]
        print("\n" + "=" * 60)
        print(f"SIMPLE {args.env.upper()} BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Environment:  {args.env}")
        print(f"Model:        {args.model}")
        print(f"Memory:       {'Enabled' if args.memory else 'Disabled'}")
        print(f"Tasks:        {len(results_list)}")
        if sql_summary is not None:
            print(f"Counted:      {sql_summary['counted']}")
            print(f"Ignored:      {sql_summary['ignored_corrupt_gold']} corrupt_gold")
        print(f"Successes:    {successes}")
        print(f"Failures:     {failures}")
        print(f"Success Rate: {success_rate:.1f}%")
        if sql_summary is not None:
            if sql_summary["missing_corrupt_lookup"]:
                print(f"Missing corrupt lookup: {sql_summary['missing_corrupt_lookup']}")
            if sql_summary["missing_success"]:
                print(f"Missing success: {sql_summary['missing_success']}")
            if sql_summary["read_errors"]:
                print(f"Read/JSON errors: {sql_summary['read_errors']}")
        print(f"Total Cost:   ${total_cost:.4f}")
        print(f"Output:       {output_dir}")
        print("=" * 60)
        failed_tasks = [r for r in results_list if not r["success"]]
        if sql_summary is not None:
            ignored_task_ids = sql_summary["ignored_task_ids"]
            failed_tasks = [
                r for r in failed_tasks if r["task_id"] not in ignored_task_ids
            ]
        if failed_tasks:
            print("\nFailed tasks:")
            for r in failed_tasks[:10]:
                error_msg = f": {r['error']}" if r['error'] else ""
                print(f"  - {r['task_id']}: reward={r['reward']:.2f}, steps={r['steps']}{error_msg}")
            if len(failed_tasks) > 10:
                print(f"  ... and {len(failed_tasks) - 10} more")
        return 0 if failures == 0 else 1
    except KeyboardInterrupt:
        logger.warning("\n[SimpleBenchmark] Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"[SimpleBenchmark] Fatal error: {e}", exc_info=True)
        return 1
    finally:
        if args.env == "sql" and mysql_container is not None:
            stop_mysql_container(mysql_container, container_name="docker-env-sql_ic_ctr")
        stop_vllm_server(vllm_proc)
if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
