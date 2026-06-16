from __future__ import annotations
import asyncio
import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional
from openai import OpenAI
from opd_evolver.base.engine.async_llm import LLMConfig, LLMsConfig
from opd_evolver.base.engine.logs import logger
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVOLVELAB_ROOT = PROJECT_ROOT / "reference" / "MemEvolve" / "Flash-Searcher-main"
EVOLVELAB_MEMORY_BACKENDS: tuple[str, ...] = (
    "lightweight_memory",
    "expel",
    "agent_workflow_memory",
    "dynamic_cheatsheet",
    "memp",
    "evolver",
)
OPD_HIERARCHICAL_BACKEND = "opd_hierarchical"
REASONING_BANK_BACKEND = "reasoning_bank"
MEMRL_BACKEND = "memrl"
PROVIDER_MEMORY_BACKENDS: tuple[str, ...] = (
    *EVOLVELAB_MEMORY_BACKENDS,
    REASONING_BANK_BACKEND,
    MEMRL_BACKEND,
)
ALL_MEMORY_BACKENDS: tuple[str, ...] = (OPD_HIERARCHICAL_BACKEND, *PROVIDER_MEMORY_BACKENDS)
class SyncChatModel:
    def __init__(self, config: LLMConfig, max_completion_tokens: Optional[int] = None):
        self.config = config
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(api_key=config.key or "EMPTY", base_url=config.base_url)
    def __call__(self, messages: Any) -> SimpleNamespace:
        normalized = self._normalize_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": normalized,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.max_completion_tokens is not None:
            kwargs["max_tokens"] = self.max_completion_tokens
        response = self.client.chat.completions.create(**kwargs)
        content = ""
        if response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
            if not content and hasattr(msg, "model_dump"):
                dumped = msg.model_dump()
                content = dumped.get("reasoning_content") or dumped.get("reasoning") or ""
        return SimpleNamespace(content=str(content or ""))
    @staticmethod
    def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, dict):
            return [messages]
        if not isinstance(messages, list):
            return [{"role": "user", "content": str(messages)}]
        out: list[dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                out.append({"role": "user", "content": str(msg)})
                continue
            role = str(msg.get("role") or "user")
            content = msg.get("content", "")
            out.append({"role": role, "content": content})
        return out
def is_evolvelab_backend(name: str | None) -> bool:
    return (name or "").strip() in EVOLVELAB_MEMORY_BACKENDS
def is_reasoning_bank_backend(name: str | None) -> bool:
    return (name or "").strip() == REASONING_BANK_BACKEND
def is_memrl_backend(name: str | None) -> bool:
    return (name or "").strip() == MEMRL_BACKEND
def is_provider_backend(name: str | None) -> bool:
    return (name or "").strip() in PROVIDER_MEMORY_BACKENDS
def _ensure_evolvelab_import_path() -> None:
    if not EVOLVELAB_ROOT.exists():
        raise FileNotFoundError(f"MemEvolve reference tree not found: {EVOLVELAB_ROOT}")
    root = str(EVOLVELAB_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
def _safe_model_label(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")
def default_baseline_output_dir(bench: str, model_name: str, method: str, env_or_tasktype: str) -> Path:
    return PROJECT_ROOT / "workspace" / "baselines" / bench / _safe_model_label(model_name) / method / env_or_tasktype
def build_sync_model(model_name: str, max_completion_tokens: Optional[int] = None) -> SyncChatModel:
    return SyncChatModel(LLMsConfig.default().get(model_name), max_completion_tokens=max_completion_tokens)
@dataclass
class EvolveLabMemoryProviderAdapter:
    backend: str
    storage_dir: str | Path
    model_name: str
    max_completion_tokens: Optional[int] = None
    _provider: Any | None = field(default=None, init=False, repr=False)
    _provider_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    def __post_init__(self) -> None:
        if self.backend not in EVOLVELAB_MEMORY_BACKENDS:
            raise ValueError(
                f"Unsupported EvolveLab memory backend: {self.backend!r}. "
                f"Allowed: {', '.join(EVOLVELAB_MEMORY_BACKENDS)}"
            )
        self.storage_dir = str(Path(self.storage_dir).expanduser())
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
    def _ensure_provider(self) -> Any:
        if self._provider is None:
            self._provider = self._new_provider()
        return self._provider
    def _get_provider(self) -> Any:
        if self._provider is not None:
            return self._provider
        with self._provider_lock:
            return self._ensure_provider()
    def _new_provider(self) -> Any:
        _ensure_evolvelab_import_path()
        from EvolveLab.config import get_memory_config
        from EvolveLab.memory_types import MemoryType, PROVIDER_MAPPING
        memory_type = MemoryType(self.backend)
        class_name, module_name = PROVIDER_MAPPING[memory_type]
        module = __import__(f"EvolveLab.providers.{module_name}", fromlist=[class_name])
        provider_class = getattr(module, class_name)
        config = get_memory_config(memory_type)
        config.update(self._storage_overrides(self.backend))
        config["model"] = build_sync_model(self.model_name, self.max_completion_tokens)
        provider = provider_class(config=config)
        if not provider.initialize():
            raise RuntimeError(f"Failed to initialize EvolveLab provider: {self.backend}")
        return provider
    def _storage_overrides(self, backend: str) -> dict[str, Any]:
        base = Path(self.storage_dir)
        provider_dir = base / backend
        provider_dir.mkdir(parents=True, exist_ok=True)
        if backend == "lightweight_memory":
            return {
                "storage_dir": str(provider_dir),
                "longterm_memory_path": str(provider_dir / "longterm_memory.json"),
                "enable_longterm_provision": True,
            }
        if backend == "expel":
            return {
                "insights_file_path": str(provider_dir / "insights.json"),
                "success_trajectories_file_path": str(provider_dir / "success_trajectories.json"),
            }
        if backend == "agent_workflow_memory":
            return {
                "store_path": str(provider_dir / "workflow_memory.json"),
                "index_dir": str(provider_dir / "index"),
            }
        if backend == "dynamic_cheatsheet":
            return {
                "store_path": str(provider_dir),
                "records_file": "dynamic_cheatsheet.json",
                "cheatsheet_file": "global_cheatsheet.txt",
            }
        if backend == "memp":
            return {"store_path": str(provider_dir), "records_file": "procedural_records.json"}
        if backend == "evolver":
            return {"store_path": str(provider_dir), "records_file": "principle_records.json"}
        return {}
    async def provide_begin(self, task_description: str, context: str = "", task_id: str = "") -> str:
        return await asyncio.to_thread(
            self._provide_sync,
            task_description,
            context,
            "BEGIN",
            {"task_id": task_id, "step_number": 0},
        )
    async def provide_in(
        self,
        task_description: str,
        context: str = "",
        task_id: str = "",
        step_number: int = 0,
    ) -> str:
        return await asyncio.to_thread(
            self._provide_sync,
            task_description,
            context,
            "IN",
            {"task_id": task_id, "step_number": step_number},
        )
    async def take_in(
        self,
        *,
        task_description: str,
        trajectory: Iterable[Any],
        success: bool,
        result: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        return await asyncio.to_thread(
            self._take_in_sync,
            task_description,
            list(trajectory),
            success,
            result,
            metadata or {},
        )
    def _provide_sync(
        self,
        task_description: str,
        context: str,
        status_name: str,
        additional_params: dict[str, Any],
    ) -> str:
        _ensure_evolvelab_import_path()
        from EvolveLab.memory_types import MemoryRequest, MemoryStatus
        with self._provider_lock:
            provider = self._ensure_provider()
            status = MemoryStatus.BEGIN if status_name == "BEGIN" else MemoryStatus.IN
            response = provider.provide_memory(
                MemoryRequest(
                    query=task_description,
                    context=context,
                    status=status,
                    additional_params=additional_params,
                )
            )
        return self.format_memory_response(response)
    def _take_in_sync(
        self,
        task_description: str,
        trajectory: list[Any],
        success: bool,
        result: Any,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        _ensure_evolvelab_import_path()
        from EvolveLab.memory_types import TrajectoryData
        with self._provider_lock:
            provider = self._ensure_provider()
            md = {
                "is_correct": bool(success),
                "success": bool(success),
                "task_success": bool(success),
                "full_query": task_description,
                **metadata,
            }
            td = TrajectoryData(
                query=task_description,
                trajectory=[self._trajectory_step_to_dict(i, step) for i, step in enumerate(trajectory, 1)],
                result=result if result is not None else {"success": bool(success)},
                metadata=md,
            )
            return provider.take_in_memory(td)
    @staticmethod
    def format_memory_response(response: Any) -> str:
        memories = list(getattr(response, "memories", []) or [])
        if not memories:
            return ""
        chunks = [f"=== EVOLVELAB MEMORY ({getattr(getattr(response, 'memory_type', ''), 'value', '')}) ==="]
        for idx, item in enumerate(memories, 1):
            content = getattr(item, "content", item)
            if isinstance(content, (dict, list)):
                text = json.dumps(content, ensure_ascii=False)
            else:
                text = str(content)
            if text.strip():
                chunks.append(f"[{idx}] {text.strip()}")
        return "\n".join(chunks).strip()
    @staticmethod
    def _trajectory_step_to_dict(index: int, step: Any) -> dict[str, Any]:
        if isinstance(step, dict):
            observation = step.get("observation") or step.get("observation_before") or ""
            action = step.get("action")
            reward = step.get("reward")
            done = step.get("done")
            after = step.get("observation_after")
            raw_response = step.get("raw_response")
        else:
            observation = getattr(step, "observation", "")
            action = getattr(step, "action", None)
            reward = getattr(step, "reward", None)
            done = getattr(step, "done", None)
            after = getattr(step, "observation_after", None)
            raw_response = getattr(step, "raw_response", None)
        parts = [
            f"Observation: {observation}",
            f"Action: {action}",
            f"Reward: {reward}",
            f"Done: {done}",
        ]
        if after is not None:
            parts.append(f"Next observation: {after}")
        if raw_response:
            parts.append(f"Model response: {raw_response}")
        return {
            "type": "step",
            "step": index,
            "content": "\n".join(parts),
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "observation_after": after,
        }
