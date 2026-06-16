from __future__ import annotations
import asyncio
import inspect
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from opd_evolver.base.engine.async_llm import LLMsConfig
from opd_evolver.base.engine.logs import logger
from opd_evolver.memory.embeddings import (
    LocalHFEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    local_hf_embedding_settings_from_env,
)
MemRLServiceFactory = Callable[[], Any]
def _compact_json(value: Any, max_chars: int = 50000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n... [TRUNCATED] ...\n" + text[-tail:]
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
class _OpenAICompatibleLLM:
    def __init__(self, model_name: str, max_completion_tokens: Optional[int] = None):
        from openai import OpenAI
        self.config = LLMsConfig.default().get(model_name)
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(api_key=self.config.key or "EMPTY", base_url=self.config.base_url)
    def generate(self, messages: Any, **kwargs: Any) -> str:
        if isinstance(messages, str):
            normalized = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            normalized = [messages]
        else:
            normalized = list(messages or [])
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": normalized,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        max_tokens = kwargs.get("max_tokens", self.max_completion_tokens)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**request)
        if not response.choices:
            return ""
        msg = response.choices[0].message
        content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
        if not content and hasattr(msg, "model_dump"):
            dumped = msg.model_dump()
            content = dumped.get("reasoning_content") or dumped.get("reasoning") or ""
        return str(content or "")
    def extract_keywords(self, text: str, max_keywords: int = 8) -> list[str]:
        prompt = (
            "Extract concise search keywords for retrieving procedural memories. "
            f"Return at most {max_keywords} keywords as a JSON list.\n\n{text}"
        )
        raw = self.generate([{"role": "user", "content": prompt}], temperature=0, max_tokens=256)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed[:max_keywords] if str(x).strip()]
        except Exception:
            pass
        tokens = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_-]{2,}", raw or text)
        out: list[str] = []
        for token in tokens:
            if token.lower() not in {x.lower() for x in out}:
                out.append(token)
            if len(out) >= max_keywords:
                break
        return out
    def generate_script(self, trajectory: str) -> str:
        prompt = (
            "Summarize this MiniHack trajectory into 3-5 reusable procedural steps. "
            "Focus on map-reading, item use, doors, rivers, and failure recovery.\n\n"
            f"{trajectory}"
        )
        return self.generate([{"role": "user", "content": prompt}], temperature=0)
class _OPDEmbedder:
    def __init__(self, provider_name: str, model_id: str, max_text_len: int = 4096):
        self.max_text_len = max_text_len
        self.model = model_id
        provider = (provider_name or "local").lower()
        if provider == "local":
            device, max_len, dtype = local_hf_embedding_settings_from_env()
            self._provider = LocalHFEmbeddingProvider(
                model_id=model_id,
                device=device,
                torch_dtype=dtype,
                max_length=max_len,
            )
        elif provider == "openrouter":
            self._provider = OpenRouterEmbeddingProvider(model=model_id)
        else:
            self._provider = OpenAIEmbeddingProvider(model=model_id)
    def embed(self, texts: list[str]) -> list[list[float]]:
        async def _embed_all() -> list[list[float]]:
            return [await self._provider.embed(text) for text in texts]
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("MemRL embedding wrapper must run outside the event loop")
        return asyncio.run(_embed_all())
    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
@dataclass
class MemRLMemoryProviderAdapter:
    storage_dir: str | Path
    model_name: str
    max_completion_tokens: Optional[int] = None
    retrieval_top_k: int = 3
    min_similarity: float = 0.0
    embedding_provider: str = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    read_only: bool = False
    mos_config_path: str | Path | None = None
    build_strategy: str = "proceduralization"
    retrieve_strategy: str = "query"
    update_strategy: str = "adjustment"
    user_id: str = "minihack_memrl"
    enable_value_driven: bool = True
    service_factory: MemRLServiceFactory | None = None
    _service: Any | None = field(default=None, init=False, repr=False)
    _service_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _last_retrieved_ids: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    retrieval_count: int = field(default=0, init=False)
    write_count: int = field(default=0, init=False)
    def __post_init__(self) -> None:
        self.storage_dir = Path(self.storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    def stats(self) -> dict[str, int | str | bool]:
        return {
            "memory_backend": "memrl",
            "memory_retrieval_count": self.retrieval_count,
            "memory_write_count": self.write_count,
            "memory_read_only": self.read_only,
        }
    async def provide_begin(self, task_description: str, context: str = "", task_id: str = "") -> str:
        return await asyncio.to_thread(self._provide_sync, task_description, context, task_id)
    async def provide_in(
        self,
        task_description: str,
        context: str = "",
        task_id: str = "",
        step_number: int = 0,
    ) -> str:
        return await self.provide_begin(task_description, context=context, task_id=task_id)
    async def take_in(
        self,
        *,
        task_description: str,
        trajectory: Iterable[Any],
        success: bool,
        result: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        if self.read_only:
            return False, "MemRL adapter is read-only; skipped memory update"
        return await asyncio.to_thread(
            self._take_in_sync,
            task_description,
            list(trajectory),
            bool(success),
            result,
            metadata or {},
        )
    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service
        with self._service_lock:
            if self._service is None:
                self._service = self._new_service()
            return self._service
    def _new_service(self) -> Any:
        if self.service_factory is not None:
            return self.service_factory()
        try:
            from memrl.service.memory_service import MemoryService
            from memrl.service.strategies import StrategyConfiguration
        except Exception as exc:
            raise RuntimeError(
                "MemRL is not installed or cannot be imported. Install it in the active "
                "environment, for example: pip install git+https://github.com/MemTensor/MemRL.git"
            ) from exc
        mos_config_path = self._resolve_mos_config_path()
        strategy_config = StrategyConfiguration.from_strings(
            self.build_strategy,
            self.retrieve_strategy,
            self.update_strategy,
        )
        llm = _OpenAICompatibleLLM(self.model_name, self.max_completion_tokens)
        embedder = _OPDEmbedder(self.embedding_provider, self.embedding_model)
        old_cwd = Path.cwd()
        try:
            os.chdir(self.storage_dir)
            return MemoryService(
                mos_config_path=str(mos_config_path),
                llm_provider=llm,
                embedding_provider=embedder,
                strategy_config=strategy_config,
                user_id=self.user_id,
                enable_value_driven=self.enable_value_driven,
            )
        finally:
            os.chdir(old_cwd)
    def _resolve_mos_config_path(self) -> Path:
        candidates: list[Path] = []
        if self.mos_config_path:
            candidates.append(Path(self.mos_config_path).expanduser())
        env_path = os.getenv("MEMRL_MOS_CONFIG_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())
        candidates.append(Path("configs/mos_config_final.json"))
        try:
            import memrl
            candidates.append(Path(memrl.__file__).resolve().parents[1] / "configs" / "mos_config_final.json")
        except Exception:
            pass
        for path in candidates:
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(
            "Could not find MemRL mos_config_final.json. Set memrl_mos_config_path in "
            "the YAML config or export MEMRL_MOS_CONFIG_PATH."
        )
    def _provide_sync(self, task_description: str, context: str, task_id: str) -> str:
        with self._service_lock:
            service = self._get_service()
            query = f"{task_description}\n\n{context}".strip() if context else task_description
            if hasattr(service, "retrieve_value_aware"):
                response = service.retrieve_value_aware(
                    query,
                    k=self.retrieval_top_k,
                    threshold=self.min_similarity,
                )
            else:
                response = service.retrieve(query, k=self.retrieval_top_k, threshold=self.min_similarity)
        formatted, memory_ids = self.format_retrieval_response(response)
        key = task_id or task_description
        self._last_retrieved_ids[key] = memory_ids
        if memory_ids:
            self.retrieval_count += len(memory_ids)
        return formatted
    def _take_in_sync(
        self,
        task_description: str,
        trajectory: list[Any],
        success: bool,
        result: Any,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        with self._service_lock:
            service = self._get_service()
            task_id = str(metadata.get("task_id") or task_description)
            retrieved_ids = list(self._last_retrieved_ids.get(task_id) or [])
            trajectory_text = self._format_trajectory(trajectory)
            meta = {
                "source_benchmark": "minihack",
                "success": success,
                "task_success": success,
                **metadata,
            }
            updated_q = {}
            if retrieved_ids:
                if hasattr(service, "update_values"):
                    updated_q = service.update_values([1.0 if success else 0.0], [retrieved_ids]) or {}
                elif hasattr(service, "update_value"):
                    reward = 1.0 if success else -1.0
                    updated_q = {mem_id: service.update_value(mem_id, reward) for mem_id in retrieved_ids}
            memory_id = None
            if hasattr(service, "update_memory"):
                memory_id = self._call_service_method(
                    service.update_memory,
                    task_description=task_description,
                    trajectory=trajectory_text,
                    success=success,
                    retrieved_memory_ids=retrieved_ids,
                    metadata=meta,
                )
            if memory_id is None and hasattr(service, "build_memory"):
                memory_id = self._call_service_method(
                    service.build_memory,
                    task_description=task_description,
                    trajectory=trajectory_text,
                    metadata=meta,
                )
        if memory_id is not None or updated_q:
            self.write_count += 1
            return True, f"MemRL updated memory_id={memory_id!s} q_updates={len(updated_q)}"
        return False, "MemRL produced no memory update"
    @staticmethod
    def _call_service_method(method: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(**kwargs)
        params = signature.parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_kwargs:
            return method(**kwargs)
        filtered = {key: value for key, value in kwargs.items() if key in params}
        return method(**filtered)
    @staticmethod
    def format_retrieval_response(response: Any) -> tuple[str, list[str]]:
        if response is None:
            return "", []
        selected: list[Any]
        if isinstance(response, dict):
            selected_obj = response.get("selected")
            candidates = response.get("candidates") or []
            selected = [selected_obj] if selected_obj else list(candidates)
        elif isinstance(response, (list, tuple)):
            selected = list(response)
        else:
            selected = [response]
        chunks = ["=== MEMRL MEMORY ==="]
        memory_ids: list[str] = []
        for idx, item in enumerate(selected, 1):
            if not item:
                continue
            mem_id = MemRLMemoryProviderAdapter._field(item, "memory_id", "mem_id", "id")
            content = MemRLMemoryProviderAdapter._memory_text(item)
            similarity = MemRLMemoryProviderAdapter._field(item, "similarity", "score", "combined_score")
            q_value = MemRLMemoryProviderAdapter._field(item, "q_value", "q")
            if mem_id:
                memory_ids.append(str(mem_id))
            if content:
                suffix = []
                if similarity is not None:
                    suffix.append(f"sim={_safe_float(similarity):.3f}")
                if q_value is not None:
                    suffix.append(f"q={_safe_float(q_value):.3f}")
                label = f"[{idx}]"
                if suffix:
                    label += " " + " ".join(suffix)
                chunks.append(f"{label}\n{content}")
        if len(chunks) == 1:
            return "", []
        return "\n".join(chunks).strip(), memory_ids
    @staticmethod
    def _field(item: Any, *names: str) -> Any:
        if isinstance(item, dict):
            for name in names:
                if name in item:
                    return item[name]
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                for name in names:
                    if name in metadata:
                        return metadata[name]
            return None
        for name in names:
            if hasattr(item, name):
                return getattr(item, name)
        return None
    @staticmethod
    def _memory_text(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        for name in ("full_content", "content", "memory", "text", "script", "trajectory"):
            value = MemRLMemoryProviderAdapter._field(item, name)
            if value:
                return str(value).strip()
        metadata = MemRLMemoryProviderAdapter._field(item, "metadata")
        if isinstance(metadata, dict):
            value = metadata.get("full_content") or metadata.get("content")
            if value:
                return str(value).strip()
        return ""
    @staticmethod
    def _format_trajectory(trajectory: list[Any]) -> str:
        steps = []
        for idx, step in enumerate(trajectory, 1):
            if isinstance(step, dict):
                payload = dict(step)
            else:
                payload = {
                    "observation": getattr(step, "observation", None),
                    "action": getattr(step, "action", None),
                    "reward": getattr(step, "reward", None),
                    "done": getattr(step, "done", None),
                    "info": getattr(step, "info", None),
                    "observation_after": getattr(step, "observation_after", None),
                    "raw_response": getattr(step, "raw_response", None),
                }
            payload["step"] = idx
            steps.append(payload)
        return _compact_json(steps)
