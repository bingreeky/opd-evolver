from __future__ import annotations
import asyncio
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from opd_evolver.base.engine.logs import logger
from opd_evolver.memory.evolvelab_adapter import build_sync_model
_FILE_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
        return lock
def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
def _lexical_score(query: str, item: dict[str, Any]) -> float:
    q = _tokens(query)
    c = _tokens(str(item.get("content", "")) + "\n" + str(item.get("task_description", "")))
    if not q or not c:
        return 0.0
    return len(q & c) / max(1, len(q | c))
def _compact_json(value: Any, max_chars: int = 20000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.6)] + "\n... [TRUNCATED] ...\n" + text[-int(max_chars * 0.4) :]
REASONING_BANK_INDUCTION_PROMPT = """You maintain a reusable ReasoningBank for agents.

Given one completed task trajectory, induce concise memories that would help future
agents solve similar tasks. Focus on transferable reasoning procedures, pitfalls,
verification checks, tool-use habits, and task-specific heuristics. Avoid copying
large observations or one-off facts that only identify this exact task.

Return JSON only:
{{
  "memories": [
    "Reusable lesson 1",
    "Reusable lesson 2"
  ]
}}

## Task
{task_description}

## Outcome
success: {success}
result: {result}
metadata: {metadata}

## Trajectory
{trajectory}
"""
@dataclass
class ReasoningBankMemoryProviderAdapter:
    storage_dir: str | Path
    model_name: str
    max_completion_tokens: Optional[int] = None
    retrieval_top_k: int = 3
    min_similarity: float = 0.0
    embedding_provider: str = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    def __post_init__(self) -> None:
        self.storage_dir = Path(self.storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_dir / "reasoning_bank.jsonl"
        self.embedding_cache_path = self.storage_dir / "embedding_cache.json"
        self._embedding_provider: Any | None = None
        self._embedding_failed = False
    async def provide_begin(self, task_description: str, context: str = "", task_id: str = "") -> str:
        return await self._provide(task_description, context, task_id=task_id, step_number=0)
    async def provide_in(
        self,
        task_description: str,
        context: str = "",
        task_id: str = "",
        step_number: int = 0,
    ) -> str:
        return await self._provide(task_description, context, task_id=task_id, step_number=step_number)
    async def take_in(
        self,
        *,
        task_description: str,
        trajectory: Iterable[Any],
        success: bool,
        result: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        trace = [self._trajectory_step_to_dict(i, step) for i, step in enumerate(trajectory, 1)]
        prompt = REASONING_BANK_INDUCTION_PROMPT.format(
            task_description=task_description,
            success=bool(success),
            result=_compact_json(result, 8000),
            metadata=_compact_json(metadata or {}, 8000),
            trajectory=_compact_json(trace, 50000),
        )
        try:
            response = await asyncio.to_thread(self._call_model, prompt)
            memories = self._parse_memories(response)
            if not memories:
                memories = [self._fallback_memory(task_description, success, result)]
            records = []
            for content in memories:
                content = str(content).strip()
                if not content:
                    continue
                embedding = await self._embed_or_none(content)
                records.append(
                    {
                        "id": self._make_id(task_description, content),
                        "content": content,
                        "task_description": task_description,
                        "trajectory": trace,
                        "result": result if result is not None else {"success": bool(success)},
                        "metadata": {
                            "success": bool(success),
                            "is_correct": bool(success),
                            "task_success": bool(success),
                            **(metadata or {}),
                        },
                        "embedding": embedding,
                        "created_at": time.time(),
                        "updated_at": time.time(),
                        "use_count": 0,
                        "success_count": 1 if success else 0,
                    }
                )
            if not records:
                return False, "ReasoningBank produced no memories"
            await asyncio.to_thread(self._append_records, records)
            return True, f"stored {len(records)} reasoning_bank memories"
        except Exception as exc:
            logger.warning(f"[ReasoningBank] take_in failed: {exc}", exc_info=True)
            return False, str(exc)
    async def _provide(self, task_description: str, context: str, *, task_id: str, step_number: int) -> str:
        items = await asyncio.to_thread(self._load_records)
        if not items:
            return ""
        query = f"{task_description}\n{context}".strip()
        query_embedding = await self._embed_or_none(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            score = 0.0
            if query_embedding is not None and item.get("embedding"):
                score = _cosine(query_embedding, item.get("embedding") or [])
            if score <= 0.0:
                score = _lexical_score(query, item)
            if score >= self.min_similarity:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = scored[: max(1, int(self.retrieval_top_k))]
        if not selected:
            return ""
        await asyncio.to_thread(self._mark_used, [item["id"] for _, item in selected])
        chunks = ["=== REASONING BANK MEMORY ==="]
        for idx, (score, item) in enumerate(selected, 1):
            chunks.append(f"[{idx}] score={score:.3f}\n{str(item.get('content', '')).strip()}")
        return "\n\n".join(chunks).strip()
    def _call_model(self, prompt: str) -> str:
        model = build_sync_model(self.model_name, self.max_completion_tokens)
        response = model([{"role": "user", "content": prompt}])
        return str(getattr(response, "content", "") or "")
    @staticmethod
    def _parse_memories(response: str) -> list[str]:
        parsed = _safe_json_loads(response)
        if isinstance(parsed, dict):
            raw = parsed.get("memories") or parsed.get("memory") or parsed.get("lessons")
            if isinstance(raw, list):
                return [str(item).strip() for item in raw if str(item).strip()]
            if isinstance(raw, str) and raw.strip():
                return [raw.strip()]
        lines = []
        for line in (response or "").splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if line and not line.startswith("{") and not line.startswith("}"):
                lines.append(line)
        return lines[:5]
    @staticmethod
    def _fallback_memory(task_description: str, success: bool, result: Any) -> str:
        outcome = "succeeded" if success else "failed"
        return (
            f"Task {outcome}. For similar tasks, restate the objective, use observations "
            f"to verify each action, and check the final answer against the task constraints. "
            f"Task summary: {task_description[:500]}"
        )
    def _load_records(self) -> list[dict[str, Any]]:
        if not self.memory_file.exists():
            return []
        records = []
        with _lock_for(self.memory_file):
            with self.memory_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict) and item.get("id") and item.get("content"):
                        records.append(item)
        return records
    def _append_records(self, records: list[dict[str, Any]]) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = {item.get("id") for item in self._load_records()}
        with _lock_for(self.memory_file):
            with self.memory_file.open("a", encoding="utf-8") as f:
                for record in records:
                    if record["id"] in existing_ids:
                        continue
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    def _mark_used(self, ids: list[str]) -> None:
        if not ids or not self.memory_file.exists():
            return
        id_set = set(ids)
        records = self._load_records()
        changed = False
        now = time.time()
        for record in records:
            if record.get("id") in id_set:
                record["use_count"] = int(record.get("use_count") or 0) + 1
                record["updated_at"] = now
                changed = True
        if not changed:
            return
        with _lock_for(self.memory_file):
            with self.memory_file.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    async def _embed_or_none(self, text: str) -> list[float] | None:
        provider = self._get_embedding_provider()
        if provider is None:
            return None
        try:
            return await provider.embed(text)
        except Exception as exc:
            self._embedding_failed = True
            logger.warning(f"[ReasoningBank] embedding failed; lexical retrieval fallback active: {exc}")
            return None
    def _get_embedding_provider(self) -> Any | None:
        if self._embedding_failed:
            return None
        if self._embedding_provider is not None:
            return self._embedding_provider
        try:
            provider_name = (self.embedding_provider or "local").lower()
            if provider_name in {"none", "disabled", "off", "lexical"}:
                self._embedding_failed = True
                return None
            if provider_name == "openrouter":
                from opd_evolver.memory.embeddings import OpenRouterEmbeddingProvider
                self._embedding_provider = OpenRouterEmbeddingProvider(
                    model=self.embedding_model,
                    cache_path=str(self.embedding_cache_path),
                )
            elif provider_name == "openai":
                from opd_evolver.memory.embeddings import OpenAIEmbeddingProvider
                self._embedding_provider = OpenAIEmbeddingProvider(
                    model=self.embedding_model,
                    cache_path=str(self.embedding_cache_path),
                )
            else:
                from opd_evolver.memory.embeddings import LocalHFEmbeddingProvider, local_hf_embedding_settings_from_env
                device, max_len, dtype = local_hf_embedding_settings_from_env()
                self._embedding_provider = LocalHFEmbeddingProvider(
                    model_id=self.embedding_model,
                    device=device,
                    max_length=max_len,
                    dtype=dtype,
                )
        except Exception as exc:
            self._embedding_failed = True
            logger.warning(f"[ReasoningBank] embedding provider unavailable; using lexical fallback: {exc}")
            return None
        return self._embedding_provider
    @staticmethod
    def _make_id(task_description: str, content: str) -> str:
        digest = hashlib.sha256(f"{task_description}\n{content}".encode("utf-8")).hexdigest()[:16]
        return f"rb_{digest}"
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
        return {
            "type": "step",
            "step": index,
            "content": _compact_json(
                {
                    "observation": observation,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "observation_after": after,
                    "raw_response": raw_response,
                },
                12000,
            ),
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "observation_after": after,
            "raw_response": raw_response,
        }
