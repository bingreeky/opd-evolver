import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from opd_evolver.base.engine.logs import logger
from opd_evolver.memory.base_store import RetrievedItem
from opd_evolver.memory.embeddings import (
    EmbeddingProvider,
    LocalHFEmbeddingProvider,
    OpenAIEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    local_hf_embedding_settings_from_env,
)
from opd_evolver.memory.skill_memory import SkillMemory, SkillItem
from opd_evolver.memory.tip_memory import TipMemory, TipItem
from opd_evolver.memory.tool_memory import ToolMemory, ToolItem
from opd_evolver.memory.trajectory_memory import TrajectoryMemory, TrajectoryItem, TrajectoryStep
ALL_MEMORY_TIERS: frozenset = frozenset({"skill", "tip", "tool", "trajectory"})
@dataclass
class MemoryConfig:
    storage_dir: str = "~/.opd_evolver/memory"
    cold_start_threshold: int = 20
    retrieval_top_k: int = 3
    min_similarity: float = 0.3
    embedding_provider: str = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_cache: bool = True
    skill_weight: float = 1.0
    tip_weight: float = 0.8
    tool_weight: float = 1.2
    trajectory_weight: float = 0.9
    writer_dataset_path: Optional[str] = None
    selector_dataset_path: Optional[str] = None
    memory_tiers: Optional[Union[str, List[str], Tuple[Any, ...]]] = None
    def enabled_memory_tiers(self) -> frozenset:
        raw = self.memory_tiers
        if raw is None:
            return ALL_MEMORY_TIERS
        out: set[str] = set()
        if isinstance(raw, str):
            iterable = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            try:
                iterable = [str(x).strip() for x in raw]
            except TypeError as e:
                raise TypeError(
                    f"memory_tiers must be None, str, or sequence, got {type(raw)}"
                ) from e
        for x in iterable:
            t = x.lower()
            if t:
                out.add(t)
        if not out:
            raise ValueError(
                "memory_tiers cannot be empty; omit the key to use all tiers"
            )
        unknown = out - ALL_MEMORY_TIERS
        if unknown:
            raise ValueError(
                f"Unknown memory tier(s): {sorted(unknown)}; "
                f"allowed: {sorted(ALL_MEMORY_TIERS)}"
            )
        return frozenset(out)
    def get_storage_path(self, tier: str) -> str:
        base = os.path.expanduser(self.storage_dir)
        return os.path.join(base, f"{tier}_memory.json")
    def get_cache_path(self) -> str:
        base = os.path.expanduser(self.storage_dir)
        return os.path.join(base, "embedding_cache.json")
    def resolved_writer_dataset_path(self) -> Optional[str]:
        if not self.writer_dataset_path:
            return None
        return os.path.expanduser(self.writer_dataset_path)
    def resolved_selector_dataset_path(self) -> Optional[str]:
        if not self.selector_dataset_path:
            return None
        return os.path.expanduser(self.selector_dataset_path)
@dataclass
class RetrievalResult:
    skills: List[RetrievedItem]
    tips: List[RetrievedItem]
    tools: List[RetrievedItem]
    trajectories: List[RetrievedItem]
    def all_items(self) -> List[RetrievedItem]:
        return self.skills + self.tips + self.tools + self.trajectories
    def format_for_context(self) -> str:
        sections = []
        if self.skills:
            sections.append("=== RETRIEVED SKILLS ===")
            for item in self.skills:
                sections.append(item.format_for_context())
        if self.tips:
            sections.append("\n=== RETRIEVED TIPS ===")
            for item in self.tips:
                sections.append(item.format_for_context())
        if self.tools:
            sections.append("\n=== RETRIEVED TOOLS ===")
            for item in self.tools:
                sections.append(item.format_for_context())
        if self.trajectories:
            sections.append("\n=== RETRIEVED TRAJECTORIES ===")
            for item in self.trajectories:
                sections.append(item.format_for_context())
        return "\n".join(sections) if sections else "No relevant memories found."
    def get_item_ids(self) -> List[str]:
        return [item.item.id for item in self.all_items()]
class HierarchicalMemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.config = config or MemoryConfig()
        storage_dir = os.path.expanduser(self.config.storage_dir)
        os.makedirs(storage_dir, exist_ok=True)
        self.embedding_provider = embedding_provider
        if self.embedding_provider is None:
            cache_path = self.config.get_cache_path() if self.config.embedding_cache else None
            provider = (self.config.embedding_provider or "").lower()
            if provider == "local":
                device, max_len, dtype = local_hf_embedding_settings_from_env()
                self.embedding_provider = LocalHFEmbeddingProvider(
                    model_id=self.config.embedding_model,
                    device=device,
                    torch_dtype=dtype,
                    max_length=max_len,
                )
                logger.info(
                    f"Using local HF embedding: {self.config.embedding_model} "
                    f"(device={device}, max_length={max_len}, dtype={dtype})"
                )
            elif provider == "openrouter":
                self.embedding_provider = OpenRouterEmbeddingProvider(
                    model=self.config.embedding_model,
                    cache_path=cache_path,
                )
                logger.info(f"Using OpenRouter embedding: {self.config.embedding_model}")
            else:
                self.embedding_provider = OpenAIEmbeddingProvider(
                    model=self.config.embedding_model,
                    cache_path=cache_path,
                )
                logger.info(f"Using OpenAI embedding: {self.config.embedding_model}")
        self.skill_memory = SkillMemory(
            storage_path=self.config.get_storage_path("skill"),
            embedding_provider=self.embedding_provider,
        )
        self.tip_memory = TipMemory(
            storage_path=self.config.get_storage_path("tip"),
            embedding_provider=self.embedding_provider,
        )
        self.tool_memory = ToolMemory(
            storage_path=self.config.get_storage_path("tool"),
            embedding_provider=self.embedding_provider,
        )
        self.trajectory_memory = TrajectoryMemory(
            storage_path=self.config.get_storage_path("trajectory"),
            embedding_provider=self.embedding_provider,
        )
        self._task_counter = 0
        self._counter_path = os.path.join(storage_dir, "task_counter.txt")
        self._load_task_counter()
        self.config.enabled_memory_tiers()
        logger.info(
            f"HierarchicalMemoryManager initialized: "
            f"skills={self.skill_memory.count()}, "
            f"tips={self.tip_memory.count()}, "
            f"tools={self.tool_memory.count()}, "
            f"trajectories={self.trajectory_memory.count()}, "
            f"task_count={self._task_counter}"
        )
    @property
    def task_count(self) -> int:
        return self._task_counter
    @property
    def is_cold_start(self) -> bool:
        return self._task_counter < self.config.cold_start_threshold
    def increment_task_counter(self) -> int:
        self._task_counter += 1
        self._save_task_counter()
        if self._task_counter == self.config.cold_start_threshold:
            logger.info(
                f"Cold start period complete! "
                f"Memory retrieval now enabled after {self._task_counter} tasks."
            )
        return self._task_counter
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> RetrievalResult:
        top_k = top_k or self.config.retrieval_top_k
        min_sim = min_similarity or self.config.min_similarity
        if self.is_cold_start:
            logger.debug(
                f"Cold start mode: skipping retrieval "
                f"(task {self._task_counter}/{self.config.cold_start_threshold})"
            )
            return RetrievalResult(
                skills=[], tips=[], tools=[], trajectories=[]
            )
        logger.debug(f"Retrieving from memory for query: {query[:100]}...")
        if self.embedding_provider is None:
            return RetrievalResult(skills=[], tips=[], tools=[], trajectories=[])
        query_embedding = await self.embedding_provider.embed(query)
        tiers = self.config.enabled_memory_tiers()
        skills = (
            self.skill_memory.search_by_embedding(query_embedding, top_k, min_sim)
            if "skill" in tiers
            else []
        )
        tips = (
            self.tip_memory.search_by_embedding(query_embedding, top_k, min_sim)
            if "tip" in tiers
            else []
        )
        tools = (
            self.tool_memory.search_by_embedding(query_embedding, top_k, min_sim)
            if "tool" in tiers
            else []
        )
        trajectories = (
            self.trajectory_memory.search_by_embedding(query_embedding, top_k, min_sim)
            if "trajectory" in tiers
            else []
        )
        result = RetrievalResult(
            skills=skills,
            tips=tips,
            tools=tools,
            trajectories=trajectories,
        )
        logger.info(
            f"Retrieved: {len(skills)} skills, {len(tips)} tips, "
            f"{len(tools)} tools, {len(trajectories)} trajectories"
        )
        return result
    async def add_skill(
        self,
        description: str,
        category: str = "",
        technique: str = "",
        preconditions: str = "",
        steps: Optional[List[str]] = None,
        source_task: Optional[str] = None,
    ) -> SkillItem:
        item = await self.skill_memory.add_skill(
            description=description,
            category=category,
            technique=technique,
            preconditions=preconditions,
            steps=steps,
            source_task=source_task,
        )
        logger.debug(f"Added skill: {technique or description[:50]}")
        return item
    async def add_tip(
        self,
        content: str,
        category: str = "",
        severity: str = "info",
        trigger: str = "",
        source_task: Optional[str] = None,
    ) -> TipItem:
        item = await self.tip_memory.add_tip(
            content=content,
            category=category,
            severity=severity,
            trigger=trigger,
            source_task=source_task,
        )
        logger.debug(f"Added tip [{severity}]: {content[:50]}")
        return item
    async def add_tool(
        self,
        name: str,
        description: str,
        code: str,
        language: str = "bash",
        input_description: str = "",
        output_description: str = "",
        source_task: Optional[str] = None,
    ) -> ToolItem:
        item = await self.tool_memory.add_tool(
            name=name,
            description=description,
            code=code,
            language=language,
            input_description=input_description,
            output_description=output_description,
            source_task=source_task,
        )
        logger.debug(f"Added tool: {name}")
        return item
    async def add_trajectory(
        self,
        task_description: str,
        steps: List[TrajectoryStep],
        outcome: str,
        total_reward: float,
        key_learnings: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source_task: Optional[str] = None,
    ) -> TrajectoryItem:
        item = await self.trajectory_memory.add_trajectory(
            task_description=task_description,
            steps=steps,
            outcome=outcome,
            total_reward=total_reward,
            key_learnings=key_learnings,
            tags=tags,
            source_task=source_task,
        )
        logger.debug(f"Added trajectory: {outcome} ({len(steps)} steps)")
        return item
    def mark_item_used(self, item_id: str, tier: str) -> None:
        store = {
            "skill": self.skill_memory,
            "tip": self.tip_memory,
            "tool": self.tool_memory,
            "trajectory": self.trajectory_memory,
        }.get(tier)
        if store:
            store.increment_usage(item_id)
            item = store.get(item_id)
            if item is not None:
                from datetime import datetime as _dt
                item.last_used = _dt.now().isoformat()
                if store.storage_path:
                    store._save_to_storage()
    def mark_item_success(self, item_id: str, tier: str) -> None:
        store = {
            "skill": self.skill_memory,
            "tip": self.tip_memory,
            "tool": self.tool_memory,
            "trajectory": self.trajectory_memory,
        }.get(tier)
        if store:
            store.increment_success(item_id)
    def get_stats(self) -> Dict[str, Any]:
        return {
            "task_count": self._task_counter,
            "is_cold_start": self.is_cold_start,
            "cold_start_threshold": self.config.cold_start_threshold,
            "skills": self.skill_memory.count(),
            "tips": self.tip_memory.count(),
            "tools": self.tool_memory.count(),
            "trajectories": self.trajectory_memory.count(),
            "total_items": (
                self.skill_memory.count() +
                self.tip_memory.count() +
                self.tool_memory.count() +
                self.trajectory_memory.count()
            ),
        }
    def _load_task_counter(self) -> None:
        if os.path.exists(self._counter_path):
            try:
                with open(self._counter_path, "r") as f:
                    self._task_counter = int(f.read().strip())
            except (ValueError, IOError):
                self._task_counter = 0
    def _save_task_counter(self) -> None:
        try:
            with open(self._counter_path, "w") as f:
                f.write(str(self._task_counter))
        except IOError as e:
            logger.warning(f"Failed to save task counter: {e}")
    def reset_cold_start(self) -> None:
        self._task_counter = 0
        self._save_task_counter()
        logger.info("Task counter reset, cold start period restarted")
    def clear_all(self) -> None:
        self.skill_memory.clear()
        self.tip_memory.clear()
        self.tool_memory.clear()
        self.trajectory_memory.clear()
        self.reset_cold_start()
        logger.warning("All memory stores cleared!")
