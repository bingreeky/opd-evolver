import asyncio
import traceback
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from opd_evolver.base.engine.async_llm import AsyncLLM
from opd_evolver.base.engine.logs import logger
from opd_evolver.memory.memory_manager import (
    HierarchicalMemoryManager,
    MemoryConfig,
    RetrievalResult,
)
from opd_evolver.memory.selector_dataset import MemorySelectorDatasetLogger
from opd_evolver.memory.trajectory_memory import TrajectoryStep
from opd_evolver.memory.usage_log import UsageLogEntry, UsageLogger
from opd_evolver.memory.writer_dataset import MemoryWriterDatasetLogger
from opd_evolver.pipelines.types import FilteredContext, ReflectionResult
@dataclass
class _TaskMemoryCtx:
    task_type: str
    filtered: FilteredContext
    selector_prefill: Optional[Dict[str, Any]] = None
class MemoryAugmentedPipeline:
    def __init__(
        self,
        llm: AsyncLLM,
        memory_manager: Optional[HierarchicalMemoryManager] = None,
        config: Optional[MemoryConfig] = None,
        enabled: bool = True,
        memory_writer_llm: Optional[AsyncLLM] = None,
        memory_selector_llm: Optional[AsyncLLM] = None,
    ):
        self.llm = llm
        self._memory_writer_llm = memory_writer_llm
        self._memory_selector_llm = memory_selector_llm
        self.config = config or MemoryConfig()
        self.enabled = enabled
        self.memory_manager = memory_manager
        if self.memory_manager is None and enabled:
            self.memory_manager = HierarchicalMemoryManager(config=self.config)
        import os
        storage_dir = os.path.expanduser(self.config.storage_dir)
        self.usage_logger = UsageLogger(
            storage_path=os.path.join(storage_dir, "usage_logs.jsonl")
        )
        wpath = self.config.resolved_writer_dataset_path()
        self._writer_dataset_logger: Optional[MemoryWriterDatasetLogger] = (
            MemoryWriterDatasetLogger(wpath) if wpath else None
        )
        spath = self.config.resolved_selector_dataset_path()
        self._selector_dataset_logger: Optional[MemorySelectorDatasetLogger] = (
            MemorySelectorDatasetLogger(spath) if spath else None
        )
        self._task_context: Dict[str, _TaskMemoryCtx] = {}
        self._state_lock = asyncio.Lock()
    async def pre_execution(
        self,
        task_id: str,
        task_description: str,
        additional_context: str = "",
        task_type: str = "",
    ) -> Optional[FilteredContext]:
        if not self.enabled or self.memory_manager is None:
            return None
        stats = self.memory_manager.get_stats()
        logger.info(
            f"[MemoryPipeline] Pre-execution: task={task_id}, "
            f"cold_start={stats['is_cold_start']}, "
            f"task_count={stats['task_count']}/{stats['cold_start_threshold']}"
        )
        async def _write_started_log(
            cands: Dict[str, List[str]],
            sel: Dict[str, List[str]],
        ) -> None:
            try:
                entry = UsageLogEntry(
                    task_id=task_id,
                    task_type=task_type or "",
                    retrieved_candidates=cands,
                    selected_memory_ids=sel,
                    env_reward=0.0,
                    success=False,
                )
                async with self._state_lock:
                    self.usage_logger.log_started(entry)
                logger.debug(f"[MemoryPipeline] Pre-execution log written: {entry.to_dict()}")
            except Exception as exc:
                logger.warning(
                    f"[MemoryPipeline] Failed to write pre_execution log: {exc}\n"
                    + traceback.format_exc()
                )
        if self.memory_manager.is_cold_start:
            logger.info("[MemoryPipeline] Cold start mode: skipping retrieval")
            await _write_started_log(cands={}, sel={})
            return None
        query = f"{task_description}\n{additional_context}".strip()
        candidates: Dict[str, List[str]] = {}
        _started_written = False
        try:
            retrieval = await self.memory_manager.retrieve(query)
            for item in retrieval.skills:
                candidates.setdefault("skill", []).append(item.item.id)
            for item in retrieval.tips:
                candidates.setdefault("tip", []).append(item.item.id)
            for item in retrieval.tools:
                candidates.setdefault("tool", []).append(item.item.id)
            for item in retrieval.trajectories:
                candidates.setdefault("trajectory", []).append(item.item.id)
            total_retrieved = len(retrieval.all_items())
            if total_retrieved == 0:
                logger.info("[MemoryPipeline] No relevant memories found")
                await _write_started_log(cands=candidates, sel={})
                _started_written = True
                return None
            logger.info(f"[MemoryPipeline] Retrieved {total_retrieved} items, filtering...")
            filtered, raw_selector, candidates_context = await self._filter_retrieved(
                task_description=task_description,
                retrieval=retrieval,
            )
            selected_ids = filtered.get_all_selected_ids()
            async with self._state_lock:
                for tier, ids in selected_ids.items():
                    for item_id in ids:
                        self.memory_manager.mark_item_used(item_id, tier)
            selected_count = sum(
                len(ids) for ids in selected_ids.values()
            )
            logger.info(
                f"[MemoryPipeline] Filtered to {selected_count} items. "
                f"Reasoning: {filtered.reasoning[:100]}..."
            )
            await _write_started_log(cands=candidates, sel=selected_ids)
            _started_written = True
            selector_prefill: Optional[Dict[str, Any]] = None
            if self._selector_dataset_logger is not None:
                selector_prefill = {
                    "retrieve": {
                        "task_description": task_description,
                        "candidates": {k: list(v) for k, v in candidates.items()},
                        "candidates_context": candidates_context,
                    },
                    "select": {
                        "raw": raw_selector,
                        "selected_memory_ids": dict(selected_ids),
                        "reasoning": filtered.reasoning,
                    },
                    "selector_sample_id": str(uuid.uuid4()),
                }
            async with self._state_lock:
                self._task_context[task_id] = _TaskMemoryCtx(
                    task_type=task_type or "",
                    filtered=filtered,
                    selector_prefill=selector_prefill,
                )
            return filtered
        except Exception:
            logger.warning(
                f"[MemoryPipeline] Retrieval/filtering failed for task={task_id}:\n"
                + traceback.format_exc()
            )
            raise
        finally:
            if not _started_written:
                await _write_started_log(cands=candidates, sel={})
    async def post_execution(
        self,
        task_id: str,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        success: bool,
        total_reward: float,
        tags: Optional[List[str]] = None,
        task_type: str = "",
    ) -> ReflectionResult:
        if not self.enabled or self.memory_manager is None:
            return ReflectionResult(
                new_skills=[], new_tips=[], new_tools=[],
                key_learnings=[], should_save_trajectory=False,
                trajectory_outcome="unknown"
            )
        async with self._state_lock:
            ctx_snapshot = self._task_context.get(task_id)
        effective_task_type = task_type
        filtered_context_summary: Optional[str] = None
        if ctx_snapshot is not None:
            effective_task_type = (
                (ctx_snapshot.task_type or task_type or "").strip() or task_type
            )
            _filtered = ctx_snapshot.filtered
            if _filtered and _filtered.formatted_context:
                fc = _filtered.formatted_context
                filtered_context_summary = fc if len(fc) <= 8000 else (fc[:8000] + "\n... [truncated] ...")
        logger.info(
            f"[MemoryPipeline] Post-execution: task={task_id}, "
            f"success={success}, reward={total_reward}, steps={len(execution_trace)}"
        )
        reflection, raw_llm_response = await self._reflect_on_execution(
            task_description=task_description,
            execution_trace=execution_trace,
            success=success,
            total_reward=total_reward,
        )
        async with self._state_lock:
            created_pairs = await self._persist_learnings(
                task_id=task_id,
                task_description=task_description,
                reflection=reflection,
                execution_trace=execution_trace,
                total_reward=total_reward,
                tags=tags,
            )
            if self._writer_dataset_logger is not None:
                logger.info(f"[GenDataset] Appending writer dataset record for task={task_id}")
                self._append_writer_dataset_record(
                    task_id=task_id,
                    task_description=task_description,
                    execution_trace=execution_trace,
                    success=success,
                    total_reward=total_reward,
                    task_type=effective_task_type,
                    tags=tags,
                    filtered_context_summary=filtered_context_summary,
                    reflection=reflection,
                    raw_llm_response=raw_llm_response,
                    created_pairs=created_pairs,
                )
            ctx = self._task_context.pop(task_id, None)
            if ctx is not None:
                if (
                    self._selector_dataset_logger is not None
                    and ctx.selector_prefill is not None
                ):
                    logger.info(f"[GenDataset] Appending selector dataset record for task={task_id}")
                    self._append_selector_dataset_record(
                        task_id=task_id,
                        success=success,
                        total_reward=total_reward,
                        selector_prefill=ctx.selector_prefill,
                    )
                if success:
                    for tier, ids in ctx.filtered.get_all_selected_ids().items():
                        for item_id in ids:
                            self.memory_manager.mark_item_success(item_id, tier)
            self.memory_manager.increment_task_counter()
            try:
                self.usage_logger.log_outcome(
                    task_id=task_id,
                    env_reward=total_reward,
                    success=success,
                )
            except Exception as _log_exc:
                logger.warning(
                    f"[MemoryPipeline] Failed to write outcome log: {_log_exc}"
                )
        return reflection
    async def _filter_retrieved(
        self,
        task_description: str,
        retrieval: RetrievalResult,
    ) -> Tuple[FilteredContext, Optional[str], str]:
        from opd_evolver.pipelines.memory_selector_prompts import (
            SELECTOR_PROMPT,
            parse_selector_response,
        )
        retrieved_context = retrieval.format_for_context()
        MAX_FILTER_CONTEXT_CHARS = 20000
        if len(retrieved_context) > MAX_FILTER_CONTEXT_CHARS:
            retrieved_context = (
                retrieved_context[:MAX_FILTER_CONTEXT_CHARS]
                + f"\n\n... [TRUNCATED: {len(retrieved_context) - MAX_FILTER_CONTEXT_CHARS} chars omitted] ...\n"
            )
        prompt = SELECTOR_PROMPT.format(
            task_description=task_description,
            retrieved_context=retrieved_context,
        )
        select_llm = self._memory_selector_llm or self.llm
        try:
            response = await select_llm(prompt)
            if not response or "{" not in response:
                raise ValueError("No JSON found in selector response")
            parsed = parse_selector_response(response, retrieval)
            return parsed, response, retrieved_context
        except Exception as e:
            err_name = type(e).__name__
            logger.warning(f"[MemoryPipeline] Selector failed ({err_name}): {e}; retrying once")
            try:
                await asyncio.sleep(0)
                response = await select_llm(prompt)
                if not response or "{" not in response:
                    raise ValueError("No JSON found in selector response")
                parsed = parse_selector_response(response, retrieval)
                return parsed, response, retrieved_context
            except Exception as e2:
                err2_name = type(e2).__name__
                logger.error(f"[MemoryPipeline] Selector failed after retry ({err2_name}): {e2}")
            fb = FilteredContext(
                selected_skill_ids=[i.item.id for i in retrieval.skills[:2]],
                selected_tip_ids=[i.item.id for i in retrieval.tips[:2]],
                selected_tool_ids=[i.item.id for i in retrieval.tools[:1]],
                selected_trajectory_ids=[i.item.id for i in retrieval.trajectories[:1]],
                formatted_context=retrieved_context,
                reasoning="Fallback: using top items from each tier",
            )
            return fb, None, retrieved_context
    async def _reflect_on_execution(
        self,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        success: bool,
        total_reward: float,
    ) -> Tuple[ReflectionResult, Optional[str]]:
        from opd_evolver.pipelines.memory_prompts import REFLECTION_PROMPT, parse_reflection_response
        trace_text = self._format_trace_for_reflection(execution_trace)
        MAX_TRACE_CHARS = 50000
        if len(trace_text) > MAX_TRACE_CHARS:
            logger.warning(
                f"[MemoryPipeline] Trace too long ({len(trace_text)} chars), "
                f"truncating to {MAX_TRACE_CHARS} chars"
            )
            first_part = int(MAX_TRACE_CHARS * 0.6)
            last_part = MAX_TRACE_CHARS - first_part
            trace_text = (
                trace_text[:first_part] +
                f"\n\n... [TRUNCATED: {len(trace_text) - MAX_TRACE_CHARS} chars omitted] ...\n\n" +
                trace_text[-last_part:]
            )
        outcome = "SUCCESS" if success else ("PARTIAL" if total_reward > 0 else "FAILURE")
        prompt = REFLECTION_PROMPT.format(
            task_description=task_description,
            execution_trace=trace_text,
            outcome=outcome,
            total_reward=total_reward,
        )
        reflect_llm = self._memory_writer_llm or self.llm
        try:
            response = await reflect_llm(prompt)
            if not response:
                raise ValueError("Empty LLM response")
            try:
                parsed = parse_reflection_response(response, outcome)
                return parsed, response
            except Exception as pe:
                logger.error(f"[MemoryPipeline] Reflection parse failed: {pe}")
                return (
                    ReflectionResult(
                        new_skills=[],
                        new_tips=[],
                        new_tools=[],
                        key_learnings=[f"Task {'succeeded' if success else 'failed'}"],
                        should_save_trajectory=success,
                        trajectory_outcome=outcome.lower(),
                    ),
                    response,
                )
        except Exception as e:
            logger.error(f"[MemoryPipeline] Reflection failed: {e}")
            return (
                ReflectionResult(
                    new_skills=[],
                    new_tips=[],
                    new_tools=[],
                    key_learnings=[f"Task {'succeeded' if success else 'failed'}"],
                    should_save_trajectory=success,
                    trajectory_outcome=outcome.lower(),
                ),
                None,
            )
    async def _persist_learnings(
        self,
        task_id: str,
        task_description: str,
        reflection: ReflectionResult,
        execution_trace: List[Dict[str, Any]],
        total_reward: float,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        created: List[Tuple[str, str]] = []
        if self.memory_manager is None:
            return created
        tiers = self.config.enabled_memory_tiers()
        if "skill" in tiers:
            for skill in reflection.new_skills:
                try:
                    item = await self.memory_manager.add_skill(
                        description=skill.get("description", ""),
                        category=skill.get("category", ""),
                        technique=skill.get("technique", ""),
                        preconditions=skill.get("preconditions", ""),
                        steps=skill.get("steps", []),
                        source_task=task_id,
                    )
                    created.append(("skill", item.id))
                except Exception as e:
                    logger.warning(f"Failed to add skill: {e}")
        if "tip" in tiers:
            for tip in reflection.new_tips:
                try:
                    item = await self.memory_manager.add_tip(
                        content=tip.get("content", ""),
                        category=tip.get("category", ""),
                        severity=tip.get("severity", "info"),
                        trigger=tip.get("trigger", ""),
                        source_task=task_id,
                    )
                    created.append(("tip", item.id))
                except Exception as e:
                    logger.warning(f"Failed to add tip: {e}")
        if "tool" in tiers:
            for tool in reflection.new_tools:
                try:
                    item = await self.memory_manager.add_tool(
                        name=tool.get("name", "unnamed_tool"),
                        description=tool.get("description", ""),
                        code=tool.get("code", ""),
                        language=tool.get("language", "bash"),
                        input_description=tool.get("input_description", ""),
                        output_description=tool.get("output_description", ""),
                        source_task=task_id,
                    )
                    created.append(("tool", item.id))
                except Exception as e:
                    logger.warning(f"Failed to add tool: {e}")
        if "trajectory" in tiers and reflection.should_save_trajectory:
            try:
                steps = self._convert_trace_to_steps(execution_trace)
                item = await self.memory_manager.add_trajectory(
                    task_description=task_description,
                    steps=steps,
                    outcome=reflection.trajectory_outcome,
                    total_reward=total_reward,
                    key_learnings=reflection.key_learnings,
                    tags=tags,
                    source_task=task_id,
                )
                created.append(("trajectory", item.id))
            except Exception as e:
                logger.warning(f"Failed to add trajectory: {e}")
        cnt = Counter(t for t, _ in created)
        logger.info(
            f"[MemoryPipeline] Persisted: "
            f"{cnt['skill']} skills, "
            f"{cnt['tip']} tips, "
            f"{cnt['tool']} tools, "
            f"trajectory={'yes' if cnt['trajectory'] else 'no'}"
        )
        return created
    @staticmethod
    def _nest_created_memory_ids(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for tier, mid in pairs:
            out.setdefault(tier, []).append(mid)
        return out
    def _append_writer_dataset_record(
        self,
        task_id: str,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        success: bool,
        total_reward: float,
        task_type: str,
        tags: Optional[List[str]],
        filtered_context_summary: Optional[str],
        reflection: ReflectionResult,
        raw_llm_response: Optional[str],
        created_pairs: List[Tuple[str, str]],
    ) -> None:
        if self._writer_dataset_logger is None:
            return
        tags_list = list(tags) if tags else []
        record = {
            "task_id": task_id,
            "input": {
                "task_description": task_description,
                "success": success,
                "total_reward": total_reward,
                "task_type": task_type or "",
                "tags": tags_list,
                "execution_trace": execution_trace,
                "filtered_context_summary": filtered_context_summary,
            },
            "output_memory": raw_llm_response,
            "parsed_reflection": asdict(reflection),
            "created_memory_ids": self._nest_created_memory_ids(created_pairs),
            "score": None,
        }
        try:
            self._writer_dataset_logger.append(record)
        except Exception as exc:
            logger.warning(f"[MemoryPipeline] Failed to append writer dataset row: {exc}")
    def _append_selector_dataset_record(
        self,
        task_id: str,
        success: bool,
        total_reward: float,
        selector_prefill: Dict[str, Any],
    ) -> None:
        if self._selector_dataset_logger is None:
            return
        record = {
            "task_id": task_id,
            "retrieve": selector_prefill["retrieve"],
            "select": selector_prefill["select"],
            "selector_sample_id": selector_prefill.get("selector_sample_id"),
            "success": success,
            "total_reward": total_reward,
            "mean_selected_memory_score": None,
            "score": None,
        }
        try:
            self._selector_dataset_logger.append(record)
        except Exception as exc:
            logger.warning(f"[MemoryPipeline] Failed to append selector dataset row: {exc}")
    def _format_trace_for_reflection(
        self,
        trace: List[Dict[str, Any]],
        max_steps: int = 10,
    ) -> str:
        lines = []
        if len(trace) > max_steps:
            show_trace = trace[:4] + [{"_marker": "..."}] + trace[-4:]
        else:
            show_trace = trace
        for i, step in enumerate(show_trace):
            if step.get("_marker"):
                lines.append(f"\n... ({len(trace) - 8} steps omitted) ...\n")
                continue
            action = step.get("action", {})
            action_name = action.get("action", "unknown") if isinstance(action, dict) else str(action)
            params = action.get("params", {}) if isinstance(action, dict) else {}
            command = params.get("command", "")
            obs = step.get("observation", "")
            if isinstance(obs, dict):
                obs = obs.get("output", str(obs))
            obs_str = str(obs)
            obs_display = obs_str[:150] + "..." if len(obs_str) > 150 else obs_str
            reward = step.get("reward", 0)
            lines.append(f"Step {i + 1}:")
            lines.append(f"  Action: {action_name}")
            if command:
                cmd_display = command[:80] + "..." if len(command) > 80 else command
                lines.append(f"  Command: {cmd_display}")
            lines.append(f"  Result: {obs_display}")
            lines.append(f"  Reward: {reward}")
            lines.append("")
        return "\n".join(lines)
    def _convert_trace_to_steps(
        self,
        trace: List[Dict[str, Any]],
    ) -> List[TrajectoryStep]:
        steps = []
        for i, step in enumerate(trace):
            action = step.get("action", {})
            action_name = action.get("action", "unknown") if isinstance(action, dict) else str(action)
            params = action.get("params", {}) if isinstance(action, dict) else {}
            obs = step.get("observation", "")
            if isinstance(obs, dict):
                obs = obs.get("output", str(obs))
            steps.append(TrajectoryStep(
                step_num=i + 1,
                observation=str(obs)[:500],
                action=action_name,
                action_params=params,
                result=str(step.get("info", ""))[:500],
                reward=float(step.get("reward", 0)),
            ))
        return steps
    def get_filtered_context_for_prompt(self) -> str:
        return ""
    def get_memory_stats(self) -> Dict[str, Any]:
        if self.memory_manager:
            return self.memory_manager.get_stats()
        return {"enabled": False}
