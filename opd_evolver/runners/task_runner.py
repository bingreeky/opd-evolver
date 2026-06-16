from __future__ import annotations
import asyncio
import csv
import inspect
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from opd_evolver.base.engine.async_llm import LLMsConfig, create_llm_instance
from opd_evolver.base.engine.logs import logger, LogLevel
from opd_evolver.base.agent.memory import Memory
from opd_evolver.benchmark.common.env import BasicInfo, Environment
from opd_evolver.benchmark.common.runner import Runner, StepRecord, LevelResult
from opd_evolver.memory.evolvelab_adapter import (
    EvolveLabMemoryProviderAdapter,
    MEMRL_BACKEND,
    OPD_HIERARCHICAL_BACKEND,
    is_evolvelab_backend,
    is_memrl_backend,
    is_reasoning_bank_backend,
)
from opd_evolver.memory.memrl_adapter import MemRLMemoryProviderAdapter
from opd_evolver.memory.reasoning_bank_adapter import ReasoningBankMemoryProviderAdapter
from opd_evolver.subagents.react_agent import ReActAgent
class SimpleTaskRunner(Runner):
    def __init__(
        self,
        model: str,
        env_type: str = "ctf",
        max_steps: int = 30,
        step_timeout: float | None = 120.0,
        trajectory_dir: Path | None = None,
        csv_summary_path: Path | None = None,
        llm_max_completion_tokens: Optional[int] = None,
    ):
        self.model = model
        self.env_type = env_type
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.csv_summary_path = Path(csv_summary_path) if csv_summary_path else None
        self._csv_lock = asyncio.Lock()
        self.llm = create_llm_instance(
            LLMsConfig.default().get(model),
            max_completion_tokens=llm_max_completion_tokens,
        )
    def _create_agent(self, memory_context: str = "") -> ReActAgent:
        agent = ReActAgent(
            llm=self.llm,
            benchmark_type=self.env_type,
            memory=Memory(llm=self.llm, max_memory=10),
        )
        if memory_context:
            agent.context = memory_context
        return agent
    async def run(
        self,
        agent: ReActAgent | None,
        env: Environment,
        memory_context: str = "",
    ) -> LevelResult:
        prepare = getattr(env, "prepare", None)
        if prepare is not None:
            maybe_prepared = prepare()
            if inspect.isawaitable(maybe_prepared):
                await maybe_prepared
        info = env.get_basic_info()
        logger.info(f"[SimpleRunner] Starting task: {info.env_id}")
        if agent is None:
            agent = self._create_agent(memory_context)
        elif memory_context:
            agent.context = memory_context
        agent.task_instruction = info.instruction
        agent.original_question = info.instruction
        agent.reset(info)
        obs = await env.reset()
        history: List[StepRecord] = []
        total_reward = 0.0
        done = False
        exception_occurred = None
        try:
            for t in range(self.max_steps):
                current_step = t + 1
                logger.log_to_file(LogLevel.INFO, f"Environment Observation: {obs}")
                try:
                    if self.step_timeout:
                        step_result = await asyncio.wait_for(
                            agent.step(
                                observation=obs,
                                history=history,
                                current_step=current_step,
                                max_steps=self.max_steps,
                            ),
                            timeout=self.step_timeout,
                        )
                    else:
                        step_result = await agent.step(
                            observation=obs,
                            history=history,
                            current_step=current_step,
                            max_steps=self.max_steps,
                        )
                except asyncio.TimeoutError:
                    logger.error(f"[SimpleRunner] Step timed out after {self.step_timeout}s")
                    step_record = StepRecord(
                        observation=obs,
                        action={"error": "step_timeout"},
                        reward=0.0,
                        raw_response="step timeout",
                        done=True,
                        info={"error": "step_timeout"},
                        raw_input=None,
                        observation_after=None,
                    )
                    history.append(step_record)
                    break
                action, raw_response, raw_input = step_result
                logger.info(f"[SimpleRunner] Step {current_step}/{self.max_steps}: {action}")
                obs_before = obs
                obs_next, reward, step_done, step_info = await env.step(action)
                step_record = StepRecord(
                    observation=obs_before,
                    action=action,
                    reward=reward,
                    raw_response=raw_response,
                    done=step_done,
                    info=step_info,
                    raw_input=raw_input,
                    observation_after=obs_next,
                )
                history.append(step_record)
                total_reward += reward
                obs = obs_next
                if step_done:
                    done = True
                    break
            if not done:
                logger.info("[SimpleRunner] Max steps reached, forcing submit")
                try:
                    obs_before = obs
                    obs, reward, _, info_dict = await env.step({"action": "submit", "params": {}})
                    total_reward = float(reward)
                    done = True
                    history.append(StepRecord(
                        observation=obs_before,
                        action={"action": "submit", "params": {}},
                        reward=reward,
                        raw_response="forced_submit",
                        done=True,
                        info=info_dict,
                        raw_input=None,
                        observation_after=obs,
                    ))
                except Exception as e:
                    logger.error(f"[SimpleRunner] Forced submit failed: {e}")
                    done = True
        except Exception as e:
            logger.error(f"[SimpleRunner] Execution error: {e}", exc_info=True)
            exception_occurred = e
        finally:
            if hasattr(env, 'close'):
                try:
                    await env.close()
                except Exception as e:
                    logger.warning(f"[SimpleRunner] Cleanup error: {e}")
        success = self._determine_success(history, total_reward)
        usage = agent.llm.get_usage_summary() if agent else {}
        result = LevelResult(
            model=usage.get("model", self.model),
            total_reward=total_reward,
            steps=len(history),
            done=done,
            trace=history,
            cost=usage.get("total_cost", 0.0),
            input_tokens=usage.get("total_input_tokens", 0),
            output_tokens=usage.get("total_output_tokens", 0),
            success=success,
        )
        logger.info(f"[SimpleRunner] Task completed: success={success}, reward={total_reward:.2f}, steps={len(history)}")
        if self.trajectory_dir:
            self._save_trajectory(info, result, history)
        if self.csv_summary_path:
            await self._save_csv(info.env_id, result)
        if exception_occurred:
            raise exception_occurred
        return result
    def _determine_success(self, history: List[StepRecord], total_reward: float) -> bool:
        if self.env_type == "minihack":
            if total_reward > 0:
                return True
            return any(bool(record.info.get("success")) for record in history if record.info)
        if self.env_type == "awm":
            if total_reward >= 1.0:
                return True
            for record in reversed(history):
                info = record.info or {}
                if info.get("success"):
                    return True
                if info.get("reward_type") == "complete":
                    return True
                verify_result = info.get("verify_result")
                if isinstance(verify_result, dict):
                    for key in ("success", "passed", "pass", "is_correct", "complete"):
                        if verify_result.get(key) is True:
                            return True
            return False
        if total_reward == 1:
            return True
        threshold = 0.8 if self.env_type == "bash" else 1.0
        for record in reversed(history):
            action = record.action
            if action.get("action") == "submit":
                info = record.info or {}
                if info.get("success") or info.get("reward", 0) >= threshold:
                    return True
        return False
    def _save_trajectory(self, info: BasicInfo, result: LevelResult, history: List[StepRecord]) -> None:
        try:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            steps = []
            for i, record in enumerate(history):
                step = {
                    "step": i + 1,
                    "action": record.action,
                    "observation": record.observation,
                    "observation_before": record.observation,
                    "observation_after": record.observation_after,
                    "reward": record.reward,
                    "done": record.done,
                    "info": record.info,
                    "raw_input": record.raw_input,
                    "raw_response": record.raw_response,
                }
                steps.append(step)
            trajectory = {
                "task_id": info.env_id,
                "instruction": info.instruction,
                "model": self.model,
                "env_type": self.env_type,
                "success": result.success,
                "total_reward": result.total_reward,
                "steps": result.steps,
                "cost": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "timestamp": result.timestamp,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "action_space": info.action_space,
                "meta_data": info.meta_data,
                "execution_trace": steps,
            }
            safe_env_id = str(info.env_id).replace("/", "_").replace(":", "_")
            filename = (
                f"{safe_env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            )
            filepath = self.trajectory_dir / filename
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"[SimpleRunner] Trajectory saved: {filepath}")
        except Exception as e:
            logger.error(f"[SimpleRunner] Failed to save trajectory: {e}")
    async def _save_csv(self, task_id: str, result: LevelResult) -> None:
        async with self._csv_lock:
            try:
                self.csv_summary_path.parent.mkdir(parents=True, exist_ok=True)
                fieldnames = ["task_id", "model", "success", "reward", "steps", "cost", "timestamp"]
                need_header = not self.csv_summary_path.exists() or self.csv_summary_path.stat().st_size == 0
                if need_header:
                    with self.csv_summary_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                with self.csv_summary_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({
                        "task_id": task_id,
                        "model": result.model,
                        "success": result.success,
                        "reward": f"{result.total_reward:.4f}",
                        "steps": result.steps,
                        "cost": f"{result.cost:.6f}",
                        "timestamp": result.timestamp,
                    })
            except Exception as e:
                logger.error(f"[SimpleRunner] Failed to save CSV: {e}")
class MemoryAugmentedTaskRunner(SimpleTaskRunner):
    def __init__(
        self,
        model: str,
        env_type: str = "ctf",
        memory_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 30,
        step_timeout: float | None = 120.0,
        trajectory_dir: Path | None = None,
        csv_summary_path: Path | None = None,
        llm_max_completion_tokens: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            env_type=env_type,
            max_steps=max_steps,
            step_timeout=step_timeout,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
            llm_max_completion_tokens=llm_max_completion_tokens,
        )
        self.memory_config = memory_config or {}
        self.memory_config["env_type"] = env_type
        self.memory_backend = self.memory_config.get("memory_backend", OPD_HIERARCHICAL_BACKEND)
        self._memory_pipeline = None
        self._provider_adapter = None
        self._provider_adapter_lock = threading.Lock()
    @staticmethod
    def _steprecord_to_dict(step: StepRecord) -> Dict[str, Any]:
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
            "act_prompt": step.act_prompt,
        }
    async def _get_memory_pipeline(self):
        if (
            is_evolvelab_backend(self.memory_backend)
            or is_reasoning_bank_backend(self.memory_backend)
            or is_memrl_backend(self.memory_backend)
        ):
            return None
        if self._memory_pipeline is None:
            try:
                from opd_evolver.pipelines.memory_pipeline import MemoryAugmentedPipeline
                from opd_evolver.memory.memory_manager import MemoryConfig
                env_type = self.memory_config.get("env_type", "ctf")
                storage_dir = self.memory_config.get("storage_dir")
                if not storage_dir:
                    storage_dir = str(Path("workspace") / "memory" / env_type)
                else:
                    storage_dir = str(Path(storage_dir).expanduser())
                writer_path = self.memory_config.get("writer_dataset_path")
                if writer_path is None and self.memory_config.get("writer_dataset", False):
                    writer_path = str(
                        Path(storage_dir) / "memory_writer_dataset.jsonl"
                    )
                selector_path = self.memory_config.get("selector_dataset_path")
                if selector_path is None and self.memory_config.get(
                    "selector_dataset", False
                ):
                    selector_path = str(
                        Path(storage_dir) / "memory_selector_dataset.jsonl"
                    )
                config = MemoryConfig(
                    storage_dir=storage_dir,
                    embedding_provider=self.memory_config.get("embedding_provider", "local"),
                    embedding_model=self.memory_config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                    cold_start_threshold=self.memory_config.get("cold_start_threshold", 20),
                    retrieval_top_k=self.memory_config.get("retrieval_top_k", 3),
                    writer_dataset_path=writer_path,
                    selector_dataset_path=selector_path,
                    memory_tiers=self.memory_config.get("memory_tiers"),
                )
                writer_name = self.memory_config.get("reflection_model")
                writer_llm = None
                if writer_name:
                    writer_llm = create_llm_instance(
                        LLMsConfig.default().get(writer_name)
                    )
                selector_name = self.memory_config.get("selector_model")
                selector_llm = None
                if selector_name:
                    selector_llm = create_llm_instance(
                        LLMsConfig.default().get(selector_name)
                    )
                self._memory_pipeline = MemoryAugmentedPipeline(
                    llm=self.llm,
                    memory_writer_llm=writer_llm,
                    memory_selector_llm=selector_llm,
                    config=config,
                )
                logger.info("[MemoryRunner] Memory pipeline initialized")
            except Exception as e:
                logger.warning(f"[MemoryRunner] Memory pipeline init failed: {e}")
                self._memory_pipeline = None
        return self._memory_pipeline
    def _make_evolvelab_adapter(self) -> EvolveLabMemoryProviderAdapter:
        env_type = self.memory_config.get("env_type", self.env_type)
        storage_dir = self.memory_config.get("storage_dir")
        if not storage_dir:
            storage_dir = str(Path("workspace") / "memory" / env_type)
        return EvolveLabMemoryProviderAdapter(
            backend=self.memory_backend,
            storage_dir=storage_dir,
            model_name=self.model,
            max_completion_tokens=self.memory_config.get("llm_max_completion_tokens"),
        )
    def _make_reasoning_bank_adapter(self) -> ReasoningBankMemoryProviderAdapter:
        env_type = self.memory_config.get("env_type", self.env_type)
        storage_dir = self.memory_config.get("storage_dir")
        if not storage_dir:
            storage_dir = str(Path("workspace") / "memory" / env_type)
        storage_dir = Path(storage_dir).expanduser()
        if storage_dir.name != "reasoning_bank":
            storage_dir = storage_dir / "reasoning_bank"
        return ReasoningBankMemoryProviderAdapter(
            storage_dir=storage_dir,
            model_name=self.model,
            max_completion_tokens=self.memory_config.get("llm_max_completion_tokens"),
            retrieval_top_k=int(self.memory_config.get("retrieval_top_k", 3)),
            min_similarity=float(self.memory_config.get("min_similarity", 0.0)),
            embedding_provider=self.memory_config.get("embedding_provider", "local"),
            embedding_model=self.memory_config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
        )
    def _make_memrl_adapter(self) -> MemRLMemoryProviderAdapter:
        storage_dir = self.memory_config.get("memrl_storage_dir") or self.memory_config.get("storage_dir")
        if not storage_dir:
            storage_dir = str(Path("workspace") / "memory" / self.env_type / MEMRL_BACKEND)
        return MemRLMemoryProviderAdapter(
            storage_dir=storage_dir,
            model_name=self.model,
            max_completion_tokens=self.memory_config.get("llm_max_completion_tokens"),
            retrieval_top_k=int(self.memory_config.get("retrieval_top_k", 3)),
            min_similarity=float(self.memory_config.get("min_similarity", 0.0)),
            embedding_provider=self.memory_config.get("embedding_provider", "local"),
            embedding_model=self.memory_config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
            read_only=bool(self.memory_config.get("memory_read_only", False)),
            mos_config_path=self.memory_config.get("memrl_mos_config_path"),
            build_strategy=self.memory_config.get("memrl_build_strategy", "proceduralization"),
            retrieve_strategy=self.memory_config.get("memrl_retrieve_strategy", "query"),
            update_strategy=self.memory_config.get("memrl_update_strategy", "adjustment"),
            user_id=self.memory_config.get("memrl_user_id", f"{self.env_type}_memrl"),
            enable_value_driven=bool(self.memory_config.get("memrl_enable_value_driven", True)),
        )
    def _get_provider_adapter(self) -> Any | None:
        if self._provider_adapter is not None:
            return self._provider_adapter
        with self._provider_adapter_lock:
            if self._provider_adapter is not None:
                return self._provider_adapter
            if is_evolvelab_backend(self.memory_backend):
                self._provider_adapter = self._make_evolvelab_adapter()
            elif is_reasoning_bank_backend(self.memory_backend):
                self._provider_adapter = self._make_reasoning_bank_adapter()
            elif is_memrl_backend(self.memory_backend):
                self._provider_adapter = self._make_memrl_adapter()
        return self._provider_adapter
    def get_memory_stats(self) -> dict[str, Any]:
        if self.memory_backend == OPD_HIERARCHICAL_BACKEND:
            return {
                "memory_backend": self.memory_backend,
                "memory_retrieval_count": "",
                "memory_write_count": "",
                "memory_read_only": bool(self.memory_config.get("memory_read_only", False)),
            }
        adapter = getattr(self, "_provider_adapter", None)
        if adapter is not None and hasattr(adapter, "stats"):
            return adapter.stats()
        return {
            "memory_backend": self.memory_backend,
            "memory_retrieval_count": "",
            "memory_write_count": "",
            "memory_read_only": bool(self.memory_config.get("memory_read_only", False)),
        }
    async def run(
        self,
        agent: ReActAgent | None,
        env: Environment,
        memory_context: str = "",
    ) -> LevelResult:
        info = env.get_basic_info()
        task_id = info.env_id
        task_type = (info.meta_data or {}).get("hardness") or self.env_type
        pipeline = await self._get_memory_pipeline()
        provider_adapter = self._get_provider_adapter()
        augmented_context = memory_context
        if provider_adapter is not None:
            try:
                begin_context = await provider_adapter.provide_begin(
                    task_description=info.instruction,
                    context=str(info.meta_data or {}),
                    task_id=task_id,
                )
                if begin_context:
                    augmented_context = begin_context
                    logger.info(
                        f"[MemoryRunner] Retrieved provider memory context "
                        f"backend={self.memory_backend} chars={len(augmented_context)}"
                    )
            except Exception as e:
                logger.warning(f"[MemoryRunner] Provider pre-execution failed: {e}")
        elif pipeline:
            try:
                pre_result = await pipeline.pre_execution(
                    task_id=task_id,
                    task_description=info.instruction,
                    additional_context=str(info.meta_data or {}),
                    task_type=str(task_type),
                )
                if pre_result and pre_result.formatted_context:
                    augmented_context = pre_result.formatted_context
                    logger.info(f"[MemoryRunner] Retrieved memory context ({len(augmented_context)} chars)")
            except Exception as e:
                logger.warning(f"[MemoryRunner] Pre-execution failed: {e}")
        result = await super().run(agent, env, augmented_context)
        if provider_adapter is not None:
            try:
                trace_dicts = [self._steprecord_to_dict(step) for step in result.trace]
                ok, msg = await provider_adapter.take_in(
                    task_description=info.instruction,
                    trajectory=trace_dicts,
                    success=result.success,
                    result={"success": result.success, "reward": result.total_reward},
                    metadata={
                        "task_id": task_id,
                        "task_type": str(task_type),
                        "env_type": self.env_type,
                        "memory_backend": self.memory_backend,
                    },
                )
                logger.info(
                    f"[MemoryRunner] Provider post-execution backend={self.memory_backend} "
                    f"ok={ok} msg={msg}"
                )
            except Exception as e:
                logger.warning(f"[MemoryRunner] Provider post-execution failed: {e}")
        elif pipeline:
            try:
                trace_dicts = [self._steprecord_to_dict(step) for step in result.trace]
                await pipeline.post_execution(
                    task_id=task_id,
                    task_description=info.instruction,
                    execution_trace=trace_dicts,
                    success=result.success,
                    total_reward=result.total_reward,
                    tags=info.meta_data.get("tags") if info.meta_data else None,
                    task_type=str(task_type),
                )
                logger.info(f"[MemoryRunner] Post-execution completed for {task_id}")
            except Exception as e:
                logger.warning(f"[MemoryRunner] Post-execution failed: {e}")
        return result
class SimpleTaskCompatibilityRunner(SimpleTaskRunner):
    def __init__(
        self,
        model: str,
        env_type: str = "ctf",
        max_steps: int = 30,
        step_timeout: float | None = 120.0,
        trajectory_dir: Path | None = None,
        csv_summary_path: Path | None = None,
    ):
        super().__init__(
            model=model,
            env_type=env_type,
            max_steps=max_steps,
            step_timeout=step_timeout,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
        )
class MemoryAugmentedTaskCompatibilityRunner(MemoryAugmentedTaskRunner):
    pass
SimpleRunnerRunner = SimpleTaskCompatibilityRunner
SimpleIntercodeRunner = SimpleTaskRunner
MemoryAugmentedIntercodeRunner = MemoryAugmentedTaskRunner
SimpleInterCodeRunner = SimpleTaskCompatibilityRunner
MemoryAugmentedCTFRunner = MemoryAugmentedTaskCompatibilityRunner
