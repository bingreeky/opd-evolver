import asyncio
import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Type
from datetime import datetime
from opd_evolver.benchmark.common.env import Environment
import opd_evolver.base.agent.base_agent as Agent
from opd_evolver.benchmark.common.runner import Runner, LevelResult
LevelSpec = Dict[str, Any]
class Benchmark:
    def list_levels(self) -> List[LevelSpec]:
        raise NotImplementedError
    def make_env(self, level: LevelSpec) -> "Environment":
        raise NotImplementedError
    async def run(
        self,
        agent_cls: Type[Agent],
        agent_kwargs: Optional[Dict[str, Any]] = None,
        runner: Runner | None = None,
        levels: Optional[List[LevelSpec]] = None,
        max_concurrency: int = 1,
        on_task_complete: Optional[callable] = None,
    ):
        if runner is None:
            runner = getattr(self, '_runner', Runner())
        levels = levels or self.list_levels()
        agent_kwargs = agent_kwargs or {}
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        async def run_level(level: LevelSpec):
            async with semaphore:
                env = self.make_env(level)
                agent = agent_cls(**agent_kwargs)
                result = await runner.run(agent, env)
                return level, result
        tasks = [asyncio.create_task(run_level(level)) for level in levels]
        results = {}
        for task in asyncio.as_completed(tasks):
            level, result = await task
            level_id = level.get("id", str(level))
            results[level_id] = result
            if on_task_complete:
                try:
                    await_result = on_task_complete(level_id, result)
                    if asyncio.iscoroutine(await_result):
                        await await_result
                except Exception as e:
                    from opd_evolver.base.engine.logs import logger
                    logger.warning(f"on_task_complete callback failed for {level_id}: {e}")
        return results
