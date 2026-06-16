import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from opd_evolver.benchmark.common.env import Action, BasicInfo, Environment, Observation
from opd_evolver.base.agent.base_agent import BaseAgent
from opd_evolver.base.engine.logs import logger, LogLevel
@dataclass
class StepRecord:
    observation: Observation
    action: Action
    reward: float
    raw_response: str
    done: bool
    info: Dict[str, Any]
    raw_input: Optional[str] = None
    act_prompt: Optional[str] = None
    observation_after: Optional[Observation] = None
@dataclass
class LevelResult:
    model: str
    total_reward: float
    steps: int
    done: bool
    trace: List[StepRecord]
    cost: float
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    success: bool = False
class Runner:
    step_timeout: Optional[float] = 600.0
    async def run(self, agent: BaseAgent, env: Environment) -> LevelResult:
        start_time = datetime.now().isoformat()
        info = env.get_basic_info()
        agent.reset(info)
        reset_result = env.reset()
        obs = await reset_result if inspect.isawaitable(reset_result) else reset_result
        history: List[StepRecord] = []
        total_reward = 0.0
        max_steps = info.max_steps
        for t in range(max_steps):
            logger.log_to_file(LogLevel.INFO, f"Environment Observation:{obs}")
            try:
                if self.step_timeout:
                    step_result = await asyncio.wait_for(
                        agent.step(observation=obs, history=history),
                        timeout=self.step_timeout,
                    )
                else:
                    step_result = await agent.step(observation=obs, history=history)
            except asyncio.TimeoutError:
                logger.error(f"Agent step timed out after {self.step_timeout} seconds; terminating level early.")
                step_record = StepRecord(
                    observation=obs,
                    action={"error": "step_timeout"},
                    reward=0.0,
                    raw_response="step timeout",
                    done=True,
                    info={"error": "step_timeout", "timeout_s": self.step_timeout},
                    raw_input=None,
                    observation_after=None,
                )
                history.append(step_record)
                break
            if isinstance(step_result, (list, tuple)):
                if len(step_result) == 3:
                    action, raw_response, raw_input = step_result
                elif len(step_result) == 2:
                    action, raw_response = step_result
                    raw_input = None
                else:
                    raise ValueError(f"agent.step returned {len(step_result)} values, expected 2 or 3")
            else:
                raise TypeError(f"agent.step returned unsupported type: {type(step_result)}")
            obs_next, reward, done, step_info = await env.step(action)
            step_record = StepRecord(
                observation=obs,
                action=action,
                reward=reward,
                raw_response=raw_response,
                done=done,
                info=step_info,
                raw_input=raw_input,
                observation_after=obs_next,
            )
            history.append(step_record)
            total_reward += reward
            obs = obs_next
            if done:
                break
        end_time = datetime.now().isoformat()
        usage_summary = agent.llm.get_usage_summary()
        return LevelResult(
            model=usage_summary.get("model", ""),
            total_reward=total_reward,
            steps=len(history),
            done=history[-1].done if history else False,
            trace=history,
            cost=usage_summary.get("total_cost", 0.0),
            input_tokens=usage_summary.get("total_input_tokens", 0),
            output_tokens=usage_summary.get("total_output_tokens", 0),
            start_time=start_time,
            end_time=end_time,
        )
