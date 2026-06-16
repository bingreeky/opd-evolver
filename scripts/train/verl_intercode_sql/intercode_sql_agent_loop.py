from __future__ import annotations
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4
from opd_evolver.base.engine.utils import parse_llm_action_response
from opd_evolver.benchmark.bench_intercode import InterCodeEnvironment
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
ALLOWED_ACTIONS = {"execute", "submit"}
@register("intercode_sql_agent")
class InterCodeSqlAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        max_steps: int | None = None,
        invalid_action_reward: float = -1.0,
        per_turn_max_tokens: int | None = None,
        **kwargs,
    ):
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.config_max_steps = max_steps
        self.invalid_action_reward = float(invalid_action_reward)
        env_cap = os.getenv("INTERCODE_SQL_PER_TURN_MAX_TOKENS")
        if env_cap:
            per_turn_max_tokens = int(env_cap)
        elif per_turn_max_tokens is None:
            per_turn_max_tokens = 512
        self.per_turn_max_tokens = max(1, int(per_turn_max_tokens))
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        interaction_kwargs = dict(kwargs.get("interaction_kwargs") or {})
        if not interaction_kwargs:
            extra_info = kwargs.get("extra_info") or {}
            interaction_kwargs = dict(extra_info.get("interaction_kwargs") or {})
        task_idx = int(interaction_kwargs.get("task_idx", kwargs.get("index", 0)))
        task_id = str(interaction_kwargs.get("task_id") or f"sql_{task_idx}")
        max_steps = int(interaction_kwargs.get("max_steps") or self.config_max_steps or 30)
        env = InterCodeEnvironment(
            level={"id": task_id, "index": task_idx},
            env_type=str(interaction_kwargs.get("env_type", "sql")),
            image_name=str(interaction_kwargs.get("image_name", "docker-env-sql")),
            data_path=str(interaction_kwargs["dataset_path"]),
            traj_dir=None,
            verbose=False,
            max_steps=max_steps,
            ctf_tasks=None,
            sql_service_mode=str(interaction_kwargs.get("sql_service_mode", os.getenv("SQL_SERVICE_MODE", "docker"))),
            sql_host=str(interaction_kwargs.get("sql_host", os.getenv("SQL_HOST", "127.0.0.1"))),
            sql_port=int(interaction_kwargs.get("sql_port", os.getenv("SQL_PORT", "3307"))),
            sql_user=str(interaction_kwargs.get("sql_user", os.getenv("SQL_USER", "admin"))),
            sql_password=str(interaction_kwargs.get("sql_password", os.getenv("SQL_PASSWORD", "admin"))),
        )
        messages = [dict(item) for item in kwargs["raw_prompt"]]
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        turn_scores: list[float] = []
        env_infos: list[dict[str, Any]] = []
        parsed_actions: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}
        cumulative_reward = 0.0
        assistant_turns = 0
        user_turns = 0
        submitted = False
        terminal_error = False
        try:
            initial_observation = await env.reset()
            messages = self._inject_initial_observation(messages, initial_observation)
            prompt_ids = await self.apply_chat_template(messages)
            context_ids = list(prompt_ids)
            done = False
            while not done and assistant_turns < max_steps and len(response_ids) < self.response_length:
                remaining = self.response_length - len(response_ids)
                if remaining <= 0:
                    break
                turn_sampling_params = dict(sampling_params)
                turn_sampling_params["max_tokens"] = min(self.per_turn_max_tokens, remaining)
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=context_ids,
                        sampling_params=turn_sampling_params,
                    )
                if metrics.get("num_preempted") is None:
                    metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
                elif output.num_preempted is not None:
                    metrics["num_preempted"] += output.num_preempted
                generated_ids = list(output.token_ids)
                assistant_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                assistant_turns += 1
                self._append_response_tokens(
                    response_ids=response_ids,
                    response_mask=response_mask,
                    token_ids=generated_ids,
                    mask_value=1,
                    response_logprobs=response_logprobs,
                    token_logprobs=output.log_probs,
                )
                context_ids.extend(generated_ids)
                parsed_action = parse_llm_action_response(assistant_text)
                parsed_actions.append(parsed_action)
                action_name = str(parsed_action.get("action", "")).strip().lower()
                params = parsed_action.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                if action_name not in ALLOWED_ACTIONS:
                    observation = {
                        "error": "invalid_action",
                        "message": "Use only InterCode SQL JSON actions: execute or submit.",
                        "parsed_action": parsed_action,
                        "current_step": assistant_turns,
                        "max_steps": max_steps,
                    }
                    reward = self.invalid_action_reward
                    info = {"parsed_action": parsed_action, "error": "invalid_action"}
                    done = True
                    terminal_error = True
                elif action_name == "execute":
                    command = params.get("command")
                    if not isinstance(command, str) or not command.strip():
                        observation = {
                            "error": "missing_command",
                            "message": "Action execute requires params.command as a non-empty SQL string.",
                            "current_step": assistant_turns,
                            "max_steps": max_steps,
                        }
                        reward = self.invalid_action_reward
                        info = {"parsed_action": parsed_action, "error": "missing_command"}
                        done = True
                        terminal_error = True
                    else:
                        observation, reward, done, info = await env.step(
                            {"action": "execute", "params": {"command": command}}
                        )
                else:
                    observation, reward, done, info = await env.step({"action": "submit", "params": {}})
                    submitted = True
                    done = True
                reward_f = float(reward)
                cumulative_reward += reward_f
                turn_scores.append(reward_f)
                env_infos.append(dict(info or {}))
                if done:
                    break
                if len(response_ids) >= self.response_length:
                    break
                observation_ids = await self._encode_observation(observation)
                user_turns += 1
                self._append_response_tokens(
                    response_ids=response_ids,
                    response_mask=response_mask,
                    token_ids=observation_ids,
                    mask_value=0,
                    response_logprobs=response_logprobs,
                    token_logprobs=[0.0] * len(observation_ids),
                )
                context_ids.extend(observation_ids)
            if not submitted and not terminal_error and assistant_turns > 0:
                try:
                    submit_observation, submit_reward, _submit_done, submit_info = await env.step(
                        {"action": "submit", "params": {}}
                    )
                    final_reward = float(submit_reward)
                    cumulative_reward += final_reward
                    turn_scores.append(final_reward)
                    env_infos.append({"auto_submit": True, **dict(submit_info or {})})
                    submitted = True
                    if len(response_ids) < self.response_length:
                        observation_ids = await self._encode_observation(
                            {
                                "message": "auto_submitted",
                                "submit_observation": submit_observation,
                            }
                        )
                        user_turns += 1
                        self._append_response_tokens(
                            response_ids=response_ids,
                            response_mask=response_mask,
                            token_ids=observation_ids,
                            mask_value=0,
                            response_logprobs=response_logprobs,
                            token_logprobs=[0.0] * len(observation_ids),
                        )
                except Exception as exc:
                    logger.warning("Failed to auto-submit SQL task %s: %s", task_id, exc)
            while len(response_logprobs) < len(response_ids):
                response_logprobs.append(0.0)
            response_logprobs = (response_logprobs + [0.0] * self.response_length)[: self.response_length]
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs if response_ids else None,
                reward_score=float(cumulative_reward),
                num_turns=assistant_turns + user_turns + 1,
                metrics=metrics,
                extra_fields={
                    "turn_scores": turn_scores,
                    "tool_rewards": turn_scores,
                    "parsed_actions": parsed_actions,
                    "env_infos": env_infos,
                    "cumulative_reward": cumulative_reward,
                    "submitted": submitted,
                    "task_id": task_id,
                    "task_idx": task_idx,
                    "db": interaction_kwargs.get("db", ""),
                },
            )
        finally:
            try:
                await env.close()
            except Exception as exc:
                logger.warning("Failed to close InterCode SQL env for %s: %s", task_id, exc)
    def _inject_initial_observation(self, messages: list[dict[str, Any]], observation: Any) -> list[dict[str, Any]]:
        observation_text = self._format_observation(observation)
        if not messages:
            return [{"role": "user", "content": f"INITIAL_OBSERVATION:\n{observation_text}"}]
        first = dict(messages[0])
        content = first.get("content", "")
        if isinstance(content, str):
            first["content"] = f"{content}\n\nINITIAL_OBSERVATION:\n{observation_text}"
        else:
            first["content"] = content
        return [first, *messages[1:]]
    async def _encode_observation(self, observation: Any) -> list[int]:
        observation_text = (
            "OBSERVATION:\n"
            f"{self._format_observation(observation)}\n\n"
            "Return the next InterCode SQL action as exactly one JSON object."
        )
        return await self.apply_chat_template(
            [{"role": "user", "content": observation_text}],
            remove_system_prompt=True,
        )
    def _format_observation(self, payload: Any) -> str:
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False, default=self._json_default)
        return str(payload)
    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            if obj.is_finite():
                return float(obj)
            return str(obj)
        return str(obj)
    def _append_response_tokens(
        self,
        *,
        response_ids: list[int],
        response_mask: list[int],
        token_ids: list[int],
        mask_value: int,
        response_logprobs: list[float] | None = None,
        token_logprobs: list[float] | None = None,
    ) -> None:
        remaining = self.response_length - len(response_ids)
        if remaining <= 0:
            return
        clipped = token_ids[:remaining]
        response_ids.extend(clipped)
        response_mask.extend([mask_value] * len(clipped))
        if response_logprobs is not None:
            if token_logprobs:
                response_logprobs.extend(token_logprobs[: len(clipped)])
            else:
                response_logprobs.extend([0.0] * len(clipped))
