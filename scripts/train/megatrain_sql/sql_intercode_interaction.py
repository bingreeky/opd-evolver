from __future__ import annotations
import json
import logging
import os
from decimal import Decimal
from datetime import date, datetime
from typing import Any, Optional
from uuid import uuid4
from opd_evolver.base.engine.utils import parse_llm_action_response
from opd_evolver.benchmark.bench_intercode import InterCodeEnvironment
from verl.interactions.base import BaseInteraction
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
class SqlIntercodeInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        task_idx = int(kwargs["task_idx"])
        task_id = str(kwargs.get("task_id") or f"sql_{task_idx}")
        dataset_path = str(kwargs["dataset_path"])
        image_name = str(kwargs.get("image_name", self.config.get("image_name", "docker-env-sql")))
        env_type = str(kwargs.get("env_type", self.config.get("env_type", "sql")))
        max_steps = int(kwargs.get("max_steps", self.config.get("max_steps", 30)))
        sql_service_mode = str(
            kwargs.get("sql_service_mode")
            or os.getenv("SQL_SERVICE_MODE")
            or self.config.get("sql_service_mode", "docker")
        )
        sql_host = str(
            kwargs.get("sql_host")
            or os.getenv("SQL_HOST")
            or self.config.get("sql_host", "127.0.0.1")
        )
        sql_port = int(
            kwargs.get("sql_port")
            or os.getenv("SQL_PORT")
            or self.config.get("sql_port", 3307)
        )
        sql_user = str(
            kwargs.get("sql_user")
            or os.getenv("SQL_USER")
            or self.config.get("sql_user", "admin")
        )
        sql_password = str(
            kwargs.get("sql_password")
            or os.getenv("SQL_PASSWORD")
            or self.config.get("sql_password", "admin")
        )
        level = {"id": task_id, "index": task_idx}
        env = InterCodeEnvironment(
            level=level,
            env_type=env_type,
            image_name=image_name,
            data_path=dataset_path,
            traj_dir=None,
            verbose=False,
            max_steps=max_steps,
            ctf_tasks=None,
            sql_service_mode=sql_service_mode,
            sql_host=sql_host,
            sql_port=sql_port,
            sql_user=sql_user,
            sql_password=sql_password,
        )
        initial_observation = await env.reset()
        self._instance_dict[instance_id] = {
            "env": env,
            "task_id": task_id,
            "task_idx": task_idx,
            "query": kwargs.get("query", ""),
            "ground_truth": kwargs.get("ground_truth", ""),
            "db": kwargs.get("db", ""),
            "current_observation": initial_observation,
            "last_turn_reward": 0.0,
            "cumulative_reward": 0.0,
            "submitted": False,
            "turn_count": 0,
            "last_action": None,
        }
        return instance_id
    def _get_state(self, instance_id: str) -> dict[str, Any]:
        if instance_id not in self._instance_dict:
            raise KeyError(f"Unknown interaction instance_id: {instance_id}")
        return self._instance_dict[instance_id]
    def _find_last_assistant_content(self, messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, str):
                    return content
                return json.dumps(content, ensure_ascii=False)
        return ""
    def _format_observation(self, payload: Any) -> str:
        if isinstance(payload, dict):
            def _json_default(obj: Any) -> Any:
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, Decimal):
                    if obj.is_finite():
                        return float(obj)
                    return str(obj)
                raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
            return json.dumps(payload, ensure_ascii=False, default=_json_default)
        return str(payload)
    async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._get_state(instance_id)
        state["turn_count"] += 1
        assistant_content = self._find_last_assistant_content(messages)
        parsed_action = parse_llm_action_response(assistant_content)
        state["last_action"] = parsed_action
        action_name = str(parsed_action.get("action", "")).strip().lower()
        params = parsed_action.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if action_name == "execute":
            command = params.get("command")
            if not isinstance(command, str) or not command.strip():
                state["last_turn_reward"] = -1.0
                response = {
                    "error": "missing_command",
                    "message": "Action execute requires params.command as a non-empty SQL string.",
                    "current_step": state["turn_count"],
                    "max_steps": state["env"].max_steps,
                }
                state["current_observation"] = response
                return True, self._format_observation(response), -1.0, {"parsed_action": parsed_action}
            observation, reward, done, info = await state["env"].step({"action": "execute", "params": {"command": command}})
            execute_reward = float(reward)
            state["current_observation"] = observation
            state["last_turn_reward"] = execute_reward
            state["cumulative_reward"] += execute_reward
            if done and not state.get("submitted", False):
                submit_observation, submit_reward, submit_done, submit_info = await state["env"].step(
                    {"action": "submit", "params": {}}
                )
                final_reward = float(submit_reward)
                state["submitted"] = True
                state["last_turn_reward"] = final_reward
                state["cumulative_reward"] += final_reward
                merged_observation = {
                    "message": "terminal_execute_auto_submitted",
                    "execute_observation": observation,
                    "submit_observation": submit_observation,
                }
                state["current_observation"] = merged_observation
                return bool(submit_done), self._format_observation(merged_observation), final_reward, {
                    "parsed_action": parsed_action,
                    "env_info": info,
                    "auto_submit": True,
                    "submit_info": submit_info,
                }
            return done, self._format_observation(observation), execute_reward, {
                "parsed_action": parsed_action,
                "env_info": info,
            }
        if action_name == "submit":
            observation, reward, done, info = await state["env"].step({"action": "submit", "params": {}})
            final_reward = float(reward)
            state["current_observation"] = observation
            state["last_turn_reward"] = final_reward
            state["cumulative_reward"] += final_reward
            state["submitted"] = True
            return True, self._format_observation(observation), final_reward, {
                "parsed_action": parsed_action,
                "env_info": info,
            }
        state["last_turn_reward"] = -1.0
        response = {
            "error": "invalid_action",
            "message": "Only execute and submit are allowed.",
            "parsed_action": parsed_action,
            "current_step": state["turn_count"],
            "max_steps": state["env"].max_steps,
        }
        state["current_observation"] = response
        return True, self._format_observation(response), -1.0, {"parsed_action": parsed_action}
    async def calculate_score(self, *args, **kwargs) -> float:
        instance_id = kwargs.get("instance_id")
        if instance_id is None and args:
            instance_id = args[0]
        if instance_id is not None and instance_id in self._instance_dict:
            state = self._instance_dict[instance_id]
            return float(state.get("last_turn_reward", 0.0))
        return 0.0
    async def finalize_interaction(self, *args, **kwargs) -> None:
        instance_id = kwargs.get("instance_id")
        if instance_id is None and args:
            instance_id = args[0]
        if instance_id is not None:
            state_items = [(instance_id, self._instance_dict.pop(instance_id, None))]
        else:
            state_items = list(self._instance_dict.items())
            self._instance_dict.clear()
        for active_instance_id, state in state_items:
            if not state:
                continue
            env = state.get("env")
            if env is None:
                continue
            try:
                await env.close()
            except Exception as exc:
                logger.warning("Failed to close SQL interaction env for %s: %s", active_instance_id, exc)
