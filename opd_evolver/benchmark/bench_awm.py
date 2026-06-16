from __future__ import annotations
import inspect
import json
from typing import Any, Dict, Iterable, List, Tuple
from opd_evolver.benchmark.common.env import Action, BasicInfo, Environment, Observation
_HIDDEN_AWM_TOOLS = {"verify", "done", "__list_scenarios__"}
def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
def _to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return _to_plain(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): _to_plain(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)
def _json_dumps(value: Any) -> str:
    return json.dumps(_to_plain(value), ensure_ascii=False, indent=2, sort_keys=True)
class AWMEnvironment(Environment):
    def __init__(
        self,
        base_url: str,
        scenario: str,
        task_idx: int,
        max_steps: int = 30,
        verifier_mode: str = "code",
        keep_session: bool = False,
        reward_config: Dict[str, float] | None = None,
        *,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.scenario = scenario
        self.task_idx = int(task_idx)
        self.max_steps = max_steps
        self.verifier_mode = verifier_mode
        self.keep_session = keep_session
        self.reward_config = reward_config
        self.connect_timeout_s = float(connect_timeout_s)
        self.message_timeout_s = float(message_timeout_s)
        self._awm_env_factory = None
        self._call_tool_action_cls = None
        self._client_manager = None
        self._client = None
        self._prepared = False
        self._closed = False
        self._task = ""
        self._tools: List[Any] = []
        self._tool_names: set[str] = set()
        self._action_space = "Available MCP tools:\n\nNo tools loaded yet."
        self._initial_observation: Observation | None = None
    async def prepare(self) -> None:
        if self._prepared:
            return
        await self._ensure_client()
        reset_kwargs: Dict[str, Any] = {
            "scenario": self.scenario,
            "task_idx": self.task_idx,
        }
        if self.reward_config is not None:
            reset_kwargs["reward_config"] = self.reward_config
        result = await self._maybe_await(self._client.reset(**reset_kwargs))
        reset_observation = self._normalize_result(
            result,
            last_tool=None,
            default_done=False,
            default_reward=0.0,
        )
        self._task = str(reset_observation.get("task") or "")
        tools = await self._maybe_await(self._client.list_tools())
        self._tools = list(tools or [])
        self._tool_names = {
            str(_get_value(tool, "name"))
            for tool in self._tools
            if _get_value(tool, "name") and str(_get_value(tool, "name")) not in _HIDDEN_AWM_TOOLS
        }
        self._action_space = self._render_action_space(self._tools)
        reset_observation.update(
            {
                "scenario": self.scenario,
                "task_idx": self.task_idx,
                "task": self._task,
                "num_tools": len(self._tool_names),
                "done": False,
            }
        )
        self._initial_observation = reset_observation
        self._prepared = True
    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=f"{self.scenario}:{self.task_idx}",
            instruction=self._task or f"AWM task {self.scenario}:{self.task_idx}",
            action_space=self._action_space,
            max_steps=self.max_steps,
            meta_data={
                "env_type": "awm",
                "scenario": self.scenario,
                "task_idx": self.task_idx,
                "base_url": self.base_url,
                "verifier_mode": self.verifier_mode,
            },
        )
    async def reset(self, seed: int | None = None) -> Observation:
        del seed
        await self.prepare()
        return dict(self._initial_observation or {})
    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        await self.prepare()
        action_name = str((action or {}).get("action") or "").strip()
        params = (action or {}).get("params") or {}
        if not isinstance(params, dict):
            params = {"value": params}
        is_submit = action_name in {"submit", "finish"}
        if is_submit:
            tool_name = "verify"
            final_answer = (
                params.get("answer")
                or params.get("final_answer")
                or params.get("message")
                or params.get("reason")
                or ""
            )
            arguments = {
                "verifier_mode": self.verifier_mode,
                "final_answer": str(final_answer),
            }
        else:
            tool_name = action_name
            arguments = params
            if tool_name not in self._tool_names:
                observation = self._base_observation(
                    last_tool=tool_name,
                    reward=-1.0,
                    done=False,
                    error=f"Unknown AWM MCP tool: {tool_name!r}",
                )
                return observation, -1.0, False, self._info_from_observation(observation)
        try:
            call_tool_action = self._call_tool_action_cls(
                tool_name=tool_name,
                arguments=arguments,
            )
            result = await self._maybe_await(self._client.step(call_tool_action))
            observation = self._normalize_result(
                result,
                last_tool=tool_name,
                default_done=is_submit,
                default_reward=0.0,
            )
            reward = float(observation.get("reward") or 0.0)
            done = bool(observation.get("done")) or is_submit
            observation["done"] = done
            info = self._info_from_observation(observation)
            return observation, reward, done, info
        except Exception as exc:
            reward = -1.0
            observation = self._base_observation(
                last_tool=tool_name,
                reward=reward,
                done=is_submit,
                error=f"{type(exc).__name__}: {exc}",
            )
            return observation, reward, is_submit, self._info_from_observation(observation)
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._call_tool_action_cls is not None:
            try:
                done_action = self._call_tool_action_cls(
                    tool_name="done",
                    arguments={"keep_session": self.keep_session},
                )
                await self._maybe_await(self._client.step(done_action))
            except Exception:
                pass
        if self._client_manager is not None and hasattr(self._client_manager, "__aexit__"):
            try:
                await self._client_manager.__aexit__(None, None, None)
            except Exception:
                pass
    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from agent_world_model_env import AWMEnv
            from openenv.core.env_server.mcp_types import CallToolAction
        except ImportError as exc:
            raise ImportError(
                "AWM/OpenEnv dependencies are not installed. Install OpenEnv with "
                "the agent_world_model_env package and start the AWM server before "
                "running scripts/eval/bench_simple_awm.py."
            ) from exc
        self._awm_env_factory = AWMEnv
        self._call_tool_action_cls = CallToolAction
        self._client_manager = AWMEnv(
            base_url=self.base_url,
            connect_timeout_s=self.connect_timeout_s,
            message_timeout_s=self.message_timeout_s,
        )
        if hasattr(self._client_manager, "__aenter__"):
            self._client = await self._client_manager.__aenter__()
        else:
            self._client = self._client_manager
    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value
    def _base_observation(
        self,
        last_tool: str | None,
        reward: float,
        done: bool,
        error: str | None = None,
        tool_result: Any = None,
        verify_result: Any = None,
    ) -> Observation:
        return {
            "scenario": self.scenario,
            "task_idx": self.task_idx,
            "task": self._task,
            "last_tool": last_tool,
            "tool_result": _to_plain(tool_result),
            "error": error,
            "reward": reward,
            "done": done,
            "verify_result": _to_plain(verify_result),
        }
    def _normalize_result(
        self,
        result: Any,
        last_tool: str | None,
        default_done: bool,
        default_reward: float,
    ) -> Observation:
        observation_obj = _get_value(result, "observation", result)
        observation = _to_plain(observation_obj)
        if not isinstance(observation, dict):
            observation = {"tool_result": observation}
        reward = _get_value(result, "reward", observation.get("reward", default_reward))
        done = _get_value(result, "done", observation.get("done", default_done))
        tool_name = observation.get("tool_name") or observation.get("last_tool") or last_tool
        normalized = self._base_observation(
            last_tool=tool_name,
            reward=float(reward or 0.0),
            done=bool(done),
            error=observation.get("error"),
            tool_result=observation.get("tool_result"),
            verify_result=observation.get("verify_result"),
        )
        normalized.update(observation)
        normalized["last_tool"] = normalized.get("last_tool") or normalized.get("tool_name") or last_tool
        normalized["reward"] = float(reward or 0.0)
        normalized["done"] = bool(done)
        return normalized
    def _info_from_observation(self, observation: Observation) -> Dict[str, Any]:
        verify_result = observation.get("verify_result")
        success = self._is_successful_verify(verify_result) or observation.get("reward_type") == "complete"
        return {
            "env_type": "awm",
            "scenario": self.scenario,
            "task_idx": self.task_idx,
            "tool_name": observation.get("last_tool") or observation.get("tool_name"),
            "reward": observation.get("reward", 0.0),
            "reward_type": observation.get("reward_type"),
            "success": bool(success),
            "error": observation.get("error"),
            "verify_result": verify_result,
            "trajectory_path": observation.get("trajectory_path"),
            "session_dir": observation.get("session_dir"),
        }
    def _is_successful_verify(self, verify_result: Any) -> bool:
        plain = _to_plain(verify_result)
        if not isinstance(plain, dict):
            return False
        for key in ("success", "passed", "pass", "is_correct", "complete"):
            if plain.get(key) is True:
                return True
        return False
    def _render_action_space(self, tools: Iterable[Any]) -> str:
        blocks = ["Available MCP tools:"]
        for tool in tools:
            name = str(_get_value(tool, "name", "") or "")
            if not name or name in _HIDDEN_AWM_TOOLS:
                continue
            description = _get_value(tool, "description", "") or "No description provided."
            input_schema = (
                _get_value(tool, "input_schema")
                or _get_value(tool, "inputSchema")
                or _get_value(tool, "schema")
                or {}
            )
            blocks.append(
                "\n".join(
                    [
                        f"### {name}",
                        f"Description: {description}",
                        "Input schema:",
                        _json_dumps(input_schema),
                    ]
                )
            )
        blocks.append(
            "\n".join(
                [
                    "### submit",
                    "Description: Run AWM verification when the task is complete.",
                    "Input schema:",
                    _json_dumps(
                        {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "Optional brief final answer or final state summary.",
                                }
                            },
                        }
                    ),
                ]
            )
        )
        return "\n\n".join(blocks)
