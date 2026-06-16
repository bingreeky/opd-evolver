from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Callable
class StepLike(Protocol):
    action: Dict[str, Any]
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]
class ActionFormatter(ABC):
    @property
    @abstractmethod
    def action_type(self) -> str:
        ...
    @abstractmethod
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        ...
class ObservationFormatter(ABC):
    @abstractmethod
    def can_format(self, obs: Dict[str, Any]) -> bool:
        ...
    @abstractmethod
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        ...
class ExecuteActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "execute"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        cmd = params.get("command", "")[:max_len]
        return f'execute(command="{cmd}")'
class FinishActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "finish"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        status = params.get("status", "done")
        msg = params.get("message", "")[:60]
        return f'finish(status="{status}", msg="{msg}")'
class SubmitActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "submit"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return "submit()"
class ExitCodeObservationFormatter(ObservationFormatter):
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "exit_code" in obs
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        exit_code = obs.get("exit_code", "N/A")
        output = str(obs.get("output", ""))
        return f"exit_code={exit_code}", output
class GoogleSearchActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "GoogleSearchAction"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        query = params.get("query", "")[:80]
        return f'GoogleSearch(query="{query}")'
class ExtractUrlActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "ExtractUrlContentAction"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        url = params.get("url", "")[:60]
        browse_query = params.get("browse_query", "")[:40]
        return f'ExtractUrl(url="{url}", query="{browse_query}")'
class ExecuteCodeActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "ExecuteCodeAction"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        code = params.get("code", "")[:80].replace("\n", " ")
        return f'ExecuteCode(code="{code}...")'
class SuccessObservationFormatter(ObservationFormatter):
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "success" in obs
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        success = obs.get("success", False)
        output = str(obs.get("output", obs.get("error", "")))
        return f"success={success}", output
class ACICommandActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "aci_command"
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        cmd = params.get("command", "")
        if "\n" in cmd:
            first_line = cmd.split("\n")[0][:60]
            return f'aci_command("{first_line}...")'
        return f'aci_command("{cmd[:max_len]}")'
class SWEBenchObservationFormatter(ObservationFormatter):
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "state_info" in obs or "command" in obs
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        state_info = obs.get("state_info", "")
        output = str(obs.get("output", ""))
        exit_code = obs.get("exit_code", "N/A")
        return f"exit_code={exit_code}, {state_info}", output
class FallbackActionFormatter(ActionFormatter):
    def __init__(self, action_type: str = "unknown"):
        self._action_type = action_type
    @property
    def action_type(self) -> str:
        return self._action_type
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        param_keys = list(params.keys())[:3]
        return f'{self._action_type}({param_keys})'
class FallbackObservationFormatter(ObservationFormatter):
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return True
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        return "", str(obs)
class TraceFormatter:
    def __init__(self):
        self._action_formatters: Dict[str, ActionFormatter] = {}
        self._obs_formatters: List[ObservationFormatter] = []
        self._fallback_obs_formatter = FallbackObservationFormatter()
    def register_action_formatter(self, formatter: ActionFormatter) -> "TraceFormatter":
        self._action_formatters[formatter.action_type] = formatter
        return self
    def register_obs_formatter(self, formatter: ObservationFormatter) -> "TraceFormatter":
        self._obs_formatters.append(formatter)
        return self
    def format_action(self, action: Dict[str, Any], max_len: int = 100) -> str:
        action_type = action.get("action", "unknown")
        params = action.get("params", {})
        formatter = self._action_formatters.get(action_type)
        if formatter:
            return formatter.format(params, max_len)
        return FallbackActionFormatter(action_type).format(params, max_len)
    def format_observation(self, obs: Any, max_len: int = 300) -> tuple[str, str]:
        if not isinstance(obs, dict):
            return "", str(obs)
        for formatter in self._obs_formatters:
            if formatter.can_format(obs):
                return formatter.format(obs, max_len)
        return self._fallback_obs_formatter.format(obs, max_len)
    def format_trace(self, trace: List[StepLike], max_output_len: int = 300) -> str:
        if not trace:
            return "No steps executed"
        lines = []
        for i, step in enumerate(trace, 1):
            action_str = self.format_action(step.action)
            lines.append(f"Step {i}: {action_str}")
            status_line, output = self.format_observation(step.observation, max_output_len)
            if status_line:
                lines.append(f"  → {status_line}")
            if len(output) > max_output_len:
                output = output[:max_output_len] + f"...[+{len(output)-max_output_len} chars]"
            output = output.replace("\n", " ").strip()
            lines.append(f"  → output: {output}")
            lines.append("")
        return "\n".join(lines)
def create_terminalbench_formatter() -> TraceFormatter:
    return (
        TraceFormatter()
        .register_action_formatter(ExecuteActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_action_formatter(SubmitActionFormatter())
        .register_obs_formatter(ExitCodeObservationFormatter())
    )
def create_gaia_formatter() -> TraceFormatter:
    return (
        TraceFormatter()
        .register_action_formatter(GoogleSearchActionFormatter())
        .register_action_formatter(ExtractUrlActionFormatter())
        .register_action_formatter(ExecuteCodeActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_action_formatter(SubmitActionFormatter())
        .register_obs_formatter(SuccessObservationFormatter())
    )
def create_swebench_formatter() -> TraceFormatter:
    return (
        TraceFormatter()
        .register_action_formatter(ACICommandActionFormatter())
        .register_action_formatter(ExecuteActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_action_formatter(SubmitActionFormatter())
        .register_obs_formatter(SWEBenchObservationFormatter())
        .register_obs_formatter(ExitCodeObservationFormatter())
    )
def create_universal_formatter() -> TraceFormatter:
    return (
        TraceFormatter()
        .register_action_formatter(ExecuteActionFormatter())
        .register_action_formatter(ACICommandActionFormatter())
        .register_action_formatter(GoogleSearchActionFormatter())
        .register_action_formatter(ExtractUrlActionFormatter())
        .register_action_formatter(ExecuteCodeActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_action_formatter(SubmitActionFormatter())
        .register_obs_formatter(SWEBenchObservationFormatter())
        .register_obs_formatter(ExitCodeObservationFormatter())
        .register_obs_formatter(SuccessObservationFormatter())
    )
