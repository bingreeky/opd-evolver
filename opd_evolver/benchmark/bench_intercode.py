from __future__ import annotations
import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from opd_evolver.base.engine.logs import logger
from opd_evolver.benchmark.benchmark import Benchmark, LevelSpec
from opd_evolver.benchmark.common.env import Action, BasicInfo, Environment, Observation
from opd_evolver.benchmark.common.runner import LevelResult, StepRecord
from intercode.envs import BashEnv, SqlEnv, CTFEnv
from intercode.utils import get_container
from intercode.assets import (
    bash_build_docker, bash_image_name, bash_test_data,
    sql_build_docker, sql_image_name, sql_test_data,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_CLASSES = {
    "bash": BashEnv,
    "sql": SqlEnv,
    "ctf": CTFEnv,
}
DEFAULT_IMAGES = {
    "bash": "intercode-nl2bash",
    "sql": "docker-env-sql",
    "ctf": "intercode-ctf",
}
class _NoopContainer:
    def stop(self) -> None:
        return None
def _configure_local_sql_service(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
) -> None:
    import intercode.envs.ic_env as ic_env
    from intercode.envs.sql import sql_env
    sql_env.SQL_CONFIG.update(
        {
            "host": host,
            "port": int(port),
            "user": user,
            "password": password,
        }
    )
    ic_env.get_container = lambda container_name, image_name: _NoopContainer()
def get_action_space_description(env_type: str) -> str:
    if env_type == "bash":
        return (
            "Execute bash commands to complete the task.\n"
            "Actions:\n"
            '  - {"action": "execute", "params": {"command": "your bash command"}}\n'
            '  - {"action": "submit", "params": {}}\n'
            '  - {"action": "finish", "params": {"status": "done|partial|blocked", "message": "..."}}\n'
            "\nExamples: ls, grep, awk, find, cat, echo, etc."
        )
    elif env_type == "sql":
        return (
            "Execute SQL queries to complete the task.\n"
            "Actions:\n"
            '  - {"action": "execute", "params": {"command": "your SQL query"}}\n'
            '  - {"action": "submit", "params": {}}\n'
            '  - {"action": "finish", "params": {"status": "done|partial|blocked", "message": "..."}}\n'
            "\nExamples: SELECT, INSERT, UPDATE, DELETE, SHOW TABLES, etc."
        )
    elif env_type == "python":
        return (
            "Execute Python code to complete the task.\n"
            "Actions:\n"
            '  - {"action": "execute", "params": {"command": "your Python code"}}\n'
            '  - {"action": "submit", "params": {}}\n'
            '  - {"action": "finish", "params": {"status": "done|partial|blocked", "message": "..."}}\n'
            "\nWrite complete Python functions or code blocks."
        )
    elif env_type == "ctf":
        return (
            "Solve CTF (Capture The Flag) challenges.\n"
            "Actions:\n"
            '  - {"action": "execute", "params": {"command": "your command or answer"}}\n'
            '  - {"action": "submit", "params": {}}\n'
            '  - {"action": "finish", "params": {"status": "done|partial|blocked", "message": "..."}}\n'
            "\nSubmit flags or execute commands to find them."
        )
    return "Execute commands to complete the task."
class InterCodeEnvironment(Environment):
    def __init__(
        self,
        level: LevelSpec,
        env_type: str,
        image_name: str,
        data_path: str | None,
        traj_dir: str | None,
        verbose: bool,
        max_steps: int,
        ctf_tasks: List[Dict] | None = None,
        sql_service_mode: str = "docker",
        sql_host: str = "127.0.0.1",
        sql_port: int = 3307,
        sql_user: str = "admin",
        sql_password: str = "admin",
    ):
        self.level = level
        self.env_type = env_type
        self.max_steps = max_steps
        self.task_idx = level["index"]
        self.task_id = level["id"]
        self.sql_service_mode = (sql_service_mode or "docker").strip().lower()
        if self.sql_service_mode not in {"docker", "local"}:
            raise ValueError(f"Unsupported sql_service_mode: {sql_service_mode!r}")
        env_cls = ENV_CLASSES.get(env_type)
        if not env_cls:
            raise ValueError(f"Unknown env_type: {env_type}")
        if env_type == "bash":
            bash_build_docker()
            image_name = image_name or bash_image_name
            data_path = data_path or bash_test_data
        elif env_type == "sql":
            if self.sql_service_mode == "docker":
                sql_build_docker()
            image_name = image_name or sql_image_name
            data_path = data_path or sql_test_data
        unique_id = uuid.uuid4().hex[:8]
        unique_container_name = f"{image_name}_ic_ctr_{self.task_idx}_{unique_id}"
        if env_type == "ctf":
            if ctf_tasks and self.task_idx < len(ctf_tasks):
                task_data = ctf_tasks[self.task_idx]
                self._query = task_data.get("query", f"CTF Challenge {self.task_idx}")
                self._gold = task_data.get("gold", "")
                self._tags = task_data.get("tags", [])
                self._setup = task_data.get("setup", "")
                if self._setup:
                    self._query = f"SETUP: First run this command to prepare the environment:\n{self._setup}\n\nTASK: {self._query}"
            else:
                raise ValueError(f"CTF task {self.task_idx} not found in ctf_tasks")
            task = {"query": self._query, "gold": self._gold}
            self._ic_env = env_cls(
                image_name=image_name,
                task=task,
                traj_dir=traj_dir,
                verbose=verbose,
            )
            try:
                if hasattr(self._ic_env, 'container') and self._ic_env.container:
                    self._ic_env.container_name = unique_container_name
                    self._ic_env.container = get_container(unique_container_name, image_name)
                    logger.info(f"CTF task {self.task_idx}: using container {unique_container_name}")
            except Exception as e:
                logger.warning(f"Failed to create unique container, using default: {e}")
            workdir = f"/ctf/{self.task_idx}"
            self._ic_env.workdir = workdir
            logger.info(f"CTF task {self.task_idx}: workdir={workdir}, gold={self._gold[:20]}...")
        else:
            if env_type == "sql" and self.sql_service_mode == "local":
                _configure_local_sql_service(
                    host=sql_host,
                    port=sql_port,
                    user=sql_user,
                    password=sql_password,
                )
            self._ic_env = env_cls(
                image_name=image_name,
                data_path=data_path,
                traj_dir=traj_dir,
                verbose=verbose,
            )
            if env_type == "bash":
                try:
                    if hasattr(self._ic_env, 'container') and self._ic_env.container:
                        self._ic_env.container_name = unique_container_name
                        self._ic_env.container = get_container(unique_container_name, image_name)
                        logger.info(f"BASH task {self.task_idx}: using container {unique_container_name}")
                        if hasattr(self._ic_env, 'container_eval'):
                            eval_container_name = f"{image_name}_ic_ctr_eval_{self.task_idx}_{unique_id}"
                            self._ic_env.ctr_name_eval = eval_container_name
                            self._ic_env.container_eval = get_container(eval_container_name, image_name)
                            logger.info(f"BASH task {self.task_idx}: using eval container {eval_container_name}")
                except Exception as e:
                    logger.warning(f"Failed to create unique container for bash, using default: {e}")
            task_data = self._ic_env.data_loader.data.iloc[self.task_idx]
            self._query = task_data.get("query",
                f"Complete InterCode {self.env_type} task {self.task_idx}")
            if env_type == "sql" and "db" in task_data and task_data["db"]:
                db_name = task_data["db"]
                self._query = f"DATABASE: {db_name}\n\nQUERY: {self._query}"
                logger.info(f"SQL task {self.task_idx}: database={db_name}")
        self._done = False
        self._steps = 0
        self._submitted = False
        self._observation = None
        self._container_started = True
    def get_basic_info(self) -> BasicInfo:
        meta = {
            "env_type": self.env_type,
            "task_index": self.task_idx,
        }
        try:
            if hasattr(self, "_ic_env") and hasattr(self._ic_env, "data_loader"):
                task_data = self._ic_env.data_loader.data.iloc[self.task_idx]
                hardness = task_data.get("hardness")
                if hardness:
                    meta["hardness"] = str(hardness)
        except Exception:
            pass
        return BasicInfo(
            env_id=self.task_id,
            instruction=self._query or f"Complete InterCode {self.env_type} task {self.task_idx}",
            action_space=get_action_space_description(self.env_type),
            max_steps=self.max_steps,
            meta_data=meta,
        )
    async def reset(self, seed: int | None = None) -> Observation:
        self._done = False
        self._steps = 0
        self._submitted = False
        if self.env_type == "ctf":
            self._observation = self._query
        else:
            obs, info = self._ic_env.reset(index=self.task_idx)
            self._observation = obs
            if hasattr(self._ic_env, 'query'):
                self._query = self._ic_env.query
        return {
            "message": "Environment ready.",
            "instruction": self._query,
            "observation": self._observation,
            "current_step": 0,
            "max_steps": self.max_steps,
        }
    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self._steps += 1
        action_type = action.get("action", "")
        params = action.get("params", {})
        if self._done and action_type != "submit":
            return (
                {"error": "Environment already finished"},
                0.0,
                True,
                {"error": "already_done"},
            )
        if action_type == "execute":
            command = params.get("command", "")
            if not command:
                return (
                    {"error": "No command provided"},
                    0.0,
                    False,
                    {"error": "no_command"},
                )
            obs, reward, done, info = self._ic_env.step(command)
            self._observation = obs
            if self._steps >= self.max_steps:
                self._done = True
                return (
                    {
                        "command": command,
                        "output": obs,
                        "current_step": self._steps,
                        "max_steps": self.max_steps,
                        "message": "Max steps reached",
                    },
                    0.0,
                    True,
                    {"max_steps_reached": True},
                )
            return (
                {
                    "command": command,
                    "output": obs,
                    "current_step": self._steps,
                    "max_steps": self.max_steps,
                },
                0.0,
                False,
                {"action_exec": info.get("ACTION_EXEC", True)},
            )
        elif action_type == "submit":
            if self.env_type == "ctf":
                flag = params.get("flag", "")
                if flag and not (flag.startswith("picoCTF{") and flag.endswith("}")):
                    logger.warning(
                        f"[InterCodeEnvironment] Invalid flag format: {flag}. "
                        f"Expected 'picoCTF{ ...} ' format. Attempting to fix..."
                    )
                    if not flag.startswith("picoCTF{"):
                        flag = f"picoCTF{ {flag}} "
                    elif not flag.endswith("}"):
                        flag = flag + "}"
                    logger.info(f"[InterCodeEnvironment] Corrected flag to: {flag}")
                if not flag and self._observation:
                    obs_str = str(self._observation)
                    if "picoCTF{" in obs_str:
                        start = obs_str.find("picoCTF{")
                        end = obs_str.find("}", start) + 1
                        if end > start:
                            flag = obs_str[start:end]
                            logger.info(f"[InterCodeEnvironment] Extracted flag from output: {flag}")
                if not flag:
                    flag = ""
                    logger.warning(
                        "[InterCodeEnvironment] No flag provided in submit params and unable to extract from output. "
                        "Submitting empty flag (will be marked incorrect)."
                    )
                submit_cmd = f"submit {flag}"
            else:
                submit_cmd = "submit"
            logger.info(f">>> Submitting solution with command: {submit_cmd}")
            obs, reward, done, info = self._ic_env.step(submit_cmd)
            self._done = True
            self._submitted = True
            return (
                {
                    "message": "Solution submitted",
                    "output": obs,
                    "reward": reward,
                    "current_step": self._steps,
                },
                float(reward),
                True,
                {"submitted": True, "reward": reward},
            )
        elif action_type == "finish":
            status = params.get("status", "done")
            message = params.get("message", "")
            completed = params.get("completed", [])
            issues = params.get("issues", [])
            finish_result = {
                "status": status,
                "message": message,
                "completed": completed,
                "issues": issues,
            }
            self._done = True
            return (
                {
                    "message": f"Subtask finished: {status}",
                    "finish_result": finish_result,
                    "current_step": self._steps,
                },
                0.0,
                True,
                {"finished": True, "finish_result": finish_result},
            )
        else:
            return (
                {"error": f"Unknown action type: {action_type}"},
                0.0,
                False,
                {"error": f"unknown_action: {action_type}"},
            )
    async def close(self):
        try:
            if self.env_type == "sql":
                if hasattr(self._ic_env, 'cur'):
                    self._ic_env.cur.close()
                if hasattr(self._ic_env, 'cnx'):
                    self._ic_env.cnx.close()
            else:
                self._ic_env.close()
        except Exception as e:
            logger.warning(f"Failed to close InterCode env: {e}")
class InterCodeBenchmark(Benchmark):
    def __init__(
        self,
        env_type: str = "bash",
        image_name: str | None = None,
        data_path: str | None = None,
        traj_dir: str | None = None,
        verbose: bool = False,
        max_steps: int = 15,
        max_tasks: int | None = None,
        sql_service_mode: str = "docker",
        sql_host: str = "127.0.0.1",
        sql_port: int = 3307,
        sql_user: str = "admin",
        sql_password: str = "admin",
    ):
        self.env_type = env_type
        self.image_name = image_name or DEFAULT_IMAGES.get(env_type)
        self.data_path = data_path
        self.traj_dir = traj_dir
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        self.sql_service_mode = (sql_service_mode or "docker").strip().lower()
        if self.sql_service_mode not in {"docker", "local"}:
            raise ValueError(f"Unsupported sql_service_mode: {sql_service_mode!r}")
        self.sql_host = sql_host
        self.sql_port = int(sql_port)
        self.sql_user = sql_user
        self.sql_password = sql_password
        self._ctf_tasks: List[Dict] = []
        self._sql_task_ids: List[str] | None = None
        self._init_data_loader()
    def _init_data_loader(self):
        env_cls = ENV_CLASSES.get(self.env_type)
        if not env_cls:
            raise ValueError(f"Unknown env_type: {self.env_type}")
        if self.env_type == "bash":
            bash_build_docker()
            image_name = self.image_name or bash_image_name
            data_path = self.data_path or bash_test_data
        elif self.env_type == "sql":
            if self.sql_service_mode == "docker":
                sql_build_docker()
            image_name = self.image_name or sql_image_name
            data_path = self.data_path or sql_test_data
            from intercode.utils import IntercodeDataLoader
            loader = IntercodeDataLoader(data_path)
            self._task_count = len(loader)
            self._resolved_image = image_name
            self._resolved_data_path = data_path
            try:
                df = getattr(loader, "data", None)
                if df is not None and "id" in df.columns:
                    raw = df["id"].tolist()
                    if len(raw) == self._task_count and all(
                        x is not None and str(x).strip() for x in raw
                    ):
                        self._sql_task_ids = [str(x).strip() for x in raw]
            except Exception as exc:
                logger.debug(f"SQL task ids from column 'id' not used: {exc}")
                self._sql_task_ids = None
            return
        elif self.env_type == "ctf":
            image_name = self.image_name or "intercode-ctf"
            data_path = self.data_path or str(PROJECT_ROOT / "data" / "ctf" / "ic_ctf.json")
            json_path = Path(data_path)
            if json_path.exists():
                with open(json_path, "r") as f:
                    self._ctf_tasks = json.load(f)
                self._task_count = len(self._ctf_tasks)
                logger.info(f"Loaded {self._task_count} CTF tasks from {json_path}")
            else:
                logger.warning(f"CTF data file not found: {json_path}, using default count")
                self._task_count = 50
            self._resolved_image = image_name
            self._resolved_data_path = data_path
            return
        else:
            image_name = self.image_name
            data_path = self.data_path
        temp_env = env_cls(
            image_name=image_name,
            data_path=data_path,
            verbose=False,
        )
        self._task_count = len(temp_env.data_loader)
        temp_env.close()
        self._resolved_image = image_name
        self._resolved_data_path = data_path
    def list_levels(self) -> List[LevelSpec]:
        count = self._task_count
        if self.max_tasks:
            count = min(count, self.max_tasks)
        def _level_id(i: int) -> str:
            if self.env_type == "sql" and self._sql_task_ids is not None:
                return self._sql_task_ids[i]
            return f"{self.env_type}_{i}"
        return [
            {
                "id": _level_id(i),
                "index": i,
            }
            for i in range(count)
        ]
    def make_env(self, level: LevelSpec) -> Environment:
        return InterCodeEnvironment(
            level=level,
            env_type=self.env_type,
            image_name=self._resolved_image,
            data_path=self._resolved_data_path,
            traj_dir=self.traj_dir,
            verbose=self.verbose,
            max_steps=self.max_steps,
            ctf_tasks=self._ctf_tasks if self.env_type == "ctf" else None,
            sql_service_mode=self.sql_service_mode,
            sql_host=self.sql_host,
            sql_port=self.sql_port,
            sql_user=self.sql_user,
            sql_password=self.sql_password,
        )
