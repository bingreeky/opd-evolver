from __future__ import annotations
import ast
import asyncio
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from opd_evolver.base.engine.logs import logger
from opd_evolver.benchmark.benchmark import Benchmark, LevelSpec
from opd_evolver.benchmark.common.env import Action, BasicInfo, Environment, Observation
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = PROJECT_ROOT / "reference" / "LifelongAgentBench"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "lifelong" / "processed"
TASK_TYPES = ("db", "os", "kg")
DEFAULT_MAX_STEPS = {"db": 6, "os": 8, "kg": 18}
def _ensure_reference_on_path() -> None:
    if not REFERENCE_ROOT.exists():
        raise FileNotFoundError(
            f"LifelongAgentBench reference repo not found at {REFERENCE_ROOT}. "
            "Keep the reference checkout in place or set up equivalent task dependencies."
        )
    ref = str(REFERENCE_ROOT)
    if ref not in sys.path:
        sys.path.insert(0, ref)
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared LifelongAgentBench split not found: {path}. "
            "Run scripts/dataset/prepare_lifelong_agent_bench.py on the server first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
def _action_names(actions: list[Any]) -> list[str]:
    names: list[str] = []
    for action in actions:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(action))
        if match:
            names.append(match.group(1))
    return sorted(set(names))
def _compact_action_space(task_type: str) -> str:
    if task_type == "db":
        return (
            "Execute SQL against the initialized MySQL database.\n"
            'Actions:\n  - {"action":"execute","params":{"command":"SQL query"}}\n'
            '  - {"action":"submit","params":{}}\n'
            "For SELECT tasks, submit evaluates the latest query output. For INSERT/UPDATE/DELETE "
            "tasks, submit evaluates the final table state."
        )
    if task_type == "os":
        return (
            "Execute bash commands in the initialized Ubuntu container.\n"
            'Actions:\n  - {"action":"execute","params":{"command":"bash command"}}\n'
            '  - {"action":"submit","params":{}}\n'
            "Submit runs the hidden evaluation command."
        )
    if task_type == "kg":
        return (
            "Use Knowledge Graph API calls to answer the question.\n"
            'Actions:\n  - {"action":"execute","params":{"command":"get_relations(entity_or_var)"}}\n'
            '  - {"action":"execute","params":{"command":"get_neighbors(entity_or_var, relation)"}}\n'
            '  - {"action":"execute","params":{"command":"intersection(#0, #1)"}}\n'
            '  - {"action":"execute","params":{"command":"get_attributes(#0)"}}\n'
            '  - {"action":"execute","params":{"command":"argmax(#0, attribute)"}}\n'
            '  - {"action":"execute","params":{"command":"argmin(#0, attribute)"}}\n'
            '  - {"action":"execute","params":{"command":"count(#0)"}}\n'
            '  - {"action":"submit","params":{"answer":"#0"}}\n'
            "Submit a variable index such as #0, or an explicit answer string/list."
        )
    raise ValueError(f"unknown task_type: {task_type}")
def _maybe_parse_obj(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return value
def _run_with_timeout(fn: Any, timeout_s: float, label: str) -> None:
    err: list[BaseException] = []
    def _target() -> None:
        try:
            fn()
        except BaseException as exc:
            err.append(exc)
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        logger.warning(f"{label} timed out after {timeout_s:.1f}s; skipping remaining cleanup.")
        return
    if err:
        raise err[0]
class _DBRuntime:
    def __init__(self, entry: dict[str, Any]):
        _ensure_reference_on_path()
        from src.tasks.instance.db_bench.task import (
            AnswerType,
            DBBench,
            DirectTypeAnswerValidator,
        )
        from src.tasks.instance.db_bench.container import DBBenchContainer
        self.AnswerType = AnswerType
        self.DirectTypeAnswerValidator = DirectTypeAnswerValidator
        cleaned: dict[str, Any] = dict(entry)
        for key in ("answer_info", "table_info", "skill_list"):
            if key in cleaned:
                cleaned[key] = _maybe_parse_obj(cleaned[key])
        answer_info = cleaned.get("answer_info")
        if isinstance(answer_info, dict):
            for sub_key in ("direct", "sql", "md5"):
                if sub_key in answer_info:
                    answer_info[sub_key] = _maybe_parse_obj(answer_info[sub_key])
            cleaned["answer_info"] = answer_info
        table_info = cleaned.get("table_info")
        if isinstance(table_info, dict):
            for sub_key in ("row_list", "column_info_list", "name"):
                if sub_key in table_info:
                    table_info[sub_key] = _maybe_parse_obj(table_info[sub_key])
            cleaned["table_info"] = table_info
        skill_list = cleaned.get("skill_list")
        if isinstance(skill_list, str):
            skill_list = _maybe_parse_obj(skill_list)
        if skill_list is None:
            skill_list = []
        if not isinstance(skill_list, list):
            skill_list = [skill_list]
        cleaned["skill_list"] = skill_list
        self.dataset_item = DBBench._construct_dataset_item(cleaned)
        self.container = DBBenchContainer()
        self.last_output = ""
    def reset(self) -> str:
        _ensure_reference_on_path()
        from src.tasks.instance.db_bench.task import DBBench
        self.container.execute(DBBench._build_init_sql(self.dataset_item))
        return "Database initialized."
    def execute(self, sql: str) -> str:
        self.last_output = self.container.execute(sql, self.dataset_item.database_name)
        return self.last_output
    def submit(self, answer: Optional[str] = None) -> tuple[bool, str]:
        answer_info = self.dataset_item.answer_info
        candidate = self.last_output if answer is None or not str(answer).strip() else str(answer)
        if answer_info.answer_type == self.AnswerType.MD5:
            column_name_str = ",".join(
                f"`{col.name}`" for col in self.dataset_item.table_info.column_info_list
            )
            table_name = self.dataset_item.table_info.name
            md5_query = (
                "select md5(group_concat(rowhash order by rowhash)) as hash "
                f"from( SELECT substring(MD5(CONCAT_WS(',', {column_name_str})), 1, 5) AS rowhash "
                f"FROM `{table_name}`) as sub;"
            )
            candidate = self.container.execute(md5_query, self.dataset_item.database_name)
            match = re.search(r"\('?(.*?)'?,\)", candidate)
            candidate = match.group(1) if match else candidate
            ok = candidate == answer_info.answer_md5
        else:
            ok = self.DirectTypeAnswerValidator.validate(candidate, answer_info.answer_direct)
        return bool(ok), str(candidate)
    def close(self) -> None:
        try:
            self.container.execute(f"drop database `{self.dataset_item.database_name}`")
        except Exception:
            pass
        try:
            _run_with_timeout(self.container.delete, timeout_s=10.0, label="DB container delete")
        except Exception as exc:
            logger.warning(f"DB runtime cleanup failed: {exc}")
def _spawn_db_runtime_and_reset(entry: dict[str, Any]) -> tuple[_DBRuntime, str]:
    runtime = _DBRuntime(entry)
    msg = runtime.reset()
    return runtime, msg
class _OSRuntime:
    def __init__(self, entry: dict[str, Any], timeout: int):
        _ensure_reference_on_path()
        from src.tasks.instance.os_interaction.task import OSInteraction
        from src.tasks.instance.os_interaction.container import OSInteractionContainer
        from src.tasks.instance.os_interaction.utility import (
            CommandItem,
            CommandName,
        )
        self.CommandItem = CommandItem
        self.CommandName = CommandName
        cleaned = dict(entry)
        for key in ("initialization_command_item", "evaluation_info", "skill_list"):
            if key in cleaned:
                cleaned[key] = _maybe_parse_obj(cleaned[key])
        eval_info = cleaned.get("evaluation_info")
        if isinstance(eval_info, dict):
            for sub_key in ("evaluation_command_item", "extra_evaluation_command_item"):
                if sub_key in eval_info:
                    eval_info[sub_key] = _maybe_parse_obj(eval_info[sub_key])
            cleaned["evaluation_info"] = eval_info
        skills = cleaned.get("skill_list")
        if skills is None:
            skills = []
        if isinstance(skills, str):
            skills = _maybe_parse_obj(skills)
        if not isinstance(skills, list):
            skills = [skills]
        cleaned["skill_list"] = skills
        cleaned.setdefault("raw_entry_hash", "")
        self.dataset_item = OSInteraction._construct_dataset_item(cleaned)
        self.container = OSInteractionContainer(timeout)
    def reset(self) -> str:
        result = self.container.execute_independent(self.dataset_item.initialization_command_item)
        if result.timeout_flag or result.exit_code != 0:
            raise RuntimeError(
                f"OS initialization failed: exit={result.exit_code} output={result.output}"
            )
        return "Container initialized."
    def execute(self, command: str) -> str:
        result = self.container.execute_independent(
            self.CommandItem(command_name=self.CommandName.BASH, script=command)
        )
        if result.timeout_flag:
            return "The command timed out."
        return result.output or ""
    def submit(self) -> tuple[bool, str]:
        result = self.container.execute_independent(
            self.dataset_item.evaluation_info.evaluation_command_item
        )
        output = "timeout" if result.timeout_flag else (result.output or "")
        return (not result.timeout_flag and result.exit_code == 0), output
    def close(self) -> None:
        try:
            self.container.terminate()
        except Exception as exc:
            logger.warning(f"OS runtime cleanup failed: {exc}")
class _KGRuntime:
    def __init__(self, entry: dict[str, Any], sparql_url: str, ontology_dir: Path):
        _ensure_reference_on_path()
        if not ontology_dir.exists():
            raise FileNotFoundError(
                f"KG ontology dir missing: {ontology_dir}. "
                "Provide vocab.json/fb_roles before running KG tasks."
            )
        from src.tasks.instance.knowledge_graph.api import (
            KnowledgeGraphAPI,
            KnowledgeGraphAPIException,
            Variable,
        )
        from src.tasks.instance.knowledge_graph.utils.sparql_executor import (
            SparqlExecutor,
        )
        self.Variable = Variable
        self.KnowledgeGraphAPIException = KnowledgeGraphAPIException
        self.api = KnowledgeGraphAPI(str(ontology_dir), SparqlExecutor(sparql_url))
        self.entry = entry
        self.variables: list[Any] = []
        self.last_output = ""
    def reset(self) -> str:
        self.variables = []
        self.api.reset_cache()
        entities = list((self.entry.get("entity_dict") or {}).keys())
        return f"Question: {self.entry.get('question')}\nEntities: {entities}"
    @staticmethod
    def _split_args(arg_text: str) -> list[str]:
        if not arg_text.strip():
            return []
        try:
            parsed = ast.literal_eval(f"({arg_text},)")
            return [str(x) for x in parsed]
        except Exception:
            return [p.strip() for p in arg_text.split(",") if p.strip()]
    def _resolve_arg(self, raw: str) -> Any:
        text = raw.strip()
        entity_dict = self.entry.get("entity_dict") or {}
        if text in entity_dict:
            return entity_dict[text]
        if text.startswith("#"):
            return self.variables[int(text[1:])]
        lowered = text.lower()
        for prefix in ("variable#", "variable #", "var#", "var #"):
            if lowered.startswith(prefix):
                return self.variables[int(lowered[len(prefix) :])]
        return text
    def execute(self, command: str) -> str:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", command)
        if not match:
            return f"Invalid KG API call: {command}"
        api_name, arg_text = match.groups()
        if api_name not in {
            "get_relations",
            "get_neighbors",
            "intersection",
            "get_attributes",
            "argmax",
            "argmin",
            "count",
        }:
            return f"Unknown API name: {api_name}"
        args = [self._resolve_arg(x) for x in self._split_args(arg_text)]
        try:
            new_var, message = getattr(self.api, api_name)(*args)
        except Exception as exc:
            message = f"Error in executing {command}: {exc}"
            self.last_output = message
            return message
        if new_var is not None:
            self.variables.append(new_var)
            message = message.replace("<<NEW_VARIABLE>>", f"#{len(self.variables) - 1}")
        message = message.replace("<<API_STR>>", command)
        self.last_output = message
        return message
    def submit(self, answer: Optional[Any]) -> tuple[bool, str, float]:
        if answer is None or answer == "":
            answer = self.last_output
        if isinstance(answer, str) and answer.strip().startswith("#"):
            try:
                variable = self.variables[int(answer.strip()[1:])]
                answer_list = self.api.final_execute(variable)
            except Exception as exc:
                return False, f"final_execute failed: {exc}", 0.0
        elif isinstance(answer, list):
            answer_list = [str(x) for x in answer]
        else:
            text = str(answer)
            answer_list = [x for x in re.split(r"\s*<SEP>\s*|\s*,\s*", text) if x]
        gold = set(str(x) for x in _as_list(self.entry.get("answer_list")))
        pred = set(answer_list)
        exact = pred == gold
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        f1 = 0.0 if tp == 0 else 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn)))
        return exact, "<SEP>".join(answer_list), f1
    def close(self) -> None:
        self.variables = []
class LifelongAgentEnvironment(Environment):
    def __init__(
        self,
        level: LevelSpec,
        task_type: str,
        entry: dict[str, Any],
        max_steps: int,
        os_timeout: int = 20,
        sparql_url: str = "http://127.0.0.1:3001/sparql",
        ontology_dir: str | Path | None = None,
    ):
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unknown LifelongAgentBench task_type: {task_type}")
        self.level = level
        self.task_type = task_type
        self.entry = entry
        self.max_steps = max_steps
        self.os_timeout = os_timeout
        self.sparql_url = sparql_url
        self.ontology_dir = Path(ontology_dir) if ontology_dir else (
            REFERENCE_ROOT / "data" / "v0121" / "knowledge_graph" / "ontology"
        )
        self.runtime: Any = None
        self.done = False
        self.steps = 0
    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=str(self.level["id"]),
            instruction=self._instruction(),
            action_space=_compact_action_space(self.task_type),
            max_steps=self.max_steps,
            meta_data={
                "task_type": self.task_type,
                "source_index": self.entry.get("source_index"),
                "skill_tags": self.entry.get("stratify_labels")
                or self.entry.get("skill_list")
                or _action_names(_as_list(self.entry.get("action_list"))),
            },
        )
    def _instruction(self) -> str:
        if self.task_type == "db":
            table_raw = self.entry.get("table_info", {})
            table = _maybe_parse_obj(table_raw)
            if not isinstance(table, dict):
                table = {}
            col_raw = table.get("column_info_list", [])
            col_list = _maybe_parse_obj(col_raw)
            if not isinstance(col_list, list):
                col_list = []
            cols = [
                c.get("name")
                for c in col_list
                if isinstance(c, dict) and isinstance(c.get("name"), (str, int, float))
            ]
            suffix = (
                f"Table: {table.get('name')}; columns: {', '.join(str(c) for c in cols if c is not None)}."
            )
            return f"{self.entry.get('instruction', '')}\n{suffix}"
        if self.task_type == "os":
            return str(self.entry.get("instruction", ""))
        entities = list((self.entry.get("entity_dict") or {}).keys())
        return f"Question: {self.entry.get('question')}\nEntities: {entities}"
    async def reset(self, seed: int | None = None) -> Observation:
        del seed
        self.done = False
        self.steps = 0
        if self.task_type == "db":
            self.runtime, message = await asyncio.to_thread(_spawn_db_runtime_and_reset, self.entry)
        elif self.task_type == "os":
            self.runtime = _OSRuntime(self.entry, timeout=self.os_timeout)
            message = self.runtime.reset()
        else:
            self.runtime = _KGRuntime(
                self.entry,
                sparql_url=self.sparql_url,
                ontology_dir=self.ontology_dir,
            )
            message = self.runtime.reset()
        return {
            "message": message,
            "instruction": self._instruction(),
            "current_step": 0,
            "max_steps": self.max_steps,
        }
    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            return {"error": "Environment already finished"}, 0.0, True, {"error": "already_done"}
        self.steps += 1
        action_type = str(action.get("action", ""))
        params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
        if action_type == "execute":
            command = str(params.get("command", "")).strip()
            if not command:
                return {"error": "No command provided"}, 0.0, False, {"error": "no_command"}
            output = self.runtime.execute(command)
            done = self.steps >= self.max_steps
            self.done = done
            return (
                {
                    "command": command,
                    "output": output,
                    "current_step": self.steps,
                    "max_steps": self.max_steps,
                },
                0.0,
                done,
                {"max_steps_reached": done} if done else {},
            )
        if action_type == "submit":
            answer = params.get("answer")
            if self.task_type == "db":
                success, observed = self.runtime.submit(answer=str(answer) if answer is not None else None)
                info = {"submitted": True, "observed": observed}
            elif self.task_type == "os":
                success, observed = self.runtime.submit()
                info = {"submitted": True, "observed": observed}
            else:
                success, observed, f1 = self.runtime.submit(answer)
                info = {"submitted": True, "observed": observed, "f1_score": f1}
            self.done = True
            return (
                {
                    "message": "Solution submitted",
                    "success": success,
                    "output": observed,
                    "current_step": self.steps,
                },
                1.0 if success else 0.0,
                True,
                info,
            )
        if action_type == "finish":
            self.done = True
            return (
                {"message": "Agent finished without submitting.", "current_step": self.steps},
                0.0,
                True,
                {"finished": True, "finish_result": params},
            )
        return {"error": f"Unknown action type: {action_type}"}, 0.0, False, {"error": action_type}
    async def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
class LifelongAgentBenchmark(Benchmark):
    def __init__(
        self,
        task_type: str,
        split: str = "test",
        data_root: str | Path = DEFAULT_DATA_ROOT,
        max_steps: Optional[int] = None,
        max_tasks: Optional[int] = None,
        os_timeout: int = 20,
        sparql_url: str = "http://127.0.0.1:3001/sparql",
        ontology_dir: str | Path | None = None,
    ):
        if task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {task_type}")
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")
        self.task_type = task_type
        self.split = split
        self.data_root = Path(data_root)
        self.max_steps = max_steps or DEFAULT_MAX_STEPS[task_type]
        self.max_tasks = max_tasks
        self.os_timeout = os_timeout
        self.sparql_url = sparql_url
        self.ontology_dir = ontology_dir
        self.rows = _load_jsonl(self.data_root / task_type / f"{split}.jsonl")
        if max_tasks is not None:
            self.rows = self.rows[:max_tasks]
    def list_levels(self) -> List[LevelSpec]:
        return [
            {
                "id": row.get("task_id") or f"{self.task_type}_{idx}",
                "index": idx,
                "task_type": self.task_type,
            }
            for idx, row in enumerate(self.rows)
        ]
    def make_env(self, level: LevelSpec) -> Environment:
        idx = int(level["index"])
        return LifelongAgentEnvironment(
            level=level,
            task_type=self.task_type,
            entry=self.rows[idx],
            max_steps=self.max_steps,
            os_timeout=self.os_timeout,
            sparql_url=self.sparql_url,
            ontology_dir=self.ontology_dir,
        )
