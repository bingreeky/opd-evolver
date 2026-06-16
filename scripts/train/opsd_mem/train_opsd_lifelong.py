#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import ast
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opd_evolver.base.hf_snapshot import resolve_model_name_or_path
from opd_evolver.benchmark.bench_lifelong_agent import _compact_action_space
from opd_evolver.trainer import LifelongSelfDistillationDataCollator, OPSDTrainer
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "lifelong_agent_bench" / "processed"
@dataclass
class LifelongScriptArguments(ScriptArguments):
    data_root: str = field(
        default=str(DEFAULT_DATA_ROOT),
        metadata={"help": "Prepared LifelongAgentBench data root."},
    )
    split: str = field(default="train", metadata={"help": "train, test, or all."})
    task_types: str = field(
        default="db,os,kg",
        metadata={"help": "Comma-separated subset of db,os,kg."},
    )
    teacher_mode: str = field(
        default="gold",
        metadata={"help": "gold, memory_only, or both."},
    )
    memory_store_dir: str = field(
        default="",
        metadata={"help": "Hierarchical memory dir. Empty disables memory injection."},
    )
    memory_top_k: int = field(default=50, metadata={"help": "Global top-k memory fallback."})
    memory_retrieve_k: int = field(default=100, metadata={"help": "Per-example retrieve-k."})
    memory_select_k: int = field(default=10, metadata={"help": "Per-example select-k."})
    memory_embedding_model: str = field(default="Qwen/Qwen3-Embedding-0.6B")
    memory_min_score: float = field(default=0.01)
    max_train_samples: int = field(default=0, metadata={"help": "0 means all."})
    fixed_teacher: bool = field(default=True)
    use_ema_teacher: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    use_tinker_loss: bool = field(default=False)
    top_k_loss: int = field(default=0)
    jsd_token_clip: float = field(default=0.05)
    reason_first: bool = field(default=False)
    run_config: str | None = field(default=None)
def _load_sql_memory_helpers() -> tuple[Any, Any]:
    path = PROJECT_ROOT / "scripts" / "train" / "opsd_mem" / "train_opsd_sql.py"
    spec = importlib.util.spec_from_file_location("train_opsd_sql_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import SQL memory helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_top_memories, module.build_per_example_memory_contexts
def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared split not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
def _maybe_parse_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
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
def _db_problem(row: dict[str, Any]) -> str:
    table = _maybe_parse_obj(row.get("table_info", {})) or {}
    columns = table.get("column_info_list", []) if isinstance(table, dict) else []
    col_text = ", ".join(
        f"{c.get('name')} {c.get('type')}" for c in columns if isinstance(c, dict)
    )
    return (
        f"{row.get('instruction', '')}\n"
        f"Table: {table.get('name') if isinstance(table, dict) else ''}\n"
        f"Columns: {col_text}"
    )
def _db_solution(row: dict[str, Any]) -> str:
    answer_info = _maybe_parse_obj(row.get("answer_info")) or {}
    sql = ((answer_info.get("sql") if isinstance(answer_info, dict) else "") or "").strip()
    return json.dumps(
        [
            {"action": "execute", "params": {"command": sql}},
            {"action": "submit", "params": {}},
        ],
        ensure_ascii=False,
    )
def _command_script(command_item: Any) -> str:
    command_item = _maybe_parse_obj(command_item)
    if isinstance(command_item, dict):
        return str(command_item.get("script", ""))
    return str(command_item or "")
def _os_solution(row: dict[str, Any]) -> str:
    eval_info = _maybe_parse_obj(row.get("evaluation_info")) or {}
    gt = eval_info.get("ground_truth_command_item") if isinstance(eval_info, dict) else None
    script = _command_script(gt)
    return json.dumps(
        [
            {"action": "execute", "params": {"command": script}},
            {"action": "submit", "params": {}},
        ],
        ensure_ascii=False,
    )
def _kg_problem(row: dict[str, Any]) -> str:
    entities = list((row.get("entity_dict") or {}).keys())
    return f"Question: {row.get('question')}\nEntities: {entities}"
def _kg_solution(row: dict[str, Any]) -> str:
    actions = [
        {"action": "execute", "params": {"command": str(action)}}
        for action in row.get("action_list", [])
    ]
    if actions:
        actions.append({"action": "submit", "params": {"answer": f"#{len(actions) - 1}"}})
    else:
        actions.append({"action": "submit", "params": {"answer": row.get("answer_list", [])}})
    return json.dumps(actions, ensure_ascii=False)
def _problem_and_solution(task_type: str, row: dict[str, Any]) -> tuple[str, str]:
    if task_type == "db":
        return _db_problem(row), _db_solution(row)
    if task_type == "os":
        return str(row.get("instruction", "")), _os_solution(row)
    if task_type == "kg":
        return _kg_problem(row), _kg_solution(row)
    raise ValueError(f"Unknown task_type: {task_type}")
def load_lifelong_dataset(data_root: str | Path, split: str, task_types: list[str]) -> Dataset:
    records: list[dict[str, Any]] = []
    for task_type in task_types:
        for row in _jsonl_rows(Path(data_root) / task_type / f"{split}.jsonl"):
            problem, solution = _problem_and_solution(task_type, row)
            records.append(
                {
                    "task_id": row.get("task_id"),
                    "task_type": task_type,
                    "problem": problem,
                    "action_space": _compact_action_space(task_type),
                    "solution": solution,
                    "skill_tags": row.get("stratify_labels") or row.get("skill_list") or [],
                }
            )
    if not records:
        raise ValueError(f"No LifelongAgentBench records loaded from {data_root} split={split}")
    return Dataset.from_list(records)
class LifelongOPSDTrainer(OPSDTrainer):
    _name = "LIFELONG_OPSD"
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in ("task_id", "task_type", "action_space", "skill_tags", "memory_context"):
            if self._signature_columns is not None and col not in self._signature_columns:
                self._signature_columns.append(col)
def main() -> None:
    parser = TrlParser((LifelongScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.model_name_or_path = resolve_model_name_or_path(model_args.model_name_or_path)
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires --use_peft")
    if script_args.use_ema_teacher and script_args.fixed_teacher:
        raise ValueError("use_ema_teacher and fixed_teacher are mutually exclusive")
    if not training_args.output_dir:
        training_args.output_dir = "outputs/opsd_lifelong"
    if script_args.run_config:
        training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    report_to = training_args.report_to
    report_seq = report_to if isinstance(report_to, (list, tuple)) else ([report_to] if report_to else [])
    if report_seq and ("wandb" in report_seq or "all" in report_seq):
        if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = "offline"
    model_kwargs: dict[str, Any] = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation or "sdpa",
        "torch_dtype": torch.bfloat16,
    }
    qconfig = get_quantization_config(model_args)
    if qconfig is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = qconfig
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    task_types = [x.strip() for x in script_args.task_types.split(",") if x.strip()]
    invalid = sorted(set(task_types) - {"db", "os", "kg"})
    if invalid:
        raise ValueError(f"Unknown task_types: {invalid}")
    train_dataset = load_lifelong_dataset(script_args.data_root, script_args.split, task_types)
    if script_args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(script_args.max_train_samples, len(train_dataset))))
    per_example_memory = False
    global_memory_context = ""
    if script_args.memory_store_dir:
        load_top_memories, build_per_example_memory_contexts = _load_sql_memory_helpers()
        if script_args.memory_retrieve_k > 0:
            contexts = build_per_example_memory_contexts(
                problems=train_dataset["problem"],
                store_dir=script_args.memory_store_dir,
                retrieve_k=script_args.memory_retrieve_k,
                select_k=script_args.memory_select_k,
                embedding_model=script_args.memory_embedding_model,
                min_score=script_args.memory_min_score,
            )
            train_dataset = train_dataset.add_column("memory_context", contexts)
            per_example_memory = True
        else:
            global_memory_context = load_top_memories(
                script_args.memory_store_dir,
                top_k=script_args.memory_top_k,
                min_score=script_args.memory_min_score,
            )
    data_collator = LifelongSelfDistillationDataCollator(
        tokenizer=tokenizer,
        memory_context="" if per_example_memory else global_memory_context,
        max_length=training_args.max_length,
        teacher_mode=script_args.teacher_mode,
    )
    trainer = LifelongOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
    )
    print("=" * 80)
    print("Starting LifelongAgentBench executor OPSD")
    print(f"  Dataset: {len(train_dataset)} examples from {script_args.data_root} split={script_args.split}")
    print(f"  Task types: {', '.join(task_types)}")
    print(f"  Teacher mode: {script_args.teacher_mode}")
    print(f"  Memory: {'per-example' if per_example_memory else ('global' if global_memory_context else 'none')}")
    print(f"  Output: {training_args.output_dir}")
    print("=" * 80)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Saved LifelongAgentBench executor adapter to: {training_args.output_dir}")
if __name__ == "__main__":
    main()
