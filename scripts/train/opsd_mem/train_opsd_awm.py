#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
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
from opd_evolver.trainer import LifelongSelfDistillationDataCollator, OPSDTrainer
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "workspace" / "awm_runs_scale_3000_memory" / "executor_opd_tasks.jsonl"
)
@dataclass
class AWMScriptArguments(ScriptArguments):
    dataset_path: str = field(
        default=str(DEFAULT_DATASET_PATH),
        metadata={"help": "Task-level AWM OPD JSONL from build_awm_executor_opd_dataset.py."},
    )
    teacher_mode: str = field(default="both", metadata={"help": "gold, memory_only, or both."})
    memory_store_dir: str = field(default="", metadata={"help": "Hierarchical memory dir. Empty disables memory injection."})
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
def load_awm_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"AWM OPD dataset not found: {dataset_path}")
    records: list[dict[str, Any]] = []
    skipped = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            problem = row.get("problem")
            action_space = row.get("action_space")
            solution = row.get("solution")
            if not isinstance(problem, str) or not problem.strip():
                skipped += 1
                continue
            if not isinstance(action_space, str) or not action_space.strip():
                skipped += 1
                continue
            if not isinstance(solution, str) or not solution.strip():
                skipped += 1
                continue
            records.append(
                {
                    "task_id": row.get("task_id"),
                    "task_type": "awm",
                    "problem": problem,
                    "action_space": action_space,
                    "solution": solution,
                    "skill_tags": row.get("skill_tags") or ["awm", "mcp_tool_use"],
                }
            )
    if not records:
        raise ValueError(f"No valid AWM OPD records loaded from {dataset_path}; skipped={skipped}")
    print(f"Loaded AWM OPD rows: valid={len(records)}, skipped={skipped}")
    return Dataset.from_list(records)
class AWMOPSDTrainer(OPSDTrainer):
    _name = "AWM_OPSD"
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in ("task_id", "task_type", "action_space", "skill_tags", "memory_context"):
            if self._signature_columns is not None and col not in self._signature_columns:
                self._signature_columns.append(col)
def main() -> None:
    parser = TrlParser((AWMScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.model_name_or_path = resolve_model_name_or_path(model_args.model_name_or_path)
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires --use_peft")
    if script_args.use_ema_teacher and script_args.fixed_teacher:
        raise ValueError("use_ema_teacher and fixed_teacher are mutually exclusive")
    if not training_args.output_dir:
        training_args.output_dir = "outputs/opsd_awm/executor"
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
    train_dataset = load_awm_dataset(script_args.dataset_path)
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
    trainer = AWMOPSDTrainer(
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
    print("Starting AWM executor OPD")
    print(f"  Dataset: {len(train_dataset)} examples from {script_args.dataset_path}")
    print(f"  Teacher mode: {script_args.teacher_mode}")
    print(f"  Memory: {'per-example' if per_example_memory else ('global' if global_memory_context else 'none')}")
    print(f"  Output: {training_args.output_dir}")
    print("=" * 80)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Saved AWM executor adapter to: {training_args.output_dir}")
if __name__ == "__main__":
    main()
