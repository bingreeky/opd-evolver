#!/usr/bin/env python3
from __future__ import annotations
import json
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
from opd_evolver.trainer import OPSDTrainer
from opd_evolver.trainer.writer_collator import WriterSelfDistillationDataCollator
@dataclass
class WriterScriptArguments(ScriptArguments):
    dataset_path: str = field(
        default="workspace/memory/sql/memory_writer_dataset.jsonl",
        metadata={"help": "Path to writer JSONL (possibly scored)."},
    )
    topk_ratio: float = field(
        default=1.0,
        metadata={"help": "Keep top ratio of samples by score (0 < r <= 1)."},
    )
    topk_samples: int = field(
        default=0,
        metadata={"help": "Keep top-k samples by score after topk_ratio (0=disabled)."},
    )
    min_score: float = field(
        default=-1e9,
        metadata={"help": "Drop rows with score < min_score."},
    )
    hard_score_threshold: float = field(
        default=-1e9,
        metadata={"help": "Optional extra gate for hard subset; set > -1e9 to enable."},
    )
    max_trace_chars: int = field(
        default=12000,
        metadata={"help": "Max chars kept for execution trace in prompt."},
    )
    max_train_samples: int = field(
        default=0,
        metadata={"help": "Use first N samples after filtering. 0 means all."},
    )
    fixed_teacher: bool = field(
        default=True,
        metadata={
            "help": "Use initial policy as teacher (requires --use_peft).",
        },
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={"help": "Use EMA teacher (mutually exclusive with fixed_teacher)."},
    )
    ema_decay: float = field(default=0.999, metadata={"help": "EMA decay."})
    use_tinker_loss: bool = field(
        default=False,
        metadata={"help": "Use reverse-KL (Thinking-Machines style) instead of JSD."},
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict JSD to top-k teacher tokens. 0 = full vocab."},
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Per-token JSD clamp. 0 = no clipping."},
    )
def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
def _format_trace(trace: Any, max_chars: int) -> str:
    if not isinstance(trace, list):
        return "(no execution trace)"
    lines: list[str] = []
    for idx, step in enumerate(trace, 1):
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        obs = step.get("observation")
        rew = step.get("reward")
        lines.append(f"Step {idx} action={str(action)[:160]} reward={rew}")
        lines.append(f"obs={str(obs)[:240]}")
        lines.append("")
    txt = "\n".join(lines) if lines else "(empty execution trace)"
    if len(txt) > max_chars:
        txt = txt[:max_chars] + f"\n... [TRUNCATED {len(txt) - max_chars} chars]"
    return txt
def _normalize_writer_row(row: dict[str, Any], max_trace_chars: int) -> dict[str, Any] | None:
    in_block = row.get("input")
    if not isinstance(in_block, dict):
        return None
    task_description = str(in_block.get("task_description", "")).strip()
    if not task_description:
        return None
    execution_trace = in_block.get("execution_trace", [])
    trace_text = _format_trace(execution_trace, max_chars=max_trace_chars)
    filtered_context_summary = str(in_block.get("filtered_context_summary", "")).strip()
    raw_solution = row.get("output_memory")
    parsed_solution = row.get("parsed_reflection")
    if isinstance(raw_solution, str) and raw_solution.strip():
        solution = raw_solution.strip()
    elif isinstance(parsed_solution, dict):
        solution = _compact_json(parsed_solution)
    else:
        return None
    success = bool(in_block.get("success", False))
    total_reward = float(in_block.get("total_reward", 0.0))
    score = row.get("score")
    if score is None:
        score = 1.0 if success else 0.0
    score = float(score)
    created_ids = row.get("created_memory_ids", {})
    privileged = _compact_json(
        {
            "success": success,
            "total_reward": total_reward,
            "score": score,
            "created_memory_ids": created_ids,
            "quality_rubric": {
                "grounded": "memory should align with observed trajectory",
                "actionable": "prefer reusable, concise items",
                "safety": "avoid speculative tools/code",
            },
        }
    )
    context_part = (
        f"\n\nRetrieved Memory Context Summary:\n{filtered_context_summary}"
        if filtered_context_summary
        else ""
    )
    problem = (
        f"Task Description:\n{task_description}\n"
        f"\nOutcome: success={success}, total_reward={total_reward}"
        f"{context_part}\n\n"
        f"Execution Trace (truncated):\n{trace_text}\n\n"
        "Output strict JSON with keys: new_skills, new_tips, new_tools, key_learnings, "
        "should_save_trajectory, trajectory_outcome."
    )
    return {
        "problem": problem,
        "solution": solution,
        "privileged": privileged,
        "score": score,
    }
def load_writer_dataset(path: str | Path, max_trace_chars: int) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Writer dataset not found: {path}")
    records: list[dict[str, Any]] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rec = _normalize_writer_row(row, max_trace_chars=max_trace_chars)
            if rec is None:
                bad += 1
                continue
            records.append(rec)
    if not records:
        raise ValueError(f"No valid writer records in {path}")
    print(f"Loaded writer rows: valid={len(records)}, skipped={bad}")
    return Dataset.from_list(records)
def _filter_dataset(dataset: Dataset, args: WriterScriptArguments) -> Dataset:
    rows = list(dataset)
    rows = [r for r in rows if float(r.get("score", 0.0)) >= args.min_score]
    if args.hard_score_threshold > -1e9:
        rows = [r for r in rows if float(r.get("score", 0.0)) >= args.hard_score_threshold]
    rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    if args.topk_ratio < 1.0:
        keep = max(1, int(len(rows) * max(args.topk_ratio, 1e-9)))
        rows = rows[:keep]
    if args.topk_samples > 0:
        rows = rows[: args.topk_samples]
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]
    if not rows:
        raise ValueError("Filtering removed all writer samples.")
    print(f"Writer rows after filtering: {len(rows)}")
    return Dataset.from_list(rows)
class WriterOPSDTrainer(OPSDTrainer):
    _name = "WRITER_OPSD"
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in ("score",):
            if self._signature_columns is not None and col not in self._signature_columns:
                self._signature_columns.append(col)
def main() -> None:
    parser = TrlParser((WriterScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.model_name_or_path = resolve_model_name_or_path(model_args.model_name_or_path)
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires --use_peft")
    if script_args.use_ema_teacher and script_args.fixed_teacher:
        raise ValueError("use_ema_teacher and fixed_teacher are mutually exclusive")
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
    else:
        model_kwargs["device_map"] = None
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_writer_dataset(
        path=script_args.dataset_path,
        max_trace_chars=script_args.max_trace_chars,
    )
    dataset = _filter_dataset(dataset, script_args)
    collator = WriterSelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
    )
    trainer = WriterOPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
    )
    print("=" * 80)
    print("Starting writer OPD training")
    print(f"Dataset size: {len(dataset)}")
    print(f"Output dir: {training_args.output_dir}")
    _vllm_on = getattr(training_args, "use_vllm", False)
    _vllm_mode = getattr(training_args, "vllm_mode", None)
    print(f"vLLM:        {_vllm_mode if _vllm_on else 'off'}")
    print("=" * 80)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Writer model saved to {training_args.output_dir}")
if __name__ == "__main__":
    main()
