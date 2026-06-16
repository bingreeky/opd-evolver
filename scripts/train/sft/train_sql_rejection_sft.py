#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from datetime import datetime
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from trl import SFTTrainer
from opd_evolver.base.hf_snapshot import resolve_model_name_or_path
try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None
DEFAULT_MODEL_DIR: str | None = None
BEIJING_TZ = ZoneInfo("Asia/Shanghai")
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Step-level rejection-SFT JSONL path.")
    ap.add_argument("--output-dir", required=True, help="Output adapter directory.")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Base model path.")
    ap.add_argument("--max-length", type=int, default=4096, help="Max token length.")
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules.",
    )
    ap.add_argument("--report-to", default="none", help="Trainer report_to backend.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--mix-order",
        choices=["original", "interleave_prefix", "shuffle"],
        default="interleave_prefix",
        help="Ordering policy for mixed DB/OS lifelong SFT data.",
    )
    ap.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed used when --mix-order=shuffle.",
    )
    return ap.parse_args()
def _task_prefix(record: dict[str, Any]) -> str:
    task_id = record.get("task_id")
    if isinstance(task_id, str) and "_" in task_id:
        prefix = task_id.split("_", 1)[0]
        if prefix:
            return prefix
    return "unknown"
def _count_prefix_switches(records: list[dict[str, Any]]) -> int:
    if len(records) < 2:
        return 0
    return sum(
        1
        for prev, cur in zip(records, records[1:])
        if _task_prefix(prev) != _task_prefix(cur)
    )
def _reorder_records(
    records: list[dict[str, Any]],
    *,
    mix_order: str,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    if mix_order == "original":
        return records
    if mix_order == "shuffle":
        shuffled = list(records)
        random.Random(shuffle_seed).shuffle(shuffled)
        return shuffled
    if mix_order != "interleave_prefix":
        raise ValueError(f"Unsupported mix_order: {mix_order}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prefix_order: list[str] = []
    for record in records:
        prefix = _task_prefix(record)
        if prefix not in grouped:
            prefix_order.append(prefix)
        grouped[prefix].append(record)
    positions = {prefix: 0 for prefix in prefix_order}
    ordered: list[dict[str, Any]] = []
    remaining = len(records)
    while remaining:
        progressed = False
        for prefix in prefix_order:
            pos = positions[prefix]
            group = grouped[prefix]
            if pos >= len(group):
                continue
            ordered.append(group[pos])
            positions[prefix] = pos + 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return ordered
def load_jsonl_dataset(
    path: str | Path,
    *,
    mix_order: str,
    shuffle_seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    dataset_path = Path(path)
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
            prompt = row.get("prompt")
            completion = row.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped += 1
                continue
            if not isinstance(completion, str) or not completion.strip():
                skipped += 1
                continue
            record = {
                "prompt": prompt,
                "completion": completion.strip(),
            }
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                record["task_id"] = task_id
            step_index = row.get("step_index")
            if isinstance(step_index, int):
                record["step_index"] = step_index
            messages = row.get("messages")
            if isinstance(messages, list) and len(messages) == 2:
                if all(isinstance(m, dict) for m in messages):
                    record["messages"] = messages
            records.append(record)
    if not records:
        raise ValueError(f"No valid rejection-SFT rows found in {dataset_path}")
    original_prefix_counts = Counter(_task_prefix(record) for record in records)
    original_switches = _count_prefix_switches(records)
    records = _reorder_records(records, mix_order=mix_order, shuffle_seed=shuffle_seed)
    ordered_switches = _count_prefix_switches(records)
    load_stats = {
        "valid_rows": len(records),
        "skipped_rows": skipped,
        "mix_order": mix_order,
        "shuffle_seed": shuffle_seed,
        "prefix_counts": dict(original_prefix_counts),
        "prefix_switches_before_ordering": original_switches,
        "prefix_switches_after_ordering": ordered_switches,
    }
    print(
        "Loaded rejection-SFT rows: "
        f"valid={len(records)}, skipped={skipped}, mix_order={mix_order}, "
        f"prefix_switches={original_switches}->{ordered_switches}"
    )
    return Dataset.from_list(records), load_stats
def _format_compact_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("+", "")
def build_default_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(args.dataset).name.lower()
    cleaned_suffix = "-cleaned" if "cleaned" in dataset_name else ""
    return (
        f"sft{cleaned_suffix}"
        f"-mix{args.mix_order}"
        f"-bj{timestamp}"
        f"-lr{_format_compact_float(args.learning_rate)}"
        f"-ep{_format_compact_float(args.num_train_epochs)}"
        f"-bs{args.per_device_train_batch_size}"
        f"-ga{args.gradient_accumulation_steps}"
        f"-len{args.max_length}"
        f"-lora{args.lora_r}"
    )
def resolve_run_name(args: argparse.Namespace) -> str:
    for key in ("WANDB_NAME", "RUN_NAME"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return build_default_run_name(args)
def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)
def render_prompt_text(
    tokenizer: Any,
    prompt: str,
    messages: list[dict[str, str]] | None,
) -> str:
    if messages and getattr(tokenizer, "chat_template", None):
        prompt_messages = [messages[0]]
        return _apply_chat_template(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
    if getattr(tokenizer, "chat_template", None):
        prompt_messages = [{"role": "user", "content": prompt}]
        return _apply_chat_template(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
    return prompt
def tokenize_rejection_dataset(dataset: Dataset, tokenizer: Any, max_length: int) -> tuple[Dataset, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "total_rows": len(dataset),
        "kept_rows": 0,
        "skipped_prompt_too_long": 0,
        "skipped_empty_completion_after_truncation": 0,
        "truncated_target": 0,
        "max_seq_len": 0,
        "avg_seq_len": 0.0,
        "min_label_len": 0,
        "max_label_len": 0,
        "avg_label_len": 0.0,
        "total_label_tokens": 0,
        "avg_supervised_token_ratio": 0.0,
    }
    seq_lens: list[int] = []
    label_lens: list[int] = []
    supervised_ratios: list[float] = []
    eos_token = tokenizer.eos_token or ""
    for row in dataset:
        messages = row.get("messages")
        if not isinstance(messages, list):
            messages = None
        prompt_text = render_prompt_text(
            tokenizer=tokenizer,
            prompt=row["prompt"],
            messages=messages,
        )
        target_text = row["completion"]
        if eos_token and not target_text.endswith(eos_token):
            target_text += eos_token
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(prompt_ids) >= max_length:
            stats["skipped_prompt_too_long"] += 1
            continue
        target_ids = tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        remaining_target_len = max_length - len(prompt_ids)
        if len(target_ids) > remaining_target_len:
            target_ids = target_ids[:remaining_target_len]
            stats["truncated_target"] += 1
        if not target_ids:
            stats["skipped_empty_completion_after_truncation"] += 1
            continue
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)
        row_out = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }
        rows.append(row_out)
        seq_lens.append(len(input_ids))
        label_lens.append(len(target_ids))
        supervised_ratios.append(len(target_ids) / len(input_ids))
    if not rows:
        raise ValueError("All SFT rows were filtered out during tokenization.")
    stats["kept_rows"] = len(rows)
    stats["max_seq_len"] = max(seq_lens)
    stats["avg_seq_len"] = sum(seq_lens) / len(seq_lens)
    stats["min_label_len"] = min(label_lens)
    stats["max_label_len"] = max(label_lens)
    stats["avg_label_len"] = sum(label_lens) / len(label_lens)
    stats["total_label_tokens"] = sum(label_lens)
    stats["avg_supervised_token_ratio"] = sum(supervised_ratios) / len(supervised_ratios)
    return Dataset.from_list(rows), stats
def build_training_args(args: argparse.Namespace) -> Any:
    run_name = resolve_run_name(args)
    common_kwargs = dict(
        output_dir=args.output_dir,
        run_name=run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        remove_unused_columns=False,
        bf16=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=args.seed,
    )
    if SFTConfig is not None:
        return SFTConfig(**common_kwargs)
    return TrainingArguments(**common_kwargs)
def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    training_args: Any,
    peft_config: LoraConfig,
) -> SFTTrainer:
    signature = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        "peft_config": peft_config,
    }
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)
def main() -> int:
    args = parse_args()
    args.model_dir = resolve_model_name_or_path(args.model_dir)
    run_name = resolve_run_name(args)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_dataset, load_stats = load_jsonl_dataset(
        args.dataset,
        mix_order=args.mix_order,
        shuffle_seed=args.shuffle_seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset, tokenization_stats = tokenize_rejection_dataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    stats_path = Path(args.output_dir) / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dataset": str(Path(args.dataset).resolve()),
                "raw_rows": len(raw_dataset),
                "tokenized_rows": len(train_dataset),
                "load": load_stats,
                "tokenization": tokenization_stats,
                "max_length": args.max_length,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Tokenized rejection-SFT dataset")
    print(f"  Raw rows:       {len(raw_dataset)}")
    print(f"  Train rows:     {len(train_dataset)}")
    print(f"  Mix order:      {args.mix_order}")
    print(f"  Avg seq len:    {tokenization_stats['avg_seq_len']:.1f}")
    print(f"  Max seq len:    {tokenization_stats['max_seq_len']}")
    print(f"  Avg label len:  {tokenization_stats['avg_label_len']:.1f}")
    print(f"  Label tokens:   {tokenization_stats['total_label_tokens']}")
    print(f"  Label ratio:    {tokenization_stats['avg_supervised_token_ratio']:.3f}")
    print(f"  Run name:       {run_name}")
    print(f"  Dataset stats:  {stats_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    training_args = build_training_args(args)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved rejection-SFT adapter to: {args.output_dir}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
