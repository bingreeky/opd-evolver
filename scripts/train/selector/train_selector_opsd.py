#!/usr/bin/env python3
from __future__ import annotations
import json
import os
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
from opd_evolver.trainer.selector_collator import SelectorSelfDistillationDataCollator
@dataclass
class SelectorScriptArguments(ScriptArguments):
    dataset_path: str = field(
        default="workspace/memory/sql/memory_selector_dataset.jsonl",
        metadata={"help": "Path to selector JSONL (possibly scored)."},
    )
    topk_ratio: float = field(
        default=1.0,
        metadata={"help": "Keep top ratio of samples by score (0 < r <= 1)."},
    )
    topk_samples: int = field(
        default=0,
        metadata={"help": "Keep top-k samples by score after topk_ratio (0=disabled)."},
    )
    rank_score_source: str = field(
        default="max_candidate_score",
        metadata={
            "help": (
                "Which value populates the training 'score' used for min_score / top-k sorting. "
                "'max_candidate_score' uses the best scored retrieved candidate, 'mean_memory' uses "
                "mean_selected_memory_score, and 'dataset' uses JSONL 'score' when present."
            ),
        },
    )
    min_score: float = field(
        default=-1e9,
        metadata={"help": "Drop rows with score < min_score."},
    )
    hard_score_threshold: float = field(
        default=-1e9,
        metadata={"help": "Optional extra gate for hard subset; set > -1e9 to enable."},
    )
    max_candidates_chars: int = field(
        default=20000,
        metadata={"help": "Max chars kept for retrieve.candidates_context."},
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
def _format_candidates_ids(cands: dict[str, Any]) -> str:
    lines: list[str] = []
    for tier in ("skill", "tip", "tool", "trajectory"):
        ids = cands.get(tier, []) if isinstance(cands, dict) else []
        if isinstance(ids, list) and ids:
            lines.append(f"- {tier}: {', '.join(str(x) for x in ids)}")
    return "\n".join(lines) if lines else "- (no candidate IDs)"
_TIER_OUTPUT_KEYS = {
    "skill": "selected_skills",
    "tip": "selected_tips",
    "tool": "selected_tools",
    "trajectory": "selected_trajectories",
}
_TIER_TAG_NAMES = {
    "skill": "SKILL",
    "tip": "TIP",
    "tool": "TOOL",
    "trajectory": "TRAJECTORY",
}
def _candidate_tag_maps(cands: Any) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    if not isinstance(cands, dict):
        return maps
    for tier, tag_name in _TIER_TAG_NAMES.items():
        ids = cands.get(tier, [])
        if not isinstance(ids, list):
            continue
        tier_map: dict[str, str] = {}
        for idx, mid in enumerate(ids, start=1):
            if isinstance(mid, str):
                tier_map[mid] = f"[RETRIEVED_{tag_name}_{idx:02d}]"
        maps[tier] = tier_map
    return maps
def _selector_json_from_selected_ids(
    selected_ids: Any,
    candidates: Any,
    reasoning: str,
) -> str:
    tag_maps = _candidate_tag_maps(candidates)
    obj: dict[str, Any] = {}
    missing: list[str] = []
    if not isinstance(selected_ids, dict):
        selected_ids = {}
    for tier, out_key in _TIER_OUTPUT_KEYS.items():
        tags: list[str] = []
        ids = selected_ids.get(tier, [])
        if isinstance(ids, list):
            for mid in ids:
                if not isinstance(mid, str):
                    continue
                tag = tag_maps.get(tier, {}).get(mid)
                if tag is None:
                    missing.append(mid)
                    continue
                tags.append(tag)
        obj[out_key] = tags
    if missing:
        suffix = f" Omitted selected IDs not present in retrieved candidates: {', '.join(missing)}."
        reasoning = (reasoning + suffix).strip()
    obj["reasoning"] = reasoning
    return _compact_json(obj)
def _score_table(table: Any) -> dict[str, dict[str, float]]:
    if not isinstance(table, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for tier, values in table.items():
        if not isinstance(values, dict):
            continue
        tier_scores: dict[str, float] = {}
        for mid, score in values.items():
            if not isinstance(mid, str):
                continue
            try:
                tier_scores[mid] = float(score)
            except (TypeError, ValueError):
                tier_scores[mid] = 0.0
        if tier_scores:
            out[str(tier)] = tier_scores
    return out
def _candidate_score_rows(
    candidates: Any,
    candidate_scores: dict[str, dict[str, float]],
    selected_ids: Any,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, dict):
        return []
    if not isinstance(selected_ids, dict):
        selected_ids = {}
    tag_maps = _candidate_tag_maps(candidates)
    rows: list[dict[str, Any]] = []
    for tier in ("skill", "tip", "tool", "trajectory"):
        ids = candidates.get(tier, [])
        if not isinstance(ids, list):
            continue
        tier_selected = selected_ids.get(tier, [])
        selected_set = (
            {mid for mid in tier_selected if isinstance(mid, str)}
            if isinstance(tier_selected, list)
            else set()
        )
        for mid in ids:
            if not isinstance(mid, str):
                continue
            rows.append(
                {
                    "tag": tag_maps.get(tier, {}).get(mid, ""),
                    "tier": tier,
                    "memory_id": mid,
                    "score": float(candidate_scores.get(tier, {}).get(mid, 0.0)),
                    "historically_selected": mid in selected_set,
                }
            )
    return rows
def _max_row_score(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return max(float(row.get("score", 0.0)) for row in rows)
def _normalize_selector_row(
    row: dict[str, Any],
    max_candidates_chars: int,
    *,
    rank_score_source: str,
) -> dict[str, Any] | None:
    retrieve = row.get("retrieve")
    select = row.get("select")
    if not isinstance(retrieve, dict) or not isinstance(select, dict):
        return None
    task_description = str(retrieve.get("task_description", "")).strip()
    if not task_description:
        return None
    candidates_context = str(retrieve.get("candidates_context", "")).strip()
    if not candidates_context:
        candidates_context = _format_candidates_ids(retrieve.get("candidates", {}))
    if len(candidates_context) > max_candidates_chars:
        candidates_context = (
            candidates_context[:max_candidates_chars]
            + f"\n... [TRUNCATED {len(candidates_context) - max_candidates_chars} chars]"
        )
    candidates = retrieve.get("candidates", {})
    selected_ids = select.get("selected_memory_ids", {})
    reasoning = str(select.get("reasoning", "")).strip()
    raw_solution = select.get("raw")
    if isinstance(raw_solution, str) and raw_solution.strip():
        solution = raw_solution.strip()
    else:
        solution = _selector_json_from_selected_ids(
            selected_ids=selected_ids,
            candidates=candidates,
            reasoning=reasoning,
        )
    raw_mean = row.get("mean_selected_memory_score")
    mean_mem_score = 0.0 if raw_mean is None else float(raw_mean)
    candidate_scores = _score_table(row.get("candidate_memory_scores"))
    selected_scores = _score_table(row.get("selected_memory_scores"))
    score_rows = _candidate_score_rows(candidates, candidate_scores, selected_ids)
    max_candidate_score = _max_row_score(score_rows)
    success = bool(row.get("success", False))
    total_reward = float(row.get("total_reward", 0.0))
    if rank_score_source == "max_candidate_score":
        if max_candidate_score is not None:
            score = max_candidate_score
        elif raw_mean is not None:
            score = float(raw_mean)
        else:
            ds = row.get("score")
            score = float(ds) if ds is not None else 0.8 * float(success)
    elif rank_score_source == "mean_memory":
        if raw_mean is not None:
            score = float(raw_mean)
        else:
            ds = row.get("score")
            score = float(ds) if ds is not None else 0.8 * float(success)
    else:
        score = row.get("score")
        if score is None:
            score = 0.8 * float(success) + 0.2 * mean_mem_score
        score = float(score)
    privileged = _compact_json(
        {
            "success": success,
            "total_reward": total_reward,
            "mean_selected_memory_score": mean_mem_score,
            "selected_memory_ids": selected_ids,
            "candidate_score_table": score_rows,
            "selected_memory_scores": selected_scores,
            "candidate_scores_available": bool(candidate_scores),
            "selection_reasoning": reasoning,
            "teacher_guidance": (
                "Use candidate_score_table as the quality signal for all retrieved candidates. "
                "The reference selector output is a historical behavior trace, not a gold label. "
                "Prefer high-score, task-relevant candidates and choose empty arrays for tiers "
                "whose candidates are low-scored or irrelevant."
            ),
        }
    )
    problem = (
        f"Task:\n{task_description}\n\n"
        f"Retrieved Candidates Context:\n{candidates_context}\n\n"
        "Return strict JSON with keys: selected_skills, selected_tips, "
        "selected_tools, selected_trajectories, reasoning."
    )
    return {
        "problem": problem,
        "solution": solution,
        "privileged": privileged,
        "score": score,
    }
def load_selector_dataset(
    path: str | Path,
    max_candidates_chars: int,
    *,
    rank_score_source: str,
) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Selector dataset not found: {path}")
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
            rec = _normalize_selector_row(
                row,
                max_candidates_chars=max_candidates_chars,
                rank_score_source=rank_score_source,
            )
            if rec is None:
                bad += 1
                continue
            records.append(rec)
    if not records:
        raise ValueError(f"No valid selector records in {path}")
    print(f"Loaded selector rows: valid={len(records)}, skipped={bad}")
    return Dataset.from_list(records)
def _filter_dataset(dataset: Dataset, args: SelectorScriptArguments) -> Dataset:
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
        raise ValueError("Filtering removed all selector samples.")
    print(f"Selector rows after filtering: {len(rows)}")
    return Dataset.from_list(rows)
class SelectorOPSDTrainer(OPSDTrainer):
    _name = "SELECTOR_OPSD"
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in ("score",):
            if self._signature_columns is not None and col not in self._signature_columns:
                self._signature_columns.append(col)
def main() -> None:
    parser = TrlParser((SelectorScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.model_name_or_path = resolve_model_name_or_path(model_args.model_name_or_path)
    if script_args.rank_score_source not in ("dataset", "mean_memory", "max_candidate_score"):
        raise ValueError("rank_score_source must be 'dataset', 'mean_memory', or 'max_candidate_score'")
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
    dataset = load_selector_dataset(
        path=script_args.dataset_path,
        max_candidates_chars=script_args.max_candidates_chars,
        rank_score_source=script_args.rank_score_source,
    )
    dataset = _filter_dataset(dataset, script_args)
    collator = SelectorSelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
    )
    trainer = SelectorOPSDTrainer(
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
    print("Starting selector OPD training")
    print(f"Dataset size: {len(dataset)}")
    print(f"Output dir: {training_args.output_dir}")
    _vllm_on = getattr(training_args, "use_vllm", False)
    _vllm_mode = getattr(training_args, "vllm_mode", None)
    print(f"vLLM:        {_vllm_mode if _vllm_on else 'off'}")
    print("=" * 80)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Selector model saved to {training_args.output_dir}")
if __name__ == "__main__":
    main()
