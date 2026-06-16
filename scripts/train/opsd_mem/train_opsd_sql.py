#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from trl.experimental.gold import GOLDConfig
from opd_evolver.base.hf_snapshot import resolve_model_name_or_path
from opd_evolver.memory.scoring import score_memory
from opd_evolver.memory.usage_log import UsageLogger
from opd_evolver.trainer import OPSDTrainer, SQLSelfDistillationDataCollator
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIDER_DATA  = PROJECT_ROOT / "data" / "sql" / "spider" / "ic_spider_dev.json"
@dataclass
class SQLScriptArguments(ScriptArguments):
    use_tinker_loss: bool = field(
        default=False,
        metadata={"help": "Use reverse-KL (Thinking-Machines style) instead of JSD."},
    )
    fixed_teacher: bool = field(
        default=True,
        metadata={
            "help": "Use the initial policy (base model w/o LoRA) as teacher. "
            "Requires --use_peft."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={"help": "Restrict JSD to top-k teacher tokens. 0 = full vocab."},
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Per-token JSD clamp. 0 = no clipping."},
    )
    presence_penalty: float = field(default=0.0, metadata={"help": "vLLM presence penalty."})
    reason_first: bool = field(default=False, metadata={"help": "Let teacher reason before teaching."})
    run_config: str | None = field(default=None, metadata={"help": "Custom run name suffix."})
    memory_store_dir: str = field(
        default="",
        metadata={"help": "Path to hierarchical memory store (JSON dir). Empty = no memories."},
    )
    memory_top_k: int = field(
        default=50,
        metadata={
            "help": "Global score-based top-K (legacy mode, used when memory_retrieve_k == 0)."
        },
    )
    memory_retrieve_k: int = field(
        default=100,
        metadata={
            "help": "Per-example retrieval: fetch this many candidates by embedding similarity "
            "before re-ranking by score.  0 = disable retrieval (fall back to global scoring)."
        },
    )
    memory_select_k: int = field(
        default=10,
        metadata={
            "help": "Number of memories to keep from the retrieve_k candidates after score "
            "re-ranking.  Must be <= memory_retrieve_k."
        },
    )
    memory_embedding_model: str = field(
        default="Qwen/Qwen3-Embedding-0.6B",
        metadata={
            "help": "HuggingFace model ID (or local path) for per-example memory retrieval "
            "query encoding.  Only used when memory_retrieve_k > 0."
        },
    )
    memory_min_score: float = field(
        default=0.0,
        metadata={
            "help": "Minimum hybrid quality score a memory must exceed to be injected into the "
            "teacher prompt.  0.0 = no gate (all retrieved memories are used).  "
            "Recommended: 0.01 to exclude zero-scored items; 0.05 to be more selective.  "
            "When the gate excludes all candidates the teacher falls back to Gold-SQL-only mode."
        },
    )
    teacher_mode: str = field(
        default="gold_sql",
        metadata={
            "help": "What privileged information the teacher sees.  "
            "gold_sql   – teacher sees problem + gold SQL (classic OPSD; works well even "
            "             without memories; memories are only added when reliably scored). "
            "memory_only – teacher sees problem + high-value memories, NO gold SQL.  "
            "             Use for Phase 4.2 OPD internalization once memories are scored. "
            "both       – teacher sees problem + gold SQL + memories.  WARNING: mixes two "
            "             privilege sources; use only when memories are high quality."
        },
    )
    max_train_samples: int = field(
        default=0,
        metadata={
            "help": "Use only the first N rows of the training split after load. "
            "0 = use the full dataset."
        },
    )
    success_filter_csv: str = field(
        default="",
        metadata={
            "help": "Path to a benchmark summary CSV (columns: task_id, success, …). "
            "When set, only dataset rows whose index N appears in a 'sql_N' task_id row "
            "with success==True are kept for training."
        },
    )
    use_ema_teacher: bool = field(default=False, metadata={"help": "EMA teacher mode."})
    ema_decay: float = field(default=0.999, metadata={"help": "EMA decay factor."})
def load_spider_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)
    records: list[dict[str, str]] = []
    for ex in raw:
        db = ex["db"]
        query = ex["query"]
        gold = ex["gold"]
        db_tables: dict[str, list[str]] = ex.get("db_tables", {})
        schema_lines = [
            f"  {tbl}({', '.join(cols)})" for tbl, cols in db_tables.items()
        ]
        schema_text = "\n".join(schema_lines) or "  (schema not available)"
        records.append({
            "problem": f"DATABASE: {db}\n\nSCHEMA:\n{schema_text}\n\nQUERY: {query}",
            "solution": gold,
            "db": db,
            "hardness": ex.get("hardness", "unknown"),
        })
    return Dataset.from_list(records)
def load_success_indices(csv_path: str | Path) -> set[int]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"success_filter_csv not found: {path}")
    indices: set[int] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            success = row.get("success", "").strip().lower()
            if success != "true":
                continue
            if task_id.startswith("sql_"):
                try:
                    indices.add(int(task_id[4:]))
                except ValueError:
                    pass
    return indices
def _simple_contribution_score(item: dict) -> float:
    usage = item.get("usage_count", 0)
    success = item.get("success_count", 0)
    if usage == 0:
        return 0.0
    ratio = success / usage
    confidence = 1 - 1 / sqrt(1 + usage)
    return confidence * ratio
def _hybrid_score(item: dict, item_id: str, tier_name: str, all_logs: list) -> float:
    if all_logs:
        relevant = [
            log for log in all_logs
            if item_id in log.all_candidate_ids()
            or item_id in log.all_selected_ids()
        ]
        if relevant:
            sc = score_memory(
                memory_id=str(item_id),
                tier=tier_name,
                last_used=item.get("last_used"),
                usage_logs=relevant,
            )
            if sc != 0.0:
                return sc
    return _simple_contribution_score(item)
def load_top_memories(
    store_dir: str,
    top_k: int = 50,
    min_score: float = 0.0,
) -> str:
    store_path = Path(os.path.expanduser(store_dir))
    if not store_path.exists():
        return ""
    usage_log_path = store_path / "usage_logs.jsonl"
    all_logs: list = []
    if usage_log_path.is_file():
        all_logs = UsageLogger(str(usage_log_path)).load_all()
    all_items: list[tuple[float, str, dict]] = []
    for json_file in store_path.glob("*_memory.json"):
        tier_name = json_file.stem.replace("_memory", "")
        try:
            with json_file.open("r", encoding="utf-8") as f:
                tier_data = json.load(f)
        except Exception:
            continue
        if isinstance(tier_data, list):
            items = tier_data
        elif isinstance(tier_data, dict) and isinstance(tier_data.get("items"), list):
            items = tier_data["items"]
        elif isinstance(tier_data, dict):
            items = list(tier_data.values())
        else:
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id:
                continue
            sc = _hybrid_score(item, str(item_id), tier_name, all_logs)
            all_items.append((sc, tier_name, item))
    if not all_items:
        return ""
    all_items.sort(key=lambda t: t[0], reverse=True)
    eligible = [t for t in all_items if t[0] > min_score]
    top = eligible[:top_k] if eligible else []
    if not top:
        return ""
    sections: list[str] = []
    for rank, (sc, tier, item) in enumerate(top, 1):
        content = item.get("content", "(no content)")
        sections.append(f"[{tier.upper()} #{rank}] (score={sc:.3f})\n{content}")
    return "\n\n".join(sections)
def build_per_example_memory_contexts(
    problems: list[str],
    store_dir: str,
    retrieve_k: int = 100,
    select_k: int = 10,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    min_score: float = 0.0,
) -> list[str]:
    import torch
    import numpy as np
    from opd_evolver.memory.base_store import GpuEmbeddingIndex
    from opd_evolver.memory.embeddings import (
        LocalHFEmbeddingProvider,
        local_hf_embedding_settings_from_env,
    )
    store_path = Path(os.path.expanduser(store_dir))
    if not store_path.exists():
        return [""] * len(problems)
    usage_log_path = store_path / "usage_logs.jsonl"
    all_logs: list = []
    if usage_log_path.is_file():
        all_logs = UsageLogger(str(usage_log_path)).load_all()
    all_items: list[tuple[float, str, dict, list]] = []
    for json_file in store_path.glob("*_memory.json"):
        tier_name = json_file.stem.replace("_memory", "")
        try:
            with json_file.open("r", encoding="utf-8") as f:
                tier_data = json.load(f)
        except Exception:
            continue
        if isinstance(tier_data, list):
            items = tier_data
        elif isinstance(tier_data, dict) and isinstance(tier_data.get("items"), list):
            items = tier_data["items"]
        elif isinstance(tier_data, dict):
            items = list(tier_data.values())
        else:
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            embedding = item.get("embedding")
            if not item_id or not embedding:
                continue
            sc = _hybrid_score(item, str(item_id), tier_name, all_logs)
            all_items.append((sc, tier_name, item, embedding))
    if not all_items:
        return [""] * len(problems)
    actual_retrieve_k = min(retrieve_k, len(all_items))
    actual_select_k = min(select_k, actual_retrieve_k)
    scored_count = sum(1 for t in all_items if t[0] > 0.0)
    print(
        f"  Memory quality: {scored_count}/{len(all_items)} items have score > 0 "
        f"(min_score gate={min_score:.3f})"
    )
    emb_device, emb_max_len, emb_dtype = local_hf_embedding_settings_from_env()
    print(
        f"  Embedding {len(problems)} problem queries with {embedding_model} "
        f"(device={emb_device}, dtype={emb_dtype}, max_length={emb_max_len}) …"
    )
    provider = LocalHFEmbeddingProvider(
        model_id=embedding_model,
        device=emb_device,
        torch_dtype=emb_dtype,
        max_length=emb_max_len,
    )
    chunk_size = 32
    query_vecs: list[list[float]] = []
    for start in range(0, len(problems), chunk_size):
        chunk = problems[start : start + chunk_size]
        query_vecs.extend(provider._embed_batch_sync(chunk))
        if (start // chunk_size) % 5 == 0:
            print(f"    … {min(start + chunk_size, len(problems))}/{len(problems)}")
    item_embeddings = [t[3] for t in all_items]
    index_device = emb_device if torch.cuda.is_available() else "cpu"
    try:
        dtype = torch.float16 if index_device.startswith("cuda") else torch.float32
        E = torch.tensor(item_embeddings, dtype=dtype, device=index_device)
        norms = E.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
        E = E / norms
        Q = torch.tensor(query_vecs, dtype=dtype, device=index_device)
        with torch.no_grad():
            sims_gpu = Q @ E.T
        sims = sims_gpu.cpu().float().numpy()
        del E, Q, sims_gpu
        use_gpu = True
    except Exception as _gpu_err:
        print(f"  GPU GEMM failed ({_gpu_err}), falling back to numpy …")
        use_gpu = False
        item_matrix = np.array(item_embeddings, dtype=np.float32)
        norms_np = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        item_matrix = item_matrix / np.maximum(norms_np, 1e-9)
        query_matrix = np.array(query_vecs, dtype=np.float32)
        sims = query_matrix @ item_matrix.T
    print(
        f"  Similarity computed via {'GPU GEMM' if use_gpu else 'numpy'}: "
        f"shape ({len(problems)}, {len(all_items)})"
    )
    contexts: list[str] = []
    for i in range(len(problems)):
        top_idx = np.argpartition(sims[i], -actual_retrieve_k)[-actual_retrieve_k:]
        candidates = sorted(
            [
                (all_items[j][0], float(sims[i][j]), all_items[j][1], all_items[j][2])
                for j in top_idx
            ],
            key=lambda t: (t[0], t[1]),
            reverse=True,
        )
        eligible = [c for c in candidates if c[0] > min_score]
        selected = eligible[:actual_select_k] if eligible else []
        if not selected:
            contexts.append("")
            continue
        sections: list[str] = []
        for rank, (sc, _sim, tier, item) in enumerate(selected, 1):
            content = item.get("content", "(no content)")
            sections.append(f"[{tier.upper()} #{rank}] (score={sc:.3f})\n{content}")
        contexts.append("\n\n".join(sections))
    non_empty = sum(1 for c in contexts if c)
    print(
        f"  Per-example memory contexts built: retrieve_k={actual_retrieve_k}, "
        f"select_k={actual_select_k}, injected={non_empty}/{len(problems)} examples"
    )
    return contexts
class SQLOPSDTrainer(OPSDTrainer):
    _name = "SQL_OPSD"
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for col in ("db", "hardness", "memory_context"):
            if self._signature_columns is not None and col not in self._signature_columns:
                self._signature_columns.append(col)
def main() -> None:
    parser = TrlParser((SQLScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.model_name_or_path = resolve_model_name_or_path(model_args.model_name_or_path)
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_procs = int(os.environ.get("WORLD_SIZE", 1))
    eff_bs = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_procs
    )
    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{eff_bs}"
        if not training_args.output_dir.endswith(script_args.run_config):
            training_args.output_dir = str(
                Path(training_args.output_dir) / script_args.run_config
            )
    else:
        model_short = model_args.model_name_or_path.rstrip("/").split("/")[-1]
        run_name = f"opsd_sql_{model_short}_lr{lr_str}_bs{eff_bs}"
        if script_args.fixed_teacher:
            run_name += "_fixteach"
    print(f"\n{'=' * 80}")
    print(f"RUN: {run_name}")
    print(f"OUTPUT: {training_args.output_dir}")
    print(f"{'=' * 80}\n")
    _rpt = training_args.report_to
    _rpt_seq = _rpt if isinstance(_rpt, (list, tuple)) else ([_rpt] if _rpt else [])
    if _rpt_seq and ("wandb" in _rpt_seq or "all" in _rpt_seq):
        if os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true", "yes"):
            if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
                os.environ["WANDB_MODE"] = "offline"
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")
        wb_offline = os.environ.get("WANDB_MODE", "").lower() == "offline"
        wb_has_key = bool(os.environ.get("WANDB_API_KEY"))
        if not wb_disabled and (wb_has_key or wb_offline):
            try:
                import wandb
                wandb.init(
                    project=training_args.wandb_project or "opsd-sql",
                    name=run_name,
                    config={
                        "model": model_args.model_name_or_path,
                        "lr": training_args.learning_rate,
                        "eff_batch_size": eff_bs,
                        "max_length": training_args.max_length,
                        "max_completion_length": training_args.max_completion_length,
                        "use_peft": model_args.use_peft,
                        "fixed_teacher": script_args.fixed_teacher,
                        "memory_top_k": script_args.memory_top_k,
                    },
                )
            except Exception:
                pass
        elif not wb_disabled and not wb_has_key and not wb_offline:
            print(
                "Skipping wandb.init (non-interactive): set WANDB_API_KEY, "
                "or WANDB_MODE=offline, or WANDB_DISABLED=true."
            )
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher requires use_peft (LoRA).")
    model_dtype = torch.bfloat16
    model_kwargs: dict[str, Any] = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "sdpa",
        torch_dtype=model_dtype,
    )
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
    spider_path = script_args.dataset_name or str(SPIDER_DATA)
    print(f"Loading Spider SQL from: {spider_path}")
    train_dataset = load_spider_dataset(spider_path)
    print(f"  {len(train_dataset)} examples loaded")
    if script_args.success_filter_csv:
        print(f"Filtering by rollout success: {script_args.success_filter_csv}")
        success_indices = load_success_indices(script_args.success_filter_csv)
        before = len(train_dataset)
        keep = [i for i in range(len(train_dataset)) if i in success_indices]
        train_dataset = train_dataset.select(keep)
        print(f"  Kept {len(train_dataset)}/{before} examples (rollout success=True)")
    if script_args.max_train_samples and script_args.max_train_samples > 0:
        n = min(script_args.max_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n))
        print(f"  Using first {n} examples (--max_train_samples)")
    per_example_memory = False
    global_memory_context = ""
    teacher_mode = script_args.teacher_mode
    min_score = script_args.memory_min_score
    if script_args.memory_store_dir:
        if script_args.memory_retrieve_k > 0:
            print(
                f"Memory retrieval mode: retrieve_k={script_args.memory_retrieve_k}, "
                f"select_k={script_args.memory_select_k}, min_score={min_score:.3f}, "
                f"teacher_mode={teacher_mode}"
            )
            problems = train_dataset["problem"]
            contexts = build_per_example_memory_contexts(
                problems=problems,
                store_dir=script_args.memory_store_dir,
                retrieve_k=script_args.memory_retrieve_k,
                select_k=script_args.memory_select_k,
                embedding_model=script_args.memory_embedding_model,
                min_score=min_score,
            )
            train_dataset = train_dataset.add_column("memory_context", contexts)
            per_example_memory = True
        else:
            global_memory_context = load_top_memories(
                script_args.memory_store_dir,
                top_k=script_args.memory_top_k,
                min_score=min_score,
            )
            if global_memory_context:
                print(
                    f"  Global memory: top-{script_args.memory_top_k} "
                    f"({len(global_memory_context)} chars), teacher_mode={teacher_mode}"
                )
            else:
                print(
                    "  Memory store specified but no items passed min_score gate "
                    f"({min_score:.3f}) – teacher will use gold SQL only."
                )
    else:
        print("  No memory store – teacher uses gold SQL as sole privilege.")
    data_collator = SQLSelfDistillationDataCollator(
        tokenizer=tokenizer,
        memory_context="" if per_example_memory else global_memory_context,
        max_length=training_args.max_length,
        teacher_mode=teacher_mode,
    )
    training_args.presence_penalty = script_args.presence_penalty
    trainer = SQLOPSDTrainer(
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
    print(f"\n{'=' * 80}")
    print("Starting SQL OPD training …")
    print(f"  Dataset:     {len(train_dataset)} Spider SQL examples")
    if script_args.success_filter_csv:
        print(f"  Filter CSV:  {script_args.success_filter_csv}")
    print(f"  Epochs:      {training_args.num_train_epochs}")
    print(f"  Batch (eff): {eff_bs}")
    if model_args.use_peft:
        print(f"  LoRA:        r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    else:
        print("  PEFT:        off (full weights)")
    print(f"  vLLM:        {'colocate' if getattr(training_args, 'use_vllm', False) else 'off'}")
    if script_args.memory_store_dir and per_example_memory:
        print(
            f"  Memory:      per-example retrieval "
            f"(retrieve_k={script_args.memory_retrieve_k}, select_k={script_args.memory_select_k}, "
            f"min_score={min_score:.3f})"
        )
    elif script_args.memory_store_dir:
        print(f"  Memory:      global top-{script_args.memory_top_k} (min_score={min_score:.3f})")
    else:
        print("  Memory:      none")
    print(f"  Teacher:     {teacher_mode}")
    print(f"{'=' * 80}\n")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"\nModel saved to {training_args.output_dir}")
if __name__ == "__main__":
    main()
