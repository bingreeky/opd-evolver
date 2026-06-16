#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from opd_evolver.memory.memory_manager import HierarchicalMemoryManager, MemoryConfig
from opd_evolver.memory.scoring import score_all_memories
from opd_evolver.memory.usage_log import UsageLogger
def _scores_for_ids(
    ids_map: Any,
    scores: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(ids_map, dict):
        return out
    for tier, ids in ids_map.items():
        if not isinstance(ids, list):
            continue
        tier_scores: Dict[str, float] = {}
        for mid in ids:
            if not isinstance(mid, str):
                continue
            tier_scores[mid] = float(scores.get(mid, 0.0))
        if tier_scores:
            out[str(tier)] = tier_scores
    return out
def _mean_score(score_table: Dict[str, Dict[str, float]]) -> Optional[float]:
    vals: List[float] = []
    for tier_scores in score_table.values():
        vals.extend(float(v) for v in tier_scores.values())
    if not vals:
        return None
    return sum(vals) / len(vals)
def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate memory selector JSONL scores.")
    ap.add_argument("--dataset", required=True, help="Input JSONL path")
    ap.add_argument(
        "--storage-dir",
        required=True,
        help="Memory storage directory (tier JSON + usage_logs.jsonl)",
    )
    ap.add_argument("--output", default=None, help="Output path (default: overwrite input)")
    ap.add_argument("--lambda-days", type=float, default=0.1)
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip rows that already have numeric score",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="For binary mode: require mean_selected_memory_score >= this (default 0)",
    )
    ap.add_argument(
        "--score-mode",
        choices=("binary", "weighted"),
        default="weighted",
        help="binary: 1 iff success and mean>=threshold; weighted: linear combo",
    )
    ap.add_argument("--w-success", type=float, default=0.8, help="Weighted mode weight")
    ap.add_argument("--w-mean", type=float, default=0.2, help="Weighted mode weight")
    args = ap.parse_args()
    in_path = os.path.expanduser(args.dataset)
    out_path = os.path.expanduser(args.output or args.dataset)
    storage_dir = os.path.expanduser(args.storage_dir)
    cfg = MemoryConfig(storage_dir=storage_dir)
    mm = HierarchicalMemoryManager(config=cfg)
    usage = UsageLogger(storage_path=os.path.join(storage_dir, "usage_logs.jsonl"))
    all_scores = score_all_memories(mm, usage, lambda_days=args.lambda_days)
    with open(in_path, encoding="utf-8") as fin:
        raw_lines = fin.readlines()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.only_missing:
                ex = row.get("score")
                if (
                    ex is not None
                    and isinstance(ex, (int, float))
                    and isinstance(row.get("candidate_memory_scores"), dict)
                    and isinstance(row.get("selected_memory_scores"), dict)
                ):
                    fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    n += 1
                    continue
            retrieve_block = row.get("retrieve") if isinstance(row.get("retrieve"), dict) else {}
            select_block = row.get("select") if isinstance(row.get("select"), dict) else {}
            candidate_scores = _scores_for_ids(
                retrieve_block.get("candidates", {}),
                all_scores,
            )
            selected_scores = _scores_for_ids(
                select_block.get("selected_memory_ids", {}),
                all_scores,
            )
            mean_sel = _mean_score(selected_scores)
            row["candidate_memory_scores"] = candidate_scores
            row["selected_memory_scores"] = selected_scores
            row["mean_selected_memory_score"] = mean_sel
            success = bool(row.get("success", False))
            mean_val = mean_sel if mean_sel is not None else 0.0
            if args.score_mode == "binary":
                row["score"] = (
                    1.0
                    if success and mean_val >= args.threshold
                    else 0.0
                )
            else:
                row["score"] = (
                    args.w_success * float(success) + args.w_mean * mean_val
                )
            fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    print(f"Wrote {n} rows to {out_path}")
if __name__ == "__main__":
    main()
