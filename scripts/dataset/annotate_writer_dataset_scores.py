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
def _mean_scores_for_created(
    created_memory_ids: Any,
    scores: Dict[str, float],
) -> Optional[float]:
    if not isinstance(created_memory_ids, dict):
        return None
    vals: List[float] = []
    for _tier, ids in created_memory_ids.items():
        if not isinstance(ids, list):
            continue
        for mid in ids:
            if not isinstance(mid, str):
                continue
            vals.append(float(scores.get(mid, 0.0)))
    if not vals:
        return None
    return sum(vals) / len(vals)
def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate memory writer JSONL with hybrid memory scores.")
    ap.add_argument("--dataset", required=True, help="Input JSONL path")
    ap.add_argument(
        "--storage-dir",
        required=True,
        help="Memory storage directory (contains tier JSON + usage_logs.jsonl)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSONL (default: overwrite --dataset)",
    )
    ap.add_argument("--lambda-days", type=float, default=0.1, help="Recency decay for score_memory")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Only set score when missing or null; leave existing numeric scores unchanged",
    )
    args = ap.parse_args()
    in_path = os.path.expanduser(args.dataset)
    out_path = os.path.expanduser(args.output or args.dataset)
    storage_dir = os.path.expanduser(args.storage_dir)
    cfg = MemoryConfig(storage_dir=storage_dir)
    mm = HierarchicalMemoryManager(config=cfg)
    usage_path = os.path.join(storage_dir, "usage_logs.jsonl")
    usage = UsageLogger(storage_path=usage_path)
    all_scores = score_all_memories(
        mm,
        usage,
        lambda_days=args.lambda_days,
    )
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(in_path, encoding="utf-8") as fin:
        raw_lines = fin.readlines()
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.only_missing:
                existing = row.get("score")
                if existing is not None and isinstance(existing, (int, float)):
                    fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    n_written += 1
                    continue
            agg = _mean_scores_for_created(row.get("created_memory_ids"), all_scores)
            row["score"] = agg
            fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n_written += 1
    print(f"Wrote {n_written} rows to {out_path}")
if __name__ == "__main__":
    main()
