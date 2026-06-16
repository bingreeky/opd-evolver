#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from typing import Any
REQ_KEYS = [
    "new_skills",
    "new_tips",
    "new_tools",
    "key_learnings",
    "should_save_trajectory",
    "trajectory_outcome",
]
def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        cand = text[start : end + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
def _extract_pred_obj(row: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    if isinstance(row.get("parsed_reflection"), dict):
        return row.get("parsed_reflection"), True
    txt = row.get("prediction") or row.get("output") or ""
    if isinstance(txt, str):
        obj = _extract_json_from_text(txt)
        if obj is not None:
            return obj, True
        return None, False
    return None, False
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate writer predictions offline.")
    ap.add_argument("--pred", required=True, help="Prediction JSONL")
    ap.add_argument(
        "--gold",
        default=None,
        help="Optional gold writer dataset JSONL for task overlap and compression ratio",
    )
    args = ap.parse_args()
    gold_trace_chars: dict[str, int] = {}
    if args.gold:
        with open(args.gold, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id", "")).strip()
                if not task_id:
                    continue
                trace = row.get("input", {}).get("execution_trace", [])
                gold_trace_chars[task_id] = len(json.dumps(trace, ensure_ascii=False, default=str))
    total = 0
    json_valid = 0
    schema_complete = 0
    non_empty_memory = 0
    compression_sum = 0.0
    compression_count = 0
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            pred_obj, valid = _extract_pred_obj(row)
            if not valid or pred_obj is None:
                continue
            json_valid += 1
            if all(k in pred_obj for k in REQ_KEYS):
                schema_complete += 1
            mem_count = 0
            for k in ("new_skills", "new_tips", "new_tools"):
                v = pred_obj.get(k)
                if isinstance(v, list):
                    mem_count += len(v)
            if mem_count > 0:
                non_empty_memory += 1
            task_id = str(row.get("task_id", "")).strip()
            if task_id and task_id in gold_trace_chars and gold_trace_chars[task_id] > 0:
                out_chars = len(json.dumps(pred_obj, ensure_ascii=False, default=str))
                compression_sum += out_chars / gold_trace_chars[task_id]
                compression_count += 1
    if total == 0:
        raise SystemExit("Prediction file has no rows.")
    result = {
        "count": total,
        "json_valid_rate": json_valid / total,
        "schema_complete_rate": schema_complete / total,
        "non_empty_memory_rate": non_empty_memory / total,
        "avg_output_to_trace_ratio": (
            compression_sum / compression_count if compression_count > 0 else None
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
