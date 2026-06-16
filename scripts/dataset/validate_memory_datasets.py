#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from typing import Any
def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield lineno, None, exc
def _validate_selector(path: str) -> dict[str, Any]:
    required = [
        ("task_id",),
        ("retrieve", "task_description"),
        ("retrieve", "candidates"),
        ("retrieve", "candidates_context"),
        ("select", "selected_memory_ids"),
        ("success",),
        ("total_reward",),
    ]
    stats = Counter()
    missing = Counter()
    for _lineno, row, err in _iter_jsonl(path):
        stats["total"] += 1
        if err is not None:
            stats["json_error"] += 1
            continue
        if not isinstance(row, dict):
            stats["not_object"] += 1
            continue
        ok = True
        for key_path in required:
            cur = row
            for key in key_path:
                if not isinstance(cur, dict) or key not in cur:
                    missing[".".join(key_path)] += 1
                    ok = False
                    break
                cur = cur[key]
        if not ok:
            stats["schema_error"] += 1
            continue
        selected = row.get("select", {}).get("selected_memory_ids", {})
        selected_count = 0
        if isinstance(selected, dict):
            for ids in selected.values():
                if isinstance(ids, list):
                    selected_count += len(ids)
        if selected_count == 0:
            stats["empty_selected"] += 1
        cctx = str(row.get("retrieve", {}).get("candidates_context", ""))
        if len(cctx) > 20000:
            stats["overlong_candidates_context"] += 1
        stats["valid"] += 1
    return {
        "path": path,
        "type": "selector",
        "stats": dict(stats),
        "missing_fields": dict(missing),
    }
def _validate_writer(path: str) -> dict[str, Any]:
    required = [
        ("task_id",),
        ("input", "task_description"),
        ("input", "execution_trace"),
    ]
    stats = Counter()
    missing = Counter()
    for _lineno, row, err in _iter_jsonl(path):
        stats["total"] += 1
        if err is not None:
            stats["json_error"] += 1
            continue
        if not isinstance(row, dict):
            stats["not_object"] += 1
            continue
        ok = True
        for key_path in required:
            cur = row
            for key in key_path:
                if not isinstance(cur, dict) or key not in cur:
                    missing[".".join(key_path)] += 1
                    ok = False
                    break
                cur = cur[key]
        if not ok:
            stats["schema_error"] += 1
            continue
        output_memory = row.get("output_memory")
        parsed_reflection = row.get("parsed_reflection")
        if not (isinstance(output_memory, str) and output_memory.strip()) and not isinstance(parsed_reflection, dict):
            stats["missing_output"] += 1
            continue
        trace = row.get("input", {}).get("execution_trace", [])
        if isinstance(trace, list) and len(trace) == 0:
            stats["empty_trace"] += 1
        score = row.get("score")
        if score is None:
            stats["missing_score"] += 1
        stats["valid"] += 1
    return {
        "path": path,
        "type": "writer",
        "stats": dict(stats),
        "missing_fields": dict(missing),
    }
def main() -> None:
    ap = argparse.ArgumentParser(description="Validate selector/writer JSONL datasets.")
    ap.add_argument("--selector", default=None, help="Path to selector dataset JSONL")
    ap.add_argument("--writer", default=None, help="Path to writer dataset JSONL")
    args = ap.parse_args()
    if not args.selector and not args.writer:
        raise SystemExit("Provide at least one of --selector / --writer")
    reports: list[dict[str, Any]] = []
    if args.selector:
        spath = os.path.expanduser(args.selector)
        reports.append(_validate_selector(spath))
    if args.writer:
        wpath = os.path.expanduser(args.writer)
        reports.append(_validate_writer(wpath))
    print(json.dumps({"reports": reports}, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
