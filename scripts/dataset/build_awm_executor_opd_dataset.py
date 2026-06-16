#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from build_sql_rejection_sft_dataset import load_best_trajectories
PSEUDO_ACTIONS = {"error", "no_action", "step_timeout"}
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-csv", required=True, help="AWM summary.csv from bench_simple_awm.py.")
    ap.add_argument("--trajectories-dir", required=True, help="Directory containing AWM trajectory JSON files.")
    ap.add_argument("--output", required=True, help="Output task-level OPD JSONL.")
    ap.add_argument(
        "--manifest-output",
        default=None,
        help="Optional manifest JSON. Defaults to <output>.manifest.json.",
    )
    ap.add_argument(
        "--include-failed-tasks",
        action="store_true",
        help="Use every task in summary.csv instead of success=true tasks only. Intended for debugging.",
    )
    return ap.parse_args()
def load_selected_task_ids(path: str | Path, *, include_failed_tasks: bool) -> tuple[list[str], Counter]:
    csv_path = Path(path)
    selected: list[str] = []
    skipped = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                skipped["missing_task_id"] += 1
                continue
            success = str(row.get("success", "")).strip().lower() == "true"
            if include_failed_tasks or success:
                selected.append(task_id)
    return list(dict.fromkeys(selected)), skipped
def _extract_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    trace = trajectory.get("execution_trace")
    if isinstance(trace, list):
        return trace
    trace = trajectory.get("trace")
    return trace if isinstance(trace, list) else []
def _canonical_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    action_name = action.get("action")
    if not isinstance(action_name, str) or not action_name.strip():
        return None
    params = action.get("params")
    return {
        "action": action_name.strip(),
        "params": params if isinstance(params, dict) else {},
    }
def _solution_actions(trajectory: dict[str, Any], skip_stats: Counter) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in _extract_steps(trajectory):
        if not isinstance(step, dict):
            skip_stats["non_dict_step"] += 1
            continue
        if step.get("raw_response") == "forced_submit":
            skip_stats["forced_submit_step"] += 1
            continue
        action = _canonical_action(step.get("action"))
        if action is None:
            skip_stats["missing_or_malformed_action"] += 1
            continue
        if action["action"] in PSEUDO_ACTIONS:
            skip_stats["pseudo_action"] += 1
            continue
        actions.append(action)
    return actions
def _task_parts(task_id: str, trajectory: dict[str, Any]) -> tuple[str | None, int | None]:
    meta = trajectory.get("meta_data") if isinstance(trajectory.get("meta_data"), dict) else {}
    scenario = meta.get("scenario")
    task_idx = meta.get("task_idx")
    if scenario is None and ":" in task_id:
        scenario = task_id.rsplit(":", 1)[0]
    if task_idx is None and ":" in task_id:
        try:
            task_idx = int(task_id.rsplit(":", 1)[1])
        except ValueError:
            task_idx = None
    return str(scenario) if scenario is not None else None, task_idx if isinstance(task_idx, int) else None
def _skill_tags(task_id: str, trajectory: dict[str, Any], actions: list[dict[str, Any]]) -> list[str]:
    scenario, _ = _task_parts(task_id, trajectory)
    tags = ["awm", "mcp_tool_use"]
    if scenario:
        tags.append(scenario)
    for action in actions[:20]:
        name = action.get("action")
        if isinstance(name, str) and name not in tags:
            tags.append(name)
    return tags
def main() -> int:
    args = parse_args()
    summary_csv_path = Path(args.summary_csv)
    trajectories_dir = Path(args.trajectories_dir)
    output_path = Path(args.output)
    manifest_path = (
        Path(args.manifest_output)
        if args.manifest_output
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    selected_task_ids, summary_skips = load_selected_task_ids(
        summary_csv_path,
        include_failed_tasks=args.include_failed_tasks,
    )
    trajectories_by_task, trajectory_stats = load_best_trajectories(trajectories_dir)
    task_skip_stats: Counter[str, int] = Counter()
    step_skip_stats: Counter[str, int] = Counter()
    action_stats: Counter[str, int] = Counter()
    records: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        match = trajectories_by_task.get(task_id)
        if match is None:
            task_skip_stats["missing_trajectory_for_selected_task"] += 1
            continue
        trajectory_path, trajectory = match
        problem = str(trajectory.get("instruction") or "").strip()
        action_space = str(trajectory.get("action_space") or "").strip()
        actions = _solution_actions(trajectory, step_skip_stats)
        if not problem:
            task_skip_stats["missing_problem"] += 1
            continue
        if not action_space:
            task_skip_stats["missing_action_space"] += 1
            continue
        if not actions:
            task_skip_stats["no_valid_actions"] += 1
            continue
        scenario, task_idx = _task_parts(task_id, trajectory)
        for action in actions:
            action_stats[str(action.get("action"))] += 1
        records.append(
            {
                "task_id": task_id,
                "task_type": "awm",
                "scenario": scenario,
                "task_idx": task_idx,
                "problem": problem,
                "action_space": action_space,
                "solution": json.dumps(actions, ensure_ascii=False, separators=(",", ":")),
                "success": bool(trajectory.get("success")),
                "reward": trajectory.get("total_reward"),
                "trajectory_file": str(trajectory_path),
                "skill_tags": _skill_tags(task_id, trajectory, actions),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "benchmark": "agent_world_model",
        "summary_csv": str(summary_csv_path),
        "trajectories_dir": str(trajectories_dir),
        "output_path": str(output_path),
        "include_failed_tasks": bool(args.include_failed_tasks),
        "selected_tasks_in_summary": len(selected_task_ids),
        "task_level_opd_samples": len(records),
        "action_counts": dict(action_stats),
        "summary_skip_stats": dict(summary_skips),
        "trajectory_stats": dict(trajectory_stats),
        "task_skip_stats": dict(task_skip_stats),
        "step_skip_stats": dict(step_skip_stats),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Built AWM executor OPD dataset")
    print(f"  Selected tasks in summary: {len(selected_task_ids)}")
    print(f"  Task-level OPD samples:    {len(records)}")
    print(f"  Output JSONL:              {output_path}")
    print(f"  Manifest JSON:             {manifest_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
