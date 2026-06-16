#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable
from datasets import Dataset
DEFAULT_TRAIN_JSON = Path("data/sql/merged/ic_sql_merged_train_split.json")
DEFAULT_TEST_JSON = Path("data/sql/merged/ic_sql_merged_test_split.json")
DEFAULT_OUTPUT_DIR = Path("scripts/train/megatrain_sql/data_success")
ENV_IMAGE_NAME = "docker-env-sql"
ACTION_SPACE = (
    'Action space:\n'
    '  {"action": "execute", "params": {"command": "<SQL query>"}}\n'
    '  {"action": "submit", "params": {}}'
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VERL parquet datasets for InterCode SQL GRPO, filtered to tasks that were successful in a previous rollout."
    )
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--sql-service-mode", choices=["docker", "local"], default="docker")
    parser.add_argument("--sql-host", default="127.0.0.1")
    parser.add_argument("--sql-port", type=int, default=3307)
    parser.add_argument("--sql-user", default="admin")
    parser.add_argument("--sql-password", default="admin")
    parser.add_argument(
        "--train-success-summary-csv",
        type=Path,
        required=True,
        help="Rollout summary.csv whose rows include columns: task_id, success. Only success=true tasks are kept for the train split.",
    )
    parser.add_argument(
        "--train-trajectories-dir",
        type=Path,
        default=None,
        help="Optional trajectories dir. If set, only keep task_ids that have a corresponding trajectory JSON file.",
    )
    parser.add_argument(
        "--test-success-summary-csv",
        type=Path,
        default=None,
        help="Optional rollout summary.csv used to filter the test split as well.",
    )
    parser.add_argument(
        "--test-trajectories-dir",
        type=Path,
        default=None,
        help="Optional trajectories dir used to filter the test split as well.",
    )
    return parser.parse_args()
def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {path}, got {type(data)!r}")
    return data
def render_schema(db_tables: dict[str, list[str]] | None) -> str:
    if not isinstance(db_tables, dict) or not db_tables:
        return "(schema not available)"
    lines: list[str] = []
    for table_name, columns in db_tables.items():
        if isinstance(columns, list) and columns:
            cols = ", ".join(str(col) for col in columns)
        else:
            cols = ""
        lines.append(f"- {table_name}({cols})")
    return "\n".join(lines)
def build_prompt(example: dict[str, Any]) -> str:
    db = str(example.get("db", "")).strip()
    query = str(example.get("query", "")).strip()
    schema_text = render_schema(example.get("db_tables"))
    return (
        "You are solving an InterCode SQL task in a multi-turn environment.\n"
        "Return exactly one JSON object per turn. Do not output markdown, code fences, or explanations.\n\n"
        f"{ACTION_SPACE}\n\n"
        "Rules:\n"
        "- Use execute to run a SQL statement against the current database.\n"
        "- Use submit only when you want the environment to evaluate your final state.\n"
        "- You must eventually call submit to finish the episode with a meaningful score.\n"
        "- Do not use any action other than execute or submit.\n"
        "- Keep every response valid JSON.\n"
        f"- Usually you should first run `USE {db};` before querying tables in that database.\n\n"
        f"DATABASE: {db}\n\n"
        f"SCHEMA:\n{schema_text}\n\n"
        f"QUERY: {query}"
    )
def _task_id_for_example(example: dict[str, Any], task_idx: int) -> str:
    return str(example.get("id") or f"sql_{task_idx}")
def build_row(
    example: dict[str, Any],
    *,
    dataset_path: Path,
    split: str,
    task_idx: int,
    sql_service_mode: str,
    sql_host: str,
    sql_port: int,
    sql_user: str,
    sql_password: str,
) -> dict[str, Any]:
    task_id = _task_id_for_example(example, task_idx)
    db = str(example.get("db", "")).strip()
    query = str(example.get("query", "")).strip()
    gold = str(example.get("gold", "")).strip()
    interaction_kwargs = {
        "name": "sql_intercode",
        "task_idx": task_idx,
        "task_id": task_id,
        "dataset_path": str(dataset_path),
        "env_type": "sql",
        "image_name": ENV_IMAGE_NAME,
        "max_steps": 30,
        "mysql_host_port": 3307,
        "mysql_container_name": "docker-env-sql_ic_ctr",
        "sql_service_mode": sql_service_mode,
        "sql_host": sql_host,
        "sql_port": sql_port,
        "sql_user": sql_user,
        "sql_password": sql_password,
        "query": query,
        "ground_truth": gold,
        "db": db,
    }
    extra_info = {
        "index": task_idx,
        "task_id": task_id,
        "task_idx": task_idx,
        "db": db,
        "query": query,
        "gold": gold,
        "hardness": str(example.get("hardness", "unknown")),
        "source_dataset": str(example.get("source_dataset", "unknown")),
        "source_index": example.get("source_index"),
        "db_tables": example.get("db_tables", {}),
        "split": split,
        "dataset_path": str(dataset_path),
        "interaction_kwargs": interaction_kwargs,
    }
    return {
        "data_source": "intercode_sql",
        "ability": "sql_agent",
        "agent_name": "tool_agent",
        "prompt": [{"role": "user", "content": build_prompt(example)}],
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": extra_info,
    }
def _read_success_task_ids(summary_csv: Path) -> set[str]:
    success_ids: set[str] = set()
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = str(row.get("task_id", "")).strip()
            success = str(row.get("success", "")).strip().lower()
            if not task_id or success != "true":
                continue
            success_ids.add(task_id)
    return success_ids
def _existing_trajectory_task_ids(trajectories_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in trajectories_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            tid = data.get("task_id")
            if isinstance(tid, str) and tid:
                ids.add(tid)
    return ids
def _filter_examples(
    examples: list[dict[str, Any]],
    *,
    max_rows: int,
    success_task_ids: set[str],
    trajectories_task_ids: set[str] | None,
) -> list[tuple[int, dict[str, Any]]]:
    if max_rows and max_rows > 0:
        examples = examples[:max_rows]
    kept: list[tuple[int, dict[str, Any]]] = []
    for task_idx, ex in enumerate(examples):
        task_id = _task_id_for_example(ex, task_idx)
        if task_id not in success_task_ids:
            continue
        if trajectories_task_ids is not None and task_id not in trajectories_task_ids:
            continue
        kept.append((task_idx, ex))
    return kept
def convert_split_success_only(
    *,
    input_path: Path,
    output_path: Path,
    split: str,
    max_rows: int,
    success_summary_csv: Path,
    trajectories_dir: Path | None,
    sql_service_mode: str,
    sql_host: str,
    sql_port: int,
    sql_user: str,
    sql_password: str,
) -> tuple[int, int]:
    examples = load_json(input_path)
    success_ids = _read_success_task_ids(success_summary_csv)
    traj_ids = _existing_trajectory_task_ids(trajectories_dir) if trajectories_dir else None
    kept = _filter_examples(examples, max_rows=max_rows, success_task_ids=success_ids, trajectories_task_ids=traj_ids)
    rows = [
        build_row(
            ex,
            dataset_path=input_path,
            split=split,
            task_idx=task_idx,
            sql_service_mode=sql_service_mode,
            sql_host=sql_host,
            sql_port=sql_port,
            sql_user=sql_user,
            sql_password=sql_password,
        )
        for task_idx, ex in kept
    ]
    dataset = Dataset.from_list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    return len(rows), len(examples if not (max_rows and max_rows > 0) else examples[:max_rows])
def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train.parquet"
    test_out = args.output_dir / "test.parquet"
    train_kept, train_total = convert_split_success_only(
        input_path=args.train_json,
        output_path=train_out,
        split="train",
        max_rows=args.max_train_rows,
        success_summary_csv=args.train_success_summary_csv,
        trajectories_dir=args.train_trajectories_dir,
        sql_service_mode=args.sql_service_mode,
        sql_host=args.sql_host,
        sql_port=args.sql_port,
        sql_user=args.sql_user,
        sql_password=args.sql_password,
    )
    print(f"Wrote {train_kept}/{train_total} successful train rows -> {train_out}")
    if args.test_success_summary_csv is None:
        examples = load_json(args.test_json)
        if args.max_test_rows and args.max_test_rows > 0:
            examples = examples[: args.max_test_rows]
        rows = [
            build_row(
                example,
                dataset_path=args.test_json,
                split="test",
                task_idx=task_idx,
                sql_service_mode=args.sql_service_mode,
                sql_host=args.sql_host,
                sql_port=args.sql_port,
                sql_user=args.sql_user,
                sql_password=args.sql_password,
            )
            for task_idx, example in enumerate(examples)
        ]
        dataset = Dataset.from_list(rows)
        test_out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(test_out))
        print(f"Wrote {len(rows)} test rows (unfiltered) -> {test_out}")
        return 0
    test_kept, test_total = convert_split_success_only(
        input_path=args.test_json,
        output_path=test_out,
        split="test",
        max_rows=args.max_test_rows,
        success_summary_csv=args.test_success_summary_csv,
        trajectories_dir=args.test_trajectories_dir,
        sql_service_mode=args.sql_service_mode,
        sql_host=args.sql_host,
        sql_port=args.sql_port,
        sql_user=args.sql_user,
        sql_password=args.sql_password,
    )
    print(f"Wrote {test_kept}/{test_total} successful test rows -> {test_out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
