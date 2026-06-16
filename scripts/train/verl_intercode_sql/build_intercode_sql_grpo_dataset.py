#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_JSON = PROJECT_ROOT / "data" / "sql" / "merged" / "ic_sql_merged_train_split.json"
DEFAULT_TEST_JSON = PROJECT_ROOT / "data" / "sql" / "merged" / "ic_sql_merged_test_split.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "train" / "verl_intercode_sql" / "data"
ENV_IMAGE_NAME = "docker-env-sql"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VERL parquet datasets for InterCode SQL GRPO.")
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--sql-service-mode", choices=["docker", "local"], default="docker")
    parser.add_argument("--sql-host", default="127.0.0.1")
    parser.add_argument("--sql-port", type=int, default=3307)
    parser.add_argument("--sql-user", default="admin")
    parser.add_argument("--sql-password", default="admin")
    parser.add_argument("--dry-run", action="store_true")
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
        cols = ", ".join(str(col) for col in columns) if isinstance(columns, list) else ""
        lines.append(f"- {table_name}({cols})")
    return "\n".join(lines)
def build_prompt(example: dict[str, Any], *, max_steps: int) -> str:
    db = str(example.get("db", "")).strip()
    query = str(example.get("query", "")).strip()
    schema_text = render_schema(example.get("db_tables"))
    return (
        "You are solving an InterCode SQL task in a multi-turn environment.\n"
        "Return exactly one JSON object per turn. Do not output markdown, code fences, or explanations.\n\n"
        "Actions:\n"
        '  - {"action":"execute","params":{"command":"<SQL query>"}}\n'
        '  - {"action":"submit","params":{}}\n\n'
        "Rules:\n"
        "- Use execute to run SQL against the current MySQL service.\n"
        "- Use submit only when you want the environment to score your latest query result.\n"
        "- Keep every response valid JSON and use only the listed actions.\n"
        f"- You have at most {max_steps} assistant turns.\n"
        f"- Usually you should first run `USE {db};` before querying tables in that database.\n\n"
        f"DATABASE: {db}\n\n"
        f"SCHEMA:\n{schema_text}\n\n"
        f"QUESTION: {query}"
    )
def build_row(
    example: dict[str, Any],
    *,
    dataset_path: Path,
    split: str,
    task_idx: int,
    max_steps: int,
    sql_service_mode: str,
    sql_host: str,
    sql_port: int,
    sql_user: str,
    sql_password: str,
) -> dict[str, Any]:
    task_id = str(example.get("id") or f"sql_{task_idx}")
    db = str(example.get("db", "")).strip()
    query = str(example.get("query", "")).strip()
    gold = str(example.get("gold", "")).strip()
    interaction_kwargs = {
        "name": "intercode_sql",
        "task_id": task_id,
        "task_idx": task_idx,
        "dataset_path": str(dataset_path),
        "env_type": "sql",
        "image_name": ENV_IMAGE_NAME,
        "max_steps": max_steps,
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
        "max_steps": max_steps,
        "dataset_path": str(dataset_path),
        "interaction_kwargs": interaction_kwargs,
    }
    return {
        "data_source": "intercode_sql",
        "ability": "sql_agent",
        "agent_name": "intercode_sql_agent",
        "prompt": [{"role": "user", "content": build_prompt(example, max_steps=max_steps)}],
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": extra_info,
    }
def build_split(
    input_path: Path,
    *,
    split: str,
    max_rows: int,
    max_steps: int,
    sql_service_mode: str,
    sql_host: str,
    sql_port: int,
    sql_user: str,
    sql_password: str,
) -> list[dict[str, Any]]:
    examples = load_json(input_path)
    if max_rows and max_rows > 0:
        examples = examples[:max_rows]
    return [
        build_row(
            example,
            dataset_path=input_path,
            split=split,
            task_idx=task_idx,
            max_steps=max_steps,
            sql_service_mode=sql_service_mode,
            sql_host=sql_host,
            sql_port=sql_port,
            sql_user=sql_user,
            sql_password=sql_password,
        )
        for task_idx, example in enumerate(examples)
    ]
def write_split(rows: list[dict[str, Any]], output_path: Path) -> None:
    from datasets import Dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))
def main() -> int:
    args = parse_args()
    common = {
        "max_steps": args.max_steps,
        "sql_service_mode": args.sql_service_mode,
        "sql_host": args.sql_host,
        "sql_port": args.sql_port,
        "sql_user": args.sql_user,
        "sql_password": args.sql_password,
    }
    train_rows = build_split(args.train_json, split="train", max_rows=args.max_train_rows, **common)
    test_rows = build_split(args.test_json, split="test", max_rows=args.max_test_rows, **common)
    train_out = args.output_dir / "train.parquet"
    test_out = args.output_dir / "test.parquet"
    if args.dry_run:
        print(f"Would write {len(train_rows)} train rows -> {train_out}")
        print(f"Would write {len(test_rows)} test rows -> {test_out}")
        return 0
    write_split(train_rows, train_out)
    write_split(test_rows, test_out)
    print(f"Wrote {len(train_rows)} train rows -> {train_out}")
    print(f"Wrote {len(test_rows)} test rows -> {test_out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
