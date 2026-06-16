#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-train-dir", required=True, help="Run dir for base eval on train-1200.")
    ap.add_argument("--base-test-dir", required=True, help="Run dir for base eval on test-300.")
    ap.add_argument("--rs-train-dir", required=True, help="Run dir for rejection-SFT eval on train-1200.")
    ap.add_argument("--rs-test-dir", required=True, help="Run dir for rejection-SFT eval on test-300.")
    ap.add_argument("--dataset-manifest", required=True, help="Manifest JSON from build_sql_rejection_sft_dataset.py.")
    ap.add_argument("--adapter-path", required=True, help="Adapter path for rejection-SFT row.")
    ap.add_argument("--output-csv", default=None, help="Optional output CSV path.")
    ap.add_argument("--output-json", default=None, help="Optional output JSON path.")
    return ap.parse_args()
def _load_success_module() -> Any:
    module_path = PROJECT_ROOT / "scripts" / "calc_success_rate_summary_csv.py"
    spec = importlib.util.spec_from_file_location("calc_success_rate_summary_csv", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
def compute_success_rate(run_dir: str | Path) -> float:
    run_path = Path(run_dir)
    summary_csv = run_path / "summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"summary.csv not found in {run_path}")
    module = _load_success_module()
    stats = module.compute_one_csv(
        summary_csv,
        logs_ic_sql_dir=None,
        auto_logs_ic_sql=True,
        corrupt_cache={},
    )
    return float(stats.success_rate)
def main() -> int:
    args = parse_args()
    with Path(args.dataset_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    shared_counts = {
        "successful_train_tasks_used_for_sft": int(
            manifest.get("successful_tasks_with_valid_sft_steps", 0)
        ),
        "total_sft_steps": int(manifest.get("total_step_samples", 0)),
    }
    rows = [
        {
            "model": "base",
            "train1200_success_rate": compute_success_rate(args.base_train_dir),
            "test300_success_rate": compute_success_rate(args.base_test_dir),
            "successful_train_tasks_used_for_sft": shared_counts["successful_train_tasks_used_for_sft"],
            "total_sft_steps": shared_counts["total_sft_steps"],
            "adapter_path": "",
        },
        {
            "model": "rejection_sft",
            "train1200_success_rate": compute_success_rate(args.rs_train_dir),
            "test300_success_rate": compute_success_rate(args.rs_test_dir),
            "successful_train_tasks_used_for_sft": shared_counts["successful_train_tasks_used_for_sft"],
            "total_sft_steps": shared_counts["total_sft_steps"],
            "adapter_path": str(Path(args.adapter_path)),
        },
    ]
    print("SQL rejection-SFT summary")
    for row in rows:
        print(
            f"  {row['model']}: train1200={row['train1200_success_rate']:.2f}% "
            f"test300={row['test300_success_rate']:.2f}% "
            f"sft_tasks={row['successful_train_tasks_used_for_sft']} "
            f"sft_steps={row['total_sft_steps']}"
        )
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "train1200_success_rate",
                    "test300_success_rate",
                    "successful_train_tasks_used_for_sft",
                    "total_sft_steps",
                    "adapter_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote CSV:  {output_csv}")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"  Wrote JSON: {output_json}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
