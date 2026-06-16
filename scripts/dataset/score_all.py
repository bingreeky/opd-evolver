#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
import sys
def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))
def _run(py: str, argv: list[str]) -> None:
    cmd = [py, *argv]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
def main() -> None:
    here = _script_dir()
    writer_py = os.path.join(here, "annotate_writer_dataset_scores.py")
    selector_py = os.path.join(here, "annotate_selector_dataset_scores.py")
    ap = argparse.ArgumentParser(
        description="Annotate writer and/or selector memory JSONL with hybrid scores.",
    )
    ap.add_argument(
        "--storage-dir",
        required=True,
        help="Memory storage directory (tier JSON + usage_logs.jsonl)",
    )
    ap.add_argument("--writer-dataset", default=None, help="Writer JSONL input path")
    ap.add_argument(
        "--writer-output",
        default=None,
        help="Writer JSONL output (default: overwrite --writer-dataset)",
    )
    ap.add_argument("--selector-dataset", default=None, help="Selector JSONL input path")
    ap.add_argument(
        "--selector-output",
        default=None,
        help="Selector JSONL output (default: overwrite --selector-dataset)",
    )
    ap.add_argument("--lambda-days", type=float, default=0.1, help="Forwarded to both annotators")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Forwarded to both annotators",
    )
    ap.add_argument("--threshold", type=float, default=0.0, help="Selector: binary threshold")
    ap.add_argument(
        "--score-mode",
        choices=("binary", "weighted"),
        default="binary",
        help="Selector: composite score mode",
    )
    ap.add_argument("--w-success", type=float, default=1.0, help="Selector: weighted mode")
    ap.add_argument("--w-mean", type=float, default=1.0, help="Selector: weighted mode")
    args = ap.parse_args()
    if not args.writer_dataset and not args.selector_dataset:
        ap.error("Provide at least one of --writer-dataset or --selector-dataset")
    storage = os.path.expanduser(args.storage_dir)
    py = sys.executable
    if args.writer_dataset:
        w_argv = [
            writer_py,
            "--dataset",
            os.path.expanduser(args.writer_dataset),
            "--storage-dir",
            storage,
            "--lambda-days",
            str(args.lambda_days),
        ]
        if args.writer_output:
            w_argv.extend(["--output", os.path.expanduser(args.writer_output)])
        if args.only_missing:
            w_argv.append("--only-missing")
        _run(py, w_argv)
    if args.selector_dataset:
        s_argv = [
            selector_py,
            "--dataset",
            os.path.expanduser(args.selector_dataset),
            "--storage-dir",
            storage,
            "--lambda-days",
            str(args.lambda_days),
            "--threshold",
            str(args.threshold),
            "--score-mode",
            args.score_mode,
            "--w-success",
            str(args.w_success),
            "--w-mean",
            str(args.w_mean),
        ]
        if args.selector_output:
            s_argv.extend(["--output", os.path.expanduser(args.selector_output)])
        if args.only_missing:
            s_argv.append("--only-missing")
        _run(py, s_argv)
if __name__ == "__main__":
    main()
