#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()
def _script_dir() -> Path:
    return Path(__file__).resolve().parent
def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
def _scored_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_scored{dataset_path.suffix}")
def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
def _concat_jsonl(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", encoding="utf-8") as fout:
        for path in inputs:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    fout.write(line if line.endswith("\n") else f"{line}\n")
                    rows += 1
    return rows
def _maybe_score_writer(
    *,
    py: str,
    script: Path,
    dataset_path: Path | None,
    storage_dir: Path,
    output_path: Path,
    lambda_days: float,
    only_missing: bool,
    skip_missing: bool,
) -> int | None:
    if dataset_path is None:
        return None
    if not dataset_path.is_file():
        if skip_missing:
            print(f"[skip] writer dataset missing: {dataset_path}", flush=True)
            return None
        raise FileNotFoundError(f"Writer dataset missing: {dataset_path}")
    cmd = [
        py,
        str(script),
        "--dataset",
        str(dataset_path),
        "--storage-dir",
        str(storage_dir),
        "--output",
        str(output_path),
        "--lambda-days",
        str(lambda_days),
    ]
    if only_missing:
        cmd.append("--only-missing")
    _run(cmd)
    return _count_jsonl(output_path)
def _maybe_score_selector(
    *,
    py: str,
    script: Path,
    dataset_path: Path | None,
    storage_dir: Path,
    output_path: Path,
    lambda_days: float,
    only_missing: bool,
    threshold: float,
    score_mode: str,
    w_success: float,
    w_mean: float,
    skip_missing: bool,
) -> int | None:
    if dataset_path is None:
        return None
    if not dataset_path.is_file():
        if skip_missing:
            print(f"[skip] selector dataset missing: {dataset_path}", flush=True)
            return None
        raise FileNotFoundError(f"Selector dataset missing: {dataset_path}")
    cmd = [
        py,
        str(script),
        "--dataset",
        str(dataset_path),
        "--storage-dir",
        str(storage_dir),
        "--output",
        str(output_path),
        "--lambda-days",
        str(lambda_days),
        "--threshold",
        str(threshold),
        "--score-mode",
        score_mode,
        "--w-success",
        str(w_success),
        "--w-mean",
        str(w_mean),
    ]
    if only_missing:
        cmd.append("--only-missing")
    _run(cmd)
    return _count_jsonl(output_path)
def main() -> None:
    here = _script_dir()
    writer_py = here / "annotate_writer_dataset_scores.py"
    selector_py = here / "annotate_selector_dataset_scores.py"
    ap = argparse.ArgumentParser(
        description="Score grouped AWM memory writer/selector datasets and merge them.",
    )
    ap.add_argument(
        "--manifest",
        default="workspace/memory/awm_scale_3000/memory_groups.json",
        help="Path to memory_groups.json produced by the AWM grouped rollout.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Directory for merged scored JSONL files (default: manifest base_storage_dir).",
    )
    ap.add_argument("--no-writer", action="store_true", help="Skip writer dataset scoring")
    ap.add_argument("--no-selector", action="store_true", help="Skip selector dataset scoring")
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip groups whose writer/selector JSONL is missing instead of failing.",
    )
    ap.add_argument("--lambda-days", type=float, default=0.1)
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="Forwarded to both annotators.",
    )
    ap.add_argument("--threshold", type=float, default=0.0, help="Selector binary threshold")
    ap.add_argument(
        "--score-mode",
        choices=("binary", "weighted"),
        default="binary",
        help="Selector composite score mode.",
    )
    ap.add_argument("--w-success", type=float, default=1.0, help="Selector weighted mode")
    ap.add_argument("--w-mean", type=float, default=1.0, help="Selector weighted mode")
    args = ap.parse_args()
    score_writer = not args.no_writer
    score_selector = not args.no_selector
    if not score_writer and not score_selector:
        ap.error("Nothing to score: remove --no-writer or --no-selector")
    manifest_path = _expand_path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    groups = manifest.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"Manifest has no groups list: {manifest_path}")
    base_storage_dir = _expand_path(manifest.get("base_storage_dir") or manifest_path.parent)
    output_dir = _expand_path(args.output_dir) if args.output_dir else base_storage_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    writer_outputs: list[Path] = []
    selector_outputs: list[Path] = []
    group_summaries: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id", ""))
        storage_dir = _expand_path(group["storage_dir"])
        summary: dict[str, Any] = {
            "group_id": group_id,
            "storage_dir": str(storage_dir),
            "start_job_index": group.get("start_job_index"),
            "end_job_index": group.get("end_job_index"),
        }
        if score_writer:
            raw_writer = group.get("writer_dataset_path")
            writer_dataset = _expand_path(raw_writer) if raw_writer else None
            writer_output = _scored_path(writer_dataset) if writer_dataset is not None else None
            writer_rows = (
                _maybe_score_writer(
                    py=py,
                    script=writer_py,
                    dataset_path=writer_dataset,
                    storage_dir=storage_dir,
                    output_path=writer_output,
                    lambda_days=args.lambda_days,
                    only_missing=args.only_missing,
                    skip_missing=args.skip_missing,
                )
                if writer_output is not None
                else None
            )
            summary["writer_dataset_path"] = str(writer_dataset) if writer_dataset else None
            summary["writer_scored_path"] = str(writer_output) if writer_output else None
            summary["writer_rows"] = writer_rows
            if writer_rows is not None and writer_output is not None:
                writer_outputs.append(writer_output)
        if score_selector:
            raw_selector = group.get("selector_dataset_path")
            selector_dataset = _expand_path(raw_selector) if raw_selector else None
            selector_output = _scored_path(selector_dataset) if selector_dataset is not None else None
            selector_rows = (
                _maybe_score_selector(
                    py=py,
                    script=selector_py,
                    dataset_path=selector_dataset,
                    storage_dir=storage_dir,
                    output_path=selector_output,
                    lambda_days=args.lambda_days,
                    only_missing=args.only_missing,
                    threshold=args.threshold,
                    score_mode=args.score_mode,
                    w_success=args.w_success,
                    w_mean=args.w_mean,
                    skip_missing=args.skip_missing,
                )
                if selector_output is not None
                else None
            )
            summary["selector_dataset_path"] = str(selector_dataset) if selector_dataset else None
            summary["selector_scored_path"] = str(selector_output) if selector_output else None
            summary["selector_rows"] = selector_rows
            if selector_rows is not None and selector_output is not None:
                selector_outputs.append(selector_output)
        group_summaries.append(summary)
    writer_merged = output_dir / "memory_writer_dataset_scored.jsonl"
    selector_merged = output_dir / "memory_selector_dataset_scored.jsonl"
    writer_rows = _concat_jsonl(writer_outputs, writer_merged) if score_writer else None
    selector_rows = _concat_jsonl(selector_outputs, selector_merged) if score_selector else None
    scored_manifest = {
        "input_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "writer_output": str(writer_merged) if score_writer else None,
        "writer_rows": writer_rows,
        "selector_output": str(selector_merged) if score_selector else None,
        "selector_rows": selector_rows,
        "groups": group_summaries,
    }
    scored_manifest_path = output_dir / "memory_groups_scored_manifest.json"
    with scored_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(scored_manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if score_writer:
        print(f"Merged writer rows: {writer_rows} -> {writer_merged}", flush=True)
    if score_selector:
        print(f"Merged selector rows: {selector_rows} -> {selector_merged}", flush=True)
    print(f"Wrote scored manifest: {scored_manifest_path}", flush=True)
if __name__ == "__main__":
    main()
