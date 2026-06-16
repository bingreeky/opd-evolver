#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES_ROOT = PROJECT_ROOT / "workspace" / "baselines" / "ama_bench"
CSV_FIELDS = [
    "gen_model",
    "method",
    "dataset",
    "backend",
    "total_questions",
    "accuracy",
    "avg_score",
    "answers_file",
    "results_path",
]
def parse_run_metadata(results_path: Path, baselines_root: Path) -> tuple[str, str, str, str]:
    rel = results_path.relative_to(baselines_root)
    parts = rel.parts
    if len(parts) < 3:
        return "unknown", "unknown", "unknown", "unknown"
    gen_model = parts[0]
    method = parts[1]
    dataset = parts[2]
    if method == "memory_provider" and len(parts) >= 4:
        backend = parts[3]
    elif method == "longcontext":
        backend = "longcontext"
    elif method == "opd_evolver":
        backend = "opd_hierarchical"
    else:
        backend = parts[3] if len(parts) >= 4 else method
    return gen_model, method, dataset, backend
def find_best_results_file(results_dir: Path, answers_stem: str) -> Path | None:
    stable = results_dir / f"results_{answers_stem}.json"
    if stable.is_file():
        return stable
    stamped = sorted(
        results_dir.glob(f"results_{answers_stem}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return stamped[0] if stamped else None
def discover_result_files(baselines_root: Path) -> dict[Path, Path]:
    chosen: dict[Path, Path] = {}
    for answers_path in sorted(baselines_root.glob("**/answers_*.jsonl")):
        results_path = find_best_results_file(answers_path.parent, answers_path.stem)
        if results_path is not None:
            chosen[answers_path] = results_path
    return chosen
def load_summary(results_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data
def build_rows(baselines_root: Path) -> list[dict[str, Any]]:
    mapping = discover_result_files(baselines_root)
    rows: list[dict[str, Any]] = []
    for answers_path, results_path in sorted(mapping.items(), key=lambda x: str(x[0])):
        summary = load_summary(results_path)
        if summary is None:
            continue
        overall = summary.get("overall") or {}
        gen_model, method, dataset, backend = parse_run_metadata(results_path, baselines_root)
        rows.append(
            {
                "gen_model": gen_model,
                "method": method,
                "dataset": dataset,
                "backend": backend,
                "total_questions": overall.get("total_questions", 0),
                "accuracy": overall.get("accuracy", 0.0),
                "avg_score": overall.get("avg_score", 0.0),
                "answers_file": str(answers_path.relative_to(baselines_root)),
                "results_path": str(results_path.relative_to(baselines_root)),
            }
        )
    rows.sort(key=lambda r: (-float(r.get("accuracy") or 0), str(r.get("gen_model")), str(r.get("backend"))))
    return rows
def build_pivot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pivot: dict[str, dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        gen_model = row["gen_model"]
        key = f"{row['method']}/{row['backend']}"
        pivot[gen_model][key] = row["accuracy"]
    return {gm: dict(backends) for gm, backends in sorted(pivot.items())}
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate AMA-Bench judge results.")
    ap.add_argument(
        "--baselines-root",
        type=Path,
        default=DEFAULT_BASELINES_ROOT,
    )
    ap.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing files.")
    return ap.parse_args()
def main() -> int:
    args = parse_args()
    baselines_root = args.baselines_root.expanduser().resolve()
    if not baselines_root.is_dir():
        raise SystemExit(f"Baselines root not found: {baselines_root}")
    rows = build_rows(baselines_root)
    payload = {
        "baselines_root": str(baselines_root),
        "run_count": len(rows),
        "rows": rows,
        "pivot_by_gen_model": build_pivot(rows),
    }
    if args.stdout:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    csv_path = baselines_root / "judge_leaderboard.csv"
    json_path = baselines_root / "judge_leaderboard.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {json_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
