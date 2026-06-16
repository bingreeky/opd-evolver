#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    s = str(value).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None
@dataclass(frozen=True)
class CsvStats:
    file: str
    total_rows: int
    ignored_corrupt_gold: int
    counted: int
    success_true: int
    success_rate: float
    read_errors: int
    missing_success: int
    missing_corrupt_lookup: int
    corrupt_source: str
def _iter_csv_files(dir_path: Path, pattern: str, recursive: bool) -> list[Path]:
    if dir_path.is_file():
        return [dir_path]
    glob_pat = pattern
    if recursive and not pattern.startswith("**/"):
        glob_pat = f"**/{pattern}"
    files = sorted(dir_path.glob(glob_pat))
    return [p for p in files if p.is_file() and p.suffix.lower() == ".csv"]
def _resolve_input_csvs(
    p: Path,
    pattern: str,
    recursive: bool,
    use_glob: bool,
) -> tuple[list[Path], str]:
    if p.is_file():
        if p.suffix.lower() != ".csv":
            return [], "file (not .csv)"
        return [p], "csv file"
    if not p.is_dir():
        return [], "invalid"
    summary = p / "summary.csv"
    if summary.is_file() and not use_glob:
        return [summary.resolve()], "run directory (summary.csv)"
    files = _iter_csv_files(p, pattern=pattern, recursive=recursive)
    return files, "directory (glob)"
DEFAULT_CSV_PATTERN = "*.csv"
def load_corrupt_gold_from_ic_logs(logs_dir: Path) -> tuple[dict[str, bool], int]:
    corrupt_map: dict[str, bool] = {}
    parse_errors = 0
    if not logs_dir.is_dir():
        return corrupt_map, parse_errors
    for fp in sorted(logs_dir.glob("*.json")):
        if not fp.is_file():
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        info = data.get("info") or {}
        raw_cg = info.get("corrupt_gold", False)
        cg = _to_bool(raw_cg)
        if cg is None:
            cg = bool(raw_cg)
        task_id = data.get("task_id")
        if not task_id and data.get("query_idx") is not None:
            try:
                task_id = f"sql_{int(data['query_idx'])}"
            except (TypeError, ValueError):
                task_id = None
        if task_id is None:
            continue
        corrupt_map[str(task_id)] = cg is True
    return corrupt_map, parse_errors
def _resolve_corrupt_map(
    csv_path: Path,
    logs_ic_sql_dir: Path | None,
    auto_logs_ic_sql: bool,
    cache: dict[str, tuple[dict[str, bool], int]],
) -> tuple[dict[str, bool] | None, int, str]:
    if logs_ic_sql_dir is not None:
        d = logs_ic_sql_dir.expanduser().resolve()
    elif auto_logs_ic_sql:
        d = (csv_path.parent / "trajectories" / "logs_ic_sql").resolve()
    else:
        return None, 0, "(csv column corrupt_gold only)"
    if not d.is_dir():
        return None, 0, f"(no dir {d})"
    key = str(d)
    if key not in cache:
        m, pe = load_corrupt_gold_from_ic_logs(d)
        cache[key] = (m, pe)
        return m, pe, str(d)
    m, _ = cache[key]
    return m, 0, str(d)
def compute_one_csv(
    file_path: Path,
    logs_ic_sql_dir: Path | None,
    auto_logs_ic_sql: bool,
    corrupt_cache: dict[str, tuple[dict[str, bool], int]],
) -> CsvStats:
    total_rows = 0
    ignored_corrupt_gold = 0
    counted = 0
    success_true = 0
    read_errors = 0
    missing_success = 0
    missing_corrupt_lookup = 0
    corrupt_map, json_parse_errors, corrupt_source = _resolve_corrupt_map(
        file_path, logs_ic_sql_dir, auto_logs_ic_sql, corrupt_cache
    )
    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return CsvStats(
                    file=str(file_path),
                    total_rows=0,
                    ignored_corrupt_gold=0,
                    counted=0,
                    success_true=0,
                    success_rate=0.0,
                    read_errors=0,
                    missing_success=0,
                    missing_corrupt_lookup=0,
                    corrupt_source=corrupt_source,
                )
            for row in reader:
                total_rows += 1
                if corrupt_map is not None:
                    tid = row.get("task_id")
                    if tid is None or str(tid).strip() == "":
                        missing_corrupt_lookup += 1
                        corrupt_val = False
                    elif str(tid) not in corrupt_map:
                        missing_corrupt_lookup += 1
                        corrupt_val = False
                    else:
                        corrupt_val = corrupt_map[str(tid)]
                else:
                    corrupt_val = _to_bool(row.get("corrupt_gold"))
                    if corrupt_val is None:
                        corrupt_val = False
                if corrupt_val is True:
                    ignored_corrupt_gold += 1
                    continue
                counted += 1
                success_val = _to_bool(row.get("success"))
                if success_val is None:
                    missing_success += 1
                    continue
                if success_val is True:
                    success_true += 1
    except OSError:
        read_errors += 1
    read_errors += json_parse_errors
    success_rate = (success_true / counted * 100.0) if counted else 0.0
    return CsvStats(
        file=str(file_path),
        total_rows=total_rows,
        ignored_corrupt_gold=ignored_corrupt_gold,
        counted=counted,
        success_true=success_true,
        success_rate=success_rate,
        read_errors=read_errors,
        missing_success=missing_success,
        missing_corrupt_lookup=missing_corrupt_lookup,
        corrupt_source=corrupt_source,
    )
def _sum_stats(stats: Iterable[CsvStats]) -> CsvStats:
    total_rows = 0
    ignored_corrupt_gold = 0
    counted = 0
    success_true = 0
    read_errors = 0
    missing_success = 0
    missing_corrupt_lookup = 0
    for s in stats:
        total_rows += s.total_rows
        ignored_corrupt_gold += s.ignored_corrupt_gold
        counted += s.counted
        success_true += s.success_true
        read_errors += s.read_errors
        missing_success += s.missing_success
        missing_corrupt_lookup += s.missing_corrupt_lookup
    success_rate = (success_true / counted * 100.0) if counted else 0.0
    return CsvStats(
        file="__TOTAL__",
        total_rows=total_rows,
        ignored_corrupt_gold=ignored_corrupt_gold,
        counted=counted,
        success_true=success_true,
        success_rate=success_rate,
        read_errors=read_errors,
        missing_success=missing_success,
        missing_corrupt_lookup=missing_corrupt_lookup,
        corrupt_source="(aggregated)",
    )
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute success==True from summary CSV; exclude corrupt_gold using "
            "trajectories/logs_ic_sql/*.json (info.corrupt_gold), aligned by task_id."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help=(
            "SQL run directory (contains summary.csv and trajectories/logs_ic_sql), "
            "or a single .csv path. If --path is a directory and summary.csv exists, "
            "only that file is used unless --use-glob is set."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_CSV_PATTERN,
        help=(
            f'Glob pattern with --use-glob on a directory (default: "{DEFAULT_CSV_PATTERN}").'
        ),
    )
    parser.add_argument(
        "--use-glob",
        action="store_true",
        help=(
            "When --path is a directory, match CSVs by --pattern instead of using only summary.csv."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for CSV files under --path (only when --path is a directory).",
    )
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="Print per-file breakdown.",
    )
    parser.add_argument(
        "--logs-ic-sql-dir",
        default=None,
        help=(
            "Override: directory with logs_ic_sql JSON. "
            "Default: <parent of summary.csv>/trajectories/logs_ic_sql."
        ),
    )
    parser.add_argument(
        "--no-auto-logs-ic-sql",
        action="store_true",
        help="Do not auto-resolve trajectories/logs_ic_sql next to each CSV; use CSV corrupt_gold column only unless --logs-ic-sql-dir is set.",
    )
    args = parser.parse_args()
    p = Path(args.path).expanduser().resolve()
    if not p.exists():
        print(f"[ERROR] Path not found: {p}")
        return 2
    csv_files, input_mode = _resolve_input_csvs(
        p, pattern=args.pattern, recursive=args.recursive, use_glob=args.use_glob
    )
    if not csv_files:
        if p.is_dir():
            print(
                f"[ERROR] No CSV to read under: {p} "
                f"(mode={input_mode!r}; add summary.csv or use --use-glob with --pattern)"
            )
        else:
            print(f"[ERROR] Not a CSV file: {p}")
        return 2
    logs_ic = Path(args.logs_ic_sql_dir).expanduser().resolve() if args.logs_ic_sql_dir else None
    corrupt_cache: dict[str, tuple[dict[str, bool], int]] = {}
    per_file = [
        compute_one_csv(
            fp,
            logs_ic_sql_dir=logs_ic,
            auto_logs_ic_sql=not args.no_auto_logs_ic_sql,
            corrupt_cache=corrupt_cache,
        )
        for fp in csv_files
    ]
    total = _sum_stats(per_file)
    print("=" * 72)
    print("CSV Success Summary (exclude corrupt_gold==True)")
    print("=" * 72)
    print(f"Input path:                {str(p)}")
    print(f"Input mode:                {input_mode}")
    print(f"Matched CSV files:         {len(csv_files)}")
    print(f"Total rows:                {total.total_rows}")
    print(f"Ignored corrupt_gold rows: {total.ignored_corrupt_gold}")
    print(f"Counted rows (denominator):{total.counted}")
    print(f"Success==True count:       {total.success_true}")
    print(f"Success rate:              {total.success_rate:.2f}%")
    if total.missing_corrupt_lookup:
        print(
            "CSV rows with empty task_id or no logs_ic_sql entry for that task_id "
            f"(corrupt_gold unknown; treated as non-corrupt): {total.missing_corrupt_lookup}"
        )
    if total.missing_success:
        print(f"Rows missing/invalid success (counted as non-success): {total.missing_success}")
    if total.read_errors:
        print(f"Read/JSON parse errors:    {total.read_errors}")
    print("=" * 72)
    if args.print_files:
        for s in per_file:
            print(
                f"{s.file}\n"
                f"  corrupt_source={s.corrupt_source}\n"
                f"  rows={s.total_rows}, ignored_corrupt_gold={s.ignored_corrupt_gold}, "
                f"counted={s.counted}, success_true={s.success_true}, rate={s.success_rate:.2f}%, "
                f"missing_success={s.missing_success}, missing_corrupt_lookup={s.missing_corrupt_lookup}, "
                f"read_errors={s.read_errors}"
            )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
