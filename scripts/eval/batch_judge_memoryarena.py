#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINES_ROOT = PROJECT_ROOT / "workspace" / "baselines" / "memoryarena"
OPD_EVOLVER_MARKER = "/opd_evolver/"
def _load_bench_module():
    bench_path = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_memoryarena.py"
    spec = importlib.util.spec_from_file_location("bench_simple_memoryarena", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {bench_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_simple_memoryarena"] = mod
    spec.loader.exec_module(mod)
    return mod
@dataclass
class AnswersJob:
    answers_path: Path
    line_count: int
    gen_model: str
    method: str
    dataset: str
    backend: str
    results_path: Path
    skip_reason: str | None = None
def parse_run_metadata(answers_path: Path, baselines_root: Path) -> tuple[str, str, str, str]:
    rel = answers_path.relative_to(baselines_root)
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
def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
def results_path_for(answers_path: Path) -> Path:
    return answers_path.parent / f"results_{answers_path.stem}.json"
def has_existing_results(answers_path: Path) -> bool:
    stable = results_path_for(answers_path)
    if stable.is_file():
        return True
    return bool(list(answers_path.parent.glob(f"results_{answers_path.stem}_*.json")))
def discover_jobs(
    baselines_root: Path,
    *,
    answers_file: Path | None,
    include_opd_evolver: bool,
    skip_empty: bool,
    force: bool,
) -> list[AnswersJob]:
    if answers_file is not None:
        candidates = [answers_file.resolve()]
    else:
        candidates = sorted(baselines_root.glob("**/answers_*.jsonl"))
    jobs: list[AnswersJob] = []
    for path in candidates:
        try:
            path.relative_to(baselines_root.resolve())
        except ValueError as exc:
            raise SystemExit(f"--answers-file must be under {baselines_root}: {path}") from exc
        gen_model, method, dataset, backend = parse_run_metadata(path, baselines_root)
        line_count = count_lines(path)
        results_path = results_path_for(path)
        skip_reason: str | None = None
        if OPD_EVOLVER_MARKER in path.as_posix() and not include_opd_evolver:
            skip_reason = "skipped_opd_evolver"
        elif skip_empty and line_count == 0:
            skip_reason = "skipped_empty"
        elif not force and has_existing_results(path):
            skip_reason = "skipped_existing"
        jobs.append(
            AnswersJob(
                answers_path=path,
                line_count=line_count,
                gen_model=gen_model,
                method=method,
                dataset=dataset,
                backend=backend,
                results_path=results_path,
                skip_reason=skip_reason,
            )
        )
    return jobs
def check_judge_health(base_url: str) -> None:
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    health_url = f"{url}/health"
    try:
        proc = subprocess.run(
            ["curl", "-sf", health_url],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return
    if proc.returncode != 0:
        raise SystemExit(
            f"Judge health check failed: {health_url}\n"
            "Start vLLM on port 8006 with served model qwen/qwen3-32b, then retry."
        )
def manifest_record(job: AnswersJob, *, status: str, elapsed_s: float = 0.0, overall: dict | None = None, error: str = "") -> dict[str, Any]:
    rec: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "answers_file": str(job.answers_path),
        "results_file": str(job.results_path) if status == "ok" else "",
        "gen_model": job.gen_model,
        "method": job.method,
        "dataset": job.dataset,
        "backend": job.backend,
        "line_count": job.line_count,
        "elapsed_s": round(elapsed_s, 2),
        "error": error,
    }
    if overall:
        rec["total"] = overall.get("total_questions")
        rec["correct"] = overall.get("correct")
        rec["accuracy"] = overall.get("accuracy")
    return rec
async def judge_one_file(
    bench: Any,
    job: AnswersJob,
    *,
    judge_model: str,
    base_url: str,
    api_key: str,
    judge_max_concurrency: int,
) -> dict[str, Any]:
    return await bench.evaluate_answers(
        answers_file=job.answers_path,
        output_file=job.results_path,
        judge_model=judge_model,
        base_url=base_url,
        api_key=api_key,
        judge_max_concurrency=judge_max_concurrency,
    )
def append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
def run_aggregate() -> None:
    agg_script = PROJECT_ROOT / "scripts" / "eval" / "aggregate_memoryarena_judge_results.py"
    subprocess.run([sys.executable, str(agg_script)], check=True, cwd=str(PROJECT_ROOT))
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch judge MemoryArena answers with Qwen3-32B.")
    ap.add_argument(
        "--baselines-root",
        type=Path,
        default=Path(os.environ.get("BASELINES_ROOT", DEFAULT_BASELINES_ROOT)),
    )
    ap.add_argument("--answers-file", type=Path, default=None, help="Judge a single answers JSONL file.")
    ap.add_argument("--dry-run", action="store_true", help="List jobs without calling the judge.")
    ap.add_argument("--force", action="store_true", help="Re-judge even if results file exists.")
    ap.add_argument("--skip-empty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--include-opd-evolver",
        action="store_true",
        help="Include answers under opd_evolver/ (default: skip).",
    )
    ap.add_argument("--no-aggregate", action="store_true", help="Skip leaderboard aggregation after batch.")
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8006/v1"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "qwen/qwen3-32b"))
    ap.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    ap.add_argument(
        "--judge-max-concurrency",
        type=int,
        default=int(os.environ.get("JUDGE_MAX_CONCURRENCY", "8")),
    )
    ap.add_argument(
        "--file-workers",
        type=int,
        default=int(os.environ.get("FILE_WORKERS", "1")),
        help="Parallel answers files (default 1 for single vLLM).",
    )
    return ap.parse_args()
async def _judge_files_serial(
    bench: Any,
    jobs: list[AnswersJob],
    args: argparse.Namespace,
    base_url: str,
    manifest_path: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with log_path.open("a", encoding="utf-8") as log_f:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {job.answers_path} ({job.line_count} sessions)")
            t0 = time.perf_counter()
            try:
                summary = await judge_one_file(
                    bench,
                    job,
                    judge_model=args.judge_model,
                    base_url=base_url,
                    api_key=args.judge_api_key,
                    judge_max_concurrency=args.judge_max_concurrency,
                )
                elapsed = time.perf_counter() - t0
                overall = summary.get("overall", {})
                rec = manifest_record(job, status="ok", elapsed_s=elapsed, overall=overall)
                line = (
                    f"ok {job.answers_path} accuracy={overall.get('accuracy')} "
                    f"({overall.get('correct')}/{overall.get('total_questions')}) elapsed={elapsed:.1f}s\n"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                rec = manifest_record(job, status="error", elapsed_s=elapsed, error=str(exc))
                line = f"error {job.answers_path} {exc}\n"
                print(line, file=sys.stderr)
            records.append(rec)
            append_manifest(manifest_path, rec)
            log_f.write(line)
            log_f.flush()
    return records
def _judge_file_sync(
    bench: Any,
    job: AnswersJob,
    args: argparse.Namespace,
    base_url: str,
) -> tuple[AnswersJob, dict[str, Any] | None, str | None]:
    try:
        summary = asyncio.run(
            judge_one_file(
                bench,
                job,
                judge_model=args.judge_model,
                base_url=base_url,
                api_key=args.judge_api_key,
                judge_max_concurrency=args.judge_max_concurrency,
            )
        )
        return job, summary, None
    except Exception as exc:
        return job, None, str(exc)
def _judge_files_parallel(
    bench: Any,
    jobs: list[AnswersJob],
    args: argparse.Namespace,
    base_url: str,
    manifest_path: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    workers = max(1, args.file_workers)
    with log_path.open("a", encoding="utf-8") as log_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_judge_file_sync, bench, job, args, base_url): job for job in jobs
            }
            for fut in as_completed(futures):
                job = futures[fut]
                t0 = time.perf_counter()
                job, summary, err = fut.result()
                elapsed = time.perf_counter() - t0
                if err:
                    rec = manifest_record(job, status="error", elapsed_s=elapsed, error=err)
                    line = f"error {job.answers_path} {err}\n"
                    print(line, file=sys.stderr)
                else:
                    overall = (summary or {}).get("overall", {})
                    rec = manifest_record(job, status="ok", elapsed_s=elapsed, overall=overall)
                    line = (
                        f"ok {job.answers_path} accuracy={overall.get('accuracy')} "
                        f"({overall.get('correct')}/{overall.get('total_questions')}) elapsed={elapsed:.1f}s\n"
                    )
                records.append(rec)
                append_manifest(manifest_path, rec)
                log_f.write(line)
                log_f.flush()
                print(line, end="")
    return records
async def async_main() -> int:
    args = parse_args()
    baselines_root = args.baselines_root.expanduser().resolve()
    if not baselines_root.is_dir():
        raise SystemExit(f"Baselines root not found: {baselines_root}")
    answers_file = args.answers_file.expanduser().resolve() if args.answers_file else None
    if answers_file and OPD_EVOLVER_MARKER in answers_file.as_posix() and not args.include_opd_evolver:
        raise SystemExit(
            f"Refusing to judge opd_evolver answers without --include-opd-evolver: {answers_file}"
        )
    jobs = discover_jobs(
        baselines_root,
        answers_file=answers_file,
        include_opd_evolver=args.include_opd_evolver,
        skip_empty=args.skip_empty,
        force=args.force,
    )
    to_run = [j for j in jobs if j.skip_reason is None]
    skipped = [j for j in jobs if j.skip_reason is not None]
    if args.dry_run:
        print(f"baselines_root={baselines_root}")
        print(f"judge={args.judge_model} @ {args.judge_base_url}")
        print(f"include_opd_evolver={args.include_opd_evolver}")
        print(f"to_judge={len(to_run)} skipped={len(skipped)} total_discovered={len(jobs)}")
        print(f"total_sessions={sum(j.line_count for j in to_run)}")
        print("\n--- will judge ---")
        for j in to_run:
            print(
                f"  {j.line_count:4d}  {j.gen_model}  {j.method}  {j.dataset}  {j.backend}\n"
                f"         {j.answers_path}"
            )
        print("\n--- skipped ---")
        for j in skipped:
            print(f"  [{j.skip_reason}] {j.answers_path}")
        return 0
    if not to_run:
        print("No answers files to judge.")
        return 0
    bench = _load_bench_module()
    base_url = bench._normalize_base_url(args.judge_base_url)
    check_judge_health(base_url)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = baselines_root / "_judge_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"batch_{stamp}.log"
    manifest_path = baselines_root / "judge_manifest.jsonl"
    print(f"Judging {len(to_run)} files ({sum(j.line_count for j in to_run)} sessions)")
    print(f"log={log_path}")
    print(f"manifest={manifest_path}")
    for j in skipped:
        rec = manifest_record(j, status=j.skip_reason or "skipped")
        append_manifest(manifest_path, rec)
    if args.file_workers <= 1:
        await _judge_files_serial(bench, to_run, args, base_url, manifest_path, log_path)
    else:
        _judge_files_parallel(bench, to_run, args, base_url, manifest_path, log_path)
    if not args.no_aggregate:
        print("Aggregating leaderboard...")
        run_aggregate()
    return 0
def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
if __name__ == "__main__":
    main()
