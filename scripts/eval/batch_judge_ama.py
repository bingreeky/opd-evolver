#!/usr/bin/env python3
from __future__ import annotations
import argparse
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
DEFAULT_BASELINES_ROOT = PROJECT_ROOT / "workspace" / "baselines" / "ama_bench"
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "ama" / "open_end_qa_set.jsonl"
DEFAULT_JUDGE_CONFIG = PROJECT_ROOT / "reference" / "AMA-Bench" / "configs" / "llm_judge.yaml"
def _load_bench_module():
    bench_path = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_ama.py"
    spec = importlib.util.spec_from_file_location("bench_simple_ama_batch", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {bench_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_simple_ama_batch"] = mod
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
    if not stable.is_file():
        return False
    try:
        data = json.loads(stable.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    overall = data.get("overall") if isinstance(data, dict) else None
    if not isinstance(overall, dict):
        return False
    total = overall.get("total_questions", 0)
    try:
        return int(total) > 0
    except (TypeError, ValueError):
        return False
def discover_jobs(
    baselines_root: Path,
    *,
    answers_file: Path | None,
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
        if skip_empty and line_count == 0:
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
def check_judge_health(host: str, port: int) -> None:
    health_url = f"http://{host}:{port}/health"
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
            f"Start vLLM on port {port} with served model qwen/qwen3-32b, then retry."
        )
def manifest_record(
    job: AnswersJob,
    *,
    status: str,
    elapsed_s: float = 0.0,
    overall: dict | None = None,
    error: str = "",
) -> dict[str, Any]:
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
        total = overall.get("total_questions", 0)
        accuracy = overall.get("accuracy", 0.0)
        rec["total"] = total
        rec["accuracy"] = accuracy
        rec["avg_score"] = overall.get("avg_score")
        try:
            rec["correct"] = int(round(float(accuracy) * int(total)))
        except (TypeError, ValueError):
            rec["correct"] = None
    return rec
def judge_one_file(
    bench: Any,
    job: AnswersJob,
    *,
    test_file: Path,
    judge_config: Path,
    judge_server: str,
    judge_vllm_host: str,
    judge_vllm_port: int,
    judge_max_concurrency: int,
) -> dict[str, Any]:
    return bench.evaluate_answers(
        answers_file=job.answers_path,
        test_file=test_file,
        judge_config=judge_config,
        judge_server=judge_server,
        output_file=job.results_path,
        judge_max_concurrency=judge_max_concurrency,
        judge_vllm_host=judge_vllm_host,
        judge_vllm_port=judge_vllm_port,
    )
def append_manifest(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
def run_aggregate(baselines_root: Path) -> None:
    agg_script = PROJECT_ROOT / "scripts" / "eval" / "aggregate_ama_judge_results.py"
    subprocess.run(
        [sys.executable, str(agg_script), "--baselines-root", str(baselines_root)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch judge AMA-Bench answers with Qwen3-32B.")
    ap.add_argument(
        "--baselines-root",
        type=Path,
        default=Path(os.environ.get("AMA_BASELINES_ROOT", DEFAULT_BASELINES_ROOT)),
    )
    ap.add_argument(
        "--test-file",
        type=Path,
        default=Path(os.environ.get("AMA_TEST_FILE", DEFAULT_TEST_FILE)),
    )
    ap.add_argument(
        "--judge-config",
        type=Path,
        default=Path(os.environ.get("AMA_JUDGE_CONFIG", DEFAULT_JUDGE_CONFIG)),
    )
    ap.add_argument("--answers-file", type=Path, default=None, help="Judge a single answers JSONL file.")
    ap.add_argument("--dry-run", action="store_true", help="List jobs without calling the judge.")
    ap.add_argument("--force", action="store_true", help="Re-judge even if results file exists.")
    ap.add_argument("--skip-empty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-aggregate", action="store_true", help="Skip leaderboard aggregation after batch.")
    ap.add_argument("--judge-server", choices=["api", "vllm"], default="vllm")
    ap.add_argument("--judge-vllm-host", default=os.environ.get("JUDGE_VLLM_HOST", "127.0.0.1"))
    ap.add_argument("--judge-vllm-port", type=int, default=int(os.environ.get("JUDGE_VLLM_PORT", "8006")))
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
def _judge_files_serial(
    bench: Any,
    jobs: list[AnswersJob],
    args: argparse.Namespace,
    test_file: Path,
    judge_config: Path,
    manifest_path: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with log_path.open("a", encoding="utf-8") as log_f:
        for i, job in enumerate(jobs, 1):
            print(f"[{i}/{len(jobs)}] {job.answers_path} ({job.line_count} episodes)")
            t0 = time.perf_counter()
            try:
                summary = judge_one_file(
                    bench,
                    job,
                    test_file=test_file,
                    judge_config=judge_config,
                    judge_server=args.judge_server,
                    judge_vllm_host=args.judge_vllm_host,
                    judge_vllm_port=args.judge_vllm_port,
                    judge_max_concurrency=args.judge_max_concurrency,
                )
                elapsed = time.perf_counter() - t0
                overall = summary.get("overall", {})
                rec = manifest_record(job, status="ok", elapsed_s=elapsed, overall=overall)
                correct = rec.get("correct")
                line = (
                    f"ok {job.answers_path} accuracy={overall.get('accuracy')} "
                    f"({correct}/{overall.get('total_questions')}) elapsed={elapsed:.1f}s\n"
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
            print(line, end="")
    return records
def _judge_file_sync(
    bench: Any,
    job: AnswersJob,
    args: argparse.Namespace,
    test_file: Path,
    judge_config: Path,
) -> tuple[AnswersJob, dict[str, Any] | None, str | None]:
    try:
        summary = judge_one_file(
            bench,
            job,
            test_file=test_file,
            judge_config=judge_config,
            judge_server=args.judge_server,
            judge_vllm_host=args.judge_vllm_host,
            judge_vllm_port=args.judge_vllm_port,
            judge_max_concurrency=args.judge_max_concurrency,
        )
        return job, summary, None
    except Exception as exc:
        return job, None, str(exc)
def _judge_files_parallel(
    bench: Any,
    jobs: list[AnswersJob],
    args: argparse.Namespace,
    test_file: Path,
    judge_config: Path,
    manifest_path: Path,
    log_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    workers = max(1, args.file_workers)
    with log_path.open("a", encoding="utf-8") as log_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_judge_file_sync, bench, job, args, test_file, judge_config): job
                for job in jobs
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
                    correct = rec.get("correct")
                    line = (
                        f"ok {job.answers_path} accuracy={overall.get('accuracy')} "
                        f"({correct}/{overall.get('total_questions')}) elapsed={elapsed:.1f}s\n"
                    )
                records.append(rec)
                append_manifest(manifest_path, rec)
                log_f.write(line)
                log_f.flush()
                print(line, end="")
    return records
def main() -> int:
    args = parse_args()
    baselines_root = args.baselines_root.expanduser().resolve()
    if not baselines_root.is_dir():
        raise SystemExit(f"Baselines root not found: {baselines_root}")
    test_file = args.test_file.expanduser().resolve()
    if not test_file.is_file():
        raise SystemExit(f"Test file not found: {test_file}")
    judge_config = args.judge_config.expanduser().resolve()
    if not judge_config.is_file():
        raise SystemExit(f"Judge config not found: {judge_config}")
    answers_file = args.answers_file.expanduser().resolve() if args.answers_file else None
    jobs = discover_jobs(
        baselines_root,
        answers_file=answers_file,
        skip_empty=args.skip_empty,
        force=args.force,
    )
    to_run = [j for j in jobs if j.skip_reason is None]
    skipped = [j for j in jobs if j.skip_reason is not None]
    if args.dry_run:
        print(f"baselines_root={baselines_root}")
        print(f"test_file={test_file}")
        print(f"judge_config={judge_config}")
        print(f"judge=vllm @ {args.judge_vllm_host}:{args.judge_vllm_port}")
        print(f"to_judge={len(to_run)} skipped={len(skipped)} total_discovered={len(jobs)}")
        print(f"total_episodes={sum(j.line_count for j in to_run)}")
        print(f"est_qa_pairs={sum(j.line_count * 12 for j in to_run)} (12 Q/ep)")
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
        if not args.no_aggregate:
            print("Aggregating leaderboard from existing results...")
            run_aggregate(baselines_root)
        return 0
    (baselines_root / "_judge_logs").mkdir(parents=True, exist_ok=True)
    bench = _load_bench_module()
    if args.judge_server == "vllm":
        check_judge_health(args.judge_vllm_host, args.judge_vllm_port)
    else:
        print(f"judge=api config={judge_config}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = baselines_root / "_judge_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"batch_{stamp}.log"
    manifest_path = baselines_root / "judge_manifest.jsonl"
    print(f"Judging {len(to_run)} files ({sum(j.line_count for j in to_run)} episodes)")
    print(f"log={log_path}")
    print(f"manifest={manifest_path}")
    for j in skipped:
        rec = manifest_record(j, status=j.skip_reason or "skipped")
        append_manifest(manifest_path, rec)
    if args.file_workers <= 1:
        _judge_files_serial(bench, to_run, args, test_file, judge_config, manifest_path, log_path)
    else:
        _judge_files_parallel(bench, to_run, args, test_file, judge_config, manifest_path, log_path)
    if not args.no_aggregate:
        print("Aggregating leaderboard...")
        run_aggregate(baselines_root)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
