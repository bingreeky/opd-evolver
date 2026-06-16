#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_memoryarena.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from opd_evolver.memory.evolvelab_adapter import (
    EVOLVELAB_MEMORY_BACKENDS,
    REASONING_BANK_BACKEND,
)
_SHARD_DATASETS = frozenset({"formal_reasoning_math", "formal_reasoning_phys"})
def _safe_model_label(model_name: str) -> str:
    return str(model_name).replace("/", "__").replace(":", "_")
def default_output_dir(model: str, method: str, memory_backend: str | None, dataset: str) -> Path:
    base = PROJECT_ROOT / "workspace" / "baselines" / "memoryarena" / _safe_model_label(model)
    if method == "memory_provider":
        return base / "memory_provider" / dataset / (memory_backend or "unknown")
    return base / method / dataset
def _normalize_openai_base_url(host: str, port: int) -> str:
    h = host.strip() or "127.0.0.1"
    base = f"http://{h}:{port}".rstrip("/")
    return f"{base}/v1"
def _model_prefers_4b_vllm_shards(model: str) -> bool:
    m = str(model).lower().replace("_", "-")
    if "qwen3-4b" in m or "qwen3.5-4b" in m:
        return True
    if "3.5-9b" in m or "3.5_9b" in m or ("9b" in m and "4b" not in m):
        return False
    if "4b" in m:
        return True
    raise ValueError(
        f"Cannot infer 9B vs 4B vLLM shard block from --model {model!r}. "
        "Use an id like qwen/qwen3.5-9b or qwen/qwen3-4b, or omit --vllm-shard-routing "
        "and pass --openai-base-url explicitly."
    )
def _routed_openai_base_url(
    *,
    dataset: str,
    model: str,
    shard_host: str,
    shard_base_port: int,
    shard_9b_count: int,
) -> str:
    if dataset == "formal_reasoning_math":
        offset = 0
    elif dataset == "formal_reasoning_phys":
        offset = 1
    else:
        raise ValueError(dataset)
    use_4b = _model_prefers_4b_vllm_shards(model)
    port_base = shard_base_port + (shard_9b_count if use_4b else 0)
    return _normalize_openai_base_url(shard_host, port_base + offset)
def iter_jobs(
    *,
    include_longcontext: bool,
    include_opd_evolver: bool,
    include_memevolve: bool,
    include_reasoning_bank: bool,
    memevolve_backends: tuple[str, ...] | None,
) -> Iterator[tuple[str, str | None]]:
    if include_longcontext:
        yield "longcontext", None
    if include_opd_evolver:
        yield "opd_evolver", None
    backends = memevolve_backends if memevolve_backends is not None else EVOLVELAB_MEMORY_BACKENDS
    if include_memevolve:
        for b in backends:
            yield "memory_provider", b
    if include_reasoning_bank:
        yield "memory_provider", REASONING_BANK_BACKEND
def endpoint_lock_key(gen_url: str) -> str:
    raw = (gen_url or "").strip()
    p = urlparse(raw)
    if not p.netloc:
        return raw
    path = (p.path or "").rstrip("/")
    return f"{p.scheme}://{p.netloc}{path}"
def load_generation_url_map(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("generation URL map root must be a JSON object")
    out: dict[str, dict[str, str]] = {}
    for model_key, inner in data.items():
        if not isinstance(inner, dict):
            raise ValueError(f"generation URL map[{model_key!r}] must be an object")
        out[str(model_key)] = {str(ds): str(url) for ds, url in inner.items()}
    return out
def resolve_vllm_model_dir(model: str, args: argparse.Namespace) -> str | None:
    d9 = getattr(args, "vllm_model_dir_9b", None) or None
    d4 = getattr(args, "vllm_model_dir_4b", None) or None
    fallback = args.vllm_model_dir
    if d9 or d4:
        try:
            use_4b = _model_prefers_4b_vllm_shards(model)
        except ValueError:
            return fallback
        picked = d4 if use_4b else d9
        return picked or fallback
    return fallback
def build_bench_command(
    *,
    dataset: str,
    model: str,
    method: str,
    memory_backend: str | None,
    gen_url: str,
    data_root: str,
    openai_api_key: str,
    concurrency: int,
    sample_seed: int,
    vllm_model_dir: str | None,
    samples: int | None,
    task_ids: str | None,
    llm_max_completion_tokens: int | None,
    evaluate: bool,
    judge_openai_base_url: str | None,
    judge_model: str,
    judge_max_concurrency: int,
    extra: list[str],
) -> list[str]:
    out = default_output_dir(model, method, memory_backend, dataset)
    cmd: list[str] = [
        sys.executable,
        str(BENCH),
        "--datasets",
        dataset,
        "--data-root",
        data_root,
        "--method",
        method,
        "--model",
        model,
        "--no-vllm",
        "--openai-base-url",
        gen_url,
        "--openai-api-key",
        openai_api_key,
        "--concurrency",
        str(concurrency),
        "--sample-seed",
        str(sample_seed),
        "--output-dir",
        str(out),
    ]
    if vllm_model_dir:
        cmd.extend(["--vllm-model-dir", vllm_model_dir])
    if memory_backend is not None:
        cmd.extend(["--memory-backend", memory_backend])
    if samples is not None:
        cmd.extend(["--samples", str(samples)])
    if task_ids:
        cmd.extend(["--task-ids", task_ids])
    if llm_max_completion_tokens is not None:
        cmd.extend(["--llm-max-completion-tokens", str(llm_max_completion_tokens)])
    if evaluate:
        if not judge_openai_base_url:
            raise ValueError("evaluate=True requires judge_openai_base_url")
        cmd.extend(
            [
                "--judge-openai-base-url",
                judge_openai_base_url,
                "--judge-model",
                judge_model,
                "--judge-max-concurrency",
                str(judge_max_concurrency),
                "--no-judge-vllm",
            ]
        )
    else:
        cmd.append("--no-evaluate")
    cmd.extend(extra)
    return cmd
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--datasets",
        default="formal_reasoning_math,formal_reasoning_phys",
        help="Comma-separated MemoryArena dataset names (default: math + physics).",
    )
    p.add_argument("--data-root", default="data/MemoryArena")
    p.add_argument("--model", default=os.environ.get("MODEL_KEY", "qwen/qwen3.5-9b"))
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids for a multi-model matrix (overrides single --model).",
    )
    p.add_argument(
        "--generation-url-map",
        type=Path,
        default=None,
        help="JSON: {model_id: {dataset_name: openai_base_url}}. When set, overrides vLLM shard routing.",
    )
    p.add_argument(
        "--endpoint-max-concurrent",
        type=int,
        default=int(os.environ.get("MEMORYARENA_ENDPOINT_MAX_CONCURRENT", "1")),
        help="Max concurrent bench processes per distinct generation URL (default: 1).",
    )
    p.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=int(os.environ.get("MEMORYARENA_MAX_CONCURRENT_JOBS", "8")),
        help=(
            "Max concurrent bench processes globally when not using --generation-url-map "
            "(default: 8). With a generation URL map, only --endpoint-max-concurrent applies "
            "per endpoint so GPUs do not share a global cap."
        ),
    )
    p.add_argument(
        "--preset",
        choices=["thirty-two-grid"],
        default=None,
        help="thirty-two-grid: skip opd_evolver; datasets fixed to math+phys (8 methods × 2 ds × models).",
    )
    p.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible generation URL when not using --generation-url-map or shard routing.",
    )
    p.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    p.add_argument(
        "--vllm-model-dir",
        default=os.environ.get("VLLM_MODEL_DIR"),
        help="HF tokenizer path; if unset and not passed per-size, bench auto-resolves from HF_HOME for each model.",
    )
    p.add_argument(
        "--vllm-model-dir-9b",
        default=os.environ.get("VLLM_MODEL_DIR_9B"),
        help="Tokenizer snapshot for 9B-sized models (with --models).",
    )
    p.add_argument(
        "--vllm-model-dir-4b",
        default=os.environ.get("VLLM_MODEL_DIR_4B"),
        help="Tokenizer snapshot for 4B-sized models (with --models).",
    )
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "1")))
    p.add_argument("--samples", type=int, default=None, help="If set, passed as --samples to the bench.")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--task-ids", default=None, help="Optional --task-ids forwarded to the bench.")
    p.add_argument("--llm-max-completion-tokens", type=int, default=None)
    ev = p.add_mutually_exclusive_group()
    ev.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=None,
        help="Run judge after generation (default: on if judge URL set, else off).",
    )
    ev.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Skip judge; generation only.",
    )
    p.add_argument("--judge-openai-base-url", default=os.environ.get("JUDGE_OPENAI_BASE_URL"))
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-32B"))
    p.add_argument("--judge-max-concurrency", type=int, default=int(os.environ.get("JUDGE_MAX_CONCURRENCY", "1")))
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), run all jobs and report every failure at the end. "
        "If false, exit after the first failed job when aggregating results (all spawned "
        "subprocesses still finish).",
    )
    g = p.add_argument_group("Method grid filters")
    g.add_argument("--skip-longcontext", action="store_true")
    g.add_argument("--skip-opd-evolver", action="store_true")
    g.add_argument("--skip-memevolve", action="store_true")
    g.add_argument("--skip-reasoning-bank", action="store_true")
    g.add_argument(
        "--memevolve-backends",
        default=None,
        help="Comma-separated subset of MemEvolve backends (default: all in EVOLVELAB_MEMORY_BACKENDS).",
    )
    sh = p.add_argument_group("vLLM shard layout (run_vllm_qwen35_9b_sharded_servers.sh)")
    sh.add_argument(
        "--vllm-shard-routing",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("VLLM_SHARD_ROUTING", "").strip().lower() in ("1", "true", "yes"),
        help="Per-dataset generation URL: math uses shard 0 (9B or 4B block), physics uses shard 1.",
    )
    sh.add_argument(
        "--shard-host",
        default=os.environ.get("VLLM_SHARD_HOST", os.environ.get("HOST", "127.0.0.1")),
        help="Host for routed URLs (default: HOST or 127.0.0.1).",
    )
    sh.add_argument(
        "--shard-base-port",
        type=int,
        default=int(os.environ.get("VLLM_SHARD_BASE_PORT", os.environ.get("BASE_PORT", "8000"))),
        help="BASE_PORT for first 9B shard (default: 8000).",
    )
    sh.add_argument(
        "--shard-9b-count",
        type=int,
        default=int(os.environ.get("VLLM_SHARD_9B_COUNT", os.environ.get("GPU_COUNT_9B", "4"))),
        help="Number of 9B shards; first 4B port = base + this (default: 4).",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args after -- are forwarded to bench_simple_memoryarena.py.",
    )
    return p
def _split_ds(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "thirty-two-grid":
        args.skip_opd_evolver = True
        args.datasets = "formal_reasoning_math,formal_reasoning_phys"
async def _run_job_async(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    global_sem: asyncio.Semaphore | None,
    ep_sem: asyncio.Semaphore,
) -> int:
    async with ep_sem:
        if global_sem is None:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, env=env)
            return await proc.wait()
        async with global_sem:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, env=env)
            return await proc.wait()
def main() -> int:
    args = build_parser().parse_args()
    _apply_preset(args)
    datasets = _split_ds(args.datasets)
    if not datasets:
        print("error: no datasets after --datasets", file=sys.stderr)
        return 2
    models = _split_ds(args.models) if args.models else [str(args.model).strip()]
    models = [m for m in models if m]
    if not models:
        print("error: no models (--model / --models)", file=sys.stderr)
        return 2
    url_map: dict[str, dict[str, str]] | None = None
    if args.generation_url_map is not None:
        try:
            url_map = load_generation_url_map(args.generation_url_map)
        except Exception as exc:
            print(f"error: failed to load --generation-url-map: {exc}", file=sys.stderr)
            return 2
    uses_routing = bool(args.vllm_shard_routing) and url_map is None
    if url_map is not None and bool(args.vllm_shard_routing):
        print(
            "warning: --generation-url-map takes precedence; ignoring --vllm-shard-routing for URL selection.",
            file=sys.stderr,
        )
    unrouted = [d for d in datasets if d not in _SHARD_DATASETS]
    if uses_routing:
        for m in models:
            try:
                _model_prefers_4b_vllm_shards(m)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
    global_default = args.openai_base_url
    if url_map is None and not uses_routing:
        if not global_default:
            if args.dry_run:
                global_default = "http://127.0.0.1:8000/v1"
                print(
                    "note: dry-run without OPENAI_BASE_URL; using placeholder URL in printed commands.",
                    file=sys.stderr,
                )
            else:
                print("error: pass --openai-base-url or set OPENAI_BASE_URL", file=sys.stderr)
                return 2
    elif url_map is None and uses_routing and unrouted and not global_default:
        if args.dry_run:
            global_default = "http://127.0.0.1:8000/v1"
            print(
                "note: dry-run: non-math/phys datasets fall back to placeholder OPENAI_BASE_URL.",
                file=sys.stderr,
            )
        else:
            print(
                "error: --vllm-shard-routing only selects URLs for formal_reasoning_math and "
                "formal_reasoning_phys; pass --openai-base-url for other datasets.",
                file=sys.stderr,
            )
            return 2
    if url_map is not None:
        for m in models:
            if m not in url_map:
                print(f"error: generation URL map missing model key {m!r}", file=sys.stderr)
                return 2
            for d in datasets:
                if d not in url_map[m]:
                    print(
                        f"error: generation URL map missing dataset {d!r} under model {m!r}",
                        file=sys.stderr,
                    )
                    return 2
    evaluate = args.evaluate
    if evaluate is None:
        evaluate = bool(args.judge_openai_base_url)
    if evaluate and not args.judge_openai_base_url:
        print(
            "error: --evaluate requires --judge-openai-base-url (or env JUDGE_OPENAI_BASE_URL)",
            file=sys.stderr,
        )
        return 2
    memevolve_backends: tuple[str, ...] | None = None
    if args.memevolve_backends:
        memevolve_backends = tuple(_split_ds(args.memevolve_backends))
    jobs = list(
        iter_jobs(
            include_longcontext=not args.skip_longcontext,
            include_opd_evolver=not args.skip_opd_evolver,
            include_memevolve=not args.skip_memevolve,
            include_reasoning_bank=not args.skip_reasoning_bank,
            memevolve_backends=memevolve_backends,
        )
    )
    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    failures: list[str] = []
    total = len(models) * len(datasets) * len(jobs)
    idx = 0
    type JobTuple = tuple[int, int, str, str, str, str, list[str]]
    pending: list[JobTuple] = []
    for model in models:
        vdir = resolve_vllm_model_dir(model, args)
        for dataset in datasets:
            if url_map is not None:
                gen_url = url_map[model][dataset]
            elif uses_routing and dataset in _SHARD_DATASETS:
                gen_url = _routed_openai_base_url(
                    dataset=dataset,
                    model=model,
                    shard_host=args.shard_host,
                    shard_base_port=args.shard_base_port,
                    shard_9b_count=args.shard_9b_count,
                )
            else:
                gen_url = global_default
                if not gen_url:
                    print("error: missing --openai-base-url for this dataset", file=sys.stderr)
                    return 2
            for method, memory_backend in jobs:
                idx += 1
                try:
                    cmd = build_bench_command(
                        dataset=dataset,
                        model=model,
                        method=method,
                        memory_backend=memory_backend,
                        gen_url=gen_url,
                        data_root=args.data_root,
                        openai_api_key=args.openai_api_key,
                        concurrency=args.concurrency,
                        sample_seed=args.sample_seed,
                        vllm_model_dir=vdir,
                        samples=args.samples,
                        task_ids=args.task_ids or "",
                        llm_max_completion_tokens=args.llm_max_completion_tokens,
                        evaluate=bool(evaluate),
                        judge_openai_base_url=args.judge_openai_base_url,
                        judge_model=args.judge_model,
                        judge_max_concurrency=args.judge_max_concurrency,
                        extra=extra,
                    )
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                tag = method if memory_backend is None else f"{method}:{memory_backend}"
                pending.append((idx, total, model, dataset, tag, gen_url, cmd))
    endpoint_sems: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(max(1, args.endpoint_max_concurrent))
    )
    global_sem: asyncio.Semaphore | None
    if url_map is not None:
        global_sem = None
    else:
        global_sem = asyncio.Semaphore(max(1, args.max_concurrent_jobs))
    for idx_j, total_j, model, dataset, tag, gen_url, cmd in pending:
        print(f"[{idx_j}/{total_j}] model={model} dataset={dataset} {tag} openai={gen_url}", flush=True)
        print("  ", " ".join(cmd), flush=True)
    if args.dry_run:
        print("All MemoryArena matrix jobs finished OK (dry-run).")
        return 0
    async def _run_all() -> list[int]:
        env = os.environ.copy()
        async def _one(item: JobTuple) -> tuple[JobTuple, int]:
            _, _, model, dataset, tag, gen_url, cmd = item
            rc = await _run_job_async(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                global_sem=global_sem,
                ep_sem=endpoint_sems[endpoint_lock_key(gen_url)],
            )
            return item, rc
        results = await asyncio.gather(*[_one(it) for it in pending])
        return [rc for _, rc in results]
    try:
        codes = asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    for item, rc in zip(pending, codes, strict=True):
        idx_j, total_j, model, dataset, tag, _, _ = item
        if rc != 0:
            msg = f"FAILED ({rc}) model={model} dataset={dataset} {tag}"
            failures.append(msg)
            print(msg, file=sys.stderr)
            if not args.continue_on_error:
                return rc or 1
    if failures:
        print("\nSome jobs failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("All MemoryArena matrix jobs finished OK.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
