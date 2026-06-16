#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH = PROJECT_ROOT / "scripts" / "eval" / "bench_simple_ama.py"
DEFAULT_TEST_FILE = str(PROJECT_ROOT / "data" / "ama" / "open_end_qa_set.jsonl")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from opd_evolver.memory.evolvelab_adapter import (
    EVOLVELAB_MEMORY_BACKENDS,
    REASONING_BANK_BACKEND,
)
def _safe_model_label(model_name: str) -> str:
    return str(model_name).replace("/", "__").replace(":", "_")
def default_output_dir(model: str, method: str, memory_backend: str | None, subset: str) -> Path:
    base = PROJECT_ROOT / "workspace" / "baselines" / "ama_bench" / _safe_model_label(model)
    if method == "memory_provider":
        return base / "memory_provider" / subset / (memory_backend or "unknown")
    return base / method / subset
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
@dataclass(frozen=True)
class SubsetEndpoint:
    url: str
    vllm_max_model_len: int | None = None
    llm_max_completion_tokens: int | None = None
def _parse_subset_endpoint(raw: Any) -> SubsetEndpoint:
    if isinstance(raw, str):
        return SubsetEndpoint(url=raw.strip())
    if isinstance(raw, dict):
        url = raw.get("url") or raw.get("openai_base_url")
        if not url:
            raise ValueError("endpoint object requires 'url' or 'openai_base_url'")
        max_len = raw.get("vllm_max_model_len", raw.get("max_model_len"))
        max_out = raw.get("llm_max_completion_tokens", raw.get("max_completion_tokens"))
        return SubsetEndpoint(
            url=str(url).strip(),
            vllm_max_model_len=int(max_len) if max_len is not None else None,
            llm_max_completion_tokens=int(max_out) if max_out is not None else None,
        )
    raise ValueError(f"endpoint must be a URL string or object, got {type(raw)}")
def load_generation_url_map(path: Path) -> dict[str, dict[str, SubsetEndpoint]]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("generation URL map root must be a JSON object")
    out: dict[str, dict[str, SubsetEndpoint]] = {}
    for model_key, inner in data.items():
        if not isinstance(inner, dict):
            raise ValueError(f"generation URL map[{model_key!r}] must be an object")
        out[str(model_key)] = {str(subset): _parse_subset_endpoint(raw) for subset, raw in inner.items()}
    return out
def resolve_context_limits(
    model: str,
    args: argparse.Namespace,
    endpoint: SubsetEndpoint | None,
) -> tuple[int, int]:
    try:
        use_4b = _model_prefers_4b_vllm_shards(model)
    except ValueError:
        use_4b = False
    if use_4b:
        max_len = (
            (endpoint.vllm_max_model_len if endpoint else None)
            or getattr(args, "vllm_max_model_len_4b", None)
            or args.vllm_max_model_len
            or 131072
        )
        max_out = (
            (endpoint.llm_max_completion_tokens if endpoint else None)
            or getattr(args, "llm_max_completion_tokens_4b", None)
            or args.llm_max_completion_tokens
            or 2048
        )
    else:
        max_len = (
            (endpoint.vllm_max_model_len if endpoint else None)
            or getattr(args, "vllm_max_model_len_9b", None)
            or args.vllm_max_model_len
            or 131072
        )
        max_out = (
            (endpoint.llm_max_completion_tokens if endpoint else None)
            or getattr(args, "llm_max_completion_tokens_9b", None)
            or args.llm_max_completion_tokens
            or 4096
        )
    return int(max_len), int(max_out)
def endpoint_lock_key(gen_url: str) -> str:
    raw = (gen_url or "").strip()
    p = urlparse(raw)
    if not p.netloc:
        return raw
    path = (p.path or "").rstrip("/")
    return f"{p.scheme}://{p.netloc}{path}"
def _model_prefers_4b_vllm_shards(model: str) -> bool:
    m = str(model).lower().replace("_", "-")
    if "qwen3-4b" in m or "qwen3.5-4b" in m:
        return True
    if "3.5-9b" in m or "3.5_9b" in m or ("9b" in m and "4b" not in m):
        return False
    if "4b" in m:
        return True
    raise ValueError(
        f"Cannot infer 9B vs 4B from --model {model!r}. "
        "Use an id like qwen/qwen3.5-9b or qwen/qwen3-4b."
    )
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
    subset: str,
    model: str,
    method: str,
    memory_backend: str | None,
    gen_url: str,
    test_file: str,
    openai_api_key: str,
    concurrency: int,
    sample_seed: int,
    vllm_model_dir: str | None,
    samples: int | None,
    llm_max_completion_tokens: int | None,
    vllm_max_model_len: int | None,
    evaluate: bool,
    extra: list[str],
) -> list[str]:
    out = default_output_dir(model, method, memory_backend, subset)
    cmd: list[str] = [
        sys.executable,
        str(BENCH),
        "--subset",
        subset,
        "--test-file",
        test_file,
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
    if llm_max_completion_tokens is not None:
        cmd.extend(["--llm-max-completion-tokens", str(llm_max_completion_tokens)])
    if vllm_max_model_len is not None:
        cmd.extend(["--vllm-max-model-len", str(vllm_max_model_len)])
    if evaluate:
        cmd.append("--evaluate")
    else:
        cmd.append("--no-evaluate")
    cmd.extend(extra)
    return cmd
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--subset",
        choices=["openend", "mcq"],
        default=os.environ.get("AMA_SUBSET", "openend"),
        help="AMA-Bench subset (default: openend).",
    )
    p.add_argument(
        "--test-file",
        default=os.environ.get("AMA_TEST_FILE", DEFAULT_TEST_FILE),
        help="AMA JSONL test file path.",
    )
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
        help="JSON: {model_id: {subset: openai_base_url}}. When set, each URL is scheduled independently.",
    )
    p.add_argument(
        "--endpoint-max-concurrent",
        type=int,
        default=int(os.environ.get("AMA_ENDPOINT_MAX_CONCURRENT", "1")),
        help="Max concurrent bench processes per distinct generation URL (default: 1).",
    )
    p.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=int(os.environ.get("AMA_MAX_CONCURRENT_JOBS", "8")),
        help=(
            "Max concurrent bench processes globally when not using --generation-url-map "
            "(default: 8). With a generation URL map, only --endpoint-max-concurrent applies."
        ),
    )
    p.add_argument(
        "--preset",
        choices=["sixteen-grid"],
        default=None,
        help="sixteen-grid: skip opd_evolver; subset fixed to openend (8 methods × models).",
    )
    p.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible generation URL when not using --generation-url-map.",
    )
    p.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    p.add_argument(
        "--vllm-model-dir",
        default=os.environ.get("VLLM_MODEL_DIR"),
        help="HF tokenizer path for prompt length estimation.",
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
    p.add_argument("--llm-max-completion-tokens", type=int, default=None)
    p.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=int(os.environ.get("AMA_VLLM_MAX_MODEL_LEN", "0")) or None,
        help="Prompt truncation budget; must match served vLLM --max-model-len (both models if per-size unset).",
    )
    p.add_argument(
        "--vllm-max-model-len-9b",
        type=int,
        default=int(os.environ.get("AMA_VLLM_MAX_MODEL_LEN_9B", "0")) or None,
    )
    p.add_argument(
        "--vllm-max-model-len-4b",
        type=int,
        default=int(os.environ.get("AMA_VLLM_MAX_MODEL_LEN_4B", "0")) or None,
    )
    p.add_argument(
        "--llm-max-completion-tokens-9b",
        type=int,
        default=int(os.environ.get("AMA_LLM_MAX_COMPLETION_TOKENS_9B", "0")) or None,
    )
    p.add_argument(
        "--llm-max-completion-tokens-4b",
        type=int,
        default=int(os.environ.get("AMA_LLM_MAX_COMPLETION_TOKENS_4B", "0")) or None,
        help="4B default 2048 when unset (leaves more room for long trajectories).",
    )
    ev = p.add_mutually_exclusive_group()
    ev.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        default=None,
        help="Run AMA judge after generation (default: off).",
    )
    ev.add_argument(
        "--no-evaluate",
        dest="evaluate",
        action="store_false",
        help="Skip judge; generation only.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), run all jobs and report failures at the end.",
    )
    g = p.add_argument_group("Method grid filters")
    g.add_argument("--skip-longcontext", action="store_true")
    g.add_argument("--skip-opd-evolver", action="store_true")
    g.add_argument("--skip-memevolve", action="store_true")
    g.add_argument("--skip-reasoning-bank", action="store_true")
    g.add_argument(
        "--memevolve-backends",
        default=None,
        help="Comma-separated subset of MemEvolve backends (default: all).",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args after -- are forwarded to bench_simple_ama.py.",
    )
    return p
def _split_csv(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "sixteen-grid":
        args.skip_opd_evolver = True
        args.subset = "openend"
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
    subset = str(args.subset).strip()
    if not subset:
        print("error: no subset", file=sys.stderr)
        return 2
    models = _split_csv(args.models) if args.models else [str(args.model).strip()]
    models = [m for m in models if m]
    if not models:
        print("error: no models (--model / --models)", file=sys.stderr)
        return 2
    url_map: dict[str, dict[str, SubsetEndpoint]] | None = None
    if args.generation_url_map is not None:
        try:
            url_map = load_generation_url_map(args.generation_url_map)
        except Exception as exc:
            print(f"error: failed to load --generation-url-map: {exc}", file=sys.stderr)
            return 2
    global_default = args.openai_base_url
    if url_map is None:
        if not global_default:
            if args.dry_run:
                global_default = "http://127.0.0.1:8002/v1"
                print(
                    "note: dry-run without OPENAI_BASE_URL; using placeholder URL in printed commands.",
                    file=sys.stderr,
                )
            else:
                print("error: pass --openai-base-url or set OPENAI_BASE_URL", file=sys.stderr)
                return 2
    if url_map is not None:
        for m in models:
            if m not in url_map:
                print(f"error: generation URL map missing model key {m!r}", file=sys.stderr)
                return 2
            if subset not in url_map[m]:
                print(
                    f"error: generation URL map missing subset {subset!r} under model {m!r}",
                    file=sys.stderr,
                )
                return 2
    evaluate = args.evaluate if args.evaluate is not None else False
    memevolve_backends: tuple[str, ...] | None = None
    if args.memevolve_backends:
        memevolve_backends = tuple(_split_csv(args.memevolve_backends))
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
    total = len(models) * len(jobs)
    idx = 0
    type JobTuple = tuple[int, int, str, str, str, list[str]]
    pending: list[JobTuple] = []
    for model in models:
        vdir = resolve_vllm_model_dir(model, args)
        endpoint: SubsetEndpoint | None = None
        if url_map is not None:
            endpoint = url_map[model][subset]
            gen_url = endpoint.url
        else:
            gen_url = global_default
            if not gen_url:
                print("error: missing --openai-base-url", file=sys.stderr)
                return 2
        max_model_len, max_completion = resolve_context_limits(model, args, endpoint)
        for method, memory_backend in jobs:
            idx += 1
            cmd = build_bench_command(
                subset=subset,
                model=model,
                method=method,
                memory_backend=memory_backend,
                gen_url=gen_url,
                test_file=args.test_file,
                openai_api_key=args.openai_api_key,
                concurrency=args.concurrency,
                sample_seed=args.sample_seed,
                vllm_model_dir=vdir,
                samples=args.samples,
                llm_max_completion_tokens=max_completion,
                vllm_max_model_len=max_model_len,
                evaluate=bool(evaluate),
                extra=extra,
            )
            tag = method if memory_backend is None else f"{method}:{memory_backend}"
            pending.append((idx, total, model, tag, gen_url, cmd))
    endpoint_sems: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(max(1, args.endpoint_max_concurrent))
    )
    global_sem: asyncio.Semaphore | None
    if url_map is not None:
        global_sem = None
    else:
        global_sem = asyncio.Semaphore(max(1, args.max_concurrent_jobs))
    for idx_j, total_j, model, tag, gen_url, cmd in pending:
        print(f"[{idx_j}/{total_j}] model={model} subset={subset} {tag} openai={gen_url}", flush=True)
        print("  ", " ".join(cmd), flush=True)
    if args.dry_run:
        print("All AMA matrix jobs finished OK (dry-run).")
        return 0
    async def _run_all() -> list[int]:
        env = os.environ.copy()
        async def _one(item: JobTuple) -> tuple[JobTuple, int]:
            _, _, model, tag, gen_url, cmd = item
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
        idx_j, total_j, model, tag, _, _ = item
        if rc != 0:
            msg = f"FAILED ({rc}) model={model} subset={subset} {tag}"
            failures.append(msg)
            print(msg, file=sys.stderr)
            if not args.continue_on_error:
                return rc or 1
    if failures:
        print("\nSome jobs failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("All AMA matrix jobs finished OK.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
