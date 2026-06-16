#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from opd_evolver.memory.evolvelab_adapter import (
    ALL_MEMORY_BACKENDS,
    EVOLVELAB_MEMORY_BACKENDS,
    REASONING_BANK_BACKEND,
    EvolveLabMemoryProviderAdapter,
)
from opd_evolver.memory.reasoning_bank_adapter import ReasoningBankMemoryProviderAdapter
class DummyModel:
    def __call__(self, messages):
        text = json.dumps(
            {
                "selected_indices": [1],
                "guidance": "Use the task instructions carefully and verify the final answer format.",
                "strategic": ["Verify task constraints before final submission."],
                "operational": ["Keep intermediate observations concise."],
            }
        )
        return SimpleNamespace(content=text)
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--storage-dir", default="/tmp/memevolve_provider_check")
    ap.add_argument(
        "--initialize",
        action="store_true",
        help="Also instantiate and initialize each provider. This may load embedding models.",
    )
    args = ap.parse_args()
    print("Configured memory backends:")
    for backend in ALL_MEMORY_BACKENDS:
        print(f"  - {backend}")
    if not args.initialize:
        print("Import/mapping check complete. Re-run with --initialize for dependency smoke.")
        return 0
    adapter_model_name = "__dummy__"
    import opd_evolver.memory.evolvelab_adapter as adapter_module
    original_build = adapter_module.build_sync_model
    adapter_module.build_sync_model = lambda *_args, **_kwargs: DummyModel()
    try:
        for backend in EVOLVELAB_MEMORY_BACKENDS:
            storage = Path(args.storage_dir) / backend
            adapter = EvolveLabMemoryProviderAdapter(
                backend=backend,
                storage_dir=storage,
                model_name=adapter_model_name,
            )
            provider = adapter._new_provider()
            print(f"initialized {backend}: {provider.__class__.__name__}")
        rb_storage = Path(args.storage_dir) / REASONING_BANK_BACKEND
        import opd_evolver.memory.reasoning_bank_adapter as rb_module
        original_rb_build = rb_module.build_sync_model
        rb_module.build_sync_model = lambda *_args, **_kwargs: DummyModel()
        try:
            rb = ReasoningBankMemoryProviderAdapter(
                storage_dir=rb_storage,
                model_name=adapter_model_name,
                embedding_provider="none",
            )
            import asyncio
            ok, msg = asyncio.run(
                rb.take_in(
                    task_description="Toy task",
                    trajectory=[{"observation": "obs", "action": {"action": "submit"}, "reward": 1}],
                    success=True,
                    result={"success": True},
                    metadata={"task_id": "toy"},
                )
            )
            context = asyncio.run(rb.provide_begin("Toy task", context="obs", task_id="toy2"))
            if not ok or "REASONING BANK" not in context:
                raise RuntimeError(f"ReasoningBank smoke failed: ok={ok} msg={msg} context={context!r}")
            print(f"initialized {REASONING_BANK_BACKEND}: ReasoningBankMemoryProviderAdapter")
        finally:
            rb_module.build_sync_model = original_rb_build
    finally:
        adapter_module.build_sync_model = original_build
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
