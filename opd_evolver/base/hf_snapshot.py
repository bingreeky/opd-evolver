from __future__ import annotations
import os
from pathlib import Path
_LEGACY_QWEN35_9B_SNAPSHOT = (
    "/path/to/models/Qwen3.5-9B"
    "/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
def _is_hf_snapshot_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()
def _pick_latest_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in candidates:
        if _is_hf_snapshot_dir(cand):
            return cand
    return None
def hf_hub_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HF_HUB_CACHE"),
        str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
        str(Path.home() / ".cache" / "huggingface" / "hub"),
        "/root/.cache/huggingface/hub",
        "/path/to/hf/hub",
    ):
        if raw:
            roots.append(Path(raw))
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique
def hf_snapshot_cache_dir(model_id: str, hub_root: Path) -> Path:
    return hub_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
def resolve_hf_snapshot(
    model_id: str = "Qwen/Qwen3.5-9B",
    hint: str | None = None,
) -> str | None:
    hints: list[Path] = []
    if hint:
        expanded = os.path.expandvars(os.path.expanduser(str(hint).strip()))
        if expanded and expanded.lower() not in {"null", "none"}:
            path = Path(expanded)
            hints.append(path)
    hints.append(Path(_LEGACY_QWEN35_9B_SNAPSHOT))
    for hub_root in hf_hub_cache_roots():
        hints.append(hf_snapshot_cache_dir(model_id, hub_root))
    for candidate in hints:
        if _is_hf_snapshot_dir(candidate):
            return str(candidate.resolve())
        latest = _pick_latest_snapshot(candidate)
        if latest is not None:
            return str(latest.resolve())
    return None
def resolve_model_name_or_path(
    path: str | None,
    model_id: str = "Qwen/Qwen3.5-9B",
) -> str:
    resolved = resolve_hf_snapshot(model_id, hint=path)
    if resolved:
        if path:
            expanded = os.path.expandvars(os.path.expanduser(str(path)))
            if os.path.normpath(resolved) != os.path.normpath(expanded):
                print(f"Resolved model path -> {resolved}")
        return resolved
    if path:
        expanded = os.path.expandvars(os.path.expanduser(str(path)))
        if _is_hf_snapshot_dir(Path(expanded)):
            return expanded
    raise FileNotFoundError(
        f"No local {model_id} snapshot found. Set MODEL_DIR or HF_HOME, or pass a directory "
        "that contains config.json."
    )
