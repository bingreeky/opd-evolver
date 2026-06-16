from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
@dataclass
class UsageLogEntry:
    task_id: str
    task_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    retrieved_candidates: Dict[str, List[str]] = field(default_factory=dict)
    selected_memory_ids: Dict[str, List[str]] = field(default_factory=dict)
    env_reward: float = 0.0
    success: bool = False
    status: Optional[str] = None
    def to_dict(self) -> dict:
        return asdict(self)
    @classmethod
    def from_dict(cls, data: dict) -> "UsageLogEntry":
        return cls(
            task_id=data.get("task_id", ""),
            task_type=data.get("task_type", ""),
            timestamp=data.get("timestamp", ""),
            retrieved_candidates=data.get("retrieved_candidates", {}),
            selected_memory_ids=data.get("selected_memory_ids", {}),
            env_reward=float(data.get("env_reward", 0.0)),
            success=bool(data.get("success", False)),
            status=data.get("status"),
        )
    def all_candidate_ids(self) -> List[str]:
        ids: List[str] = []
        for tier_ids in self.retrieved_candidates.values():
            ids.extend(tier_ids)
        return ids
    def all_selected_ids(self) -> List[str]:
        ids: List[str] = []
        for tier_ids in self.selected_memory_ids.values():
            ids.extend(tier_ids)
        return ids
    @property
    def is_complete(self) -> bool:
        return self.status in (None, "complete")
class UsageLogger:
    def __init__(self, storage_path: str) -> None:
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
        if not os.path.exists(storage_path):
            open(storage_path, "a").close()
        self._pending: Dict[str, dict] = {}
    def log(self, entry: UsageLogEntry) -> None:
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    def log_started(self, entry: UsageLogEntry) -> None:
        d = entry.to_dict()
        self._pending[entry.task_id] = d
    def log_outcome(self, task_id: str, env_reward: float, success: bool) -> None:
        pending = self._pending.pop(task_id, None)
        if pending is not None:
            d = dict(pending)
            d["env_reward"] = env_reward
            d["success"] = success
            d["status"] = "complete"
        else:
            d = {
                "task_id": task_id,
                "env_reward": env_reward,
                "success": success,
                "status": "complete",
            }
        line = json.dumps(d, ensure_ascii=False)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    def load_all(self) -> List[UsageLogEntry]:
        if not os.path.exists(self.storage_path):
            return []
        started: Dict[str, dict] = {}
        outcomes: Dict[str, dict] = {}
        legacy: Dict[str, dict] = {}
        task_order: List[str] = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    import warnings
                    warnings.warn(
                        f"UsageLogger: skipping corrupt line {lineno} in "
                        f"{self.storage_path}: {exc}"
                    )
                    continue
                task_id = data.get("task_id", "")
                status = data.get("status")
                if status == "complete":
                    if task_id not in started and task_id not in outcomes and task_id not in legacy:
                        task_order.append(task_id)
                    outcomes[task_id] = data
                elif status == "started":
                    if task_id not in started:
                        task_order.append(task_id)
                    started[task_id] = data
                else:
                    if task_id not in legacy and task_id not in started and task_id not in outcomes:
                        task_order.append(task_id)
                    legacy[task_id] = data
        entries: List[UsageLogEntry] = []
        for task_id in task_order:
            if task_id in started:
                base = dict(started[task_id])
                if task_id in outcomes:
                    out = outcomes[task_id]
                    if out.get("retrieved_candidates") or out.get("selected_memory_ids"):
                        base = dict(out)
                    else:
                        base["env_reward"] = out.get("env_reward", base.get("env_reward", 0.0))
                        base["success"] = out.get("success", base.get("success", False))
                        base["status"] = "complete"
                try:
                    entries.append(UsageLogEntry.from_dict(base))
                except (KeyError, TypeError) as exc:
                    import warnings
                    warnings.warn(
                        f"UsageLogger: skipping malformed started record for "
                        f"task_id={task_id!r}: {exc}"
                    )
            elif task_id in outcomes:
                try:
                    entries.append(UsageLogEntry.from_dict(outcomes[task_id]))
                except (KeyError, TypeError) as exc:
                    import warnings
                    warnings.warn(
                        f"UsageLogger: skipping malformed complete record for "
                        f"task_id={task_id!r}: {exc}"
                    )
            else:
                data = legacy.get(task_id)
                if data is None:
                    continue
                try:
                    entries.append(UsageLogEntry.from_dict(data))
                except (KeyError, TypeError) as exc:
                    import warnings
                    warnings.warn(
                        f"UsageLogger: skipping malformed legacy record for "
                        f"task_id={task_id!r}: {exc}"
                    )
        return entries
    def load_for_memory(self, memory_id: str) -> List[UsageLogEntry]:
        needle = memory_id.encode("utf-8")
        candidate_task_ids: set[str] = set()
        if not os.path.exists(self.storage_path):
            return []
        with open(self.storage_path, "rb") as f:
            for raw_bytes in f:
                if needle not in raw_bytes:
                    continue
                try:
                    data = json.loads(raw_bytes.strip().decode("utf-8"))
                    tid = data.get("task_id", "")
                    if tid:
                        candidate_task_ids.add(tid)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        if not candidate_task_ids:
            return []
        return [
            entry for entry in self.load_all()
            if entry.task_id in candidate_task_ids
            and (
                memory_id in entry.all_candidate_ids()
                or memory_id in entry.all_selected_ids()
            )
        ]
    def count(self) -> int:
        return len(self.load_all())
    def count_lines(self) -> int:
        if not os.path.exists(self.storage_path):
            return 0
        with open(self.storage_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
