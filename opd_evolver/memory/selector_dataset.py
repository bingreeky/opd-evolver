from __future__ import annotations
import json
import os
from typing import Any, Dict
class MemorySelectorDatasetLogger:
    def __init__(self, storage_path: str) -> None:
        self.storage_path = os.path.expanduser(storage_path)
        parent = os.path.dirname(os.path.abspath(self.storage_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.storage_path):
            open(self.storage_path, "a", encoding="utf-8").close()
    def append(self, record: Dict[str, Any]) -> None:
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
