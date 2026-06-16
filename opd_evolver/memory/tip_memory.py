from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from opd_evolver.memory.base_store import BaseMemoryStore, MemoryItem
@dataclass
class TipItem(MemoryItem):
    category: str = ""
    severity: str = "info"
    trigger: str = ""
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "category": self.category,
            "severity": self.severity,
            "trigger": self.trigger,
        })
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TipItem":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            source_task=data.get("source_task"),
            created_at=data.get("created_at", ""),
            success_count=data.get("success_count", 0),
            usage_count=data.get("usage_count", 0),
            last_used=data.get("last_used"),
            category=data.get("category", ""),
            severity=data.get("severity", "info"),
            trigger=data.get("trigger", ""),
        )
class TipMemory(BaseMemoryStore):
    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
    ):
        super().__init__(
            tier_name="tip",
            storage_path=storage_path,
            embedding_provider=embedding_provider,
        )
    def _create_item(self, content: str, metadata: Dict[str, Any]) -> MemoryItem:
        return TipItem(
            content=content,
            metadata=metadata,
            category=metadata.get("category", ""),
            severity=metadata.get("severity", "info"),
            trigger=metadata.get("trigger", ""),
        )
    async def add_tip(
        self,
        content: str,
        category: str = "",
        severity: str = "info",
        trigger: str = "",
        source_task: Optional[str] = None,
    ) -> TipItem:
        metadata = {
            "category": category,
            "severity": severity,
            "trigger": trigger,
        }
        item = await self.add(
            content=content,
            metadata=metadata,
            source_task=source_task,
        )
        return item
    def get_by_severity(self, severity: str) -> List[TipItem]:
        return [
            item for item in self._items.values()
            if isinstance(item, TipItem) and item.severity == severity
        ]
    def get_critical_tips(self) -> List[TipItem]:
        return self.get_by_severity("critical")
    def format_for_prompt(self, items: List[TipItem], max_items: int = 5) -> str:
        if not items:
            return "No relevant tips found."
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_items = sorted(
            items[:max_items],
            key=lambda x: severity_order.get(x.severity, 3)
        )
        lines = ["## Retrieved Tips & Caveats"]
        for i, item in enumerate(sorted_items):
            icon = {"critical": "🚨", "warning": "⚠️", "info": "💡"}.get(item.severity, "📝")
            lines.append(f"\n{icon} **Tip {i + 1}** [{item.severity.upper()}]")
            lines.append(f"Category: {item.category or 'General'}")
            lines.append(f"Content: {item.content}")
            if item.trigger:
                lines.append(f"Applies when: {item.trigger}")
        return "\n".join(lines)
