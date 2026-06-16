from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from opd_evolver.memory.base_store import BaseMemoryStore, MemoryItem
@dataclass
class SkillItem(MemoryItem):
    category: str = ""
    technique: str = ""
    preconditions: str = ""
    steps: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "category": self.category,
            "technique": self.technique,
            "preconditions": self.preconditions,
            "steps": self.steps,
        })
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillItem":
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
            technique=data.get("technique", ""),
            preconditions=data.get("preconditions", ""),
            steps=data.get("steps", []),
        )
class SkillMemory(BaseMemoryStore):
    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
    ):
        super().__init__(
            tier_name="skill",
            storage_path=storage_path,
            embedding_provider=embedding_provider,
        )
    def _create_item(self, content: str, metadata: Dict[str, Any]) -> MemoryItem:
        return SkillItem(
            content=content,
            metadata=metadata,
            category=metadata.get("category", ""),
            technique=metadata.get("technique", ""),
            preconditions=metadata.get("preconditions", ""),
            steps=metadata.get("steps", []),
        )
    async def add_skill(
        self,
        description: str,
        category: str = "",
        technique: str = "",
        preconditions: str = "",
        steps: Optional[List[str]] = None,
        source_task: Optional[str] = None,
    ) -> SkillItem:
        metadata = {
            "category": category,
            "technique": technique,
            "preconditions": preconditions,
            "steps": steps or [],
        }
        item = await self.add(
            content=description,
            metadata=metadata,
            source_task=source_task,
        )
        return item
    def get_by_category(self, category: str) -> List[SkillItem]:
        return [
            item for item in self._items.values()
            if isinstance(item, SkillItem) and item.category == category
        ]
    def format_for_prompt(self, items: List[SkillItem], max_items: int = 5) -> str:
        if not items:
            return "No relevant skills found."
        lines = ["## Retrieved Skills"]
        for i, item in enumerate(items[:max_items]):
            lines.append(f"\n### Skill {i + 1}: {item.technique or 'Unnamed'}")
            lines.append(f"Category: {item.category or 'General'}")
            lines.append(f"Description: {item.content}")
            if item.preconditions:
                lines.append(f"When to use: {item.preconditions}")
            if item.steps:
                lines.append("Steps:")
                for j, step in enumerate(item.steps):
                    lines.append(f"  {j + 1}. {step}")
            lines.append(f"(Success rate: {item.success_count} uses)")
        return "\n".join(lines)
