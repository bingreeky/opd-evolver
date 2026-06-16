from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from opd_evolver.memory.base_store import BaseMemoryStore, MemoryItem
@dataclass
class ToolItem(MemoryItem):
    name: str = ""
    language: str = "bash"
    code: str = ""
    input_description: str = ""
    output_description: str = ""
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "name": self.name,
            "language": self.language,
            "code": self.code,
            "input_description": self.input_description,
            "output_description": self.output_description,
        })
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolItem":
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
            name=data.get("name", ""),
            language=data.get("language", "bash"),
            code=data.get("code", ""),
            input_description=data.get("input_description", ""),
            output_description=data.get("output_description", ""),
        )
class ToolMemory(BaseMemoryStore):
    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
    ):
        super().__init__(
            tier_name="tool",
            storage_path=storage_path,
            embedding_provider=embedding_provider,
        )
    def _create_item(self, content: str, metadata: Dict[str, Any]) -> MemoryItem:
        return ToolItem(
            content=content,
            metadata=metadata,
            name=metadata.get("name", ""),
            language=metadata.get("language", "bash"),
            code=metadata.get("code", ""),
            input_description=metadata.get("input_description", ""),
            output_description=metadata.get("output_description", ""),
        )
    async def add_tool(
        self,
        name: str,
        description: str,
        code: str,
        language: str = "bash",
        input_description: str = "",
        output_description: str = "",
        source_task: Optional[str] = None,
    ) -> ToolItem:
        metadata = {
            "name": name,
            "language": language,
            "code": code,
            "input_description": input_description,
            "output_description": output_description,
        }
        item = await self.add(
            content=description,
            metadata=metadata,
            source_task=source_task,
        )
        return item
    def get_by_language(self, language: str) -> List[ToolItem]:
        return [
            item for item in self._items.values()
            if isinstance(item, ToolItem) and item.language == language
        ]
    def get_by_name(self, name: str) -> Optional[ToolItem]:
        for item in self._items.values():
            if isinstance(item, ToolItem) and item.name == name:
                return item
        return None
    def format_for_prompt(self, items: List[ToolItem], max_items: int = 3) -> str:
        if not items:
            return "No relevant tools found."
        lines = ["## Retrieved Tools"]
        for i, item in enumerate(items[:max_items]):
            lines.append(f"\n### Tool {i + 1}: {item.name}")
            lines.append(f"Language: {item.language}")
            lines.append(f"Description: {item.content}")
            if item.input_description:
                lines.append(f"Input: {item.input_description}")
            if item.output_description:
                lines.append(f"Output: {item.output_description}")
            lines.append(f"```{item.language}")
            lines.append(item.code)
            lines.append("```")
            lines.append(f"(Used {item.usage_count} times, {item.success_count} successes)")
        return "\n".join(lines)
