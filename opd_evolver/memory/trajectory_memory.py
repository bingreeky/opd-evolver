from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from opd_evolver.memory.base_store import BaseMemoryStore, MemoryItem
@dataclass
class TrajectoryStep:
    step_num: int
    observation: str
    action: str
    action_params: Dict[str, Any]
    result: str
    reward: float
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_num": self.step_num,
            "observation": self.observation,
            "action": self.action,
            "action_params": self.action_params,
            "result": self.result,
            "reward": self.reward,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            step_num=data.get("step_num", 0),
            observation=data.get("observation", ""),
            action=data.get("action", ""),
            action_params=data.get("action_params", {}),
            result=data.get("result", ""),
            reward=data.get("reward", 0.0),
        )
@dataclass
class TrajectoryItem(MemoryItem):
    task_description: str = ""
    steps: List[TrajectoryStep] = field(default_factory=list)
    outcome: str = ""
    total_reward: float = 0.0
    key_learnings: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "task_description": self.task_description,
            "steps": [s.to_dict() for s in self.steps],
            "outcome": self.outcome,
            "total_reward": self.total_reward,
            "key_learnings": self.key_learnings,
            "tags": self.tags,
        })
        return data
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryItem":
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
            task_description=data.get("task_description", ""),
            steps=[TrajectoryStep.from_dict(s) for s in data.get("steps", [])],
            outcome=data.get("outcome", ""),
            total_reward=data.get("total_reward", 0.0),
            key_learnings=data.get("key_learnings", []),
            tags=data.get("tags", []),
        )
    def get_summary(self, max_steps: int = 5) -> str:
        lines = [
            f"Task: {self.task_description[:200]}...",
            f"Outcome: {self.outcome} (reward: {self.total_reward})",
            f"Steps: {len(self.steps)}",
        ]
        if self.key_learnings:
            lines.append("Key learnings:")
            for learning in self.key_learnings[:3]:
                lines.append(f"  - {learning}")
        if self.steps:
            lines.append("\nKey steps:")
            show_steps = self.steps[:max_steps]
            if len(self.steps) > max_steps:
                show_steps = self.steps[:2] + self.steps[-2:]
            for step in show_steps:
                action_str = f"{step.action}"
                if step.action_params:
                    cmd = step.action_params.get("command", "")
                    if cmd:
                        action_str += f": {cmd[:50]}..."
                lines.append(f"  {step.step_num}. {action_str}")
        return "\n".join(lines)
class TrajectoryMemory(BaseMemoryStore):
    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
    ):
        super().__init__(
            tier_name="trajectory",
            storage_path=storage_path,
            embedding_provider=embedding_provider,
        )
    def _create_item(self, content: str, metadata: Dict[str, Any]) -> MemoryItem:
        steps_data = metadata.get("steps", [])
        steps = [
            TrajectoryStep.from_dict(s) if isinstance(s, dict) else s
            for s in steps_data
        ]
        return TrajectoryItem(
            content=content,
            metadata=metadata,
            task_description=metadata.get("task_description", ""),
            steps=steps,
            outcome=metadata.get("outcome", ""),
            total_reward=metadata.get("total_reward", 0.0),
            key_learnings=metadata.get("key_learnings", []),
            tags=metadata.get("tags", []),
        )
    async def add_trajectory(
        self,
        task_description: str,
        steps: List[TrajectoryStep],
        outcome: str,
        total_reward: float,
        key_learnings: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        source_task: Optional[str] = None,
    ) -> TrajectoryItem:
        content_parts = [task_description]
        if key_learnings:
            content_parts.extend(key_learnings)
        content = " ".join(content_parts)
        metadata = {
            "task_description": task_description,
            "steps": [s.to_dict() for s in steps],
            "outcome": outcome,
            "total_reward": total_reward,
            "key_learnings": key_learnings or [],
            "tags": tags or [],
        }
        item = await self.add(
            content=content,
            metadata=metadata,
            source_task=source_task,
        )
        return item
    def get_successful(self) -> List[TrajectoryItem]:
        return [
            item for item in self._items.values()
            if isinstance(item, TrajectoryItem) and item.outcome == "success"
        ]
    def get_by_tag(self, tag: str) -> List[TrajectoryItem]:
        return [
            item for item in self._items.values()
            if isinstance(item, TrajectoryItem) and tag in item.tags
        ]
    def format_for_prompt(
        self,
        items: List[TrajectoryItem],
        max_items: int = 2,
        max_steps_per_item: int = 5,
    ) -> str:
        if not items:
            return "No relevant trajectories found."
        sorted_items = sorted(
            items[:max_items * 2],
            key=lambda x: (x.outcome == "success", x.total_reward),
            reverse=True
        )[:max_items]
        lines = ["## Retrieved Trajectories (Past Executions)"]
        for i, item in enumerate(sorted_items):
            icon = "✅" if item.outcome == "success" else "❌"
            lines.append(f"\n### Trajectory {i + 1} {icon}")
            lines.append(f"Task: {item.task_description[:200]}")
            lines.append(f"Outcome: {item.outcome} (reward: {item.total_reward})")
            if item.key_learnings:
                lines.append("Key learnings:")
                for learning in item.key_learnings[:3]:
                    lines.append(f"  - {learning}")
            if item.steps:
                lines.append(f"Execution ({len(item.steps)} steps):")
                show_steps = item.steps[:max_steps_per_item]
                for step in show_steps:
                    cmd = step.action_params.get("command", "")
                    if cmd:
                        lines.append(f"  {step.step_num}. {step.action}: `{cmd[:60]}`")
                        if step.result:
                            lines.append(f"     → {step.result[:100]}")
                if len(item.steps) > max_steps_per_item:
                    lines.append(f"  ... ({len(item.steps) - max_steps_per_item} more steps)")
        return "\n".join(lines)
