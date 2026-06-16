from dataclasses import dataclass
from typing import Any, Dict, List
@dataclass
class FilteredContext:
    selected_skill_ids: List[str]
    selected_tip_ids: List[str]
    selected_tool_ids: List[str]
    selected_trajectory_ids: List[str]
    formatted_context: str
    reasoning: str
    def get_all_selected_ids(self) -> Dict[str, List[str]]:
        return {
            "skill": self.selected_skill_ids,
            "tip": self.selected_tip_ids,
            "tool": self.selected_tool_ids,
            "trajectory": self.selected_trajectory_ids,
        }
@dataclass
class ReflectionResult:
    new_skills: List[Dict[str, Any]]
    new_tips: List[Dict[str, Any]]
    new_tools: List[Dict[str, Any]]
    key_learnings: List[str]
    should_save_trajectory: bool
    trajectory_outcome: str
