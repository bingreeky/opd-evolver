import json
import re
from typing import Any, Dict, List
from opd_evolver.memory.memory_manager import RetrievalResult
from opd_evolver.pipelines.memory_selector_prompts import SELECTOR_PROMPT
from opd_evolver.pipelines.types import FilteredContext, ReflectionResult
FILTERING_PROMPT = SELECTOR_PROMPT
REFLECTION_PROMPT = """You are a learning extraction assistant for an AI agent system.

## Executed Task
{task_description}

## Execution Outcome
**Result: {outcome}** (Reward: {total_reward})

## Execution Trace
{execution_trace}

## Your Job
Analyze this execution and extract reusable knowledge:

1. **Skills**: Methodological approaches that worked (or failed)
2. **Tips**: Gotchas, caveats, or heuristics discovered
3. **Tools**: Reusable code snippets or commands
4. **Key Learnings**: Main takeaways from this execution

## Output Format
Respond with a JSON object:
```json
{{
  "new_skills": [
    {{
      "description": "How to approach this type of problem",
      "category": "cryptography",
      "technique": "fernet_decryption",
      "preconditions": "When encountering encrypted payloads",
      "steps": ["Step 1", "Step 2", "Step 3"]
    }}
  ],
  "new_tips": [
    {{
      "content": "The specific gotcha or heuristic",
      "category": "debugging",
      "severity": "warning",
      "trigger": "When this situation occurs"
    }}
  ],
  "new_tools": [
    {{
      "name": "tool_name",
      "description": "What the tool does",
      "language": "python",
      "code": "the actual code",
      "input_description": "expected inputs",
      "output_description": "expected outputs"
    }}
  ],
  "key_learnings": [
    "Main takeaway 1",
    "Main takeaway 2"
  ],
  "should_save_trajectory": true,
  "trajectory_outcome": "success"
}}
```

Rules:
- Only extract genuinely reusable knowledge
- For failed tasks, focus on what went wrong and how to avoid it
- Skills should be abstract enough to apply to similar tasks
- Tools should be self-contained and executable
- Tips should be actionable warnings or heuristics
- Set should_save_trajectory=true only if this execution is worth remembering
- trajectory_outcome should be "success", "failure", or "partial"

Output JSON only, no other text:"""
def _robust_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        fixed = text.replace("\\'", "'")
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    try:
        fixed = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return json.loads(text)
def parse_filtering_response(
    response: str,
    retrieval: RetrievalResult,
) -> FilteredContext:
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        raise ValueError("No JSON found in filtering response")
    try:
        data = _robust_json_loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in filtering response: {e}")
    def extract_ids(tags: List[str], items: List) -> List[str]:
        ids = []
        for tag in tags:
            for item in items:
                if item.tag == tag:
                    ids.append(item.item.id)
                    break
        return ids
    selected_skill_ids = extract_ids(
        data.get("selected_skills", []),
        retrieval.skills
    )
    selected_tip_ids = extract_ids(
        data.get("selected_tips", []),
        retrieval.tips
    )
    selected_tool_ids = extract_ids(
        data.get("selected_tools", []),
        retrieval.tools
    )
    selected_trajectory_ids = extract_ids(
        data.get("selected_trajectories", []),
        retrieval.trajectories
    )
    formatted_parts = []
    for item in retrieval.skills:
        if item.item.id in selected_skill_ids:
            formatted_parts.append(item.format_for_context())
    for item in retrieval.tips:
        if item.item.id in selected_tip_ids:
            formatted_parts.append(item.format_for_context())
    for item in retrieval.tools:
        if item.item.id in selected_tool_ids:
            formatted_parts.append(item.format_for_context())
    for item in retrieval.trajectories:
        if item.item.id in selected_trajectory_ids:
            formatted_parts.append(item.format_for_context())
    return FilteredContext(
        selected_skill_ids=selected_skill_ids,
        selected_tip_ids=selected_tip_ids,
        selected_tool_ids=selected_tool_ids,
        selected_trajectory_ids=selected_trajectory_ids,
        formatted_context="\n\n".join(formatted_parts),
        reasoning=data.get("reasoning", ""),
    )
def parse_reflection_response(
    response: str,
    outcome: str,
) -> ReflectionResult:
    if response is None:
        raise ValueError("LLM returned None response")
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        raise ValueError("No JSON found in reflection response")
    try:
        data = _robust_json_loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in reflection response: {e}")
    return ReflectionResult(
        new_skills=data.get("new_skills", []),
        new_tips=data.get("new_tips", []),
        new_tools=data.get("new_tools", []),
        key_learnings=data.get("key_learnings", []),
        should_save_trajectory=data.get("should_save_trajectory", outcome == "SUCCESS"),
        trajectory_outcome=data.get("trajectory_outcome", outcome.lower()),
    )
