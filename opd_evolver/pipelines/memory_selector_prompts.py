from __future__ import annotations
from opd_evolver.memory.memory_manager import RetrievalResult
from opd_evolver.pipelines.types import FilteredContext
SELECTOR_PROMPT = """You are a memory filtering assistant for an AI agent system.

## Task
The agent is about to execute the following task:
{task_description}

## Retrieved Memories
The following items were retrieved from the agent's memory stores:
{retrieved_context}

## Your Job
Select the MOST RELEVANT items that will help the agent complete this task.
Be selective - only choose items that are directly applicable.

## Output Format
Respond with a JSON object:
```json
{{
  "selected_skills": ["[RETRIEVED_SKILL_01]", "[RETRIEVED_SKILL_02]"],
  "selected_tips": ["[RETRIEVED_TIP_01]"],
  "selected_tools": ["[RETRIEVED_TOOL_01]"],
  "selected_trajectories": ["[RETRIEVED_TRAJECTORY_01]"],
  "reasoning": "Brief explanation of why these items were selected"
}}
```

Rules:
- Use the exact tags from the retrieved items (e.g., [RETRIEVED_SKILL_01])
- Select 0-3 items per category
- If no items are relevant for a category, use an empty array []
- Be concise in your reasoning

Output JSON only, no other text:"""
def parse_selector_response(
    response: str,
    retrieval: RetrievalResult,
) -> FilteredContext:
    from opd_evolver.pipelines.memory_prompts import parse_filtering_response
    return parse_filtering_response(response, retrieval)
