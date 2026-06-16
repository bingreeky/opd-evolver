from __future__ import annotations
from typing import Any
def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    source = str(data_source)
    if source != "intercode_sql":
        raise NotImplementedError(f"Reward function is not implemented for data_source={source!r}")
    info = extra_info or {}
    score = info.get("cumulative_reward")
    if score is None:
        turn_scores = info.get("turn_scores") or info.get("tool_rewards") or []
        score = sum(float(item) for item in turn_scores)
    score_f = float(score)
    return {
        "score": score_f,
        "acc": 1.0 if score_f >= 1.0 else 0.0,
        "cumulative_reward": score_f,
    }
