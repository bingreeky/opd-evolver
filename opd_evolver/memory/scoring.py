from __future__ import annotations
from datetime import datetime
from math import exp, sqrt
from statistics import mean
from typing import Dict, List, Optional, TYPE_CHECKING
from opd_evolver.memory.usage_log import UsageLogEntry, UsageLogger
if TYPE_CHECKING:
    from opd_evolver.memory.memory_manager import HierarchicalMemoryManager
def _group_by_task_type(
    logs: List[UsageLogEntry],
) -> Dict[str, List[UsageLogEntry]]:
    groups: Dict[str, List[UsageLogEntry]] = {}
    for log in logs:
        groups.setdefault(log.task_type, []).append(log)
    return groups
def contribution_score(
    memory_id: str,
    usage_logs: List[UsageLogEntry],
) -> float:
    contributions: List[float] = []
    for _task_type, logs in _group_by_task_type(usage_logs).items():
        pool = [
            log for log in logs
            if memory_id in log.all_candidate_ids()
        ]
        if len(pool) < 2:
            continue
        selected = [log.env_reward for log in pool if memory_id in log.all_selected_ids()]
        not_selected = [
            log.env_reward for log in pool
            if memory_id not in log.all_selected_ids()
        ]
        if not selected or not not_selected:
            continue
        baseline = mean(log.env_reward for log in pool)
        delta = mean(selected) - baseline
        weight = len(selected) / len(pool)
        contributions.append(delta * weight)
    return mean(contributions) if contributions else 0.0
def score_memory(
    memory_id: str,
    tier: str,
    last_used: Optional[str],
    usage_logs: List[UsageLogEntry],
    tier_weights: Optional[Dict[str, float]] = None,
    lambda_days: float = 0.1,
    now: Optional[datetime] = None,
) -> float:
    if tier_weights is None:
        tier_weights = {
            "skill": 1.0,
            "tip": 0.8,
            "tool": 1.2,
            "trajectory": 0.9,
        }
    tw = tier_weights.get(tier, 1.0)
    contribution = contribution_score(memory_id, usage_logs)
    n_selected = sum(
        1 for log in usage_logs
        if memory_id in log.all_selected_ids()
    )
    confidence = 1.0 - 1.0 / sqrt(1.0 + n_selected)
    if last_used is None:
        recency = 0.0
    else:
        now = now or datetime.now()
        try:
            lu_dt = datetime.fromisoformat(last_used)
            days = max((now - lu_dt).total_seconds() / 86400.0, 0.0)
            recency = exp(-lambda_days * days)
        except (ValueError, TypeError):
            recency = 0.0
    recency = 1.0
    score = tw * confidence * contribution * recency
    return score
def score_all_memories(
    memory_manager: "HierarchicalMemoryManager",
    usage_logger: UsageLogger,
    tier_weights: Optional[Dict[str, float]] = None,
    lambda_days: float = 0.1,
    now: Optional[datetime] = None,
) -> Dict[str, float]:
    all_logs = usage_logger.load_all()
    scores: Dict[str, float] = {}
    now = now or datetime.now()
    tier_store_pairs = [
        ("skill", memory_manager.skill_memory),
        ("tip", memory_manager.tip_memory),
        ("tool", memory_manager.tool_memory),
        ("trajectory", memory_manager.trajectory_memory),
    ]
    for tier, store in tier_store_pairs:
        for item in store.get_all():
            relevant_logs = [
                log for log in all_logs
                if item.id in log.all_candidate_ids() or item.id in log.all_selected_ids()
            ]
            scores[item.id] = score_memory(
                memory_id=item.id,
                tier=tier,
                last_used=item.last_used,
                usage_logs=relevant_logs,
                tier_weights=tier_weights,
                lambda_days=lambda_days,
                now=now,
            )
    return scores
