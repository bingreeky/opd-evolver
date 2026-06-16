#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from typing import Any
def _flatten_ids(ids_map: Any) -> list[str]:
    out: list[str] = []
    if isinstance(ids_map, dict):
        for ids in ids_map.values():
            if isinstance(ids, list):
                out.extend(str(x) for x in ids)
    elif isinstance(ids_map, list):
        out.extend(str(x) for x in ids_map)
    return out
_ONLINE_KEYS = {
    "selected_skills": "skill",
    "selected_tips": "tip",
    "selected_tools": "tool",
    "selected_trajectories": "trajectory",
}
_TAG_NAMES = {
    "skill": "SKILL",
    "tip": "TIP",
    "tool": "TOOL",
    "trajectory": "TRAJECTORY",
}
def _tag_maps_from_candidates(cands: Any) -> dict[str, str]:
    tag_to_id: dict[str, str] = {}
    if not isinstance(cands, dict):
        return tag_to_id
    for tier, tag_name in _TAG_NAMES.items():
        ids = cands.get(tier, [])
        if not isinstance(ids, list):
            continue
        for idx, mid in enumerate(ids, start=1):
            if isinstance(mid, str):
                tag_to_id[f"[RETRIEVED_{tag_name}_{idx:02d}]"] = mid
    return tag_to_id
def _extract_online_schema_ids(
    obj: dict[str, Any],
    tag_to_id: dict[str, str],
) -> list[str]:
    ids: list[str] = []
    for key in _ONLINE_KEYS:
        tags = obj.get(key, [])
        if not isinstance(tags, list):
            continue
        for tag in tags:
            if not isinstance(tag, str):
                continue
            ids.append(tag_to_id.get(tag, tag))
    return ids
def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        cand = text[start : end + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
def _extract_predicted_ids(
    row: dict[str, Any],
    tag_to_id: dict[str, str],
) -> tuple[list[str], list[str]]:
    if isinstance(row.get("selected_memory_ids"), (dict, list)):
        ids = _flatten_ids(row.get("selected_memory_ids"))
        return ids, ids
    direct_online_ids = _extract_online_schema_ids(row, tag_to_id)
    if direct_online_ids:
        return direct_online_ids, direct_online_ids
    txt = row.get("prediction") or row.get("output") or ""
    if isinstance(txt, str):
        obj = _extract_json_from_text(txt)
        if isinstance(obj, dict):
            online_ids = _extract_online_schema_ids(obj, tag_to_id)
            if online_ids:
                return online_ids, online_ids
            ranked = obj.get("ranked_ids")
            selected = obj.get("selected_ids") or obj.get("selected_memory_ids")
            ranked_ids = _flatten_ids(ranked)
            selected_ids = _flatten_ids(selected)
            if not selected_ids:
                selected_ids = ranked_ids
            return ranked_ids, selected_ids
    return [], []
def _safe_div(x: float, y: float) -> float:
    return x / y if y > 0 else 0.0
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate selector predictions offline.")
    ap.add_argument("--gold", required=True, help="Gold selector dataset JSONL")
    ap.add_argument("--pred", required=True, help="Prediction JSONL")
    ap.add_argument("--k", type=int, default=3, help="k for P@k / R@k / F1@k")
    args = ap.parse_args()
    gold_by_task: dict[str, dict[str, Any]] = {}
    with open(args.gold, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            gold_ids = _flatten_ids(row.get("select", {}).get("selected_memory_ids", {}))
            gold_by_task[task_id] = {
                "ids": gold_ids,
                "tag_to_id": _tag_maps_from_candidates(row.get("retrieve", {}).get("candidates", {})),
            }
    if not gold_by_task:
        raise SystemExit("No gold rows with task_id found.")
    n = 0
    exact_match = 0
    sum_prec = 0.0
    sum_rec = 0.0
    sum_f1 = 0.0
    sum_p_at_k = 0.0
    sum_r_at_k = 0.0
    sum_f1_at_k = 0.0
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id", "")).strip()
            if not task_id or task_id not in gold_by_task:
                continue
            gold = gold_by_task[task_id]
            ranked_ids, pred_ids = _extract_predicted_ids(row, gold["tag_to_id"])
            gold_ids = gold["ids"]
            pred_set = set(pred_ids)
            gold_set = set(gold_ids)
            inter = len(pred_set & gold_set)
            prec = _safe_div(inter, len(pred_set))
            rec = _safe_div(inter, len(gold_set))
            f1 = _safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
            topk = ranked_ids[: args.k] if ranked_ids else pred_ids[: args.k]
            topk_set = set(topk)
            inter_k = len(topk_set & gold_set)
            p_at_k = _safe_div(inter_k, len(topk_set))
            r_at_k = _safe_div(inter_k, len(gold_set))
            f1_at_k = _safe_div(2 * p_at_k * r_at_k, (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0
            if pred_set == gold_set:
                exact_match += 1
            sum_prec += prec
            sum_rec += rec
            sum_f1 += f1
            sum_p_at_k += p_at_k
            sum_r_at_k += r_at_k
            sum_f1_at_k += f1_at_k
            n += 1
    if n == 0:
        raise SystemExit("No overlapping task_id rows found between gold and pred files.")
    result = {
        "count": n,
        "exact_set_match": exact_match / n,
        "precision": sum_prec / n,
        "recall": sum_rec / n,
        "f1": sum_f1 / n,
        f"p@{args.k}": sum_p_at_k / n,
        f"r@{args.k}": sum_r_at_k / n,
        f"f1@{args.k}": sum_f1_at_k / n,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == "__main__":
    main()
