import os
import re
import json
import types
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, List
from opd_evolver.base.engine.logs import logger
def parse_xml_content(content: str, tag: str) -> dict:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return {tag: None}
    elif len(matches) == 1:
        return {tag: matches[0].strip()}
    else:
        return {tag: [m.strip() for m in matches]}
def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
def write_file_content(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
def get_env_paths(base_path: str) -> List[str]:
    env_paths = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if item.startswith("env_") and os.path.isdir(os.path.join(base_path, item)):
                env_paths.append(os.path.join(base_path, item))
    return env_paths
def archive_files(env_folder_path: str, env_id: str = None) -> bool:
    if not env_folder_path:
        raise ValueError("env_folder_path cannot be empty")
    import subprocess
    import sys
    import logging
    logger = logging.getLogger(__name__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    archive_script = os.path.join(project_root, "scripts", "run_archive_files.py")
    if env_id:
        logger.info(f"Archiving auxiliary files for environment: {env_id}")
    logger.info(f"Environment folder: {env_folder_path}")
    try:
        result = subprocess.run(
            [sys.executable, archive_script, env_folder_path],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        if result.returncode == 0:
            logger.info("Directory cleanup completed successfully")
            logger.info(f"Archive output: {result.stdout}")
            done_file_path = os.path.join(env_folder_path, "done.txt")
            write_file_content(done_file_path, "")
            logger.info(f"Created done.txt file: {done_file_path}")
            return True
        else:
            logger.error(f"Archive script failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running archive script: {e}")
        return False
def parse_llm_output(resp: str, key: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {key: None}
    if not resp:
        result["_parse_error"] = "empty_response"
        return result
    candidates: List[str] = []
    def _extract_block(marker: str) -> Optional[str]:
        start = resp.find(marker)
        if start == -1:
            return None
        start += len(marker)
        end = resp.find("```", start)
        return resp[start:end if end != -1 else None].strip()
    def _fix_bash_escapes(text: str) -> str:
        result_chars = []
        in_string = False
        i = 0
        length = len(text)
        valid_escapes = set('"\\bfnrtu/')
        while i < length:
            ch = text[i]
            if ch == '\\' and i + 1 < length:
                next_ch = text[i + 1]
                if in_string:
                    if next_ch == '"' or next_ch == '\\':
                        result_chars.append(ch)
                        result_chars.append(next_ch)
                        i += 2
                        continue
                    elif next_ch not in valid_escapes:
                        result_chars.append('\\')
                        result_chars.append('\\')
                        result_chars.append(next_ch)
                        i += 2
                        continue
                    else:
                        result_chars.append(ch)
                        result_chars.append(next_ch)
                        i += 2
                        continue
                else:
                    result_chars.append(ch)
                    i += 1
                    continue
            if ch == '"':
                in_string = not in_string
            result_chars.append(ch)
            i += 1
        return ''.join(result_chars)
    def _escape_control_chars(text: str) -> str:
        out: List[str] = []
        in_string = False
        escape = False
        for ch in text:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                continue
            if in_string:
                if ch == '\n':
                    out.append('\\n')
                    continue
                elif ch == '\r':
                    out.append('\\r')
                    continue
                elif ch == '\t':
                    out.append('\\t')
                    continue
            out.append(ch)
        return "".join(out)
    def _try_parse_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            fixed = _escape_control_chars(text)
            if fixed != text:
                return json.loads(fixed)
        except Exception:
            pass
        try:
            fixed = _fix_bash_escapes(text)
            return json.loads(fixed)
        except Exception:
            pass
        return None
    block = _extract_block("```json")
    if block:
        candidates.append(block)
    if not candidates:
        block = _extract_block("```")
        if block:
            candidates.append(block)
    candidates.append(resp.strip())
    if "{" in resp and "}" in resp:
        blob = re.search(r"\{[\s\S]*\}", resp)
        if blob:
            candidates.append(blob.group(0))
    for cand in candidates:
        obj = _try_parse_json(cand)
        if obj is None:
            continue
        if isinstance(obj, dict) and key in obj:
            return {key: obj[key]}
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and key in item:
                    return {key: item[key]}
    m = re.search(rf"{re.escape(key)}\s*[:=]\s*(.+)", resp)
    if m:
        return {key: m.group(1).strip()}
    result["_parse_error"] = "key_not_found"
    logger.warning(f"Failed to parse key '{key}' from LLM output.")
    return result
def parse_llm_action_response(resp: str) -> Dict[str, Any]:
    try:
        if not resp:
            logger.warning("Received None or empty response from LLM")
            return {"action": "no_action", "params": {}, "_parse_error": "Empty LLM response"}
        def _extract_block(marker: str) -> Optional[str]:
            start = resp.find(marker)
            if start == -1:
                return None
            start += len(marker)
            end = resp.find("```", start)
            return resp[start:end if end != -1 else None].strip()
        def _extract_balanced(text: str, open_char: str, close_char: str) -> Optional[str]:
            start = text.find(open_char)
            if start == -1:
                return None
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        return text[start:idx + 1].strip()
            return None
        def _truncate_after_balanced_json(text: str) -> Optional[str]:
            obj_start = text.find("{")
            arr_start = text.find("[")
            starts = [idx for idx in (obj_start, arr_start) if idx != -1]
            if not starts:
                return None
            start = min(starts)
            in_string = False
            escape = False
            stack: List[str] = []
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in "[{":
                    stack.append(ch)
                elif ch == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                        if not stack:
                            return text[start:idx + 1].strip()
                elif ch == "]":
                    if stack and stack[-1] == "[":
                        stack.pop()
                        if not stack:
                            return text[start:idx + 1].strip()
            return text[start:].strip()
        def _repair_partial_json(text: str) -> Optional[str]:
            candidate = text.strip().strip("`")
            if not candidate:
                return None
            action_match = re.search(r'["\']action["\']\s*:', candidate)
            if action_match:
                brace_before = candidate.rfind("{", 0, action_match.start() + 1)
                if brace_before != -1:
                    candidate = candidate[brace_before:]
                else:
                    candidate = "{" + candidate[action_match.start():]
            truncated = _truncate_after_balanced_json(candidate)
            if truncated:
                candidate = truncated
            rebuilt: List[str] = []
            stack: List[str] = []
            in_string = False
            escape = False
            for ch in candidate:
                if escape:
                    rebuilt.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    rebuilt.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    rebuilt.append(ch)
                    in_string = not in_string
                    continue
                if not in_string and ch in "[{":
                    stack.append(ch)
                    rebuilt.append(ch)
                    continue
                if not in_string and ch == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                        rebuilt.append(ch)
                    continue
                if not in_string and ch == "]":
                    if stack and stack[-1] == "[":
                        stack.pop()
                        rebuilt.append(ch)
                    continue
                rebuilt.append(ch)
            repaired = "".join(rebuilt).strip()
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            while stack:
                open_ch = stack.pop()
                repaired += "}" if open_ch == "{" else "]"
            if not repaired:
                return None
            return repaired
        def _normalize_action_data(action_data: Any) -> (Optional[Dict[str, Any]], Optional[str]):
            if isinstance(action_data, list):
                if not action_data:
                    return None, "Empty list returned by LLM"
                logger.warning("LLM returned a list of actions; taking the first entry")
                action_data = action_data[0]
            if not isinstance(action_data, dict):
                return None, "Missing 'action' key or invalid dict"
            if "action" not in action_data and "name" in action_data:
                action_data = {
                    "action": action_data.get("name"),
                    "params": action_data.get("arguments", {}),
                }
            action = action_data.get("action")
            if not isinstance(action, str) or not action.strip():
                return None, "Missing 'action' key or invalid dict"
            normalized_action = action.strip().lower()
            normalized: Dict[str, Any] = dict(action_data)
            normalized["action"] = normalized_action
            params = normalized.get("params")
            if params is None:
                params = {}
            elif not isinstance(params, dict):
                params = {"value": params}
            normalized["params"] = params
            if normalized_action == "execute":
                if "command" not in params:
                    command = None
                    for field in ("command", "query", "sql"):
                        value = normalized.get(field)
                        if isinstance(value, str) and value.strip():
                            command = value.strip()
                            break
                    if command:
                        params["command"] = command
            return normalized, None
        def _heuristic_fallback_action(text: str) -> Optional[Dict[str, Any]]:
            block_match = re.search(r"```(?:sql|bash|python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
            if block_match:
                block = block_match.group(1).strip()
                if block and not block.startswith("{") and not block.startswith("["):
                    return {
                        "action": "execute",
                        "params": {"command": block},
                        "_parse_error": "heuristic_fallback_code_block",
                    }
            sql_match = re.search(
                r"(?is)\b(select|with|show|describe|desc|use|insert|update|delete|create|drop|alter)\b[\s\S]*?;",
                text,
            )
            if sql_match:
                return {
                    "action": "execute",
                    "params": {"command": sql_match.group(0).strip()},
                    "_parse_error": "heuristic_fallback_sql",
                }
            lowered = text.lower()
            if re.search(r"\b(sql|database|table|query|schema|select|join|where)\b", lowered):
                return {
                    "action": "execute",
                    "params": {"command": "SHOW TABLES;"},
                    "_parse_error": "heuristic_fallback_sql_intent",
                }
            return None
        def _try_json_loads(text: str) -> (Optional[Any], Optional[str]):
            try:
                return json.loads(text), None
            except Exception as e:
                original_error = f"{type(e).__name__}: {e}"
                try:
                    fixed = _fix_invalid_escapes(text)
                    if fixed != text:
                        return json.loads(fixed), None
                except Exception:
                    pass
                try:
                    fixed = _escape_control_chars_in_strings(text)
                    if fixed != text:
                        return json.loads(fixed), None
                except Exception:
                    pass
                try:
                    fixed = _fix_invalid_escapes(_escape_control_chars_in_strings(text))
                    if fixed != text:
                        return json.loads(fixed), None
                except Exception:
                    pass
                return None, original_error
        def _escape_control_chars_in_strings(text: str) -> str:
            out: List[str] = []
            in_string = False
            escape = False
            for ch in text:
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    out.append(ch)
                    continue
                if in_string:
                    if ch == '\n':
                        out.append('\\n')
                        continue
                    elif ch == '\r':
                        out.append('\\r')
                        continue
                    elif ch == '\t':
                        out.append('\\t')
                        continue
                out.append(ch)
            return "".join(out)
        def _fix_invalid_escapes(text: str) -> str:
            out: List[str] = []
            in_string = False
            i = 0
            length = len(text)
            while i < length:
                ch = text[i]
                if ch == '"' and (i == 0 or text[i-1] != '\\'):
                    in_string = not in_string
                    out.append(ch)
                    i += 1
                    continue
                if in_string and ch == '\\' and i + 1 < length:
                    next_ch = text[i + 1]
                    if next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
                        out.append(ch)
                        out.append(next_ch)
                        i += 2
                        continue
                    if next_ch == 'u' and i + 5 < length:
                        out.append(text[i:i+6])
                        i += 6
                        continue
                    out.append(next_ch)
                    i += 2
                    continue
                out.append(ch)
                i += 1
            return "".join(out)
        def _escape_unescaped_inner_quotes(text: str) -> str:
            out: List[str] = []
            in_string = False
            escape = False
            length = len(text)
            for i, ch in enumerate(text):
                if escape:
                    out.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    if not in_string:
                        in_string = True
                        out.append(ch)
                        continue
                    j = i + 1
                    next_nonspace = None
                    while j < length and text[j].isspace():
                        j += 1
                    if j < length:
                        next_nonspace = text[j]
                    if next_nonspace in (",", "}", "]", None):
                        in_string = False
                        out.append(ch)
                    else:
                        out.append("\\\"")
                    continue
                out.append(ch)
            return "".join(out)
        def _append_candidate(candidates: List[str], value: Optional[str]) -> None:
            if value is None:
                return
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        candidates: List[str] = []
        block = _extract_block("```json")
        _append_candidate(candidates, block)
        if not candidates:
            block = _extract_block("```")
            _append_candidate(candidates, block)
        stripped = resp.strip()
        _append_candidate(candidates, stripped)
        for extra in (_extract_balanced(resp, "{", "}"), _extract_balanced(resp, "[", "]")):
            _append_candidate(candidates, extra)
        action_match = re.search(r'["\']action["\']\s*:', resp)
        if action_match:
            brace_before = resp.rfind("{", 0, action_match.start() + 1)
            if brace_before != -1:
                _append_candidate(candidates, resp[brace_before:])
            else:
                _append_candidate(candidates, resp[action_match.start():])
        base_candidates = list(candidates)
        for cand in base_candidates:
            _append_candidate(candidates, _repair_partial_json(cand))
        parse_errors: List[str] = []
        for cand in candidates:
            action_data, err = _try_json_loads(cand)
            if action_data is None:
                sanitized = _escape_unescaped_inner_quotes(cand)
                if sanitized != cand:
                    action_data, err2 = _try_json_loads(sanitized)
                    if action_data is None:
                        parse_errors.append(f"{err}; after sanitizing quotes -> {err2}")
                        continue
                    logger.warning("Recovered action JSON by escaping inner quotes inside strings.")
                else:
                    parse_errors.append(err)
                    continue
            normalized, normalize_err = _normalize_action_data(action_data)
            if normalized is None:
                parse_errors.append(normalize_err or "normalize_failed")
                continue
            return normalized
        heuristic_action = _heuristic_fallback_action(resp)
        if heuristic_action is not None:
            logger.warning("Recovered action with heuristic fallback after JSON parse failure.")
            return heuristic_action
        error_detail = "; ".join(parse_errors) if parse_errors else "Unknown parse failure"
        logger.warning(f"Failed to parse action JSON. Tried {len(candidates)} candidates. Errors: {error_detail}. Using default action. Raw response: {resp}")
        return {"action": "Invalid", "params": {}, "_parse_error": error_detail}
    except Exception as e:
        logger.warning(f"Unexpected error while parsing action: {e}. Using default action.")
        return {"action": "Invalid", "params": {}, "_parse_error": f"{type(e).__name__}: {e}"}
def _load_basic_info(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
def summarize_candidates(workspace_path: str) -> Dict[str, Any]:
    cdir = os.path.join(workspace_path, "candidates")
    result: Dict[str, Any] = {
        "workspace_path": workspace_path,
        "candidates": [],
        "edges": [],
        "best": None,
    }
    if not os.path.isdir(cdir):
        with open(os.path.join(workspace_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
    basics_by_round: Dict[int, Dict[str, Any]] = {}
    for name in sorted(os.listdir(cdir)):
        if not name.startswith("candidate_"):
            continue
        try:
            r = int(name.split("_")[-1])
        except Exception:
            continue
        info = _load_basic_info(os.path.join(cdir, name, "basic_info.json")) or {}
        info["folder_name"] = name
        basics_by_round[r] = info
    def _m(info: Dict[str, Any], key: str) -> Optional[float]:
        try:
            val = (info.get("metrics") or {}).get(key)
            return None if val is None else float(val)
        except Exception:
            return None
    best = {"round": None, "accuracy": -1.0, "cost": None}
    for r in sorted(basics_by_round.keys()):
        info = basics_by_round[r]
        parent = info.get("parent")
        acc = _m(info, "accuracy")
        cost = _m(info, "cost")
        parent_acc = None
        parent_cost = None
        acc_delta = None
        cost_delta = None
        success = None
        if parent is not None and parent in basics_by_round:
            pinfo = basics_by_round[parent]
            parent_acc = _m(pinfo, "accuracy")
            parent_cost = _m(pinfo, "cost")
            if parent_acc is not None and acc is not None:
                acc_delta = acc - parent_acc
            if parent_cost is not None and cost is not None:
                cost_delta = cost - parent_cost
            if acc is not None and parent_acc is not None:
                if acc > parent_acc:
                    success = True
                elif acc == parent_acc and (cost is not None and parent_cost is not None) and cost < parent_cost:
                    success = True
                else:
                    success = False
            else:
                success = False
            result["edges"].append([parent, r])
        if acc is not None and acc > best["accuracy"]:
            best = {"round": r, "accuracy": acc, "cost": cost}
        item = {
            "round": r,
            "parent": parent,
            "accuracy": acc,
            "cost": cost,
            "parent_accuracy": parent_acc,
            "parent_cost": parent_cost,
            "acc_delta": acc_delta,
            "cost_delta": cost_delta,
            "success": success,
            "trajectory_path": info.get("trajectory_path"),
        }
        result["candidates"].append(item)
        try:
            with open(os.path.join(cdir, info.get("folder_name"), "optimization_result.json"), "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    result["best"] = best if best["round"] is not None else None
    try:
        with open(os.path.join(workspace_path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result
