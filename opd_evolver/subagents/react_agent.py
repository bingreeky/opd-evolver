from __future__ import annotations
import re
from typing import Any, Dict, List
from pydantic import Field
from opd_evolver.base.agent.base_agent import BaseAgent
from opd_evolver.base.agent.memory import Memory
from opd_evolver.base.engine.utils import parse_llm_action_response, parse_llm_output
from opd_evolver.base.engine.logs import logger, LogLevel
from opd_evolver.benchmark.common.env import BasicInfo, Observation, Action
GAIA_PROMPT = """You are a specialized SubAgent. Complete the assigned task efficiently.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining {remaining_steps} steps
{budget_warning}

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context ====
{context}

==== Original Question (for reference) ====
{original_question}

==== Available Tools ====
{action_space}

==== Guidelines ====
1. Focus on completing YOUR TASK above
2. Think step by step before outputting an action
3. Write key observations to the "memory" field
4. Use print() in ExecuteCodeAction to see computation results
5. Once done, use 'finish' IMMEDIATELY

⚠️ BUDGET: When remaining_steps <= 5, use 'finish' NOW!

==== Output Format ====
```json
{{
    "action": "<tool_name>",
    "params": {{}},
    "memory": "<observations>"
}}
```

==== Memory ====
{memory}

==== Current Observation ====
{obs}
"""
TERMINALBENCH_PROMPT = """
==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", your work is lost and marked as timeout.

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context (from previous attempts) ====
{context}
Use this info: repeat what WORKED, avoid what FAILED.

==== Original Question (for reference) ====
{original_question}

==== Action Space ====
{action_space}

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{obs}

==== Thinking ====
Think step by step before outputting an action. Write key reasoning in memory for future steps.

==== Action Guidelines ====
You have TWO actions available:

1. **execute** - Run shell commands and observe results
   - Use this to install packages, configure services, verify status, etc.
   - Example: "apt update && apt install -y nginx"

2. **finish** - Report your progress to MainAgent
   - Use when task is COMPLETE (status="done")
   - Use when you made PROGRESS but need more work (status="partial")
   - ⚠️ MUST use before running out of steps! Your work is LOST if you timeout.

**What to report in finish:**
- completed: List SUCCESSFUL steps that WORKED (e.g., ["apt update succeeded", "nginx installed"])
- issues: List FAILED attempts with WHY (e.g., ["nginx -v failed: command not found"])
- message: Brief summary of current state

This info helps the NEXT SubAgent know what to repeat and what to avoid.

==== Output Format ====
⚠️ CRITICAL: You MUST reply with ONLY a JSON object. No explanations, no markdown, no other text.

For execute:
{{"action": "execute", "params": {{"command": "your shell command"}}, "memory": "key findings"}}

For finish:
{{"action": "finish", "params": {{"status": "done|partial", "completed": [...], "issues": [...], "message": "..."}}, "memory": "final notes"}}

"""
INTERCODE_PROMPT = """
==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", your work is lost and marked as timeout.

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context (from previous attempts) ====
{context}
Use this info: repeat what WORKED, avoid what FAILED.

==== Original Question (for reference) ====
{original_question}

==== Action Space ====
{action_space}

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{obs}

==== Thinking ====
Think step by step before outputting an action. Write key reasoning in memory for future steps.

==== CRITICAL SUBMISSION GUIDELINES ====
⚠️ IMPORTANT: For InterCode evaluation, your LAST command must output the EXACT answer expected.

For counting tasks: Your final command should output just the number, like:
- echo 28
- echo $((17 + 11))
- find /testbed | wc -l

DO NOT output descriptive text like "There are X files and Y directories"
The system only evaluates your LAST command's output!

==== Action Guidelines ====
You have TWO actions available:

1. **execute** - Run commands and observe results
   - Use this to explore, calculate, and solve the problem
   - For counting: use find, wc, ls commands
   - For your FINAL answer: ensure the last execute outputs just the answer

2. **finish** - Report your progress to MainAgent
   - Use when task is COMPLETE (status="done")
   - Use when you made PROGRESS but need more work (status="partial")
   - ⚠️ MUST use before running out of steps! Your work is LOST if you timeout.

**What to report in finish:**
- completed: List SUCCESSFUL steps that WORKED (e.g., ["counted files: 17", "counted dirs: 11", "total: 28"])
- issues: List FAILED attempts with WHY (e.g., ["command X failed: reason"])
- message: Brief summary of current state

This info helps the NEXT SubAgent know what to repeat and what to avoid.

==== Output Format ====
⚠️ CRITICAL: You MUST reply with ONLY a JSON object. No explanations, no markdown, no other text.

For execute:
{{"action": "execute", "params": {{"command": "your command"}}, "memory": "key findings"}}

For finish:
{{"action": "finish", "params": {{"status": "done|partial", "completed": [...], "issues": [...], "message": "..."}}, "memory": "final notes"}}

"""
BASH_PROMPT = """You are a bash expert solving tasks with shell commands.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

==== Examples ====
Q: Count all lines of all *.c files in /testbed recursively
A: find /testbed -name "*.c" -print0 | xargs -0 cat | wc -l

Q: Find duplicate md5 hashes for all .java files in /testbed
A: md5sum /testbed/*.java | awk '{{print $1}}' | sort | uniq -d

Q: Print all unique file paths under dir1 compared to dir2
A: comm -23 <(find /testbed/dir1 | sed 's#/testbed/dir1/##' | sort) \
           <(find /testbed/dir2 | sed 's#/testbed/dir2/##' | sort)

Q: Calculate md5sum of sorted md5sums of all .py files under a directory
A: find /path -type f -name "*.py" -exec md5sum {{}} + | awk '{{print $1}}' | sort | md5sum

==== Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Observation ====
{obs}

==== Pipeline Rules (CRITICAL) ====
1. Extract target field BEFORE sort/uniq/wc:
   md5sum files  → "hash  path"      → awk '{{print $1}}' | sort | uniq -d
   ls -s files   → "blocks path"     → awk '{{print $1}}' | sort -n
   wc -l files   → "count filename"  → grep total | awk '{{print $1}}'
   du -sh files  → "size  path"      → awk '{{print $1}}'

2. md5sum patterns (most common tasks):
   Duplicate hashes:    md5sum /path/*.ext | awk '{{print $1}}' | sort | uniq -d
   md5 of sorted md5s:  find /path -type f -print0 | sort -z | xargs -r0 md5sum | awk '{{print $1}}' | sort | md5sum
   md5 of all md5s:     find /path -name "*.ext" -exec md5sum {{}} + | awk '{{print $1}}' | sort | md5sum

3. Counting patterns:
   Total lines across files:  find ... | xargs cat | wc -l  (single stream, no "total" line)
   Count files (not lines):   find ... | wc -l

4. find best practices:
   - Add -type f unless dirs needed
   - Use -print0 | xargs -0 for spaces in filenames
   - Use sort -z with -print0 for reproducible order

==== Common Command Patterns ====
Set difference between files:  comm -23 <(sort file1) <(sort file2)
Merge files on matching field: join -a1 -a2 <(sort file1) <(sort file2)
Prepend dynamic value:         cmd | sed 's/^/$(hostname): /'
Copy preserving attributes:    find ... | xargs -I{{}} cp -p "{{}}" /dest/
Count files changed:           find ... -exec chmod {{}} \\; -exec echo {{}} \\; | wc -l
Recursive line count (clean):  find ... | xargs cat | wc -l

==== Debug Strategy ====
If output is empty/wrong, NEVER retry same command. Instead:
1. Inspect intermediate output: remove last pipe stage, run shortened pipeline
2. Diagnose:
   - Empty? Check: (a) find -type f missing? (b) path/glob wrong? → test with ls
   - Wrong number? Check: (a) forgot awk before uniq? (b) "total" line in wc?
   - Wrong hash? Check: (a) forgot awk '{{print $1}}'? (b) files unsorted?
3. Fix only the broken stage

Example debugging:
  Cmd: find /testbed -name "*.java" -exec md5sum {{}} \\; | sort | uniq -d
  Output: (empty)

  Debug step 1: find /testbed -name "*.java" -exec md5sum {{}} \\; | sort
  Output shows: "hash1  file1\\nhash2  file2\\nhash2  file3"

  Diagnosis: uniq -d operates on full lines, but lines differ due to filenames
  Fix: find /testbed -name "*.java" -exec md5sum {{}} \\; | awk '{{print $1}}' | sort | uniq -d

==== Output Format ====
Your LAST execute output is evaluated as the final answer. It MUST contain ONLY the answer value.
  ✓ 42
  ✓ a1b2c3d4e5f6...
  ✗ "There are 42 files"        — no descriptive text
  ✗ "a1b2c3  filename"          — strip filename with awk '{{print $1}}'
  ✗ "(empty)"                   — debug before submitting

If output is empty, DO NOT retry the same command. Shorten the pipeline and inspect
intermediate output to find the broken stage, then fix only that stage.

==== Actions ====
execute: {{"action": "execute", "params": {{"command": "cmd"}}, "memory": "observation"}}
submit:  {{"action": "submit", "params": {{}}}}

Reply with ONLY valid JSON."""
OS_PROMPT = """You are an Ubuntu system administration agent. Modify the container state so that the hidden evaluator succeeds.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

==== Scoring ====
- The final answer is NOT the last command output.
- submit runs a hidden evaluation command against the current container state.
- Success requires the requested files, users, groups, permissions, scripts, processes, and content to exist exactly as specified.
- Once the state is ready, use submit. Reserve the final step for submit instead of continuing to explore.

==== Relevant Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

==== OS Rules ====
1. Follow exact names and paths from the task. Do not invent alternative filenames, groups, users, services, or directories.
2. Prefer absolute paths for required artifacts. If the task gives a relative path, verify the working directory before relying on it.
3. Verify critical state before submit with commands such as ls -l, stat, id, getent, ps, test, cat, grep, or running the required script.
4. Do not delete backups, logs, scripts, reports, or intermediate artifacts unless the task explicitly says to remove them.
5. If creating a validation/check script, make sure the script exits 0 when the requested state is present, and create the state it checks if the task requires that too.
6. For group access, set the group explicitly: chgrp <group> <path>; then chmod the requested bits. chmod g+... alone is not enough if the group owner is wrong.
7. For symlink ownership, use chown -h owner:group <symlink>. Plain chown follows the link target.
8. For long-running background tasks, use nohup or a detached shell, write a PID file if requested, and verify the PID is still alive before submit.
9. For group membership checks, use id -nG <user> or parse getent group correctly; do not require the whole getent line to equal a username.
10. For text processing tasks, inspect sample input first and extract only the requested field.

==== Common Commands ====
Users/groups:       id user; id -nG user; getent group group; useradd; groupadd; usermod -aG group user
Ownership/perms:    stat -c '%U %G %a %n' path; chown; chgrp; chmod; chmod +x script
Files/content:      mkdir -p; touch; cp -a; grep; awk; sed; sort -u; wc -l; test -e path
Processes:          nohup cmd >/path/log 2>&1 & echo $! > /path/pid; ps -p $(cat /path/pid); kill -0 $(cat /path/pid)
Symlinks:           ln -s target link; readlink link; chown -h owner:group link; ls -l link

==== Before Submit Checklist ====
- Required paths exist exactly where requested.
- Ownership, group, and permissions match the task.
- Required content, report, log, or count is correct.
- Required scripts are executable and return the intended exit code.
- Required users and groups exist with the correct membership.
- Required background process is still running, if applicable.
- No required artifact was accidentally removed.

==== Actions ====
execute: {{"action": "execute", "params": {{"command": "bash command"}}, "memory": "what changed or verified"}}
submit:  {{"action": "submit", "params": {{}}}}

Reply with ONLY valid JSON."""
SQL_PROMPT = """You are a SQL expert solving natural language to SQL tasks.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

🚨 CRITICAL: If task shows "DATABASE: <name>", you MUST execute "USE <name>;" first!

==== Examples ====
Q: DATABASE: pets_1 | Find students who have both cat AND dog pets
A: Step 1: USE pets_1;
   Step 2: SELECT fname FROM Student JOIN Has_Pet ON Student.stuid = Has_Pet.stuid
           JOIN Pets ON Pets.petid = Has_Pet.petid WHERE pettype = 'cat'
           INTERSECT
           SELECT fname FROM Student JOIN Has_Pet ON Student.stuid = Has_Pet.stuid
           JOIN Pets ON Pets.petid = Has_Pet.petid WHERE pettype = 'dog'

Q: Find students who have a dog but NOT a cat
A: SELECT fname FROM Student JOIN Has_Pet ON ... JOIN Pets ON ... WHERE pettype = 'dog'
   AND stuid NOT IN (SELECT stuid FROM Student JOIN Has_Pet ON ... JOIN Pets ON ... WHERE pettype = 'cat')

Q: Find students who do NOT have a cat
A: SELECT stuid FROM Student
   EXCEPT
   SELECT stuid FROM Student JOIN Has_Pet ON ... JOIN Pets ON ... WHERE pettype = 'cat'

Q: Which owner paid the most in total? (top-1 aggregation)
A: SELECT owner_id FROM Owners JOIN Dogs ON ... JOIN Treatments ON ...
   GROUP BY owner_id ORDER BY sum(cost_of_treatment) DESC LIMIT 1

Q: Find battles with more than 10 killed total
A: SELECT id, name FROM battle JOIN ship ON ... JOIN death ON ...
   GROUP BY id HAVING sum(killed) > 10

==== Schema Exploration ====
Step 1: SHOW DATABASES;  -- list all databases
Step 2: USE <database_name>;  -- REQUIRED: select the database (e.g., USE pets_1;)
Step 3: SHOW TABLES;  -- list tables in selected database
Step 4: DESCRIBE <table>;  -- examine ONE table per execute (not multiple)
Step 5: SELECT * FROM <table> LIMIT 3;  -- sample data

🚨 CRITICAL:
- You MUST execute "USE <database_name>;" before any table queries
- Execute ONLY ONE SQL statement per action (NO semicolon-separated commands)
- Example: ✗ DESCRIBE t1; DESCRIBE t2;  ✓ DESCRIBE t1;  (then next step: DESCRIBE t2;)

==== Critical SQL Patterns ====

BOTH X AND Y → INTERSECT (NOT AND in WHERE):
  SELECT col FROM ... WHERE condition_A
  INTERSECT
  SELECT col FROM ... WHERE condition_B

EITHER X OR Y → UNION or OR in WHERE:
  WHERE type = 'cat' OR type = 'dog'

NOT / EXCLUDE → EXCEPT or NOT IN subquery:
  SELECT ... EXCEPT SELECT ...  -- cleaner
  WHERE id NOT IN (SELECT id FROM ... WHERE condition)

3-TABLE JOIN pattern (most common in this benchmark):
  SELECT T1.col FROM TableA AS T1
  JOIN TableB AS T2 ON T1.id = T2.id
  JOIN TableC AS T3 ON T2.other_id = T3.other_id
  WHERE T3.condition = 'value'

TOP-1 AGGREGATION:
  GROUP BY key ORDER BY SUM(col) DESC LIMIT 1

DISTINCT → use when query says "distinct" or joining may produce duplicates

SELF-JOIN (when same table appears twice with different roles):
  JOIN Highschooler AS T2 ON friend.student_id = T2.id
  JOIN Highschooler AS T3 ON friend.friend_id = T3.id

==== Keyword → Pattern Mapping ====
"both A and B"           → INTERSECT
"A but not B"            → NOT IN subquery
"neither / do not have"  → EXCEPT or NOT IN
"at least N / more than" → HAVING count(*) > N
"most / largest / top"   → ORDER BY ... DESC LIMIT 1
"distinct / different"   → SELECT DISTINCT

==== Common Mistakes to Avoid ====
✗ Executing multiple SQL statements in one action (DESCRIBE t1; DESCRIBE t2;)
  → Execute ONE statement per action, or you'll get "Commands out of sync"
✗ Using GROUP BY + COUNT when question asks to sort by an existing numeric column
  → Check if a direct column already exists before reaching for COUNT/SUM
  → WRONG: GROUP BY ... ORDER BY COUNT(*) when a numeric column exists
  → RIGHT: ORDER BY numeric_column directly
✗ Using AND for "both X and Y" on the same row when they're in different rows
  → Use INTERSECT instead
✗ Forgetting DISTINCT when joining multiple tables (duplicates appear)
✗ Using WHERE count() instead of HAVING after GROUP BY
✗ Wrong join key — always verify with DESCRIBE before writing JOIN
✗ Submitting without checking if output order matches what was asked
  → Re-read the question after seeing output, before submitting
✗ Forgetting "USE database_name;" before queries

==== Verify Before Submit ====
Before submitting, re-read the original question and check your output against it:

Checklist:
□ Does the ORDER BY column match what the question asks to sort by?
  "ordered by number of X"      → ORDER BY direct_column, NOT COUNT(*)
  "ordered by total/sum of X"   → ORDER BY SUM(col)
  "ordered by how many times"   → ORDER BY COUNT(*)

□ Are extra columns excluded?
  Question asks for "names" → output must have ONLY names, not IDs or counts

□ Is the sort direction correct?
  "ascending / lowest first"   → ASC (default, but write it explicitly)
  "descending / highest first" → DESC

□ Did you use GROUP BY unnecessarily?
  If ORDER BY is on a direct column with no aggregation needed, DROP the GROUP BY.

WRONG (over-complicated):
  SELECT Name FROM people JOIN poker_player ON ...
  GROUP BY People_ID ORDER BY COUNT(Poker_Player_ID) ASC
  -- This sorts by row count, not by the actual stat column

RIGHT (direct):
  SELECT Name FROM people JOIN poker_player ON ...
  ORDER BY Final_Table_Made ASC
  -- Sort directly on the column the question refers to

==== Output Format ====
Your LAST execute output is the evaluated answer. It must return ONLY the expected columns/values.
  ✓ Single column result when asked for one field
  ✓ Exact column names matching the question
  ✗ Extra columns not asked for
  ✗ Descriptive text mixed with results

==== Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

==== Actions ====
execute: {{"action": "execute", "params": {{"command": "SQL query"}}, "memory": "findings"}}
submit:  {{"action": "submit", "params": {{}}}}

Reply with ONLY valid JSON.
"""
CTF_PROMPT = """You are a CTF (Capture The Flag) security challenge solver.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Challenge ====
{task_instruction}

==== Relevant Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Recent Memory ====
{memory}

==== Current Observation ====
{obs}

==== Strategy ====
1. Analyze the challenge carefully before acting
2. Use appropriate tools (strings, binwalk, grep, xxd, etc.)
3. For crypto challenges: analyze the algorithm, find weaknesses
4. For forensics: examine file metadata, hidden data, steganography
5. For web: check source, cookies, headers, injection points
6. For pwn/reverse: analyze binary, find vulnerabilities
7. **Before submitting: Use echo to display the complete flag in picoCTF{{...}} format**
   Example: echo "picoCTF{{your_flag_here}}"

==== Actions ====
1. **execute** - Run bash commands to investigate and solve
   {{"action": "execute", "params": {{"command": "your command"}}, "memory": "observations"}}

2. **submit** - Submit the complete flag in picoCTF{{...}} format
   {{"action": "submit", "params": {{"flag": "picoCTF{{...}}"}}}}

   🚨 CRITICAL REQUIREMENTS:
   - MUST include the complete flag with "picoCTF{{" prefix and "}}" suffix
   - Example: {{"action": "submit", "params": {{"flag": "picoCTF{{example_flag_here}}"}}}}
   - DO NOT submit partial answers like "example_flag_here" or "p"
   - DO NOT submit without the flag in params
   - The flag parameter is REQUIRED and must match the exact format

   💡 RECOMMENDED WORKFLOW:
   Step 1: Find the flag through investigation
   Step 2: Execute: echo "picoCTF{{your_discovered_flag}}" to verify format
   Step 3: Submit: {{"action": "submit", "params": {{"flag": "picoCTF{{your_discovered_flag}}"}}}}

==== Output ====
Reply with ONLY a JSON object. No explanations.

{{"action": "execute|submit", "params": {{}}, "memory": "key observations"}}
"""
KG_PROMPT = """You are a knowledge graph reasoning agent.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

==== Relevant Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

==== Strategy ====
Use the API calls to build variables, inspect relations/attributes, and submit the final variable.
Variables returned by API calls are named #0, #1, #2, ... in the order they appear.

Common flow:
1. get_relations(entity)
2. get_neighbors(entity_or_#var, relation)
3. optionally intersection(#a, #b), get_attributes(#a), argmax(#a, attr), argmin(#a, attr), or count(#a)
4. submit the final variable: {{"action": "submit", "params": {{"answer": "#0"}}}}

==== Output Format ====
Reply with ONLY valid JSON.

For an API call:
{{"action": "execute", "params": {{"command": "get_relations(entity name)"}}, "memory": "why this call helps"}}

For final answer:
{{"action": "submit", "params": {{"answer": "#0"}}, "memory": "final variable"}}
"""
MINIHACK_PROMPT = """You are a MiniHack/NetHack navigation agent.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

==== Relevant Knowledge ====
{context}

==== Action Space ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

==== Strategy ====
Read the ASCII map carefully. The player is usually shown as @. Walls block movement, doors may need open, keys or useful items should be picked up, and the goal/stairs/exit should be reached before submitting.

For Room and MazeWalk, explore efficiently and move toward the goal.
For KeyRoom, pick up the key first. Use apply with a direction to unlock locked doors (open only opens already-unlocked doors). After unlocking and moving through, use climb_up when standing on < (stairs up) to complete the task.
For River, avoid stepping into dangerous water/lava unless the map clearly provides a safe crossing.
If blocked, try search near walls or doors, then choose a new direction.

Use submit only after the observation indicates success, positive reward, terminal state, or that the goal has been reached.

==== Output Format ====
Reply with ONLY valid JSON. No markdown and no explanation outside JSON.

Move:
{{"action": "move", "params": {{"direction": "north|south|east|west|northeast|northwest|southeast|southwest"}}, "memory": "map reading and intent"}}

Interact:
{{"action": "pickup", "params": {{}}, "memory": "why pickup helps"}}
{{"action": "open", "params": {{"direction": "east"}}, "memory": "open unlocked door in direction"}}
{{"action": "apply", "params": {{"direction": "west"}}, "memory": "unlock locked door with key in direction"}}
{{"action": "climb_up", "params": {{}}, "memory": "go up stairs when standing on <"}}
{{"action": "search", "params": {{}}, "memory": "why searching helps"}}
{{"action": "wait", "params": {{}}, "memory": "why waiting helps"}}

Finish:
{{"action": "submit", "params": {{}}, "memory": "goal reached"}}
"""
AWM_PROMPT = """You are an agent operating inside an Agent World Model MCP tool-use environment.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}

==== Task ====
{task_instruction}

==== Relevant Knowledge ====
{context}

==== Available MCP Tools ====
{action_space}

==== Memory ====
{memory}

==== Current Observation ====
{obs}

==== Strategy ====
1. This is not a shell, SQL, or browser environment. Use only the listed MCP tools.
2. The "action" value must exactly match one available MCP tool name.
3. Put tool arguments in "params" and follow the tool input schema.
4. Inspect tool results before making state-changing calls when the correct arguments are uncertain.
5. When the task is complete, use "submit"; the environment will run the verifier.

==== Output Format ====
Reply with ONLY a valid JSON object. No markdown, no explanations outside JSON.

For a tool call:
{{"action": "<mcp_tool_name>", "params": {{}}, "memory": "why this tool call helps"}}

For final completion:
{{"action": "submit", "params": {{"answer": "optional brief final answer"}}, "memory": "final state"}}
"""
class ReActAgent(BaseAgent):
    name: str = Field(default="ReActAgent")
    description: str = Field(default="ReAct-style SubAgent for Orchestra framework")
    benchmark_type: str = Field(default="terminalbench")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")
    allowed_tools: List[str] | None = Field(default=None)
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    memory: Memory = Field(default=None)
    class Config:
        arbitrary_types_allowed = True
    def reset(self, env_info: BasicInfo) -> None:
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()
        if not self.original_question:
            self.original_question = env_info.instruction
        self.current_env_instruction = env_info.instruction
        if self.allowed_tools:
            self.current_action_space = self._filter_action_space(
                env_info.action_space,
                self.allowed_tools
            )
            logger.info(f"[ReActAgent] Filtered to tools: {self.allowed_tools}")
        else:
            self.current_action_space = env_info.action_space
    def _normalize_tool_name(self, name: str) -> str:
        normalized = name.lower().replace("_", "")
        if normalized.endswith("action"):
            normalized = normalized[:-6]
        return normalized
    def _tool_matches(self, tool_name: str, allowed_tools: List[str]) -> bool:
        if tool_name in allowed_tools:
            return True
        normalized_tool = self._normalize_tool_name(tool_name)
        for allowed in allowed_tools:
            if self._normalize_tool_name(allowed) == normalized_tool:
                return True
        return False
    def _filter_action_space(self, action_space: str, allowed_tools: List[str]) -> str:
        blocks = re.split(r'\n(?=### )', action_space)
        filtered_blocks = []
        for block in blocks:
            if block.startswith("Available actions") or block.startswith("Available MCP tools"):
                filtered_blocks.append(block.rstrip())
                continue
            match = re.match(r'### (\w+)', block)
            if match:
                tool_name = match.group(1)
                if self._tool_matches(tool_name, allowed_tools):
                    filtered_blocks.append(block.rstrip())
        return "\n\n".join(filtered_blocks)
    def parse_action(self, resp: str) -> Dict[str, Any]:
        return parse_llm_action_response(resp)
    def _get_memory(self) -> str:
        return self.memory.as_text()
    def _get_budget_warning(self, remaining_steps: int) -> str:
        finish_action = "submit" if self.benchmark_type in ("minihack", "awm") else "finish"
        if remaining_steps <= 3:
            return f"🚨 CRITICAL: Only {remaining_steps} steps left! Use '{finish_action}' NOW!"
        elif remaining_steps <= 5:
            return f"⚠️ Warning: {remaining_steps} steps remaining. Plan to finish soon."
        return ""
    def _build_prompt(
        self,
        observation: Any,
        current_step: int,
        max_steps: int,
        remaining_steps: int,
        budget_warning: str,
    ) -> str:
        if self.benchmark_type == "gaia":
            return GAIA_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "None",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type == "bash":
            return BASH_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type == "os":
            return OS_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type in ("sql", "db"):
            return SQL_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type == "intercode":
            return INTERCODE_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No additional context provided.",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type == "ctf":
            return CTF_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type in ("kg", "knowledge_graph"):
            return KG_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type == "minihack":
            return MINIHACK_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        elif self.benchmark_type == "awm":
            return AWM_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No prior knowledge available.",
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        else:
            return TERMINALBENCH_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No additional context provided.",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
    async def step(
        self,
        observation: Observation,
        history: Any,
        current_step: int = 1,
        max_steps: int = 30
    ) -> tuple[Action, str, str]:
        remaining_steps = max_steps - current_step
        budget_warning = self._get_budget_warning(remaining_steps)
        prompt = self._build_prompt(
            observation=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=budget_warning,
        )
        logger.log_to_file(LogLevel.INFO, f"ReActAgent Input:\n{prompt}\n")
        try:
            logger.info(f"[ReActAgent] Calling LLM for step {current_step}...")
            resp = await self.llm(prompt)
            logger.info(f"[ReActAgent] LLM responded: {resp} ")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""
        action = self.parse_action(resp)
        try:
            normalized_action = (action or {}).get("action")
            normalized_action = normalized_action.lower().strip() if isinstance(normalized_action, str) else None
            has_any_execute = False
            if history:
                for r in history:
                    a = getattr(r, "action", None) or {}
                    a_name = a.get("action")
                    if isinstance(a_name, str) and a_name.lower().strip() == "execute":
                        has_any_execute = True
                        break
            if normalized_action == "submit" and not has_any_execute and self.benchmark_type in ("sql", "db"):
                logger.warning(
                    "[ReActAgent] Model attempted 'submit' before any 'execute' in DB/SQL env; "
                    "overriding to a safe exploration query."
                )
                action = {
                    "action": "execute",
                    "params": {"command": "SHOW TABLES;"},
                    "_parse_error": "guardrail_prevent_early_submit",
                }
        except Exception:
            pass
        thinking = None
        memory_content = parse_llm_output(resp, "memory")
        if isinstance(memory_content, dict) and memory_content.get("memory"):
            thinking = memory_content["memory"]
        elif isinstance(action, dict):
            params = action.get("params", {})
            if isinstance(params, dict) and params.get("memory"):
                thinking = params.pop("memory")
            elif action.get("memory"):
                thinking = action.get("memory")
        logger.agent_action(f"ReActAgent Action: {action}")
        agent_obs = history[-1].observation if history else None
        await self.memory.add_memory(obs=agent_obs, action=action, thinking=thinking, raw_response=resp)
        return action, resp, prompt
    async def run(self, request: str = None) -> str:
        return ""
