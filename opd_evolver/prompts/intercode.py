from typing import Any, Dict, List
from opd_evolver.main_agent import build_model_pricing_table
INTERCODE_ENV_DESCRIPTIONS = {
    "bash": """
SubAgent executes bash commands in an InterCode Docker container.

Available actions for SubAgent:
- execute: Run any bash command (ls, grep, awk, find, cat, etc.)
- finish: Report completion status to MainAgent

The goal is to solve the task by running bash commands and getting the correct output.
When ready, MainAgent will submit to verify the solution.
""".strip(),
    "sql": """
SubAgent executes SQL queries in an InterCode database environment.

Available actions for SubAgent:
- execute: Run any SQL query (SELECT, INSERT, UPDATE, SHOW TABLES, etc.)
- finish: Report completion status to MainAgent

The goal is to write SQL queries that produce the expected output.
When ready, MainAgent will submit to verify the solution.
""".strip(),
    "python": """
SubAgent writes and executes Python code in an InterCode environment.

Available actions for SubAgent:
- execute: Run Python code (functions, expressions, etc.)
- finish: Report completion status to MainAgent

The goal is to write Python code that solves the given programming task.
When ready, MainAgent will submit to verify the solution.
""".strip(),
    "ctf": """
SubAgent solves CTF (Capture The Flag) challenges in an InterCode environment.

Available actions for SubAgent:
- execute: Run commands or submit flag guesses
- finish: Report completion status to MainAgent

The goal is to find the flag through exploration and problem-solving.
When ready, MainAgent will submit the flag to verify.
""".strip(),
}
class InterCodePrompt:
    def __init__(self, env_type: str = "bash"):
        self.env_type = env_type
    def build_prompt(
        self,
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
        tools: List[Any] = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index + 1
        model_pricing_table = build_model_pricing_table(sub_models, model_to_alias)
        env_description = INTERCODE_ENV_DESCRIPTIONS.get(
            self.env_type,
            INTERCODE_ENV_DESCRIPTIONS["bash"]
        )
        if remaining_attempts <= 2:
            budget_warning = f"🚨 CRITICAL: Only {remaining_attempts} attempt(s) left! Submit now if solution looks correct."
        elif remaining_attempts <= 4:
            budget_warning = f"⚠️ Warning: {remaining_attempts} attempts remaining. Plan carefully."
        else:
            budget_warning = ""
        return f"""You are the MainAgent (Orchestrator). Your task is to solve an InterCode {self.env_type.upper()} task by delegating to SubAgents.
CRITICAL: Each SubAgent runs in a FRESH environment - previous SubAgent work is lost when delegating again.
When SubAgent reports status="done", use 'submit' immediately to evaluate the solution.
==== DECISION PROCESS ====
1. READ the TASK carefully - understand what output/result is expected
2. REVIEW SUBTASK HISTORY - check what SubAgent accomplished
3. VERIFY SubAgent's work:
   - Did SubAgent produce the expected output?
   - Did SubAgent test the solution?
   - Is the solution ready to submit?
4. DECIDE:
   - ✅ status="done" AND solution looks correct → Use 'submit'
   - ✅ status="done" BUT solution seems incomplete → Use 'delegate_task' to fix
   - ⚠️ status="partial" → Use 'delegate_task' to continue
{budget_warning}
==== MODEL SELECTION ====
{model_pricing_table}
==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts
==== TASK ====
{instruction}
==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}
==== ENVIRONMENT ====
{env_description}
==== OUTPUT ====
Return JSON:
If SubAgent status="done" AND solution looks correct:
{
  "action": "submit",
  "reasoning": "SubAgent completed the task: [summary]. Submitting for evaluation.",
  "params": {  "reason": "Task completed" }
}
If SubAgent status="done" BUT solution needs improvement:
{
  "action": "delegate_task",
  "reasoning": "SubAgent claimed done but [issue]. Need to [fix].",
  "params": {
    "task_instruction": "Fix the solution: [specific instructions]",
    "context": "Previous attempt: [what worked/failed]",
    "model": "one of {sub_models}"
  }
}
If SubAgent status="partial" or first attempt:
{
  "action": "delegate_task",
  "reasoning": "Need to [describe approach]",
  "params": {
    "task_instruction": "[Clear step-by-step instructions for SubAgent]\n\nCRITICAL: Your FINAL command must output ONLY the answer (number/result), not descriptive text.\nFor counting: end with 'echo [number]' or 'find ... | wc -l'\nFor text output: end with 'echo [exact_answer]'",
    "context": "[Any context from previous attempts]",
    "model": "one of {sub_models}"
  }
}
""".strip()
class BashPrompt(InterCodePrompt):
    def __init__(self):
        super().__init__("bash")
class SqlPrompt(InterCodePrompt):
    def __init__(self):
        super().__init__("sql")
class PythonPrompt(InterCodePrompt):
    def __init__(self):
        super().__init__("python")
class CtfPrompt(InterCodePrompt):
    def __init__(self):
        super().__init__("ctf")
def get_prompt_builder(env_type: str) -> InterCodePrompt:
    return InterCodePrompt(env_type)
