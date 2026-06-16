from __future__ import annotations
import torch
from typing import Any
TEACHER_MODES = ("gold_sql", "gold", "memory_only", "both")
SUPPORTED_ENV_TYPES = ("sql", "bash", "ctf")
DEFAULT_ACTION_SPACES = {
    "bash": (
        'Actions:\n  - {"action": "execute", "params": {"command": "bash command"}}\n'
        '  - {"action": "submit", "params": {}}'
    ),
    "ctf": (
        'Actions:\n  - {"action": "execute", "params": {"command": "command or flag"}}\n'
        '  - {"action": "submit", "params": {}}'
    ),
}
class TaskSelfDistillationDataCollator:
    def __init__(
        self,
        tokenizer: Any,
        memory_context: str = "",
        max_length: int = 4096,
        teacher_mode: str = "gold_sql",
        env_type: str = "sql",
    ):
        if teacher_mode not in TEACHER_MODES:
            raise ValueError(
                f"teacher_mode must be one of {TEACHER_MODES}, got {teacher_mode!r}"
            )
        env_type = self._normalize_env_type(env_type)
        if env_type not in SUPPORTED_ENV_TYPES:
            raise ValueError(
                f"env_type must be one of {SUPPORTED_ENV_TYPES}, got {env_type!r}"
            )
        self.tokenizer = tokenizer
        self.memory_context = memory_context
        self.max_length = max_length
        self.teacher_mode = teacher_mode
        self.env_type = env_type
        self._transition_gold = (
            "Using the reference SQL above as guidance, carefully reason about "
            "the query step by step. Think about which tables to join, what "
            "conditions to apply, and how to structure the SQL. "
            "Then write the correct SQL query."
        )
        self._transition_memory = (
            "Using the expert knowledge above as guidance, carefully reason about "
            "the query step by step. Think about which tables to join, what "
            "conditions to apply, and how to structure the SQL. "
            "Then write the correct SQL query."
        )
        self._transition_both = (
            "Using the reference SQL and expert knowledge above as guidance, "
            "carefully reason about the query step by step. Think about which "
            "tables to join, what conditions to apply, and how to structure the "
            "SQL. Then write the correct SQL query."
        )
        self._task_transitions = {
            "bash": {
                "gold": (
                    "Using the reference bash solution above as guidance, reason "
                    "about the shell tools, file paths, pipes, and verification "
                    "steps needed. Then produce the correct next JSON action."
                ),
                "memory": (
                    "Using the expert knowledge above as guidance, reason about "
                    "the shell tools, file paths, pipes, and verification steps "
                    "needed. Then produce the correct next JSON action."
                ),
                "both": (
                    "Using the reference bash solution and expert knowledge above "
                    "as guidance, reason about the shell tools, file paths, pipes, "
                    "and verification steps needed. Then produce the correct next "
                    "JSON action."
                ),
            },
            "ctf": {
                "gold": (
                    "Using the reference CTF solution above as guidance, reason "
                    "about the challenge evidence, commands, decoding steps, and "
                    "flag submission strategy. Then produce the correct next JSON "
                    "action."
                ),
                "memory": (
                    "Using the expert knowledge above as guidance, reason about "
                    "the challenge evidence, commands, decoding steps, and flag "
                    "submission strategy. Then produce the correct next JSON action."
                ),
                "both": (
                    "Using the reference CTF solution and expert knowledge above "
                    "as guidance, reason about the challenge evidence, commands, "
                    "decoding steps, and flag submission strategy. Then produce "
                    "the correct next JSON action."
                ),
            },
        }
    @staticmethod
    def _normalize_env_type(env_type: Any) -> str:
        value = str(env_type or "sql").strip().lower()
        aliases = {
            "intercode_sql": "sql",
            "mysql": "sql",
            "nl2bash": "bash",
            "intercode_bash": "bash",
            "intercode_ctf": "ctf",
        }
        return aliases.get(value, value)
    @staticmethod
    def _build_student_message(problem: str) -> str:
        return (
            f"{problem}\n\n"
            "Write a SQL query to answer the question above. "
            "Think step by step about which tables to use and how to join them."
        )
    @staticmethod
    def _default_action_space(env_type: str) -> str:
        return DEFAULT_ACTION_SPACES.get(env_type, "")
    def _build_task_student_message(
        self,
        env_type: str,
        problem: str,
        action_space: str = "",
    ) -> str:
        if env_type == "sql":
            return self._build_student_message(problem)
        action_space = action_space or self._default_action_space(env_type)
        task_name = "bash" if env_type == "bash" else "CTF"
        return (
            f"Task type: {task_name}\n\n"
            f"Task:\n{problem}\n\n"
            f"Action space:\n{action_space}\n\n"
            "Solve the task step by step and output the next valid JSON action."
        )
    def _build_teacher_message(self, problem: str, solution: str, memory_context: str = "") -> str:
        ctx = memory_context if memory_context else self.memory_context
        parts = [problem]
        if self.teacher_mode in ("gold_sql", "gold"):
            parts.append(f"\n\n=== Reference SQL ===\n{solution}\n=== End Reference ===")
            if ctx:
                parts.append(
                    f"\n\n=== Expert Knowledge (top memories) ===\n"
                    f"{ctx}\n=== End Expert Knowledge ==="
                )
            parts.append(f"\n\n{self._transition_gold}")
        elif self.teacher_mode == "memory_only":
            if ctx:
                parts.append(
                    f"\n\n=== Expert Knowledge (top memories) ===\n"
                    f"{ctx}\n=== End Expert Knowledge ==="
                )
                parts.append(f"\n\n{self._transition_memory}")
            else:
                parts.append(f"\n\n=== Reference SQL ===\n{solution}\n=== End Reference ===")
                parts.append(f"\n\n{self._transition_gold}")
        else:
            parts.append(f"\n\n=== Reference SQL ===\n{solution}\n=== End Reference ===")
            if ctx:
                parts.append(
                    f"\n\n=== Expert Knowledge (top memories) ===\n"
                    f"{ctx}\n=== End Expert Knowledge ==="
                )
            parts.append(f"\n\n{self._transition_both}")
        return "".join(parts)
    def _build_task_teacher_message(
        self,
        env_type: str,
        problem: str,
        solution: str,
        memory_context: str = "",
        action_space: str = "",
    ) -> str:
        if env_type == "sql":
            return self._build_teacher_message(problem, solution, memory_context)
        ctx = memory_context if memory_context else self.memory_context
        action_space = action_space or self._default_action_space(env_type)
        reference_label = "Reference Bash Solution" if env_type == "bash" else "Reference CTF Solution"
        transition = self._task_transitions[env_type]
        parts = [
            f"Task:\n{problem}\n\n",
            f"Action space:\n{action_space}\n",
        ]
        if self.teacher_mode in ("gold_sql", "gold"):
            parts.append(f"\n=== {reference_label} ===\n{solution}\n=== End Reference ===")
            if ctx:
                parts.append(
                    f"\n\n=== Expert Knowledge (top memories) ===\n"
                    f"{ctx}\n=== End Expert Knowledge ==="
                )
            parts.append(f"\n\n{transition['gold']}")
        elif self.teacher_mode == "memory_only":
            if ctx:
                parts.append(
                    f"\n=== Expert Knowledge (top memories) ===\n"
                    f"{ctx}\n=== End Expert Knowledge ==="
                )
                parts.append(f"\n\n{transition['memory']}")
            else:
                parts.append(f"\n=== {reference_label} ===\n{solution}\n=== End Reference ===")
                parts.append(f"\n\n{transition['gold']}")
        else:
            parts.append(f"\n=== {reference_label} ===\n{solution}\n=== End Reference ===")
            if ctx:
                parts.append(
                    f"\n\n=== Expert Knowledge (top memories) ===\n"
                    f"{ctx}\n=== End Expert Knowledge ==="
                )
            parts.append(f"\n\n{transition['both']}")
        return "".join(parts)
    def _feature_env_type(self, feat: dict[str, Any]) -> str:
        for key in ("env_type", "benchmark_type", "task_type"):
            if key not in feat:
                continue
            candidate = self._normalize_env_type(feat.get(key))
            if candidate in SUPPORTED_ENV_TYPES:
                return candidate
        return self.env_type
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        student_prompts: list[str] = []
        teacher_prompts: list[str] = []
        for feat in features:
            env_type = self._feature_env_type(feat)
            problem = str(feat["problem"])
            solution = str(feat["solution"])
            action_space = str(feat.get("action_space") or "")
            per_example_ctx: str = feat.get("memory_context") or ""
            student_msg = self._build_task_student_message(
                env_type=env_type,
                problem=problem,
                action_space=action_space,
            )
            student_chat = [{"role": "user", "content": student_msg}]
            student_prompts.append(
                self.tokenizer.apply_chat_template(
                    student_chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            teacher_msg = self._build_task_teacher_message(
                env_type=env_type,
                problem=problem,
                solution=solution,
                memory_context=per_example_ctx,
                action_space=action_space,
            )
            teacher_chat = [{"role": "user", "content": teacher_msg}]
            teacher_prompts.append(
                self.tokenizer.apply_chat_template(
                    teacher_chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        student_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_lengths = [len(ids) for ids in student_no_pad["input_ids"]]
        max_student_len = max(student_lengths)
        student_enc = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_len,
            return_tensors="pt",
        )
        teacher_no_pad = self.tokenizer(
            teacher_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        teacher_lengths = [len(ids) for ids in teacher_no_pad["input_ids"]]
        max_teacher_len = max(teacher_lengths)
        teacher_enc = self.tokenizer(
            teacher_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_teacher_len,
            return_tensors="pt",
        )
        return {
            "student_prompts": student_enc["input_ids"],
            "student_prompt_attention_mask": student_enc["attention_mask"],
            "student_prompt_length": max_student_len,
            "student_prompt_lengths_per_example": torch.tensor(student_lengths),
            "teacher_prompts": teacher_enc["input_ids"],
            "teacher_prompt_attention_mask": teacher_enc["attention_mask"],
            "teacher_prompt_length": max_teacher_len,
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths),
        }
class SQLSelfDistillationDataCollator(TaskSelfDistillationDataCollator):
    def __init__(
        self,
        tokenizer: Any,
        memory_context: str = "",
        max_length: int = 4096,
        teacher_mode: str = "gold_sql",
    ):
        super().__init__(
            tokenizer=tokenizer,
            memory_context=memory_context,
            max_length=max_length,
            teacher_mode=teacher_mode,
            env_type="sql",
        )
    def _feature_env_type(self, feat: dict[str, Any]) -> str:
        return "sql"
