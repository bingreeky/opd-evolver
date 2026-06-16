from __future__ import annotations
from typing import Any
import torch
TEACHER_MODES = ("gold", "memory_only", "both")
class LifelongSelfDistillationDataCollator:
    def __init__(
        self,
        tokenizer: Any,
        memory_context: str = "",
        max_length: int = 4096,
        teacher_mode: str = "gold",
    ):
        if teacher_mode not in TEACHER_MODES:
            raise ValueError(f"teacher_mode must be one of {TEACHER_MODES}, got {teacher_mode!r}")
        self.tokenizer = tokenizer
        self.memory_context = memory_context
        self.max_length = max_length
        self.teacher_mode = teacher_mode
    @staticmethod
    def _student_message(task_type: str, problem: str, action_space: str) -> str:
        return (
            f"Task type: {task_type}\n\n"
            f"Task:\n{problem}\n\n"
            f"Action space:\n{action_space}\n\n"
            "Solve the task step by step and output only the JSON action required by the action space."
        )
    def _teacher_message(
        self,
        task_type: str,
        problem: str,
        action_space: str,
        solution: str,
        memory_context: str,
        skill_tags: str,
    ) -> str:
        ctx = memory_context or self.memory_context
        parts = [
            f"Task type: {task_type}\n\n",
            f"Task:\n{problem}\n\n",
            f"Action space:\n{action_space}\n",
        ]
        if skill_tags:
            parts.append(f"\nSkill/action tags: {skill_tags}\n")
        if self.teacher_mode in ("gold", "both"):
            parts.append(f"\n=== Reference Solution ===\n{solution}\n=== End Reference Solution ===\n")
        if self.teacher_mode in ("memory_only", "both") and ctx:
            parts.append(f"\n=== Expert Memory ===\n{ctx}\n=== End Expert Memory ===\n")
        if self.teacher_mode == "memory_only" and not ctx:
            parts.append(f"\n=== Reference Solution ===\n{solution}\n=== End Reference Solution ===\n")
        parts.append(
            "\nUsing the privileged information above, infer the best task policy. "
            "Then produce the next correct JSON action in the environment format."
        )
        return "".join(parts)
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        student_prompts: list[str] = []
        teacher_prompts: list[str] = []
        for feat in features:
            task_type = str(feat.get("task_type", "unknown"))
            problem = str(feat["problem"])
            action_space = str(feat.get("action_space", ""))
            solution = str(feat.get("solution", ""))
            memory_context = str(feat.get("memory_context") or "")
            skill_tags = feat.get("skill_tags", [])
            if isinstance(skill_tags, (list, tuple)):
                skill_tags_text = ", ".join(str(x) for x in skill_tags)
            else:
                skill_tags_text = str(skill_tags or "")
            student_chat = [
                {
                    "role": "user",
                    "content": self._student_message(task_type, problem, action_space),
                }
            ]
            teacher_chat = [
                {
                    "role": "user",
                    "content": self._teacher_message(
                        task_type=task_type,
                        problem=problem,
                        action_space=action_space,
                        solution=solution,
                        memory_context=memory_context,
                        skill_tags=skill_tags_text,
                    ),
                }
            ]
            student_prompts.append(
                self.tokenizer.apply_chat_template(
                    student_chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
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
