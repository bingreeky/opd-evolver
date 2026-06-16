from __future__ import annotations
from typing import Any
import torch
class SelectorSelfDistillationDataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._student_instruction = (
            "Select the most useful retrieved memory tags for this task. "
            "Return strict JSON with keys: selected_skills, selected_tips, "
            "selected_tools, selected_trajectories, reasoning."
        )
        self._teacher_transition = (
            "Using the candidate score table and task context above, infer the best selection policy. "
            "The historical selector output is behavior data, not a gold label. "
            "Then produce the final selector JSON in the online selector schema."
        )
    def _student_user_message(self, problem: str) -> str:
        return (
            f"{problem}\n\n"
            f"{self._student_instruction}\n"
            "Use exact retrieved tags such as [RETRIEVED_SKILL_01]. "
            "Keep each selected_* array concise and grounded in candidates."
        )
    def _teacher_user_message(self, problem: str, solution: str, privileged: str) -> str:
        parts = [problem]
        if privileged:
            parts.append(
                "\n\n=== Privileged Hints (quality-weighted) ===\n"
                f"{privileged}\n"
                "=== End Privileged Hints ==="
            )
        parts.append(
            "\n\n=== Historical Selector Output ===\n"
            f"{solution}\n"
            "=== End Historical Output ==="
        )
        parts.append(f"\n\n{self._teacher_transition}")
        return "".join(parts)
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_prompts: list[str] = []
        teacher_prompts: list[str] = []
        for feat in features:
            problem = feat["problem"]
            solution = feat["solution"]
            privileged = feat.get("privileged", "")
            student_chat = [{"role": "user", "content": self._student_user_message(problem)}]
            teacher_chat = [
                {
                    "role": "user",
                    "content": self._teacher_user_message(problem, solution, privileged),
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
