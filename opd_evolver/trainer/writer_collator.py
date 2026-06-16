from __future__ import annotations
from typing import Any
import torch
class WriterSelfDistillationDataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._student_instruction = (
            "Generate memory reflection JSON with keys: "
            "new_skills, new_tips, new_tools, key_learnings, should_save_trajectory, trajectory_outcome."
        )
        self._teacher_transition = (
            "Use the privileged quality rubric and reference memory output above to infer better writing behavior. "
            "Then produce the final memory JSON in the same schema."
        )
    def _student_user_message(self, problem: str) -> str:
        return (
            f"{problem}\n\n"
            f"{self._student_instruction}\n"
            "Be concise, factual, and avoid hallucinating tools."
        )
    def _teacher_user_message(self, problem: str, solution: str, privileged: str) -> str:
        parts = [problem]
        if privileged:
            parts.append(
                "\n\n=== Privileged Quality Hints ===\n"
                f"{privileged}\n"
                "=== End Privileged Quality Hints ==="
            )
        parts.append(
            "\n\n=== Reference Memory Output ===\n"
            f"{solution}\n"
            "=== End Reference ==="
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
