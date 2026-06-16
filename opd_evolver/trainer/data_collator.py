import torch
class SelfDistillationDataCollator:
    def __init__(self, tokenizer, max_length=2048, reason_first=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
    def __call__(self, features):
        batch_size = len(features)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []
        for feature in features:
            problem = feature["problem"]
            solution = feature["solution"]
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{ }} ."
            student_messages = [{"role": "user", "content": student_user_message}]
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            student_prompts.append(student_prompt)
            if self.reason_first:
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)
                teacher_prompts.append("")
            else:
                teacher_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                    f"{self.transition_prompt}\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{ }} ."
                )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                teacher_prompts.append(teacher_prompt)
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )
        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }
        if self.reason_first:
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)
            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{ }} ."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)
            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )
            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )
        return result
