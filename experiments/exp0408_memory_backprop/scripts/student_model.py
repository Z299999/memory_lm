from __future__ import annotations

"""Local HuggingFace student model wrapper for inference.

Keeps the model loaded in memory across calls so we don't reload every turn.
Supports both forward inference and (via backward.py) gradient-based training.
"""

from pathlib import Path


class HFStudentModel:
    def __init__(self, model_name_or_path: str, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[student] loading {model_name_or_path} ...")
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        print(f"[student] loaded on {next(self.model.parameters()).device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


# Stub for smoke testing (no torch required)
class StubStudentModel:
    def generate(self, prompt: str, **kwargs) -> str:
        return "The answer is 42. Step 1: ... Step 2: ... Final answer: **42**."


def load_student(model_name_or_path: str, stub: bool = False) -> HFStudentModel | StubStudentModel:
    if stub:
        return StubStudentModel()
    return HFStudentModel(model_name_or_path)
