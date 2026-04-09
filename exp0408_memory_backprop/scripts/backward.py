from __future__ import annotations

"""Offline self-prediction training.

Training objective: given m_t^(0), predict m_{t+1}^(0).
Samples are (m_prev, m_next) pairs written by forward.py to training/samples.jsonl.

Currently a stub — replace the body of run_backward() with real HuggingFace
fine-tuning once the student model is a local Qwen2.5-0.5B checkpoint.
"""

import json
from pathlib import Path


def load_samples(samples_path: Path) -> list[dict]:
    samples = []
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_backward(*, samples_path: Path, run_dir: Path, student=None) -> None:
    samples = load_samples(samples_path)
    print(f"[backward] loaded {len(samples)} samples from {samples_path}")

    # TODO: replace with real training once local model is ready
    # Sketch:
    #
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # from torch.optim import AdamW
    #
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    #
    # for sample in samples:
    #     prompt = sample["m_prev"]
    #     target = sample["m_next"]
    #     inputs = tokenizer(prompt, return_tensors="pt")
    #     labels = tokenizer(target, return_tensors="pt").input_ids
    #     loss = model(**inputs, labels=labels).loss
    #     loss.backward()
    #
    # optimizer.step()

    print("[backward] (stub) training skipped — implement when local model is ready.")
