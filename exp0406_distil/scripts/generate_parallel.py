#!/usr/bin/env python3
"""Parallel generation of distillation data using ThreadPoolExecutor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import OpenAICompatClient, LLMClientError


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_data_sample(question: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
    }


def generate_single_sample(args: tuple) -> tuple[int, dict | None, str | None]:
    idx, question, model, base_url, api_key, temperature, max_tokens = args
    try:
        client = OpenAICompatClient(api_key=api_key, base_url=base_url, timeout=300)
        messages = [
            {"role": "system", "content": "Please solve the following math problem step by step. Show your reasoning clearly."},
            {"role": "user", "content": question},
        ]
        result = client.chat_completion(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        sample = format_data_sample(question, result.content)
        return (idx, sample, None)
    except Exception as e:
        return (idx, None, str(e))


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data in parallel.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: generate only 10 samples per dataset.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["data"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "distillation_data.jsonl"

    existing_questions = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    messages = item.get("messages", [])
                    if messages and messages[0].get("role") == "user":
                        existing_questions.add(messages[0].get("content", ""))
        print(f"Resuming from existing data file: {output_file}")
        print(f"  Found {len(existing_questions)} existing samples.")

    num_workers = args.workers or int(os.environ.get("GENERATE_WORKERS", "8"))
    print(f"Using {num_workers} parallel workers")

    all_questions: list[tuple[int, str]] = []
    start_count = len(existing_questions)

    datasets_to_process = config["data"].get("datasets", []) + config["data"].get("advanced_datasets", [])

    for dataset_config in datasets_to_process:
        dataset_name = dataset_config["name"]
        num_samples = dataset_config.get("num_samples", 100)
        if args.debug:
            num_samples = 10
        print(f"\nLoading dataset: {dataset_name} ({num_samples} samples)")

        subset = dataset_config.get("subset")
        split = dataset_config.get("split", "train")
        try:
            ds = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        if len(ds) > num_samples:
            ds = ds.shuffle(seed=42).select(range(num_samples))
        else:
            print(f"Dataset has only {len(ds)} samples, using all.")

        for sample in ds:
            question = sample.get("question") or sample.get("problem")
            if not question:
                text_fields = [k for k in sample.keys() if isinstance(sample[k], str)]
                question = sample[text_fields[0]] if text_fields else None
            if not question:
                continue
            if question not in existing_questions:
                all_questions.append(question)
                existing_questions.add(question)

    indexed_questions = [(start_count + i, q) for i, q in enumerate(all_questions)]
    if len(indexed_questions) == 0:
        print("\nNo new questions to generate!")
        return
    print(f"\nTotal new questions to generate: {len(all_questions)}")

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    model = config["teacher"]["model"]
    base_url = config["teacher"]["base_url"]
    temperature = config["teacher"].get("temperature", 0.2)
    max_tokens = config["teacher"].get("max_tokens", 2048)

    print(f"\nStarting parallel generation with {num_workers} workers...")
    print(f"Model: {model}, Base URL: {base_url}")

    sample_args = [(idx, q, model, base_url, api_key, temperature, max_tokens) for idx, q in indexed_questions]
    total_generated, failed = 0, 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_single_sample, args) for args in sample_args]
        with tqdm(total=len(indexed_questions), desc="Generating") as pbar:
            for future in as_completed(futures):
                idx, sample, error = future.result()
                if sample:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    total_generated += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({"generated": total_generated, "failed": failed})

    with open(output_file, "r", encoding="utf-8") as f:
        final_count = sum(1 for _ in f)
    print(f"\n=== Generation Complete ===")
    print(f"Generated {total_generated} new samples ({failed} failed)")
    print(f"Total samples in file: {final_count}")


if __name__ == "__main__":
    main()
