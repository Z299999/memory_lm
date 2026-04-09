"""Generate distillation data by calling teacher model on math problems."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import yaml
from datasets import load_dataset
from llm_client import OpenAICompatClient, LLMClientError


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_teacher_response(
    client: OpenAICompatClient,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate response from teacher model."""
    prompt = f"""Please solve the following math problem step by step. Show your reasoning clearly.

Question: {question}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves math problems with clear step-by-step reasoning."},
        {"role": "user", "content": prompt},
    ]

    result = client.chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return result.content


def format_data_sample(question: str, teacher_response: str) -> dict:
    """Format a single training sample."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": teacher_response},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate distillation data from teacher model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: generate only 10 samples.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Initialize client
    client = OpenAICompatClient(
        base_url=config["teacher"].get("base_url"),
    )

    # Prepare output directory
    output_dir = Path(config["data"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    output_file = output_dir / "distillation_data.jsonl"

    # Load existing data for resume capability
    generated_samples = []
    existing_questions = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    generated_samples.append(item)
                    # Extract question to track what we already have
                    messages = item.get("messages", [])
                    if messages and messages[0].get("role") == "user":
                        existing_questions.add(messages[0].get("content", ""))
        print(f"Resuming from existing data file: {output_file}")
        print(f"  Found {len(generated_samples)} existing samples.")

    total_samples = len(generated_samples)

    # Process each dataset
    for dataset_config in config["data"]["datasets"]:
        dataset_name = dataset_config["name"]
        num_samples = dataset_config.get("num_samples", 100)

        if args.debug:
            num_samples = 10

        print(f"\nLoading dataset: {dataset_name} ({num_samples} samples)")

        # Load dataset
        subset = dataset_config.get("subset")
        split = dataset_config.get("split", "train")

        try:
            if subset:
                ds = load_dataset(dataset_name, subset, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        # Sample data
        if len(ds) > num_samples:
            ds = ds.shuffle(seed=42).select(range(num_samples))
        else:
            print(f"Dataset has only {len(ds)} samples, using all.")

        # Generate responses
        skipped = 0
        for i, sample in enumerate(tqdm(ds, desc=f"Generating {dataset_name}")):
            # Extract question based on dataset format
            if "question" in sample:
                question = sample["question"]
            elif "problem" in sample:
                question = sample["problem"]
            else:
                # Try to find a text field
                text_fields = [k for k in sample.keys() if isinstance(sample[k], str)]
                if text_fields:
                    question = sample[text_fields[0]]
                else:
                    print(f"Skipping sample: no text field found")
                    continue

            # Skip if already generated
            if question in existing_questions:
                skipped += 1
                continue

            existing_questions.add(question)

            # Generate teacher response with retry
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = generate_teacher_response(
                        client,
                        model=config["teacher"]["model"],
                        question=question,
                        temperature=config["teacher"].get("temperature", 0.2),
                        max_tokens=config["teacher"].get("max_tokens", 2048),
                    )
                    break  # Success, exit retry loop
                except (LLMClientError, TimeoutError, Exception) as e:
                    if attempt < max_retries - 1:
                        print(f"API error on sample {i} (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"  Retrying in 10 seconds...")
                        import time
                        time.sleep(10)
                    else:
                        print(f"API error on sample {i} after {max_retries} attempts: {e}")
                        print(f"  Skipping this sample.")

            if not response:
                continue
                continue

            # Format and save
            data_sample = format_data_sample(question, response)
            generated_samples.append(data_sample)
            total_samples += 1

            # Save incrementally
            if (i + 1) % 50 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in generated_samples:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        for item in generated_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nGenerated {total_samples} samples.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
