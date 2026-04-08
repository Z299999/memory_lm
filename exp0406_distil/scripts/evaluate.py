"""Evaluate student model on math benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_answer(text: str) -> str | None:
    """Extract the final answer from model output."""
    # Look for boxed answer or final number
    patterns = [
        r"\\boxed\{([^}]*)\}",  # LaTeX boxed
        r"答案 [：:]\s*(\d+(?:\.\d+)?)",  # Chinese answer pattern
        r"answer [iI][sS]:?\s*(\d+(?:\.\d+)?)",  # English answer pattern
        r"=\s*(\d+(?:\.\d+)?)$",  # Final equals
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Try to find any number at the end
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]

    return None


def extract_ground_truth(sample: dict) -> str:
    """Extract ground truth answer from dataset sample."""
    if "answer" in sample:
        # GSM8K format: answer might have number at the end
        answer = str(sample["answer"])
        numbers = re.findall(r"\d+(?:\.\d+)?", answer)
        if numbers:
            return numbers[-1]
    if "ground_truth" in sample:
        return str(sample["ground_truth"])
    return ""


def evaluate_model(
    model,
    tokenizer,
    dataset,
    max_length: int,
    device: str,
    num_samples: int | None = None,
) -> dict:
    """Evaluate model on a dataset."""
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    correct = 0
    total = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        # Get question
        if "question" in sample:
            question = sample["question"]
        elif "problem" in sample:
            question = sample["problem"]
        else:
            continue

        # Get ground truth
        ground_truth = extract_ground_truth(sample)
        if not ground_truth:
            continue

        # Generate response
        prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt) :]  # Remove prompt

        # Extract and compare answers
        predicted = extract_answer(response)
        if predicted and ground_truth:
            if predicted == ground_truth:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model on math benchmarks.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model (override config).")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(config["train"]["output_dir"]) / "final"

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Run train_student.py first.")
        return

    print(f"Loading model from: {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Determine device - prefer CPU for stable evaluation on Mac
    if torch.backends.mps.is_available():
        # MPS can be unstable for some operations, use CPU for evaluation
        print("Note: Using CPU for stable evaluation (MPS detected but may be unstable)")
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )

    if device == "cpu":
        model = model.to("cpu")

    print(f"Model loaded on device: {next(model.parameters()).device}")

    # Evaluate on test datasets
    results = {}

    for dataset_config in config["data"]["datasets"]:
        dataset_name = dataset_config["name"]
        print(f"\nEvaluating on: {dataset_name}")

        try:
            subset = dataset_config.get("subset")
            split = "test"  # Use test split for evaluation

            if subset:
                eval_dataset = load_dataset(dataset_name, subset, split=split)
            else:
                eval_dataset = load_dataset(dataset_name, split=split)

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        # Evaluate
        result = evaluate_model(
            model,
            tokenizer,
            eval_dataset,
            max_length=config["student"].get("max_length", 1024),
            device=str(device),
            num_samples=args.num_samples,
        )

        results[dataset_name] = result
        print(f"  Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    # Save results
    output_file = Path(config["data"]["save_dir"]).parent / "eval" / "results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
