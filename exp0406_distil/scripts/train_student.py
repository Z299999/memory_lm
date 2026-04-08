"""Train student model using distillation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_prompt(example: dict) -> str:
    """Format a sample into a chat-style prompt."""
    messages = example.get("messages", [])
    if not messages:
        return ""

    formatted = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            formatted += f"User: {content}\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n"

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Train student model with distillation data.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--data-file", type=str, default=None, help="Path to training data (override config).")
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Data file path
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        data_path = Path(config["data"]["save_dir"]) / "distillation_data.jsonl"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run generate_data.py first.")
        return

    print(f"Loading training data from: {data_path}")

    # Load dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Split into train/test
    test_ratio = config["data"].get("test_split_ratio", 0.1)
    if test_ratio > 0:
        dataset = dataset.train_test_split(test_size=test_ratio, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation samples: {len(eval_dataset)}")

    # Load model and tokenizer
    student_model = config["student"]["model"]
    print(f"Loading student model: {student_model}")

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        dtype=torch.float16 if config["student"].get("load_in_8bit", False) else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Training arguments
    train_config = config["train"]
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume from
    resume_from_checkpoint = None
    if output_dir.exists():
        # Find the latest checkpoint
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            def extract_step(path):
                try:
                    return int(path.name.split("-")[-1])
                except ValueError:
                    return 0
            latest_checkpoint = max(checkpoints, key=extract_step)
            if latest_checkpoint.exists() and (latest_checkpoint / "trainer_state.json").exists():
                resume_from_checkpoint = str(latest_checkpoint)
                print(f"Found existing checkpoint: {resume_from_checkpoint}")
                print("Will resume from this checkpoint.")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_config.get("batch_size", 16)),
        per_device_eval_batch_size=int(train_config.get("batch_size", 16)),
        num_train_epochs=int(train_config.get("num_epochs", 5)),
        learning_rate=float(train_config.get("learning_rate", 5e-5)),
        warmup_steps=int(int(train_config.get("num_epochs", 5)) * len(train_dataset) / int(train_config.get("batch_size", 16)) * 0.1),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        lr_scheduler_type=str(train_config.get("lr_scheduler_type", "cosine")),
        save_steps=int(train_config.get("save_steps", 200)),
        logging_steps=int(train_config.get("logging_steps", 50)),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=int(train_config.get("save_steps", 200)),
        save_total_limit=3,
        fp16=train_config.get("fp16", True) and torch.cuda.is_available(),
        seed=int(train_config.get("seed", 42)),
        report_to="none",  # Disable wandb/tensorboard by default
    )

    # Initialize trainer with formatting function for conversational data
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=format_prompt,
    )

    # Train
    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    final_output = output_dir / "final"
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    print(f"\nTraining completed. Model saved to: {final_output}")


if __name__ == "__main__":
    main()
