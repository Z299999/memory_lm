#!/usr/bin/env python3
"""Backfill source dataset metadata for generated distillation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_question(sample: dict) -> str | None:
    question = sample.get("question") or sample.get("problem")
    if question:
        return question

    text_fields = [k for k, v in sample.items() if isinstance(v, str)]
    if text_fields:
        return sample[text_fields[0]]
    return None


def build_question_index(config: dict) -> dict[str, list[dict]]:
    datasets_to_process = config["data"].get("datasets", []) + config["data"].get("advanced_datasets", [])
    question_index: dict[str, list[dict]] = {}

    for dataset_config in datasets_to_process:
        dataset_name = dataset_config["name"]
        subset = dataset_config.get("subset")
        split = dataset_config.get("split", "train")
        num_samples = dataset_config.get("num_samples", 100)

        ds = load_dataset(dataset_name, subset, split=split) if subset else load_dataset(dataset_name, split=split)
        if len(ds) > num_samples:
            ds = ds.shuffle(seed=42).select(range(num_samples))

        for sample_index, sample in enumerate(ds):
            question = extract_question(sample)
            if not question:
                continue
            question_index.setdefault(question, []).append({
                "dataset": dataset_name,
                "subset": subset,
                "split": split,
                "sample_index": sample_index,
            })

    return question_index


def main():
    parser = argparse.ArgumentParser(description="Backfill source metadata for distillation data.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/generated/distillation_data.jsonl",
        help="Input JSONL file relative to config directory unless absolute.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/distillation_data.with_sources.jsonl",
        help="Output JSONL file relative to config directory unless absolute.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base_dir = config_path.parent
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = (base_dir / input_path).resolve()
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    question_index = build_question_index(config)

    matched = 0
    unmatched = 0
    ambiguous = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            question = None
            if messages and messages[0].get("role") == "user":
                question = messages[0].get("content")

            matches = question_index.get(question or "", [])
            item["source_candidates"] = matches
            if len(matches) == 1:
                item["source"] = matches[0]
                matched += 1
            elif len(matches) > 1:
                ambiguous += 1
            else:
                unmatched += 1

            dst.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Matched: {matched}")
    print(f"Ambiguous: {ambiguous}")
    print(f"Unmatched: {unmatched}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
