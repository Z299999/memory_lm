"""Standalone evaluation script for exp0522."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation on a completed exp0522 run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the run directory (must contain config.yaml and checkpoints/).",
    )
    return parser.parse_args()


def main() -> None:
    root = _add_src_to_path()
    from config import load_config_from_yaml
    from eval import evaluate_model

    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in run_dir: {run_dir}")

    config = load_config_from_yaml(config_path)
    summary = evaluate_model(config=config, run_dir=run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
