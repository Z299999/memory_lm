"""Standalone training script for exp0522."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an exp0522 model from config.yaml.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the yaml config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of training epochs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional override for run_name.",
    )
    return parser.parse_args()


def main() -> None:
    root = _add_src_to_path()
    from config import load_config_from_yaml
    from train import train_model

    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        if config_path.exists():
            config_path = config_path.resolve()
        else:
            config_path = (root / config_path).resolve()

    config = load_config_from_yaml(config_path)
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.run_name:
        config.run_name = str(args.run_name)

    run_dir = train_model(config=config, config_path=config_path)
    print(run_dir)


if __name__ == "__main__":
    main()
