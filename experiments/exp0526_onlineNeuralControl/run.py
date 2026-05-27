#!/usr/bin/env python3
"""Run exp0526 online neural control training."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exp0526 online neural controller.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute() and not config_path.exists():
        config_path = Path(__file__).resolve().parent / config_path
    config = load_config(config_path)
    if args.epochs is not None:
        config.train.epochs = int(args.epochs)
    if args.run_name is not None:
        config.run.run_name = str(args.run_name)

    run_dir = train_model(config, config_path)
    print(f"Wrote run to {run_dir}")


if __name__ == "__main__":
    main()
