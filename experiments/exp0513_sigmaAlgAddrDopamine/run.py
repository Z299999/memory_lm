"""Single-entry runner for exp0513."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp0513 from config.yaml.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the yaml config. Defaults to exp0513/config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    root = _add_src_to_path()
    from config import load_config_from_yaml
    from train import run_experiment

    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        if config_path.exists():
            config_path = config_path.resolve()
        else:
            config_path = (root / config_path).resolve()

    config = load_config_from_yaml(config_path)
    result = run_experiment(config=config, config_path=config_path)
    print(result["run_dir"])
    print(result["summary"])


if __name__ == "__main__":
    main()
