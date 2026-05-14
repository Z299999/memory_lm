"""Run the V1 two-phase mixed-sin experiment for exp0513."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exp0513 V1 mixed-sin phase-switch experiment.")
    parser.add_argument("--phase-a-epochs", type=int, default=1000)
    parser.add_argument("--phase-b-epochs", type=int, default=1000)
    parser.add_argument("--run-name", type=str, default="mixedsin_phase_switch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-bp", type=float, default=1e-2)
    parser.add_argument("--eta-int", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-phase-b", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    _add_src_to_path()
    from train import ExperimentConfig, run_experiment

    args = parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        lr_bp=args.lr_bp,
        eta_int=args.eta_int,
        gamma=args.gamma,
        phase_a_epochs=args.phase_a_epochs,
        phase_b_epochs=args.phase_b_epochs,
        lambda_phase_b=args.lambda_phase_b,
        run_name=args.run_name,
    )
    result = run_experiment(config=config)
    print(result["run_dir"])
    print(result["phase_a_summary"])
    print(result["phase_b_summary"])


if __name__ == "__main__":
    main()
