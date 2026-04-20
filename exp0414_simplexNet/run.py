#!/usr/bin/env python3
"""Entry point for exp0414 simplex memory network experiments.

Usage:
    python3 run.py

Trains both SMN and MLP on the task defined in params.yaml and saves a
4-panel comparison plot to runs/<date>/<name>/comparison.png.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add src to path
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config, load_config_from_yaml  # type: ignore
from data import build_dataset                    # type: ignore
from smn_fitter import SMNFitter                  # type: ignore
from mlp_fitter import MLPFitter                  # type: ignore


def main() -> None:
    # Load config
    params_path = THIS_DIR / "params.yaml"
    cfg = load_config_from_yaml(params_path)

    # Experiment output directory
    exp_name  = cfg.run_name if cfg.run_name != "default" else cfg.task_name
    day       = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M%S")
    exp_dir   = THIS_DIR / "runs" / day / f"{exp_name}_{day}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset (shared between SMN and MLP for a fair comparison)
    dataset = build_dataset(cfg)
    x_train, y_train = dataset.x_train, dataset.y_train
    x_val,   y_val   = dataset.x_val,   dataset.y_val

    # Resolve per-channel bounds for the fitters
    bounds = cfg.resolved_x_bounds

    # ------------------------------------------------------------------
    # SMN
    # ------------------------------------------------------------------
    smn = SMNFitter(
        n=cfg.n, m=cfg.m,
        n_in=cfg.n_in, n_out=cfg.n_out,
        activation=cfg.node_activation,
        x_bounds=bounds,
    )
    print(f"\n{'='*55}")
    print(f"SMN: {smn.arch_str}")
    print(f"{'='*55}")
    print("Training SMN...")
    smn.fit(
        x_train, y_train, x_val, y_val,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
    )

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------
    mlp = MLPFitter(
        layers=list(cfg.mlp_layers),
        n_in=cfg.n_in, n_out=cfg.n_out,
        activation=cfg.node_activation,
        x_bounds=bounds,
    )
    print(f"\n{'='*55}")
    print(f"MLP: {mlp.arch_str}")
    print(f"{'='*55}")
    print("Training MLP...")
    mlp.fit(
        x_train, y_train, x_val, y_val,
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
    )

    # ------------------------------------------------------------------
    # 4-panel comparison plot
    # ------------------------------------------------------------------
    comparison_path = exp_dir / "comparison.png"
    figure_title = (
        f"Task={cfg.task_name} | SMN vs MLP\n"
        f"act={cfg.node_activation} | epochs={cfg.epochs} | "
        f"batch={cfg.batch_size} | num_train={cfg.num_train} | lr={cfg.lr}"
    )

    print("\nSaving 4-panel comparison plot...")
    smn.plot(
        output_path=comparison_path,
        baseline=mlp,
        title=figure_title,
    )

    print("\n" + "=" * 55)
    print("Done.")
    print(f"SMN val loss : {smn.final_val_loss:.6f}")
    print(f"MLP val loss : {mlp.final_val_loss:.6f}")
    print(f"Plot         : {comparison_path}")


if __name__ == "__main__":
    main()
