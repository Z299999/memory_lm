#!/usr/bin/env python3
"""Entry point for exp0414 simplex memory network experiments.

Usage:
    python3 run.py

This trains both SMN and MLP models and produces a comparison plot.
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

from config import Config, load_config_from_yaml
from train import train_with_config, train_with_window
from plot import save_four_panel_plot, save_2d_comparison, smn_architecture_text, mlp_architecture_text


def clone_config(base: Config, model_type: str, run_name: str) -> Config:
    """Create a copy of config with different model_type and run_name."""
    return Config(
        model_type=model_type,
        run_name=run_name,
        n=base.n,
        m=base.m,
        n_in=base.n_in,
        n_out=base.n_out,
        mlp_layers=list(base.mlp_layers),
        node_activation=base.node_activation,
        task_name=base.task_name,
        custom_function=base.custom_function,
        lr=base.lr,
        batch_size=base.batch_size,
        epochs=base.epochs,
        x_min=base.x_min,
        x_max=base.x_max,
        window_width=base.window_width,
        window_hold=base.window_hold,
    )


def main() -> None:
    # Load config
    params_path = THIS_DIR / "params.yaml"
    base = load_config_from_yaml(params_path)

    # Create experiment directory
    experiment_name = base.run_name if base.run_name != "default" else base.task_name
    day_folder = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M%S")
    experiment_dir = THIS_DIR / "runs" / day_folder / f"{experiment_name}_{day_folder}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Clone configs for SMN and MLP
    smn_config = clone_config(base, "smn", f"{experiment_name}_smn")
    mlp_config = clone_config(base, "mlp", f"{experiment_name}_mlp")

    # Get architecture descriptions
    from graph import SimplexMemoryGraph
    smn_graph = SimplexMemoryGraph(n=smn_config.n, m=smn_config.m, n_in=smn_config.n_in, n_out=smn_config.n_out)
    smn_arch = smn_architecture_text(smn_config.n, smn_config.m, smn_graph)
    mlp_arch = mlp_architecture_text(mlp_config.mlp_layers)

    print(f"\n{'='*50}")
    print(f"SMN: {smn_arch}")
    print(f"MLP: {mlp_arch}")
    if base.use_windowed_training:
        print(f"Window training: width={base.window_width*100:.0f}%, hold={base.window_hold} epochs/position")
    print(f"{'='*50}\n")

    # Train SMN
    print("Training SMN...")
    if base.use_windowed_training:
        smn_result = train_with_window(smn_config)
    else:
        smn_result = train_with_config(smn_config)

    # Train MLP
    print("\nTraining MLP...")
    mlp_result = train_with_config(mlp_config)

    # Create comparison plot
    comparison_path = experiment_dir / "comparison.png"

    if base.n_in == 2:
        print("\nBuilding 2D comparison image...")
        save_2d_comparison(
            smn_x1_grid=smn_result["x1_grid"],
            smn_x2_grid=smn_result["x2_grid"],
            smn_y_true=smn_result["y_plot"],
            smn_y_pred=smn_result["y_pred"],
            mlp_y_true=mlp_result["y_plot"],
            mlp_y_pred=mlp_result["y_pred"],
            smn_architecture=smn_arch,
            mlp_architecture=mlp_arch,
            smn_final_val_loss=smn_result["metrics"]["final_val_loss"],
            mlp_final_val_loss=mlp_result["metrics"]["final_val_loss"],
            smn_train_losses=smn_result["train_losses"],
            smn_val_losses=smn_result["val_losses"],
            mlp_train_losses=mlp_result["train_losses"],
            mlp_val_losses=mlp_result["val_losses"],
            output_path=comparison_path,
            batch_size=base.batch_size,
        )
    else:
        print("\nBuilding 4-panel comparison image...")
        save_four_panel_plot(
            smn_train_losses=smn_result["train_losses"],
            smn_val_losses=smn_result["val_losses"],
            smn_x=smn_result["x_plot"],
            smn_y_true=smn_result["y_plot"],
            smn_y_pred=smn_result["y_pred"],
            smn_architecture=smn_arch,
            smn_final_train_loss=smn_result["metrics"]["final_train_loss"],
            smn_final_val_loss=smn_result["metrics"]["final_val_loss"],
            mlp_train_losses=mlp_result["train_losses"],
            mlp_val_losses=mlp_result["val_losses"],
            mlp_x=mlp_result["x_plot"],
            mlp_y_true=mlp_result["y_plot"],
            mlp_y_pred=mlp_result["y_pred"],
            mlp_architecture=mlp_arch,
            mlp_final_train_loss=mlp_result["metrics"]["final_train_loss"],
            mlp_final_val_loss=mlp_result["metrics"]["final_val_loss"],
            figure_title=f"Task={base.task_name} | SMN vs MLP",
            output_path=comparison_path,
            batch_size=base.batch_size,
        )

    print("\n" + "="*50)
    print("Done.")
    print(f"SMN val loss: {smn_result['metrics']['final_val_loss']:.6f}")
    print(f"MLP val loss: {mlp_result['metrics']['final_val_loss']:.6f}")
    print(f"Comparison image: {comparison_path}")


if __name__ == "__main__":
    main()
