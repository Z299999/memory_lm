from __future__ import annotations

# This is the one-click entry point for the folder.
# Run this file once and it will train both TMN and MLP using scripts/config.py, then
# build one 4-panel comparison image automatically.

import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import Config, load_config_from_yaml
from train import train_with_config
from plot import save_four_panel_plot, save_weights_plot


def clone_config(base: Config, model_type: str, run_name: str) -> Config:
    return Config(
        model_type=model_type,
        run_name=run_name,
        L=base.L,
        n_in=base.n_in,
        n_out=base.n_out,
        mlp_layers=list(base.mlp_layers),
        task_name=base.task_name,
        custom_function=base.custom_function,
        node_activation=base.node_activation,
        output_activation=base.output_activation,
        lr=base.lr,
        batch_size=base.batch_size,
        epochs=base.epochs,
        x_min=base.x_min,
        x_max=base.x_max,
    )


def tmn_architecture_text(config: Config) -> str:
    core_nodes = config.L * (config.L + 1) // 2
    edge_count = 3 * config.L * (config.L - 1) // 2 + config.n_in * config.L + config.n_out * config.L
    param_count = edge_count + core_nodes + 1
    return f"L={config.L}, core_nodes={core_nodes}, params={param_count}"


def mlp_architecture_text(config: Config) -> str:
    return f"layers={config.mlp_layers}"


def main() -> None:
    params_path = THIS_DIR / "params.yaml"
    base = load_config_from_yaml(params_path)
    experiment_name = base.run_name if base.run_name != "default" else base.task_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = THIS_DIR / "runs" / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tmn_config = clone_config(base, "tmn", f"{experiment_name}_tmn")
    mlp_config = clone_config(base, "mlp", f"{experiment_name}_mlp")

    print("Training TMN...")
    tmn_result = train_with_config(tmn_config)

    print("Training MLP...")
    mlp_result = train_with_config(mlp_config)

    comparison_path = experiment_dir / "comparison_4panel.png"
    print("Building 4-panel comparison image...")
    save_four_panel_plot(
        tmn_train_losses=tmn_result["train_losses"],
        tmn_val_losses=tmn_result["val_losses"],
        tmn_x=tmn_result["x_plot"],
        tmn_y_true=tmn_result["y_plot"],
        tmn_y_pred=tmn_result["y_pred"],
        tmn_architecture=tmn_architecture_text(tmn_config),
        tmn_final_train_loss=tmn_result["metrics"]["final_train_loss"],
        tmn_final_val_loss=tmn_result["metrics"]["final_val_loss"],
        mlp_train_losses=mlp_result["train_losses"],
        mlp_val_losses=mlp_result["val_losses"],
        mlp_x=mlp_result["x_plot"],
        mlp_y_true=mlp_result["y_plot"],
        mlp_y_pred=mlp_result["y_pred"],
        mlp_architecture=mlp_architecture_text(mlp_config),
        mlp_final_train_loss=mlp_result["metrics"]["final_train_loss"],
        mlp_final_val_loss=mlp_result["metrics"]["final_val_loss"],
        figure_title=f"Task={base.task_name if not base.custom_function else 'custom'} | TMN vs MLP",
        loss_fn="MSELoss",
        output_path=comparison_path,
    )

    weights_path = experiment_dir / "weights.png"
    print("Building weights visualization...")
    save_weights_plot(tmn_result["model"], mlp_result["model"], weights_path)

    print("Done.")
    print(f"TMN val loss: {tmn_result['metrics']['final_val_loss']:.6f}")
    print(f"MLP val loss: {mlp_result['metrics']['final_val_loss']:.6f}")
    print(f"Comparison image: {comparison_path}")
    print(f"Weights image:    {weights_path}")


if __name__ == "__main__":
    main()
