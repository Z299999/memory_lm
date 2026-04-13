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
from plot import save_four_panel_plot, save_weights_plot, save_2d_comparison_plot, save_2d_comparison_with_mlp
from model.graph import TMNGraph
from model.tmn import TMNNetwork


def build_trace_fn(config: Config):
    """Build a trace function that records top and bottom row node params each epoch."""
    model = TMNNetwork(config)
    graph = model.graph
    L = config.L

    # Get depth at layer z
    def depth_at(z: int) -> int:
        if config.depth == "tetrahedron":
            return z
        elif callable(config.depth):
            return config.depth(z)
        else:
            return config.depth if isinstance(config.depth, int) else 1

    core_to_nb_idx = {node: i for i, node in enumerate(graph.core_nodes)}
    edge_to_ew_idx = {edge: i for i, edge in enumerate(graph.edges)}

    # Bottom row (z=1): first neuron (x=1) at each position y=1..L
    bottom_nodes = [("core", 1, y, 1) for y in range(1, L + 1)]

    # Top row (z=L): all neurons at the single position y=1
    # depth(L) = L for tetrahedron mode
    top_depth = depth_at(L)
    top_nodes = [("core", x, 1, L) for x in range(1, top_depth + 1)]

    traced_nodes = bottom_nodes + top_nodes

    # For each node, get its bias idx and incoming edge idxs
    traced = []
    for node in traced_nodes:
        if node not in graph.core_nodes:
            continue  # Skip if node doesn't exist
        nb_idx = core_to_nb_idx[node]
        in_edges = graph.preds[node]
        ew_idxs = [edge_to_ew_idx[(p, node)] for p in in_edges]
        traced.append((node, nb_idx, ew_idxs))

    def trace_fn(model):
        result = {}
        for node, nb_idx, ew_idxs in traced:
            x, y, z = node[1], node[2], node[3]
            row = "bottom" if z == 1 else ("top" if z == L else f"z{z}")
            result[f"{row}_b({x},{y},{z})"] = float(model.nb[nb_idx].item())
            for i, ew_idx in enumerate(ew_idxs):
                src = graph.preds[node][i]
                if src[0] == "in":
                    src_label = f"in,{src[1]}"
                else:
                    src_label = f"{src[1]},{src[2]},{src[3]}"
                result[f"{row}_w({src_label})->({x},{y},{z})"] = float(model.ew[ew_idx].item())
        return result

    return trace_fn


def clone_config(base: Config, model_type: str, run_name: str) -> Config:
    return Config(
        model_type=model_type,
        run_name=run_name,
        L=base.L,
        n_in=base.n_in,
        n_out=base.n_out,
        depth=base.depth,
        cross_layer_mode=base.cross_layer_mode,
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
    graph = TMNGraph(
        L=config.L,
        n_in=config.n_in,
        n_out=config.n_out,
        depth=config.depth,
        cross_layer_mode=config.cross_layer_mode,
    )
    param_count = len(graph.edges) + graph.core_node_count + config.n_out
    depth_str = "depth(z)=z" if config.depth == "tetrahedron" else f"depth={config.depth}"
    return (
        f"L={config.L}, {depth_str}, mode={config.cross_layer_mode}, "
        f"edges={len(graph.edges)}, params={param_count}"
    )


def mlp_architecture_text(config: Config) -> str:
    # Calculate total parameters
    # Input is 1D (scalar), output is 1D (scalar)
    layers = config.mlp_layers
    if len(layers) == 0:
        param_count = 1  # Just input->output
    else:
        param_count = 0
        prev_size = 1  # Input is scalar
        for h_size in layers:
            param_count += prev_size * h_size + h_size  # weights + bias
            prev_size = h_size
        param_count += prev_size * 1 + 1  # Output layer (weights + bias)
    return f"layers={config.mlp_layers}, params={param_count}"


def main() -> None:
    params_path = THIS_DIR / "params.yaml"
    base = load_config_from_yaml(params_path)
    experiment_name = base.run_name if base.run_name != "default" else base.task_name
    day_folder = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M%S")
    experiment_dir = THIS_DIR / "runs" / day_folder / f"{experiment_name}_{day_folder}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    tmn_config = clone_config(base, "tmn", f"{experiment_name}_tmn")
    mlp_config = clone_config(base, "mlp", f"{experiment_name}_mlp")

    tmn_trace_fn = build_trace_fn(tmn_config)

    # Print model parameter counts before training
    tmn_arch = tmn_architecture_text(tmn_config)
    mlp_arch = mlp_architecture_text(mlp_config)
    tmn_params = tmn_arch.split("params=")[1].split(",")[0] if "params=" in tmn_arch else "N/A"
    mlp_params = mlp_arch.split("params=")[1].split(",")[0] if "params=" in mlp_arch else "N/A"
    print(f"\n{'='*50}")
    print(f"TMN: {tmn_params} parameters")
    print(f"MLP: {mlp_params} parameters")
    print(f"{'='*50}\n")

    print("Training TMN...")
    tmn_result = train_with_config(tmn_config, trace_fn=tmn_trace_fn)

    print("Training MLP...")
    mlp_result = train_with_config(mlp_config)

    comparison_path = experiment_dir / "comparison_4panel.png"

    # Check if 2D task
    is_2d = base.n_in == 2

    if is_2d:
        print("Building 2D comparison image with MLP...")
        # Get grid info for plotting
        x1_plot = tmn_result["x1_grid"]
        x2_plot = tmn_result["x2_grid"]
        save_2d_comparison_with_mlp(
            tmn_y_true=tmn_result["y_plot"],
            tmn_y_pred=tmn_result["y_pred"],
            mlp_y_true=mlp_result["y_plot"],
            mlp_y_pred=mlp_result["y_pred"],
            x1_grid=x1_plot,
            x2_grid=x2_plot,
            tmn_architecture=tmn_architecture_text(tmn_config),
            mlp_architecture=mlp_architecture_text(mlp_config),
            tmn_final_val_loss=tmn_result["metrics"]["final_val_loss"],
            mlp_final_val_loss=mlp_result["metrics"]["final_val_loss"],
            tmn_train_losses=tmn_result["train_losses"],
            tmn_val_losses=tmn_result["val_losses"],
            mlp_train_losses=mlp_result["train_losses"],
            mlp_val_losses=mlp_result["val_losses"],
            traced_params=tmn_result["traced_params"],
            output_path=comparison_path,
        )
    else:
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

    if is_2d:
        print("Skipping weights visualization for 2D task...")
    else:
        weights_path = experiment_dir / "weights.png"
        print("Building weights visualization...")
        save_weights_plot(
            tmn_result["model"],
            tmn_result["traced_params"],
            weights_path,
            tmn_architecture_text(tmn_config),
        )

    print("Done.")
    print(f"TMN val loss: {tmn_result['metrics']['final_val_loss']:.6f}")
    print(f"MLP val loss: {mlp_result['metrics']['final_val_loss']:.6f}")
    print(f"Comparison image: {comparison_path}")
    if not is_2d:
        print(f"Weights image:    {weights_path}")


if __name__ == "__main__":
    main()
