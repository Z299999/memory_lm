from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_four_panel_plot(
    tmn_train_losses: list[float],
    tmn_val_losses: list[float],
    tmn_x: np.ndarray,
    tmn_y_true: np.ndarray,
    tmn_y_pred: np.ndarray,
    tmn_architecture: str,
    tmn_final_train_loss: float,
    tmn_final_val_loss: float,
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    mlp_x: np.ndarray,
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    mlp_architecture: str,
    mlp_final_train_loss: float,
    mlp_final_val_loss: float,
    figure_title: str,
    loss_fn: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{figure_title} | loss={loss_fn}", fontsize=14)

    axes[0, 0].plot(tmn_train_losses, label="train")
    axes[0, 0].plot(tmn_val_losses, label="val")
    axes[0, 0].set_title(
        f"TMN Loss Curve\n{tmn_architecture}\n"
        f"final train={tmn_final_train_loss:.6f}  val={tmn_final_val_loss:.6f}"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel(f"{loss_fn}")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()

    axes[0, 1].plot(tmn_x, tmn_y_true, label="target")
    axes[0, 1].plot(tmn_x, tmn_y_pred, label="prediction")
    axes[0, 1].set_title(f"TMN Prediction\n{tmn_architecture}")
    axes[0, 1].set_xlabel("normalized x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].legend()

    axes[1, 0].plot(mlp_train_losses, label="train")
    axes[1, 0].plot(mlp_val_losses, label="val")
    axes[1, 0].set_title(
        f"MLP Loss Curve\n{mlp_architecture}\n"
        f"final train={mlp_final_train_loss:.6f}  val={mlp_final_val_loss:.6f}"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel(f"{loss_fn}")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()

    axes[1, 1].plot(mlp_x, mlp_y_true, label="target")
    axes[1, 1].plot(mlp_x, mlp_y_pred, label="prediction")
    axes[1, 1].set_title(f"MLP Prediction\n{mlp_architecture}")
    axes[1, 1].set_xlabel("normalized x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()


def _node_pos(node, L, depth=1):
    if node[0] == "in":
        return (-(L + 1.5), (L - 1) / 2.0)
    if node[0] == "out":
        return (L + 1.5, (L - 1) / 2.0)
    # 3D format: ("core", x, y, z)
    # Pyramid layout: bottom layer (z=1) has L positions, top layer (z=L) has 1 position
    # Center each layer horizontally
    _, x, y, z = node
    # Layer z has (L-z+1) positions, center them around x=0
    n_pos_in_layer = L - z + 1
    # Position y goes from 1 to n_pos_in_layer
    # Center: x = (y - 1) - (n_pos_in_layer - 1) / 2
    # Increase spacing between nodes
    x_pos = (y - 1) * 1.5 - (n_pos_in_layer - 1) * 1.5 / 2
    y_pos = (z - 1) * 1.2  # z=1 at bottom (y=0), z=L at top
    return (x_pos, y_pos)


def _should_visualize_node(node):
    """Return True if node should be visualized. Only visualize inner layer (x=1)."""
    if node[0] in ("in", "out"):
        return True
    # Core node: ("core", x, y, z) - only visualize x=1 (inner layer)
    return node[1] == 1


def _draw_tmn_weights(ax, model):
    graph = model.graph
    L = graph.L
    depth = graph.depth

    edge_to_idx = {edge: i for i, edge in enumerate(graph.edges)}

    # Count visible edges to decide whether to show labels
    visible_edges = [
        (src, dst) for src, dst in graph.edges
        if _should_visualize_node(src) and _should_visualize_node(dst)
    ]
    show_labels = len(visible_edges) <= 30

    for src, dst in visible_edges:
        w = model.ew[edge_to_idx[(src, dst)]].item()
        x0, y0 = _node_pos(src, L, depth)
        x1, y1 = _node_pos(dst, L, depth)
        color = "#2166ac" if w >= 0 else "#d6604d"
        lw = max(0.8, min(4.0, abs(w) * 3))
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
            zorder=2,
        )
        if show_labels:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my, f"{w:.2f}", fontsize=7, ha="center", va="center",
                    color=color, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.75, edgecolor="none"))

    core_to_bias_idx = {node: i for i, node in enumerate(graph.core_nodes)}

    for node in graph.nodes:
        # Skip nodes not in the inner layer (x=1)
        if not _should_visualize_node(node):
            continue
        x, y = _node_pos(node, L, depth)
        if node[0] == "core":
            bias = model.nb[core_to_bias_idx[node]].item()
            label = f"b={bias:.2f}"
            fc = "#d1e5f0"
        elif node[0] == "in":
            label = "in"
            fc = "#e0e0e0"
        else:
            bias = model.output_bias.item()
            label = f"b={bias:.2f}"
            fc = "#fddbc7"

        circle = mpatches.Circle((x, y), 0.45, color=fc, ec="#555555", lw=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=5)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    depth = graph.depth
    ax.set_title(f"TMN weights\nL={L}, depth={depth}, mode={graph.cross_layer_mode}")


def save_weights_plot(
    tmn_model,
    traced_params: dict[str, list[float]],
    output_path: Path,
    tmn_architecture: str,
) -> None:
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"TMN weights and training evolution | {tmn_architecture}", fontsize=13)

    # Left half: TMN DAG (final weights)
    ax_tmn = fig.add_axes([0.02, 0.05, 0.46, 0.88])
    _draw_tmn_weights(ax_tmn, tmn_model)

    # Right half: two subplots for traced params over epochs
    # Upper: edge weights, Lower: node biases
    n_epochs = len(next(iter(traced_params.values())))

    # Separate keys into edge weights and node biases
    ew_keys = sorted([k for k in traced_params if k.startswith("w(")])
    nb_keys = sorted([k for k in traced_params if k.startswith("b(")])

    ax_ew = fig.add_axes([0.52, 0.52, 0.46, 0.38])
    ax_nb = fig.add_axes([0.52, 0.08, 0.46, 0.38])

    epochs = list(range(1, n_epochs + 1))

    # Plot edge weights
    for key in ew_keys:
        ax_ew.plot(epochs, traced_params[key], label=key, linewidth=0.8)
    ax_ew.set_ylabel("edge weight")
    ax_ew.set_title("Edge weights evolution (bottom row)")
    if len(ew_keys) <= 12:
        ax_ew.legend(fontsize=6, ncol=2)

    # Plot node biases
    for key in nb_keys:
        ax_nb.plot(epochs, traced_params[key], label=key, linewidth=0.8)
    ax_nb.set_xlabel("epoch")
    ax_nb.set_ylabel("node bias")
    ax_nb.set_title("Node biases evolution (bottom row)")
    if len(nb_keys) <= 8:
        ax_nb.legend(fontsize=8)

    plt.savefig(output_path, dpi=150)
    plt.close()
