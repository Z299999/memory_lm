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
        return (-(L + 1.5), (L - 1) / 2.0 + 0.3)
    if node[0] == "out":
        return (L + 1.5, (L - 1) / 2.0 + 0.3)
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

    # For tetrahedron mode: add x-offset for depth dimension (3D projection)
    # Shift nodes with higher x to the right and slightly down for perspective
    # Compute depth_at_z: how many neurons at layer z
    if depth == "tetrahedron":
        depth_at_z = z  # depth(z) = z
    elif callable(depth):
        depth_at_z = depth(z)
    else:
        depth_at_z = depth if isinstance(depth, int) else 1

    if depth_at_z > 1:
        # Spread x layers across a range, centered around original position
        x_offset = (x - (depth_at_z + 1) / 2) * 0.5
        y_offset = -(x - 1) * 0.15  # Higher x appears slightly lower (perspective)
        x_pos += x_offset
        y_pos += y_offset
    return (x_pos, y_pos)


def _should_visualize_node(node):
    """Return True if node should be visualized. Visualize all nodes."""
    if node[0] in ("in", "out"):
        return True
    # Core node: ("core", x, y, z) - visualize all depths
    return True


def _draw_tmn_weights(ax, model):
    graph = model.graph
    L = graph.L
    depth = graph.depth

    # Get max depth for color scaling
    if depth == "tetrahedron":
        max_d = L
    elif callable(depth):
        max_d = depth(L)
    else:
        max_d = depth if isinstance(depth, int) else 1

    edge_to_idx = {edge: i for i, edge in enumerate(graph.edges)}

    # Count visible edges to decide whether to show labels
    visible_edges = [
        (src, dst) for src, dst in graph.edges
        if _should_visualize_node(src) and _should_visualize_node(dst)
    ]
    show_labels = len(visible_edges) <= 40

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
            ax.text(mx, my, f"{w:.2f}", fontsize=6, ha="center", va="center",
                    color=color, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.75, edgecolor="none"))

    core_to_bias_idx = {node: i for i, node in enumerate(graph.core_nodes)}

    # Color map for depth (x value) - from light to dark blue
    depth_colors = ["#e5f5f9", "#99d8c9", "#2ca25f", "#8856a7", "#810f7c"]

    for node in graph.nodes:
        if not _should_visualize_node(node):
            continue
        x, y = _node_pos(node, L, depth)

        if node[0] == "core":
            bias = model.nb[core_to_bias_idx[node]].item()
            label = f"b={bias:.2f}"
            # Color by depth (x value)
            x_val = node[1]
            color_idx = min(x_val - 1, len(depth_colors) - 1)
            fc = depth_colors[color_idx]
            # Size decreases with depth for perspective
            radius = 0.28 - (x_val - 1) * 0.03
        elif node[0] == "in":
            label = "in"
            fc = "#e0e0e0"
            radius = 0.35
        else:  # output
            bias = model.output_bias.item()
            label = f"b={bias:.2f}"
            fc = "#fddbc7"
            radius = 0.35

        circle = mpatches.Circle((x, y), radius, color=fc, ec="#555555", lw=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=6, zorder=5)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    depth_str = f"depth(z)=z" if depth == "tetrahedron" else f"depth={depth}"
    ax.set_title(f"TMN weights\nL={L}, {depth_str}, mode={graph.cross_layer_mode}")


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
    # New format: "bottom_w(...)" and "top_w(...)", "bottom_b(...)" and "top_b(...)"
    ew_keys = sorted([k for k in traced_params if "_w(" in k])
    nb_keys = sorted([k for k in traced_params if "_b(" in k])

    ax_ew = fig.add_axes([0.52, 0.52, 0.46, 0.38])
    ax_nb = fig.add_axes([0.52, 0.08, 0.46, 0.38])

    epochs = list(range(1, n_epochs + 1))

    # Color map for bottom (blue) and top (orange)
    row_colors = {"bottom": "#1f77b4", "top": "#ff7f0e"}

    # Plot edge weights
    for key in ew_keys:
        row = "bottom" if key.startswith("bottom_") else "top"
        # Simplify label: extract just the edge info
        # e.g., "bottom_w(in,1)->(1,1,1)" -> "bot: in->(1,1,1)" or "top: in->(1,1,1)"
        row_short = "bot" if row == "bottom" else "top"
        edge_info = key.replace(f"{row}_w(", "").rstrip(")")
        label = f"{row_short}: {edge_info}"
        ax_ew.plot(epochs, traced_params[key], label=label, linewidth=0.8, color=row_colors.get(row, None))
    ax_ew.set_ylabel("edge weight")
    ax_ew.set_title("Edge weights evolution (top and bottom rows)")
    ax_ew.legend(
        fontsize=4,
        ncol=3,
        loc="upper right",
        framealpha=0.8,
        borderpad=0.3,
        handlelength=1.2,
        labelspacing=0.2,
    )

    # Plot node biases
    for key in nb_keys:
        row = "bottom" if key.startswith("bottom_") else "top"
        row_short = "bot" if row == "bottom" else "top"
        # e.g., "bottom_b(1,3,1)" -> "bot: (1,3,1)"
        coord = key.split("(")[1].rstrip(")")
        label = f"{row_short}: ({coord})"
        ax_nb.plot(epochs, traced_params[key], label=label, linewidth=0.8, color=row_colors.get(row, None))
    ax_nb.set_xlabel("epoch")
    ax_nb.set_ylabel("node bias")
    ax_nb.set_title("Node biases evolution (top and bottom rows)")
    ax_nb.legend(
        fontsize=5,
        ncol=2,
        loc="upper right",
        framealpha=0.8,
        borderpad=0.3,
        handlelength=1.2,
        labelspacing=0.2,
    )

    plt.savefig(output_path, dpi=150)
    plt.close()


def save_2d_comparison_plot(
    tmn_y_true: np.ndarray,
    tmn_y_pred: np.ndarray,
    tmn_x1_grid: np.ndarray,
    tmn_x2_grid: np.ndarray,
    tmn_architecture: str,
    tmn_final_val_loss: float,
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> None:
    """Save 4-panel comparison plot for 2D -> 1D functions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"2D Function Fit | {tmn_architecture} | final val loss={tmn_final_val_loss:.6f}", fontsize=12)

    # Reshape for plotting
    n_side = int(np.sqrt(len(tmn_y_true)))
    y_true_2d = tmn_y_true.reshape(n_side, n_side)
    y_pred_2d = tmn_y_pred.reshape(n_side, n_side)
    error_2d = y_true_2d - y_pred_2d

    # Panel 1: Target function heatmap
    im0 = axes[0, 0].pcolormesh(tmn_x1_grid, tmn_x2_grid, y_true_2d.T, cmap='viridis', shading='auto')
    axes[0, 0].set_title("Target Function")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")
    fig.colorbar(im0, ax=axes[0, 0])

    # Panel 2: TMN prediction heatmap
    im1 = axes[0, 1].pcolormesh(tmn_x1_grid, tmn_x2_grid, y_pred_2d.T, cmap='viridis', shading='auto')
    axes[0, 1].set_title(f"TMN Prediction")
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")
    fig.colorbar(im1, ax=axes[0, 1])

    # Panel 3: Error heatmap
    im2 = axes[1, 0].pcolormesh(tmn_x1_grid, tmn_x2_grid, error_2d.T, cmap='RdBu_r', shading='auto')
    axes[1, 0].set_title("Error (Target - Prediction)")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("x2")
    fig.colorbar(im2, ax=axes[1, 0])

    # Panel 4: Loss curves
    axes[1, 1].plot(train_losses, label="train")
    axes[1, 1].plot(val_losses, label="val")
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
