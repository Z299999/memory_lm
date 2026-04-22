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
    batch_size: int = 64,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{figure_title} | batch_size={batch_size} | loss={loss_fn}", fontsize=14)

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

    # TMN prediction plot - support multi-output
    if tmn_y_true.ndim == 1 or tmn_y_true.shape[1] == 1:
        # Single output
        axes[0, 1].plot(tmn_x, tmn_y_true, label="target")
        axes[0, 1].plot(tmn_x, tmn_y_pred, label="prediction")
    else:
        # Multi-output: plot each dimension with different color
        n_out = tmn_y_true.shape[1]
        for i in range(n_out):
            axes[0, 1].plot(tmn_x, tmn_y_true[:, i], label=f"target dim {i+1}", alpha=0.7)
            axes[0, 1].plot(tmn_x, tmn_y_pred[:, i], label=f"pred dim {i+1}", alpha=0.7, linestyle='--')
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

    # MLP prediction plot - support multi-output
    if mlp_y_true.ndim == 1 or mlp_y_true.shape[1] == 1:
        # Single output
        axes[1, 1].plot(mlp_x, mlp_y_true, label="target")
        axes[1, 1].plot(mlp_x, mlp_y_pred, label="prediction")
    else:
        # Multi-output
        n_out = mlp_y_true.shape[1]
        for i in range(n_out):
            axes[1, 1].plot(mlp_x, mlp_y_true[:, i], label=f"target dim {i+1}", alpha=0.7)
            axes[1, 1].plot(mlp_x, mlp_y_pred[:, i], label=f"pred dim {i+1}", alpha=0.7, linestyle='--')
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
            # Multi-output support: show first output bias
            if model.output_bias.numel() == 1:
                bias = model.output_bias.item()
            else:
                bias = model.output_bias[0].item()
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


def save_2d_comparison_with_mlp(
    tmn_y_true: np.ndarray,
    tmn_y_pred: np.ndarray,
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    tmn_architecture: str,
    mlp_architecture: str,
    tmn_final_val_loss: float,
    mlp_final_val_loss: float,
    tmn_train_losses: list[float],
    tmn_val_losses: list[float],
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    traced_params: dict[str, list[float]] | None,
    output_path: Path,
    batch_size: int = 64,
) -> None:
    """Save 2D comparison plot with TMN vs MLP.

    Layout for n_out=1:
    Row 0: TMN Target | TMN Pred | MLP Target | MLP Pred
    Row 1: TMN Error  |          | MLP Error  |
    Row 2: Bottom params | Top params | Loss comparison (span 2 cols)

    For n_out > 1: all dimensions on one figure (one row per output dim + params + loss).
    """
    import re

    # Check if multi-output
    if tmn_y_true.ndim > 1 and tmn_y_true.shape[1] > 1:
        # Multi-output: single figure with all dimensions
        _save_2d_comparison_multi_output(
            tmn_y_true, tmn_y_pred,
            mlp_y_true, mlp_y_pred,
            x1_grid, x2_grid,
            tmn_architecture, mlp_architecture,
            tmn_final_val_loss, mlp_final_val_loss,
            tmn_train_losses, tmn_val_losses,
            mlp_train_losses, mlp_val_losses,
            traced_params, output_path,
            batch_size
        )
        return

    # Single output case - existing logic
    _save_2d_comparison_single_output(
        tmn_y_true, tmn_y_pred,
        mlp_y_true, mlp_y_pred,
        x1_grid, x2_grid,
        tmn_architecture, mlp_architecture,
        tmn_final_val_loss, mlp_final_val_loss,
        tmn_train_losses, tmn_val_losses,
        mlp_train_losses, mlp_val_losses,
        traced_params, output_path,
        batch_size=batch_size
    )


def _save_2d_comparison_multi_output(
    tmn_y_true: np.ndarray,
    tmn_y_pred: np.ndarray,
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    tmn_architecture: str,
    mlp_architecture: str,
    tmn_final_val_loss: float,
    mlp_final_val_loss: float,
    tmn_train_losses: list[float],
    tmn_val_losses: list[float],
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    traced_params: dict[str, list[float]] | None,
    output_path: Path,
    batch_size: int = 64,
) -> None:
    """Save 2D comparison for multi-output: all dimensions on one figure.

    Layout (example for n_out=2):
    Row 0: TMN Target dim1 | TMN Pred dim1 | MLP Target dim1 | MLP Pred dim1
    Row 1: TMN Target dim2 | TMN Pred dim2 | MLP Target dim2 | MLP Pred dim2
    Row 2: TMN Error dim1 | TMN Error dim2 | MLP Error dim1 | MLP Error dim2
    Row 3: Bottom params | Top params | Loss comparison (span 2 cols)
    """
    import re

    n_out = tmn_y_true.shape[1]
    n_side = int(np.sqrt(len(tmn_y_true)))
    x_min = float(x1_grid.min())
    x_max = float(x1_grid.max())

    # Parse L for param titles
    l_match = re.search(r'L=(\d+)', tmn_architecture)
    L = int(l_match.group(1)) if l_match else 3

    # Use 12 columns:
    # - Heatmap rows: 4 panels × 3 cols each = 12 cols
    # - Bottom row: 3 panels × 4 cols each = 12 cols (all equal width!)
    fig = plt.figure(figsize=(22, 5 * (n_out + 2)))
    total_rows = n_out + 2
    height_ratios = [1.0] * total_rows
    gs = fig.add_gridspec(total_rows, 12, height_ratios=height_ratios, hspace=0.30, wspace=0.25)

    # Determine common color scale across all dimensions
    all_values = []
    for i in range(n_out):
        all_values.extend([
            tmn_y_true[:, i].ravel(), tmn_y_pred[:, i].ravel(),
            mlp_y_true[:, i].ravel(), mlp_y_pred[:, i].ravel()
        ])
    all_values = np.concatenate(all_values)
    vmin, vmax = float(all_values.min()), float(all_values.max())

    # === Rows 0 to n_out-1: Predictions for each dimension ===
    # 4 panels: TMN Target | TMN Pred | MLP Target | MLP Pred
    # Each spans 3 cols: [0:3], [3:6], [6:9], [9:12]
    for dim in range(n_out):
        tmn_true_2d = tmn_y_true[:, dim].reshape(n_side, n_side)
        tmn_pred_2d = tmn_y_pred[:, dim].reshape(n_side, n_side)
        mlp_true_2d = mlp_y_true[:, dim].reshape(n_side, n_side)
        mlp_pred_2d = mlp_y_pred[:, dim].reshape(n_side, n_side)

        # TMN Target (cols 0-3)
        ax_target = fig.add_subplot(gs[dim, 0:3])
        im = ax_target.imshow(tmn_true_2d.T, origin='lower',
                              extent=[x_min, x_max, x_min, x_max],
                              cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax_target.set_title(f"TMN Target (dim {dim+1})")
        ax_target.set_xlabel("x1")
        ax_target.set_ylabel("x2")
        fig.colorbar(im, ax=ax_target, fraction=0.046, pad=0.04)

        # TMN Prediction (cols 3-6)
        ax_pred = fig.add_subplot(gs[dim, 3:6])
        im = ax_pred.imshow(tmn_pred_2d.T, origin='lower',
                            extent=[x_min, x_max, x_min, x_max],
                            cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax_pred.set_title(f"TMN Prediction (dim {dim+1})")
        ax_pred.set_xlabel("x1")
        ax_pred.set_ylabel("x2")
        fig.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04)

        # MLP Target (cols 6-9)
        ax_mlp_target = fig.add_subplot(gs[dim, 6:9])
        im = ax_mlp_target.imshow(mlp_true_2d.T, origin='lower',
                                  extent=[x_min, x_max, x_min, x_max],
                                  cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax_mlp_target.set_title(f"MLP Target (dim {dim+1})")
        ax_mlp_target.set_xlabel("x1")
        ax_mlp_target.set_ylabel("x2")
        fig.colorbar(im, ax=ax_mlp_target, fraction=0.046, pad=0.04)

        # MLP Prediction (cols 9-12)
        ax_mlp_pred = fig.add_subplot(gs[dim, 9:12])
        im = ax_mlp_pred.imshow(mlp_pred_2d.T, origin='lower',
                                extent=[x_min, x_max, x_min, x_max],
                                cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax_mlp_pred.set_title(f"MLP Prediction (dim {dim+1})")
        ax_mlp_pred.set_xlabel("x1")
        ax_mlp_pred.set_ylabel("x2")
        fig.colorbar(im, ax=ax_mlp_pred, fraction=0.046, pad=0.04)

    # === Row n_out: Errors (4 panels, each 3 cols) ===
    error_row = n_out
    err_max = 0.0
    for dim in range(n_out):
        tmn_err = np.abs(tmn_y_true[:, dim] - tmn_y_pred[:, dim]).max()
        mlp_err = np.abs(mlp_y_true[:, dim] - mlp_y_pred[:, dim]).max()
        err_max = max(err_max, tmn_err, mlp_err)

    for dim in range(n_out):
        tmn_error_2d = (tmn_y_true[:, dim].reshape(n_side, n_side) -
                        tmn_y_pred[:, dim].reshape(n_side, n_side))
        mlp_error_2d = (mlp_y_true[:, dim].reshape(n_side, n_side) -
                        mlp_y_pred[:, dim].reshape(n_side, n_side))

        # TMN Error (cols 0:3 for dim 0, etc. - spread across row)
        ax_tmn_err = fig.add_subplot(gs[error_row, dim*3:(dim+1)*3])
        im = ax_tmn_err.imshow(tmn_error_2d.T, origin='lower',
                               extent=[x_min, x_max, x_min, x_max],
                               cmap='RdBu_r', vmin=-err_max, vmax=err_max, interpolation='bilinear')
        ax_tmn_err.set_title(f"TMN Error (dim {dim+1})")
        ax_tmn_err.set_xlabel("x1")
        ax_tmn_err.set_ylabel("x2")
        fig.colorbar(im, ax=ax_tmn_err, fraction=0.046, pad=0.04)

    # === Row n_out+1: Params and Loss (3 equal-width plots) ===
    # Each plot spans 4 cols: [0:4], [4:8], [8:12]
    param_row = n_out + 1
    epochs = list(range(1, len(tmn_train_losses) + 1))

    if traced_params:
        ew_keys = sorted([k for k in traced_params if "_w(" in k])
        nb_keys = sorted([k for k in traced_params if "_b(" in k])

        # Bottom params (cols 0-4)
        ax_bottom = fig.add_subplot(gs[param_row, 0:4])
        for key in ew_keys:
            if key.startswith("bottom_"):
                edge_info = key.replace("bottom_w(", "").rstrip(")")
                ax_bottom.plot(epochs, traced_params[key], label=f"w: {edge_info}", linewidth=0.8)
        for key in nb_keys:
            if key.startswith("bottom_"):
                coord = key.split("(")[1].rstrip(")")
                ax_bottom.plot(epochs, traced_params[key], label=f"b: ({coord})", linewidth=0.8, linestyle='--')
        ax_bottom.set_ylabel("bottom params")
        ax_bottom.set_title(f"Bottom row (z=1)")
        ax_bottom.legend(fontsize=4, ncol=1, loc="upper right", framealpha=0.8)

        # Top params (cols 4-8)
        ax_top = fig.add_subplot(gs[param_row, 4:8])
        for key in ew_keys:
            if key.startswith("top_"):
                edge_info = key.replace("top_w(", "").rstrip(")")
                ax_top.plot(epochs, traced_params[key], label=f"w: {edge_info}", linewidth=0.8)
        for key in nb_keys:
            if key.startswith("top_"):
                coord = key.split("(")[1].rstrip(")")
                ax_top.plot(epochs, traced_params[key], label=f"b: ({coord})", linewidth=0.8, linestyle='--')
        ax_top.set_ylabel("top params")
        ax_top.set_title(f"Top row (z={L})")
        ax_top.legend(fontsize=4, ncol=2, loc="upper right", framealpha=0.8)

        # Loss (cols 8-12)
        ax_loss = fig.add_subplot(gs[param_row, 8:12])
    else:
        ax_loss = fig.add_subplot(gs[param_row, :])

    ax_loss.plot(tmn_train_losses, label="TMN train", color='#1f77b4', linewidth=1)
    ax_loss.plot(tmn_val_losses, label="TMN val", color='#1f77b4', linestyle='--', linewidth=1)
    ax_loss.plot(mlp_train_losses, label="MLP train", color='#ff7f0e', linewidth=1)
    ax_loss.plot(mlp_val_losses, label="MLP val", color='#ff7f0e', linestyle='--', linewidth=1)
    ax_loss.set_title("Training Loss Comparison")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="upper right", ncol=2)

    # Extract MLP layers from architecture string
    mlp_layers_match = mlp_architecture.split('layers=')[1].split('],')[0] + ']' if 'layers=' in mlp_architecture else 'N/A'

    plt.suptitle(
        f"2D Function Fit | TMN vs MLP | Task: {tmn_architecture.split(',')[0]} | n_out={n_out} | "
        f"batch_size={batch_size} | "
        f"TMN params={tmn_architecture.split('params=')[1].split(',')[0] if 'params=' in tmn_architecture else 'N/A'} | "
        f"MLP {mlp_layers_match} params={mlp_architecture.split('params=')[1].split(',')[0] if 'params=' in mlp_architecture else 'N/A'}",
        fontsize=14, y=0.98
    )
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def _save_2d_comparison_single_output(
    tmn_y_true: np.ndarray,
    tmn_y_pred: np.ndarray,
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    tmn_architecture: str,
    mlp_architecture: str,
    tmn_final_val_loss: float,
    mlp_final_val_loss: float,
    tmn_train_losses: list[float],
    tmn_val_losses: list[float],
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    traced_params: dict[str, list[float]] | None,
    output_path: Path,
    output_idx: int = 0,
    n_out: int = 1,
    batch_size: int = 64,
) -> None:
    """Helper function to save 2D comparison for a single output dimension."""
    import re

    # Parse L from tmn_architecture string
    l_match = re.search(r'L=(\d+)', tmn_architecture)
    L = int(l_match.group(1)) if l_match else 3

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.5, 0.5], hspace=0.35, wspace=0.3)

    # Reshape for plotting
    n_side = int(np.sqrt(len(tmn_y_true)))
    tmn_true_2d = tmn_y_true.reshape(n_side, n_side)
    tmn_pred_2d = tmn_y_pred.reshape(n_side, n_side)
    tmn_error_2d = tmn_true_2d - tmn_pred_2d

    mlp_true_2d = mlp_y_true.reshape(n_side, n_side)
    mlp_pred_2d = mlp_y_pred.reshape(n_side, n_side)
    mlp_error_2d = mlp_true_2d - mlp_pred_2d

    # Get axis limits from grids
    x_min = float(x1_grid.min())
    x_max = float(x1_grid.max())

    # Determine common color scale for all heatmaps
    all_values = np.concatenate([tmn_true_2d.ravel(), tmn_pred_2d.ravel(),
                                  mlp_true_2d.ravel(), mlp_pred_2d.ravel()])
    vmin, vmax = float(all_values.min()), float(all_values.max())

    # Error color scale (diverging, centered at 0)
    err_max = max(abs(tmn_error_2d).max(), abs(mlp_error_2d).max())

    # === Row 0: Predictions ===

    # TMN Target (0, 0)
    ax_tmn_target = fig.add_subplot(gs[0, 0])
    im0 = ax_tmn_target.imshow(tmn_true_2d.T, origin='lower',
                                extent=[x_min, x_max, x_min, x_max],
                                cmap='viridis', vmin=vmin, vmax=vmax,
                                interpolation='bilinear')
    title_suffix = f" (dim {output_idx+1}/{n_out})" if n_out > 1 else ""
    ax_tmn_target.set_title(f"TMN Target{title_suffix}\n{tmn_architecture}")
    ax_tmn_target.set_xlabel("x1")
    ax_tmn_target.set_ylabel("x2")
    fig.colorbar(im0, ax=ax_tmn_target, fraction=0.046, pad=0.04)

    # TMN Prediction (0, 1)
    ax_tmn_pred = fig.add_subplot(gs[0, 1])
    im1 = ax_tmn_pred.imshow(tmn_pred_2d.T, origin='lower',
                              extent=[x_min, x_max, x_min, x_max],
                              cmap='viridis', vmin=vmin, vmax=vmax,
                              interpolation='bilinear')
    ax_tmn_pred.set_title(f"TMN Prediction{title_suffix}\nval loss={tmn_final_val_loss:.6f}")
    ax_tmn_pred.set_xlabel("x1")
    ax_tmn_pred.set_ylabel("x2")
    fig.colorbar(im1, ax=ax_tmn_pred, fraction=0.046, pad=0.04)

    # MLP Target (0, 2)
    ax_mlp_target = fig.add_subplot(gs[0, 2])
    im2 = ax_mlp_target.imshow(mlp_true_2d.T, origin='lower',
                                extent=[x_min, x_max, x_min, x_max],
                                cmap='viridis', vmin=vmin, vmax=vmax,
                                interpolation='bilinear')
    ax_mlp_target.set_title(f"MLP Target{title_suffix}\n{mlp_architecture}")
    ax_mlp_target.set_xlabel("x1")
    ax_mlp_target.set_ylabel("x2")
    fig.colorbar(im2, ax=ax_mlp_target, fraction=0.046, pad=0.04)

    # MLP Prediction (0, 3)
    ax_mlp_pred = fig.add_subplot(gs[0, 3])
    im3 = ax_mlp_pred.imshow(mlp_pred_2d.T, origin='lower',
                              extent=[x_min, x_max, x_min, x_max],
                              cmap='viridis', vmin=vmin, vmax=vmax,
                              interpolation='bilinear')
    ax_mlp_pred.set_title(f"MLP Prediction{title_suffix}\nval loss={mlp_final_val_loss:.6f}")
    ax_mlp_pred.set_xlabel("x1")
    ax_mlp_pred.set_ylabel("x2")
    fig.colorbar(im3, ax=ax_mlp_pred, fraction=0.046, pad=0.04)

    # === Row 1: Errors ===

    # TMN Error (1, 0)
    ax_tmn_error = fig.add_subplot(gs[1, 0])
    im4 = ax_tmn_error.imshow(tmn_error_2d.T, origin='lower',
                               extent=[x_min, x_max, x_min, x_max],
                               cmap='RdBu_r', vmin=-err_max, vmax=err_max,
                               interpolation='bilinear')
    ax_tmn_error.set_title("TMN Error (Target - Prediction)")
    ax_tmn_error.set_xlabel("x1")
    ax_tmn_error.set_ylabel("x2")
    fig.colorbar(im4, ax=ax_tmn_error, fraction=0.046, pad=0.04)

    # MLP Error (1, 2) - span columns 2-3
    ax_mlp_error = fig.add_subplot(gs[1, 2:4])
    im5 = ax_mlp_error.imshow(mlp_error_2d.T, origin='lower',
                               extent=[x_min, x_max, x_min, x_max],
                               cmap='RdBu_r', vmin=-err_max, vmax=err_max,
                               interpolation='bilinear')
    ax_mlp_error.set_title("MLP Error (Target - Prediction)")
    ax_mlp_error.set_xlabel("x1")
    ax_mlp_error.set_ylabel("x2")
    fig.colorbar(im5, ax=ax_mlp_error, fraction=0.046, pad=0.04)

    # === Row 2-3: Parameter evolution and Loss curves ===
    epochs = list(range(1, len(tmn_train_losses) + 1))

    # Separate traced params into edge weights and node biases
    if traced_params:
        ew_keys = sorted([k for k in traced_params if "_w(" in k])
        nb_keys = sorted([k for k in traced_params if "_b(" in k])

        # Color map for bottom (blue) and top (orange)
        row_colors = {"bottom": "#1f77b4", "top": "#ff7f0e"}

        # Bottom row params (2, 0) - span 2 columns
        ax_bottom = fig.add_subplot(gs[2, 0:2])
        for key in ew_keys:
            if key.startswith("bottom_"):
                row_short = "bot"
                edge_info = key.replace(f"bottom_w(", "").rstrip(")")
                label = f"{row_short} w: {edge_info}"
                ax_bottom.plot(epochs, traced_params[key], label=label, linewidth=0.8)
        for key in nb_keys:
            if key.startswith("bottom_"):
                row_short = "bot"
                coord = key.split("(")[1].rstrip(")")
                label = f"{row_short} b: ({coord})"
                ax_bottom.plot(epochs, traced_params[key], label=label, linewidth=0.8, linestyle='--')
        ax_bottom.set_ylabel("bottom params")
        ax_bottom.set_title("Bottom row (z=1) - Edge weights (solid) and Biases (dashed)")
        ax_bottom.legend(fontsize=5, ncol=2, loc="upper right", framealpha=0.8)

        # Top row params (2, 2) - span 2 columns
        ax_top = fig.add_subplot(gs[2, 2:4])
        for key in ew_keys:
            if key.startswith("top_"):
                row_short = "top"
                edge_info = key.replace(f"top_w(", "").rstrip(")")
                label = f"{row_short} w: {edge_info}"
                ax_top.plot(epochs, traced_params[key], label=label, linewidth=0.8)
        for key in nb_keys:
            if key.startswith("top_"):
                row_short = "top"
                coord = key.split("(")[1].rstrip(")")
                label = f"{row_short} b: ({coord})"
                ax_top.plot(epochs, traced_params[key], label=label, linewidth=0.8, linestyle='--')
        ax_top.set_ylabel("top params")
        ax_top.set_title(f"Top row (z={L}) - Edge weights (solid) and Biases (dashed)")
        ax_top.legend(fontsize=5, ncol=2, loc="upper right", framealpha=0.8)

        # Row 3: Loss comparison (span all 4 columns)
        ax_loss = fig.add_subplot(gs[3, :])
    else:
        # No traced params, use row 2 for loss
        ax_loss = fig.add_subplot(gs[2, :])

    ax_loss.plot(tmn_train_losses, label="TMN train", color='#1f77b4', linewidth=1)
    ax_loss.plot(tmn_val_losses, label="TMN val", color='#1f77b4', linestyle='--', linewidth=1)
    ax_loss.plot(mlp_train_losses, label="MLP train", color='#ff7f0e', linewidth=1)
    ax_loss.plot(mlp_val_losses, label="MLP val", color='#ff7f0e', linestyle='--', linewidth=1)
    ax_loss.set_title("Training Loss Comparison")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale("log")
    ax_loss.legend(loc="upper right", ncol=4)

    # Modify output path for multi-output
    if n_out > 1:
        output_path = output_path.parent / f"{output_path.stem}_dim{output_idx+1}{output_path.suffix}"

    plt.suptitle(
        f"2D Function Fit | TMN vs MLP | Task: {tmn_architecture.split(',')[0]} | "
        f"batch_size={batch_size}",
        fontsize=14, y=0.98
    )
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
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
    """Save 4-panel comparison plot for 2D -> 1D functions (TMN only, legacy)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"2D Function Fit | {tmn_architecture} | final val loss={tmn_final_val_loss:.6f}", fontsize=12)

    # Reshape for plotting
    n_side = int(np.sqrt(len(tmn_y_true)))
    y_true_2d = tmn_y_true.reshape(n_side, n_side)
    y_pred_2d = tmn_y_pred.reshape(n_side, n_side)
    error_2d = y_true_2d - y_pred_2d

    # Get axis limits
    x_min = float(tmn_x1_grid.min())
    x_max = float(tmn_x1_grid.max())

    # Panel 1: Target function heatmap (using imshow with interpolation)
    im0 = axes[0, 0].imshow(y_true_2d.T, origin='lower',
                            extent=[x_min, x_max, x_min, x_max],
                            cmap='viridis', interpolation='bilinear')
    axes[0, 0].set_title("Target Function")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")
    fig.colorbar(im0, ax=axes[0, 0])

    # Panel 2: TMN prediction heatmap
    im1 = axes[0, 1].imshow(y_pred_2d.T, origin='lower',
                            extent=[x_min, x_max, x_min, x_max],
                            cmap='viridis', interpolation='bilinear')
    axes[0, 1].set_title(f"TMN Prediction")
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")
    fig.colorbar(im1, ax=axes[0, 1])

    # Panel 3: Error heatmap
    im2 = axes[1, 0].imshow(error_2d.T, origin='lower',
                            extent=[x_min, x_max, x_min, x_max],
                            cmap='RdBu_r', interpolation='bilinear')
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
    plt.savefig(output_path, dpi=200)
    plt.close()
