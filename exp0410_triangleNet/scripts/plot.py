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


def _node_pos(node, L):
    if node[0] == "in":
        return (-(L + 1), (L - 1) / 2.0)
    if node[0] == "out":
        return (L + 1, (L - 1) / 2.0)
    _, r, c = node
    return (2 * c - r - 1, L - r)


def _draw_tmn_weights(ax, model):
    graph = model.graph
    L = graph.L
    show_labels = len(graph.edges) <= 30

    edge_to_idx = {edge: i for i, edge in enumerate(graph.edges)}

    for src, dst in graph.edges:
        w = model.ew[edge_to_idx[(src, dst)]].item()
        x0, y0 = _node_pos(src, L)
        x1, y1 = _node_pos(dst, L)
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
        x, y = _node_pos(node, L)
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

        circle = mpatches.Circle((x, y), 0.38, color=fc, ec="#555555", lw=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5, zorder=5)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    ax.set_title(f"TMN weights  (L={L})")


def save_weights_plot(tmn_model, mlp_model, output_path: Path) -> None:
    linears = [m for m in mlp_model.network if isinstance(m, nn.Linear)]
    n_layers = len(linears)

    fig = plt.figure(figsize=(16, max(6, n_layers * 2)))
    fig.suptitle("Learned weights", fontsize=13)

    # Left half: TMN DAG
    ax_tmn = fig.add_axes([0.02, 0.05, 0.46, 0.88])
    _draw_tmn_weights(ax_tmn, tmn_model)

    # Right half: MLP weight heatmaps, one per layer
    if n_layers == 0:
        return

    vmax = max(m.weight.detach().abs().max().item() for m in linears)
    h = 0.82 / n_layers
    pad = 0.03

    for i, layer in enumerate(linears):
        W = layer.weight.detach().cpu().numpy()
        bottom = 0.08 + (n_layers - 1 - i) * (h + pad)
        ax = fig.add_axes([0.54, bottom, 0.38, h])
        im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"W{i + 1}  {W.shape[0]}×{W.shape[1]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(f"MLP weights — W{i + 1}  {W.shape[0]}×{W.shape[1]}", fontsize=9)

    fig.colorbar(im, ax=fig.axes[-1], shrink=0.8, label="weight", pad=0.02)
    plt.savefig(output_path, dpi=150)
    plt.close()
