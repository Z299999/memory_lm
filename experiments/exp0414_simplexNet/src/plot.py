"""Visualization for simplex memory network experiments.

This module provides functions for creating comparison plots between
SMN and MLP models.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smn_architecture_text(n: int, m: int, graph, n_out: int = 1) -> str:
    """Get architecture description string for SMN."""
    # SMN parameters: edge weights + core node biases + output biases
    n_edges = graph.edge_count
    n_core = graph.core_node_count
    n_out = len(graph.output_nodes)
    n_params = n_edges + n_core + n_out
    return f"SMN(n={n}, m={m}), nodes={n_core}, edges={n_edges}, params={n_params}"


def mlp_architecture_text(layers: list[int], n_in: int = 1, n_out: int = 1) -> str:
    """Get architecture description string for MLP."""
    # Calculate parameters
    if len(layers) == 0:
        param_count = n_in * n_out + n_out  # Just input->output
    else:
        param_count = 0
        prev_size = n_in
        for h_size in layers:
            param_count += prev_size * h_size + h_size
            prev_size = h_size
        param_count += prev_size * n_out + n_out

    # Calculate nodes and edges
    # Nodes: input + hidden + output
    n_nodes = n_in + sum(layers) + n_out

    # Edges: connections between consecutive layers
    n_edges = 0
    prev_size = n_in
    for h_size in layers:
        n_edges += prev_size * h_size
        prev_size = h_size
    n_edges += prev_size * n_out

    return f"MLP layers={layers}, nodes={n_nodes}, edges={n_edges}, params={param_count}"


def _draw_scatter(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    architecture: str,
    final_val_loss: float,
) -> None:
    """Draw ||y_true|| vs ||y_pred|| scatter plot on ax.

    Uses L2 norm of the output vector so the plot scales to any n_out
    (including 100i100o) without needing multiple colors.
    For n_out=1 the norm equals |y|, identical to a plain scatter.
    A dashed y=x diagonal marks perfect prediction.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    n_out = y_true.shape[1]

    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)

    ax.scatter(norm_true, norm_pred, s=3, alpha=0.4, color='steelblue')

    lo = float(min(norm_true.min(), norm_pred.min()))
    hi = float(max(norm_true.max(), norm_pred.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y=x')

    xlabel = '||y_true||' if n_out > 1 else 'True'
    ylabel = '||y_pred||' if n_out > 1 else 'Predicted'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{model_name} Prediction\n{architecture}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def save_four_panel_plot(
    smn_train_losses: list[float],
    smn_val_losses: list[float],
    smn_y_true: np.ndarray,
    smn_y_pred: np.ndarray,
    smn_architecture: str,
    smn_final_train_loss: float,
    smn_final_val_loss: float,
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    mlp_architecture: str,
    mlp_final_train_loss: float,
    mlp_final_val_loss: float,
    figure_title: str,
    output_path: Path,
    batch_size: int = 64,
) -> None:
    """Create 4-panel comparison plot.

    Layout:
    [Top-left]   SMN: True vs Predicted scatter
    [Top-right]  MLP: True vs Predicted scatter
    [Bottom-left]  SMN: Train/Val Loss
    [Bottom-right] MLP: Train/Val Loss

    Works for any n_in / n_out combination.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: SMN scatter
    _draw_scatter(axes[0, 0], smn_y_true, smn_y_pred, 'SMN', smn_architecture, smn_final_val_loss)

    # Top-right: MLP scatter
    _draw_scatter(axes[0, 1], mlp_y_true, mlp_y_pred, 'MLP', mlp_architecture, mlp_final_val_loss)

    # Bottom-left: SMN losses
    ax = axes[1, 0]
    epochs = range(1, len(smn_train_losses) + 1)
    ax.plot(epochs, smn_train_losses, 'b-', label='Train', linewidth=1)
    ax.plot(epochs, smn_val_losses, 'r-', label='Val', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'SMN Losses\nfinal_val={smn_final_val_loss:.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Bottom-right: MLP losses
    ax = axes[1, 1]
    epochs = range(1, len(mlp_train_losses) + 1)
    ax.plot(epochs, mlp_train_losses, 'b-', label='Train', linewidth=1)
    ax.plot(epochs, mlp_val_losses, 'g-', label='Val', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'MLP Losses\nfinal_val={mlp_final_val_loss:.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.suptitle(figure_title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_2d_comparison(
    smn_x1_grid: np.ndarray,
    smn_x2_grid: np.ndarray,
    smn_y_true: np.ndarray,
    smn_y_pred: np.ndarray,
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    smn_architecture: str,
    mlp_architecture: str,
    smn_final_val_loss: float,
    mlp_final_val_loss: float,
    smn_train_losses: list[float],
    smn_val_losses: list[float],
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    output_path: Path,
    batch_size: int = 64,
) -> None:
    """Create 2D task comparison plot.

    Layout:
    [Top row] SMN: True surface, Pred surface, Error
    [Bottom row] MLP: True surface, Pred surface, Error
    Plus loss curves on the side.
    """
    n_side = int(np.sqrt(len(smn_y_true)))

    fig = plt.figure(figsize=(14, 10))

    # Reshape for plotting
    smn_y_true_2d = smn_y_true.reshape(n_side, n_side)
    smn_y_pred_2d = smn_y_pred.reshape(n_side, n_side)
    smn_error_2d = np.abs(smn_y_true_2d - smn_y_pred_2d)

    mlp_y_true_2d = mlp_y_true.reshape(n_side, n_side)
    mlp_y_pred_2d = mlp_y_pred.reshape(n_side, n_side)
    mlp_error_2d = np.abs(mlp_y_true_2d - mlp_y_pred_2d)

    x1 = smn_x1_grid[:, 0]
    x2 = smn_x2_grid[0, :]

    # SMN row (top)
    ax1 = fig.add_subplot(2, 4, 1)
    surf1 = ax1.contourf(x1, x2, smn_y_true_2d, levels=20, cmap='viridis')
    ax1.set_title(f'SMN True\n{smn_architecture}')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    plt.colorbar(surf1, ax=ax1)

    ax2 = fig.add_subplot(2, 4, 2)
    surf2 = ax2.contourf(x1, x2, smn_y_pred_2d, levels=20, cmap='viridis')
    ax2.set_title(f'SMN Pred\nval_loss={smn_final_val_loss:.4f}')
    ax2.set_xlabel('x1')
    plt.colorbar(surf2, ax=ax2)

    ax3 = fig.add_subplot(2, 4, 3)
    surf3 = ax3.contourf(x1, x2, smn_error_2d, levels=20, cmap='Reds')
    ax3.set_title('SMN Error')
    ax3.set_xlabel('x1')
    plt.colorbar(surf3, ax=ax3)

    # Loss curves for SMN
    ax4 = fig.add_subplot(2, 4, 4)
    epochs = range(1, len(smn_train_losses) + 1)
    ax4.plot(epochs, smn_train_losses, 'b-', label='Train', linewidth=1)
    ax4.plot(epochs, smn_val_losses, 'r-', label='Val', linewidth=1)
    ax4.set_title(f'SMN Losses')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # MLP row (bottom)
    ax5 = fig.add_subplot(2, 4, 5)
    surf5 = ax5.contourf(x1, x2, mlp_y_true_2d, levels=20, cmap='viridis')
    ax5.set_title(f'MLP True\n{mlp_architecture}')
    ax5.set_xlabel('x1')
    ax5.set_ylabel('x2')
    plt.colorbar(surf5, ax=ax5)

    ax6 = fig.add_subplot(2, 4, 6)
    surf6 = ax6.contourf(x1, x2, mlp_y_pred_2d, levels=20, cmap='viridis')
    ax6.set_title(f'MLP Pred\nval_loss={mlp_final_val_loss:.4f}')
    ax6.set_xlabel('x1')
    plt.colorbar(surf6, ax=ax6)

    ax7 = fig.add_subplot(2, 4, 7)
    surf7 = ax7.contourf(x1, x2, mlp_error_2d, levels=20, cmap='Reds')
    ax7.set_title('MLP Error')
    ax7.set_xlabel('x1')
    plt.colorbar(surf7, ax=ax7)

    # Loss curves for MLP
    ax8 = fig.add_subplot(2, 4, 8)
    epochs = range(1, len(mlp_train_losses) + 1)
    ax8.plot(epochs, mlp_train_losses, 'b-', label='Train', linewidth=1)
    ax8.plot(epochs, mlp_val_losses, 'g-', label='Val', linewidth=1)
    ax8.set_title(f'MLP Losses')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Loss')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale('log')

    fig.suptitle('2D Task: SMN vs MLP Comparison', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
