from __future__ import annotations

# This file contains shared helpers for reproducibility, directory creation,
# and the single comparison plot used by run.py.

import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_four_panel_plot(
    tmn_train_losses: list[float],
    tmn_val_losses: list[float],
    tmn_x: np.ndarray,
    tmn_y_true: np.ndarray,
    tmn_y_pred: np.ndarray,
    tmn_architecture: str,
    mlp_train_losses: list[float],
    mlp_val_losses: list[float],
    mlp_x: np.ndarray,
    mlp_y_true: np.ndarray,
    mlp_y_pred: np.ndarray,
    mlp_architecture: str,
    figure_title: str,
    output_path: Path,
) -> None:
    # This is the main comparison figure used by run.py.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(figure_title, fontsize=14)

    axes[0, 0].plot(tmn_train_losses, label="train")
    axes[0, 0].plot(tmn_val_losses, label="val")
    axes[0, 0].set_title(f"TMN Loss Curve\n{tmn_architecture}")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(tmn_x, tmn_y_true, label="target")
    axes[0, 1].plot(tmn_x, tmn_y_pred, label="prediction")
    axes[0, 1].set_title(f"TMN Prediction\n{tmn_architecture}")
    axes[0, 1].set_xlabel("normalized x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].legend()

    axes[1, 0].plot(mlp_train_losses, label="train")
    axes[1, 0].plot(mlp_val_losses, label="val")
    axes[1, 0].set_title(f"MLP Loss Curve\n{mlp_architecture}")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MSE Loss")
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
