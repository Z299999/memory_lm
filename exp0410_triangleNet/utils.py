from __future__ import annotations

# This file contains shared helpers for reproducibility, directory creation,
# JSON output, and plot generation for the experiment scripts.

import json
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


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_loss_curve(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_prediction_plot(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_true, label="target")
    plt.plot(x, y_pred, label="prediction")
    plt.xlabel("normalized x")
    plt.ylabel("y")
    plt.title("Function Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
