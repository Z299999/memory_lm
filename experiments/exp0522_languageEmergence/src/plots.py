"""Plotting utilities for exp0522."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Keep matplotlib/font caches inside the experiment workspace.
_RUNTIME_CACHE = Path(__file__).resolve().parents[1] / ".runtime_cache"
_MPL_CACHE = _RUNTIME_CACHE / "mpl"
_XDG_CACHE = _RUNTIME_CACHE / "xdg"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .config import ExperimentConfig
except ImportError:  # pragma: no cover - script mode
    from config import ExperimentConfig


def _slice_rollout(rollout: dict[str, Any], num_steps: int) -> dict[str, Any]:
    sliced = dict(rollout)
    for key in ("phase", "target", "prediction", "messages", "message_norm"):
        value = rollout.get(key)
        if value is not None:
            sliced[key] = value[:num_steps]
    return sliced


def plot_training_curves(
    *,
    full_history: list[dict[str, float]],
    baseline_history: list[dict[str, float]],
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    fig, ax = plt.subplots(
        figsize=(config.plot_training_fig_width, config.plot_training_fig_height),
    )
    epochs = [row["epoch"] for row in full_history]
    ax.plot(
        epochs,
        [row["train_loss"] for row in full_history],
        label="full train",
        linewidth=config.plot_series_linewidth,
    )
    ax.plot(
        epochs,
        [row["val_loss"] for row in full_history],
        label="full val",
        linewidth=config.plot_aux_linewidth,
    )
    ax.plot(
        epochs,
        [row["train_loss"] for row in baseline_history],
        label="baseline train",
        linewidth=config.plot_series_linewidth,
    )
    ax.plot(
        epochs,
        [row["val_loss"] for row in baseline_history],
        label="baseline val",
        linewidth=config.plot_aux_linewidth,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("exp0522 training curves")
    ax.grid(True, alpha=config.plot_grid_alpha)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)


def plot_rollout_diagnostics(
    *,
    short_full: dict[str, Any],
    short_baseline: dict[str, Any],
    short_mute_deaf: dict[str, Any],
    long_full: dict[str, Any],
    long_baseline: dict[str, Any],
    long_mute_deaf: dict[str, Any],
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    short_full = _slice_rollout(short_full, config.plot_short_steps)
    short_baseline = _slice_rollout(short_baseline, config.plot_short_steps)
    short_mute_deaf = _slice_rollout(short_mute_deaf, config.plot_short_steps)
    long_full = _slice_rollout(long_full, config.plot_long_steps)
    long_baseline = _slice_rollout(long_baseline, config.plot_long_steps)
    long_mute_deaf = _slice_rollout(long_mute_deaf, config.plot_long_steps)
    error_full = _slice_rollout(short_full, config.plot_error_steps)
    error_baseline = _slice_rollout(short_baseline, config.plot_error_steps)
    error_mute_deaf = _slice_rollout(short_mute_deaf, config.plot_error_steps)
    message_rollout = _slice_rollout(short_full, config.plot_message_steps)

    short_steps = np.arange(len(short_full["target"]))
    long_steps = np.arange(len(long_full["target"]))
    error_steps = np.arange(len(error_full["target"]))
    message_steps = np.arange(len(message_rollout["target"]))

    short_target = short_full["target"].numpy()
    long_target = long_full["target"].numpy()
    short_full_pred = short_full["prediction"].numpy()
    short_baseline_pred = short_baseline["prediction"].numpy()
    short_mute_pred = short_mute_deaf["prediction"].numpy()
    long_full_pred = long_full["prediction"].numpy()
    long_baseline_pred = long_baseline["prediction"].numpy()
    long_mute_pred = long_mute_deaf["prediction"].numpy()

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(config.plot_diag_fig_width, config.plot_diag_fig_height),
        gridspec_kw={"height_ratios": [1.35, 1.35, 1.0, 1.0, 0.85]},
    )

    axes[0].plot(
        short_steps,
        short_target,
        label="target",
        linewidth=config.plot_target_linewidth,
        color="black",
    )
    axes[0].plot(short_steps, short_full_pred, label="full", linewidth=config.plot_series_linewidth)
    axes[0].plot(short_steps, short_baseline_pred, label="baseline", linewidth=config.plot_aux_linewidth)
    axes[0].plot(short_steps, short_mute_pred, label="mute_deaf", linewidth=config.plot_aux_linewidth)
    axes[0].set_title("Short rollout")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=config.plot_grid_alpha)
    axes[0].legend(loc="upper right", ncol=config.plot_prediction_legend_ncols)

    axes[1].plot(
        long_steps,
        long_target,
        label="target",
        linewidth=config.plot_target_linewidth,
        color="black",
    )
    axes[1].plot(long_steps, long_full_pred, label="full", linewidth=config.plot_series_linewidth)
    axes[1].plot(long_steps, long_baseline_pred, label="baseline", linewidth=config.plot_aux_linewidth)
    axes[1].plot(long_steps, long_mute_pred, label="mute_deaf", linewidth=config.plot_aux_linewidth)
    axes[1].set_title("Long rollout")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=config.plot_grid_alpha)
    axes[1].legend(loc="upper right", ncol=config.plot_prediction_legend_ncols)

    error_target = error_full["target"].numpy()
    axes[2].plot(
        error_steps,
        error_full["prediction"].numpy() - error_target,
        label="full error",
        linewidth=config.plot_series_linewidth,
    )
    axes[2].plot(
        error_steps,
        error_baseline["prediction"].numpy() - error_target,
        label="baseline error",
        linewidth=config.plot_aux_linewidth,
    )
    axes[2].plot(
        error_steps,
        error_mute_deaf["prediction"].numpy() - error_target,
        label="mute_deaf error",
        linewidth=config.plot_aux_linewidth,
    )
    axes[2].axhline(0.0, color="black", linewidth=config.plot_zero_linewidth, alpha=0.7)
    axes[2].set_title("Short rollout error")
    axes[2].set_ylabel("pred - target")
    axes[2].grid(True, alpha=config.plot_grid_alpha)
    axes[2].legend(loc="upper right", ncol=config.plot_error_legend_ncols)

    messages = message_rollout["messages"].numpy()
    if messages.shape[1] > 0:
        for idx in range(messages.shape[1]):
            axes[3].plot(
                message_steps,
                messages[:, idx],
                linewidth=config.plot_aux_linewidth,
                label=f"m{idx}",
            )
        axes[3].legend(loc="upper right", ncol=config.plot_message_legend_ncols)
    axes[3].set_title("Language channel traces")
    axes[3].set_ylabel("message")
    axes[3].grid(True, alpha=config.plot_grid_alpha)

    axes[4].plot(
        message_steps,
        message_rollout["message_norm"].numpy(),
        color="black",
        linewidth=config.plot_series_linewidth,
    )
    axes[4].set_title("Message norm")
    axes[4].set_xlabel("step")
    axes[4].set_ylabel("||m_t||")
    axes[4].grid(True, alpha=config.plot_grid_alpha)

    fig.suptitle("exp0522 rollout diagnostics", fontsize=config.plot_title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)
