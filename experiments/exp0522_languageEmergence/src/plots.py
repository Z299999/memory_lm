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


_TRAINING_SERIES_SPECS = {
    "full_train": ("full train", "train_loss", "full", "main"),
    "full_val": ("full val", "val_loss", "full", "aux"),
    "baseline_train": ("baseline train", "train_loss", "baseline", "main"),
    "baseline_val": ("baseline val", "val_loss", "baseline", "aux"),
}

_ROLLOUT_SERIES_SPECS = {
    "target": ("target", "target", None, "target"),
    "full": ("full", "prediction", None, "main"),
    "baseline": ("baseline", "prediction", None, "aux"),
    "mute_deaf": ("mute_deaf", "prediction", None, "aux"),
}

_ERROR_SERIES_SPECS = {
    "full": ("full error", "prediction", "main"),
    "baseline": ("baseline error", "prediction", "aux"),
    "mute_deaf": ("mute_deaf error", "prediction", "aux"),
}


def _slice_rollout(rollout: dict[str, Any] | None, num_steps: int) -> dict[str, Any] | None:
    if rollout is None:
        return None
    sliced = dict(rollout)
    for key in ("phase", "target", "prediction", "messages", "message_norm"):
        value = rollout.get(key)
        if value is not None:
            sliced[key] = value[:num_steps]
    return sliced


def _linewidth(config: ExperimentConfig, width_kind: str) -> float:
    if width_kind == "target":
        return config.plot_target_linewidth
    if width_kind == "main":
        return config.plot_series_linewidth
    return config.plot_aux_linewidth


def _linestyle(config: ExperimentConfig, width_kind: str) -> str:
    if width_kind == "target":
        return config.plot_target_linestyle
    return "-"


def _build_rollout_panels(config: ExperimentConfig) -> list[tuple[str, float]]:
    panels: list[tuple[str, float]] = [("short", 1.35), ("long", 1.35)]
    if config.plot_error_series:
        panels.append(("error", 1.0))
    if config.plot_show_message_traces:
        panels.append(("messages", 1.0))
    if config.plot_show_message_norm:
        panels.append(("message_norm", 0.85))
    return panels


def _maybe_add_legend(ax: plt.Axes, *, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", ncol=ncol)


def plot_training_curves(
    *,
    full_history: list[dict[str, float]],
    baseline_history: list[dict[str, float]] | None,
    output_path: Path,
    config: ExperimentConfig,
    full_label: str = "full",
    baseline_label: str = "baseline",
) -> None:
    fig, ax = plt.subplots(
        figsize=(config.plot_training_fig_width, config.plot_training_fig_height),
    )
    epochs = [row["epoch"] for row in full_history]
    histories = {
        "full": full_history,
        "baseline": baseline_history,
    }
    for series_name in config.plot_training_series:
        label, metric_key, history_key, width_kind = _TRAINING_SERIES_SPECS[series_name]
        history = histories[history_key]
        if history is None:
            continue
        if history_key == "full":
            display_label = label.replace("full", full_label, 1)
        else:
            display_label = label.replace("baseline", baseline_label, 1)
        ax.plot(
            epochs,
            [row[metric_key] for row in history],
            label=display_label,
            linewidth=_linewidth(config, width_kind),
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("exp0522 training curves")
    ax.grid(True, alpha=config.plot_grid_alpha)
    _maybe_add_legend(ax, ncol=min(len(config.plot_training_series), 4))
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)


def plot_training_timeline(
    *,
    timeline_payload: dict[str, Any],
    output_path: Path,
    config: ExperimentConfig,
    full_label: str = "full",
) -> None:
    panels = timeline_payload.get("panels", [])
    if not panels:
        return

    n_panels = len(panels)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.plot_training_fig_width, max(3.4 * nrows, 6.8)),
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for ax, panel in zip(flat_axes, panels):
        global_steps = np.asarray(panel["global_step"], dtype=float)
        target = np.asarray(panel["target"], dtype=float)
        prediction = np.asarray(panel["prediction"], dtype=float)
        update_steps = [int(step) for step in panel.get("update_steps", [])]

        ax.plot(
            global_steps,
            target,
            color=config.plot_target_color,
            linestyle=config.plot_target_linestyle,
            linewidth=config.plot_target_linewidth,
            label="target",
        )
        ax.plot(
            global_steps,
            prediction,
            linewidth=config.plot_series_linewidth,
            label=full_label,
        )
        for update_step in update_steps:
            ax.axvline(
                update_step,
                color="#bdbdbd",
                linestyle=":",
                linewidth=config.plot_zero_linewidth,
                alpha=0.9,
            )
        ax.set_title(
            f"t={int(panel['start_step'])}..{int(panel['end_step'])}",
            fontsize=max(config.plot_title_fontsize - 2, 10),
        )
        ax.set_xlabel("global step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(ax, ncol=2)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("exp0522 training timeline", fontsize=config.plot_title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)


def plot_rollout_diagnostics(
    *,
    short_full: dict[str, Any],
    short_baseline: dict[str, Any] | None,
    short_mute_deaf: dict[str, Any] | None,
    long_full: dict[str, Any],
    long_baseline: dict[str, Any] | None,
    long_mute_deaf: dict[str, Any] | None,
    continuous_long_full: dict[str, Any] | None,
    continuous_long_baseline: dict[str, Any] | None,
    continuous_long_mute_deaf: dict[str, Any] | None,
    output_path: Path,
    config: ExperimentConfig,
    full_label: str = "full",
    baseline_label: str = "baseline",
    mute_label: str = "mute_deaf",
) -> None:
    short_full = _slice_rollout(short_full, config.plot_short_steps)
    short_baseline = _slice_rollout(short_baseline, config.plot_short_steps)
    short_mute_deaf = _slice_rollout(short_mute_deaf, config.plot_short_steps)
    long_full = _slice_rollout(long_full, config.plot_long_steps)
    long_baseline = _slice_rollout(long_baseline, config.plot_long_steps)
    long_mute_deaf = _slice_rollout(long_mute_deaf, config.plot_long_steps)
    if continuous_long_full is not None:
        continuous_long_full = _slice_rollout(continuous_long_full, config.plot_long_steps)
        continuous_long_baseline = _slice_rollout(continuous_long_baseline, config.plot_long_steps)
        continuous_long_mute_deaf = _slice_rollout(continuous_long_mute_deaf, config.plot_long_steps)
    error_full = _slice_rollout(short_full, config.plot_error_steps)
    error_baseline = _slice_rollout(short_baseline, config.plot_error_steps)
    error_mute_deaf = _slice_rollout(short_mute_deaf, config.plot_error_steps)
    message_source = continuous_long_full if continuous_long_full is not None else short_full
    message_rollout = _slice_rollout(message_source, config.plot_message_steps)

    short_steps = np.arange(len(short_full["target"]))
    long_steps = np.arange(len(long_full["target"]))
    error_steps = np.arange(len(error_full["target"]))
    message_steps = np.arange(len(message_rollout["target"]))

    short_target = short_full["target"].numpy()
    long_target = long_full["target"].numpy()
    short_full_pred = short_full["prediction"].numpy()
    short_baseline_pred = short_baseline["prediction"].numpy() if short_baseline is not None else None
    short_mute_pred = short_mute_deaf["prediction"].numpy() if short_mute_deaf is not None else None
    long_full_pred = long_full["prediction"].numpy()
    long_baseline_pred = long_baseline["prediction"].numpy() if long_baseline is not None else None
    long_mute_pred = long_mute_deaf["prediction"].numpy() if long_mute_deaf is not None else None

    panels = _build_rollout_panels(config)
    if continuous_long_full is not None:
        panels.insert(2, ("continuous_long", 1.35))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(config.plot_diag_fig_width, config.plot_diag_fig_height),
        gridspec_kw={"height_ratios": [height for _, height in panels]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axis_by_panel = {panel_name: ax for ax, (panel_name, _) in zip(axes, panels)}

    rollout_predictions = {
        "target": short_target,
        "full": short_full_pred,
        "baseline": short_baseline_pred,
        "mute_deaf": short_mute_pred,
    }
    short_axis = axis_by_panel["short"]
    for series_name in config.plot_rollout_series:
        if rollout_predictions[series_name] is None:
            continue
        label, _, color, width_kind = _ROLLOUT_SERIES_SPECS[series_name]
        if series_name == "full":
            display_label = full_label
        elif series_name == "baseline":
            display_label = baseline_label
        elif series_name == "mute_deaf":
            display_label = mute_label
        else:
            display_label = label
        short_axis.plot(
            short_steps,
            rollout_predictions[series_name],
            label=display_label,
            color=config.plot_target_color if series_name == "target" else color,
            linestyle=_linestyle(config, width_kind),
            linewidth=_linewidth(config, width_kind),
        )
    short_axis.set_title("Reset short eval" if continuous_long_full is not None else "Short rollout")
    short_axis.set_ylabel("value")
    short_axis.grid(True, alpha=config.plot_grid_alpha)
    _maybe_add_legend(short_axis, ncol=config.plot_prediction_legend_ncols)

    long_predictions = {
        "target": long_target,
        "full": long_full_pred,
        "baseline": long_baseline_pred,
        "mute_deaf": long_mute_pred,
    }
    long_axis = axis_by_panel["long"]
    for series_name in config.plot_rollout_series:
        if long_predictions[series_name] is None:
            continue
        label, _, color, width_kind = _ROLLOUT_SERIES_SPECS[series_name]
        if series_name == "full":
            display_label = full_label
        elif series_name == "baseline":
            display_label = baseline_label
        elif series_name == "mute_deaf":
            display_label = mute_label
        else:
            display_label = label
        long_axis.plot(
            long_steps,
            long_predictions[series_name],
            label=display_label,
            color=config.plot_target_color if series_name == "target" else color,
            linestyle=_linestyle(config, width_kind),
            linewidth=_linewidth(config, width_kind),
        )
    long_axis.set_title("Reset long eval" if continuous_long_full is not None else "Long rollout")
    long_axis.set_ylabel("value")
    long_axis.grid(True, alpha=config.plot_grid_alpha)
    _maybe_add_legend(long_axis, ncol=config.plot_prediction_legend_ncols)

    if "continuous_long" in axis_by_panel:
        continuous_target = continuous_long_full["target"].numpy()
        continuous_steps = np.arange(len(continuous_target))
        continuous_predictions = {
            "target": continuous_target,
            "full": continuous_long_full["prediction"].numpy(),
            "baseline": None if continuous_long_baseline is None else continuous_long_baseline["prediction"].numpy(),
            "mute_deaf": None if continuous_long_mute_deaf is None else continuous_long_mute_deaf["prediction"].numpy(),
        }
        continuous_axis = axis_by_panel["continuous_long"]
        for series_name in config.plot_rollout_series:
            if continuous_predictions[series_name] is None:
                continue
            label, _, color, width_kind = _ROLLOUT_SERIES_SPECS[series_name]
            if series_name == "full":
                display_label = full_label
            elif series_name == "baseline":
                display_label = baseline_label
            elif series_name == "mute_deaf":
                display_label = mute_label
            else:
                display_label = label
            continuous_axis.plot(
                continuous_steps,
                continuous_predictions[series_name],
                label=display_label,
                color=config.plot_target_color if series_name == "target" else color,
                linestyle=_linestyle(config, width_kind),
                linewidth=_linewidth(config, width_kind),
            )
        continuous_axis.set_title("Continuous long eval")
        continuous_axis.set_ylabel("value")
        continuous_axis.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(continuous_axis, ncol=config.plot_prediction_legend_ncols)

    if "error" in axis_by_panel:
        error_target = error_full["target"].numpy()
        error_rollouts = {
            "full": error_full,
            "baseline": error_baseline,
            "mute_deaf": error_mute_deaf,
        }
        error_axis = axis_by_panel["error"]
        for series_name in config.plot_error_series:
            if error_rollouts[series_name] is None:
                continue
            label, prediction_key, width_kind = _ERROR_SERIES_SPECS[series_name]
            if series_name == "full":
                display_label = label.replace("full", full_label, 1)
            elif series_name == "baseline":
                display_label = label.replace("baseline", baseline_label, 1)
            else:
                display_label = label.replace("mute_deaf", mute_label, 1)
            error_axis.plot(
                error_steps,
                error_rollouts[series_name][prediction_key].numpy() - error_target,
                label=display_label,
                linewidth=_linewidth(config, width_kind),
            )
        error_axis.axhline(0.0, color="black", linewidth=config.plot_zero_linewidth, alpha=0.7)
        error_axis.set_title("Short rollout error")
        error_axis.set_ylabel("pred - target")
        error_axis.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(error_axis, ncol=config.plot_error_legend_ncols)

    if "messages" in axis_by_panel:
        messages = message_rollout["messages"].numpy()
        message_axis = axis_by_panel["messages"]
        if messages.shape[1] > 0:
            for idx in range(messages.shape[1]):
                message_axis.plot(
                    message_steps,
                    messages[:, idx],
                    linewidth=config.plot_aux_linewidth,
                    label=f"m{idx}",
                )
            _maybe_add_legend(message_axis, ncol=config.plot_message_legend_ncols)
        message_axis.set_title("Language channel traces")
        message_axis.set_ylabel("message")
        message_axis.grid(True, alpha=config.plot_grid_alpha)

    if "message_norm" in axis_by_panel:
        norm_axis = axis_by_panel["message_norm"]
        norm_axis.plot(
            message_steps,
            message_rollout["message_norm"].numpy(),
            color="black",
            linewidth=config.plot_series_linewidth,
        )
        norm_axis.set_title("Message norm")
        norm_axis.set_ylabel("||m_t||")
        norm_axis.grid(True, alpha=config.plot_grid_alpha)

    axes[-1].set_xlabel("step")

    fig.suptitle("exp0522 rollout diagnostics", fontsize=config.plot_title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)
