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


def _condition_width_kind(condition_name: str) -> str:
    return "main" if condition_name == "full" else "aux"


_CONDITION_COLORS: dict[str, str] = {
    "full":        "#1f77b4",  # matplotlib C0 blue
    "sole_eye":    "#ff7f0e",  # matplotlib C1 orange
    "sole_speech": "#2ca02c",  # matplotlib C2 green
    "neither":     "#d62728",  # matplotlib C3 red
    "late_blind":  "#c05030",  # muted red-orange
    "late_mute":   "#bcbd22",  # olive yellow
    "blink":       "#17becf",  # teal
}

_LATE_TRANSITION_VLINE_COLORS: dict[str, str] = {
    "late_blind": _CONDITION_COLORS["late_blind"],
    "late_mute":  _CONDITION_COLORS["late_mute"],
}


def _build_rollout_panels(config: ExperimentConfig, *, has_continuous: bool) -> list[tuple[str, float]]:
    panels: list[tuple[str, float]] = [("short", 1.35), ("long", 1.35)]
    if has_continuous:
        panels.append(("continuous_long", 1.35))
    if config.eval_conditions:
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
    ncols = max(1, int(config.plot_training_timeline_ncols))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.plot_training_timeline_fig_width, max(3.4 * nrows, 6.8)),
        squeeze=False,
    )
    flat_axes = list(axes.flat)

    for ax, panel in zip(flat_axes, panels):
        global_steps = np.asarray(panel["global_step"], dtype=float)
        target = np.asarray(panel["target"], dtype=float)
        prediction = np.asarray(panel["prediction"], dtype=float)
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
    reset_evals: dict[str, dict] | None,
    reset_long_evals: dict[str, dict] | None,
    continuous_evals: dict[str, dict] | None,
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    has_reset = reset_evals is not None and "full" in reset_evals
    has_continuous = continuous_evals is not None and "full" in continuous_evals

    # Determine the primary reference rollout (for messages / norm panels)
    if has_reset:
        primary_short = _slice_rollout(reset_evals["full"], config.plot_short_steps)
        primary_long = _slice_rollout(reset_long_evals["full"] if reset_long_evals else None, config.plot_long_steps)
    else:
        primary_short = _slice_rollout(continuous_evals["full"] if has_continuous else None, config.plot_short_steps)
        primary_long = primary_short

    if primary_short is None:
        return

    message_source = (
        _slice_rollout(continuous_evals["full"], config.plot_message_steps)
        if has_continuous
        else _slice_rollout(primary_short, config.plot_message_steps)
    )

    short_steps = np.arange(len(primary_short["target"]))
    long_steps = np.arange(len(primary_long["target"])) if primary_long is not None else short_steps
    error_rollout = _slice_rollout(primary_short, config.plot_error_steps)
    error_steps = np.arange(len(error_rollout["target"]))
    message_steps = np.arange(len(message_source["target"]))

    panels = _build_rollout_panels(config, has_continuous=has_continuous)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(config.plot_diag_fig_width, config.plot_diag_fig_height),
        gridspec_kw={"height_ratios": [height for _, height in panels]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axis_by_panel = {panel_name: ax for ax, (panel_name, _) in zip(axes, panels)}

    def _build_transition_vlines(series: tuple[str, ...], num_steps: int) -> list[tuple[int, str, str]]:
        vlines = []
        for condition, color in _LATE_TRANSITION_VLINE_COLORS.items():
            if condition not in series:
                continue
            step = config.eval_late_blind_step if condition == "late_blind" else config.eval_late_mute_step
            if step < num_steps:
                vlines.append((step - 1, f"↓{condition}", color))
        if "blink" in series:
            color = _CONDITION_COLORS["blink"]
            if config.eval_blink_blind_start < num_steps:
                vlines.append((config.eval_blink_blind_start - 1, "↓blink", color))
            if config.eval_blink_blind_end < num_steps:
                vlines.append((config.eval_blink_blind_end - 1, "↑blink", color))
        return vlines

    def _plot_rollout_panel(ax: plt.Axes, evals: dict[str, dict] | None, num_steps: int, title: str) -> None:
        if evals is None:
            return
        ref = _slice_rollout(evals.get("full"), num_steps)
        if ref is None:
            return
        steps = np.arange(len(ref["target"]))
        ax.plot(
            steps,
            ref["target"].numpy(),
            label="target",
            color=config.plot_target_color,
            linestyle=config.plot_target_linestyle,
            linewidth=config.plot_target_linewidth,
        )
        for condition in config.eval_conditions:
            rollout = _slice_rollout(evals.get(condition), num_steps)
            if rollout is None:
                continue
            width_kind = _condition_width_kind(condition)
            ax.plot(
                steps,
                rollout["prediction"].numpy(),
                label=condition,
                linewidth=_linewidth(config, width_kind),
                color=_CONDITION_COLORS.get(condition),
                zorder=3 if condition == "full" else 2,
            )
        for vline_step, vline_label, vline_color in _build_transition_vlines(config.eval_conditions, num_steps):
            ax.axvline(vline_step, color=vline_color, linestyle="--", linewidth=1.0, alpha=0.55, label=vline_label)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(ax, ncol=config.plot_prediction_legend_ncols)

    _plot_rollout_panel(
        axis_by_panel["short"],
        reset_evals,
        config.plot_short_steps,
        "Reset short eval" if has_continuous else "Short rollout",
    )
    _plot_rollout_panel(
        axis_by_panel["long"],
        reset_long_evals,
        config.plot_long_steps,
        "Reset long eval" if has_continuous else "Long rollout",
    )
    if "continuous_long" in axis_by_panel:
        _plot_rollout_panel(
            axis_by_panel["continuous_long"],
            continuous_evals,
            config.plot_long_steps,
            "Continuous long eval",
        )

    if "error" in axis_by_panel:
        error_axis = axis_by_panel["error"]
        ref_evals = reset_evals if has_reset else continuous_evals
        if ref_evals is not None:
            ref = _slice_rollout(ref_evals.get("full"), config.plot_error_steps)
            if ref is not None:
                error_target = ref["target"].numpy()
                for condition in config.eval_conditions:
                    rollout = _slice_rollout(ref_evals.get(condition), config.plot_error_steps)
                    if rollout is None:
                        continue
                    width_kind = _condition_width_kind(condition)
                    error_axis.plot(
                        error_steps,
                        rollout["prediction"].numpy() - error_target,
                        label=f"{condition} error",
                        linewidth=_linewidth(config, width_kind),
                        color=_CONDITION_COLORS.get(condition),
                        zorder=3 if condition == "full" else 2,
                    )
        error_axis.axhline(0.0, color="black", linewidth=config.plot_zero_linewidth, alpha=0.7)
        for vline_step, vline_label, vline_color in _build_transition_vlines(config.eval_conditions, config.plot_error_steps):
            error_axis.axvline(vline_step, color=vline_color, linestyle="--", linewidth=1.0, alpha=0.55, label=vline_label)
        error_axis.set_title("Short rollout error")
        error_axis.set_ylabel("pred - target")
        error_axis.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(error_axis, ncol=config.plot_error_legend_ncols)

    if "messages" in axis_by_panel:
        messages = message_source["messages"].numpy()
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
        message_axis.set_title("Full model language channel traces")
        message_axis.set_ylabel("message")
        message_axis.grid(True, alpha=config.plot_grid_alpha)

    if "message_norm" in axis_by_panel:
        norm_axis = axis_by_panel["message_norm"]
        norm_axis.plot(
            message_steps,
            message_source["message_norm"].numpy(),
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
