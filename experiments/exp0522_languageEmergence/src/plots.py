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
    from .config import ExperimentConfig, parse_condition
except ImportError:  # pragma: no cover - script mode
    from config import ExperimentConfig, parse_condition


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


_ROLLOUT_DIAG_BASE_HEIGHT_UNITS = (4 * 1.35) + 1.0 + 1.0 + 0.85


def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


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
    base, _ = parse_condition(condition_name)
    return "main" if base == "full" else "aux"


_CONDITION_COLORS: dict[str, str] = {
    "full":        "#1f77b4",  # matplotlib C0 blue
    "sole_eye":    "#17becf",  # cyan
    "sole_speech": "#2ca02c",  # matplotlib C2 green
    "neither":     "#d62728",  # matplotlib C3 red
    "late_blind":  "#c05030",  # muted red-orange
    "late_mute":   "#bcbd22",  # olive yellow
    "blink":       "#ff7f0e",  # vivid orange
    "dim":         "#f4a261",  # lighter orange for soft blink
    "stutter":     "#e377c2",  # pink
}

def _resolve_top_rollout_panels(
    config: ExperimentConfig,
    *,
    has_reset: bool,
    has_continuous: bool,
) -> list[str]:
    reset_panels = ["reset_short", "reset_long"] if has_reset else []
    continuous_panels = ["continuous_short", "continuous_long"] if has_continuous else []

    if config.plot_rollout_top_mode == "all_available":
        return reset_panels + continuous_panels

    if config.plot_rollout_top_mode == "match_eval":
        if config.eval_phase_mode == "reset":
            return reset_panels
        if config.eval_phase_mode == "continuous":
            return continuous_panels
        return reset_panels + continuous_panels

    if config.train_phase_mode == "continuous" and continuous_panels:
        return continuous_panels
    if config.train_phase_mode == "reset" and reset_panels:
        return reset_panels
    return reset_panels + continuous_panels


def _resolve_aux_horizons(config: ExperimentConfig) -> list[str]:
    if config.plot_aux_horizon == "both":
        return ["short", "long"]
    return [config.plot_aux_horizon]


def _build_rollout_panels(
    config: ExperimentConfig,
    *,
    top_panels: list[str],
    aux_horizons: list[str],
    has_language_panels: bool,
    has_eval_conditions: bool,
) -> list[tuple[str, float]]:
    panels: list[tuple[str, float]] = []
    panels.extend((panel_name, 1.35) for panel_name in top_panels)
    if has_eval_conditions:
        panels.extend((f"error_{horizon}", 1.0) for horizon in aux_horizons)
    if has_language_panels and config.plot_show_message_traces:
        panels.extend((f"messages_{horizon}", 1.0) for horizon in aux_horizons)
    if has_language_panels and config.plot_show_message_norm:
        panels.extend((f"message_norm_{horizon}", 0.85) for horizon in aux_horizons)
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
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(config.plot_training_fig_width, config.plot_training_fig_height + 2.0),
        gridspec_kw={"height_ratios": [1.65, 1.0]},
        sharex=True,
    )
    loss_axis, steps_axis = axes
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
        loss_axis.plot(
            epochs,
            [row[metric_key] for row in history],
            label=display_label,
            linewidth=_linewidth(config, width_kind),
        )
    loss_axis.set_yscale("log")
    loss_axis.set_ylabel("MSE")
    loss_axis.set_title("exp0522 training curves")
    loss_axis.grid(True, alpha=config.plot_grid_alpha)
    _maybe_add_legend(loss_axis, ncol=min(len(config.plot_training_series), 4))

    train_steps = [row["train_steps"] for row in full_history]
    rolling_window = max(3, min(25, len(train_steps) // 10 if len(train_steps) >= 10 else len(train_steps)))
    steps_axis.plot(
        epochs,
        train_steps,
        color="#1f77b4",
        linewidth=config.plot_series_linewidth,
        alpha=0.8,
        label="train steps",
    )
    if len(train_steps) >= 2:
        steps_axis.plot(
            epochs,
            _rolling_mean(train_steps, rolling_window),
            color="#0d3b66",
            linewidth=config.plot_aux_linewidth,
            linestyle="--",
            label=f"rolling mean ({rolling_window})",
        )
    steps_axis.set_xlabel("Epoch")
    steps_axis.set_ylabel("Train steps")
    steps_axis.set_title("Realized window length")
    steps_axis.grid(True, alpha=config.plot_grid_alpha)
    _maybe_add_legend(steps_axis, ncol=2)
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

    if config.plot_training_timeline_shared_ylim and panels:
        all_vals = np.concatenate([
            np.concatenate([panel["target"], panel["prediction"]])
            for panel in panels
        ])
        _y_margin = (all_vals.max() - all_vals.min()) * 0.05 + 1e-6
        _shared_ylim = (float(all_vals.min() - _y_margin), float(all_vals.max() + _y_margin))
    else:
        _shared_ylim = None

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
        for update_step in panel.get("update_steps", []):
            ax.axvline(
                float(update_step),
                color="#c7c7c7",
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
            )
        ax.set_title(
            f"t={int(panel['start_step'])}..{int(panel['end_step'])}",
            fontsize=max(config.plot_title_fontsize - 2, 10),
        )
        if _shared_ylim is not None:
            ax.set_ylim(_shared_ylim)
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

    top_panels = _resolve_top_rollout_panels(
        config,
        has_reset=has_reset,
        has_continuous=has_continuous,
    )
    if not top_panels:
        return

    def _panel_rollout(panel_name: str) -> tuple[dict[str, dict] | None, int, str]:
        if panel_name == "reset_short":
            return reset_evals, config.plot_short_steps, "Reset short eval"
        if panel_name == "reset_long":
            return reset_long_evals, config.plot_long_steps, "Reset long eval"
        if panel_name == "continuous_short":
            return continuous_evals, config.plot_short_steps, "Continuous short eval"
        if panel_name == "continuous_long":
            return continuous_evals, config.plot_long_steps, "Continuous long eval"
        raise ValueError(f"Unknown rollout panel name: {panel_name}")

    def _aux_source(horizon: str, kind: str) -> tuple[dict[str, dict] | None, int, str]:
        if horizon == "short":
            if has_continuous:
                evals = continuous_evals
                phase_label = "Continuous"
            else:
                evals = reset_evals
                phase_label = "Reset"
            num_steps = config.plot_error_steps if kind == "error" else config.plot_message_steps
            return evals, num_steps, phase_label
        if has_continuous:
            return continuous_evals, config.plot_long_steps, "Continuous"
        return reset_long_evals, config.plot_long_steps, "Reset"

    has_language_panels = False
    if config.plot_show_message_traces or config.plot_show_message_norm:
        long_message_source, long_message_steps, _ = _aux_source("long", "messages")
        short_message_source, short_message_steps, _ = _aux_source("short", "messages")
        probe = _slice_rollout(
            (long_message_source or short_message_source)["full"]
            if (long_message_source or short_message_source) is not None
            else None,
            long_message_steps if long_message_source is not None else short_message_steps,
        )
        has_language_panels = bool(probe is not None and probe["messages"].shape[1] > 0)

    aux_horizons = _resolve_aux_horizons(config)
    panels = _build_rollout_panels(
        config,
        top_panels=top_panels,
        aux_horizons=aux_horizons,
        has_language_panels=has_language_panels,
        has_eval_conditions=bool(config.eval_conditions),
    )
    total_height_units = sum(height for _, height in panels)
    fig_height = max(
        8.0,
        config.plot_diag_fig_height * (total_height_units / _ROLLOUT_DIAG_BASE_HEIGHT_UNITS),
    )
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(config.plot_diag_fig_width, fig_height),
        gridspec_kw={"height_ratios": [height for _, height in panels]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axis_by_panel = {panel_name: ax for ax, (panel_name, _) in zip(axes, panels)}

    def _build_transition_vlines(
        series: tuple[str, ...],
        num_steps: int,
        *,
        start_step: int = 0,
    ) -> list[tuple[int, str, str]]:
        vlines = []
        for condition_str in series:
            base, params = parse_condition(condition_str)
            color = _CONDITION_COLORS.get(base)
            if color is None:
                continue
            if base in ("late_blind", "late_mute") and len(params) == 1:
                step = params[0]
                if step < num_steps:
                    vlines.append((start_step + step - 1, f"↓{base}", color))
            elif base in ("blink", "stutter") and len(params) == 2:
                s, e = params[0], params[1]
                if s < num_steps:
                    vlines.append((start_step + s - 1, f"↓{base}", color))
                if e < num_steps:
                    vlines.append((start_step + e - 1, f"↑{base}", color))
            elif base == "dim" and len(params) == 3:
                s, e, pct = params
                if s < num_steps:
                    vlines.append((start_step + s - 1, f"↓dim({pct}%)", color))
                if e < num_steps:
                    vlines.append((start_step + e - 1, f"↑dim({pct}%)", color))
        return vlines

    def _plot_rollout_panel(ax: plt.Axes, evals: dict[str, dict] | None, num_steps: int, title: str) -> None:
        if evals is None:
            return
        ref = _slice_rollout(evals.get("full"), num_steps)
        if ref is None:
            return
        start_step = int(ref.get("start_step", 0))
        steps = np.arange(len(ref["target"])) + start_step
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
            base, _ = parse_condition(condition)
            width_kind = _condition_width_kind(condition)
            ax.plot(
                steps,
                rollout["prediction"].numpy(),
                label=condition,
                linewidth=_linewidth(config, width_kind),
                color=_CONDITION_COLORS.get(base),
                zorder=3 if base == "full" else 2,
            )
        for vline_step, vline_label, vline_color in _build_transition_vlines(
            config.eval_conditions,
            num_steps,
            start_step=start_step,
        ):
            ax.axvline(vline_step, color=vline_color, linestyle="--", linewidth=1.0, alpha=0.55, label=vline_label)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.grid(True, alpha=config.plot_grid_alpha)
        if len(steps) > 0:
            ax.set_xlim(steps[0], steps[-1])
        ax.margins(x=0)
        _maybe_add_legend(ax, ncol=config.plot_prediction_legend_ncols)

    for panel_name in top_panels:
        panel_axis = axis_by_panel.get(panel_name)
        if panel_axis is None:
            continue
        evals, num_steps, title = _panel_rollout(panel_name)
        _plot_rollout_panel(panel_axis, evals, num_steps, title)

    for horizon in aux_horizons:
        error_axis = axis_by_panel.get(f"error_{horizon}")
        if error_axis is None:
            continue
        ref_evals, error_steps_count, phase_label = _aux_source(horizon, "error")
        ref = None
        if ref_evals is not None:
            ref = _slice_rollout(ref_evals.get("full"), error_steps_count)
            if ref is not None:
                start_step = int(ref.get("start_step", 0))
                error_steps = np.arange(len(ref["target"])) + start_step
                error_target = ref["target"].numpy()
                for condition in config.eval_conditions:
                    rollout = _slice_rollout(ref_evals.get(condition), error_steps_count)
                    if rollout is None:
                        continue
                    base, _ = parse_condition(condition)
                    width_kind = _condition_width_kind(condition)
                    error_axis.plot(
                        error_steps,
                        rollout["prediction"].numpy() - error_target,
                        label=f"{condition} error",
                        linewidth=_linewidth(config, width_kind),
                        color=_CONDITION_COLORS.get(base),
                        zorder=3 if base == "full" else 2,
                    )
        error_axis.axhline(0.0, color="black", linewidth=config.plot_zero_linewidth, alpha=0.7)
        for vline_step, vline_label, vline_color in _build_transition_vlines(
            config.eval_conditions,
            error_steps_count,
            start_step=int(ref.get("start_step", 0)) if ref_evals is not None and ref is not None else 0,
        ):
            error_axis.axvline(vline_step, color=vline_color, linestyle="--", linewidth=1.0, alpha=0.55, label=vline_label)
        error_axis.set_title(f"{phase_label} {horizon} error")
        error_axis.set_ylabel("pred - target")
        error_axis.grid(True, alpha=config.plot_grid_alpha)
        if ref_evals is not None and ref is not None and len(error_steps) > 0:
            error_axis.set_xlim(error_steps[0], error_steps[-1])
        error_axis.margins(x=0)
        _maybe_add_legend(error_axis, ncol=config.plot_error_legend_ncols)

    for horizon in aux_horizons:
        message_axis = axis_by_panel.get(f"messages_{horizon}")
        if message_axis is None:
            continue
        ref_evals, message_steps_count, phase_label = _aux_source(horizon, "messages")
        message_source = _slice_rollout(ref_evals.get("full") if ref_evals is not None else None, message_steps_count)
        if message_source is None:
            continue
        messages = message_source["messages"].numpy()
        message_steps = np.arange(len(message_source["target"])) + int(message_source.get("start_step", 0))
        if messages.shape[1] > 0:
            for idx in range(messages.shape[1]):
                label = f"m{idx}"
                message_axis.plot(
                    message_steps,
                    messages[:, idx],
                    linewidth=config.plot_aux_linewidth,
                    label=label,
                )
            _maybe_add_legend(message_axis, ncol=config.plot_message_legend_ncols)
        message_axis.set_title(f"{phase_label} {horizon} language channel traces")
        message_axis.set_ylabel("message")
        message_axis.grid(True, alpha=config.plot_grid_alpha)
        if len(message_steps) > 0:
            message_axis.set_xlim(message_steps[0], message_steps[-1])
        message_axis.margins(x=0)

    for horizon in aux_horizons:
        norm_axis = axis_by_panel.get(f"message_norm_{horizon}")
        if norm_axis is None:
            continue
        ref_evals, message_steps_count, phase_label = _aux_source(horizon, "messages")
        message_source = _slice_rollout(ref_evals.get("full") if ref_evals is not None else None, message_steps_count)
        if message_source is None:
            continue
        message_steps = np.arange(len(message_source["target"])) + int(message_source.get("start_step", 0))
        norm = message_source["message_norm"].numpy()
        norm_axis.plot(
            message_steps,
            norm,
            color="black",
            linewidth=config.plot_series_linewidth,
        )
        norm_axis.set_title(f"{phase_label} {horizon} message norm")
        norm_axis.set_ylabel("||m_t||")
        norm_axis.grid(True, alpha=config.plot_grid_alpha)
        if len(message_steps) > 0:
            norm_axis.set_xlim(message_steps[0], message_steps[-1])
        norm_axis.margins(x=0)

    axes[-1].set_xlabel("step")

    fig.suptitle("exp0522 rollout diagnostics", fontsize=config.plot_title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)
