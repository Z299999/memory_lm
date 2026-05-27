"""Plotting utilities for exp0526 online neural control."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

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
except ImportError:  # pragma: no cover
    from config import ExperimentConfig, parse_condition


_CONDITION_COLORS = {
    "full": "#1f77b4",
    "sole_eye": "#17becf",
    "sole_speech": "#2ca02c",
    "neither": "#d62728",
    "blink": "#ff7f0e",
    "stutter": "#e377c2",
}


def _maybe_add_legend(ax: plt.Axes, *, ncol: int) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", ncol=ncol)


def _transition_vlines(config: ExperimentConfig, start_step: int, num_steps: int) -> list[tuple[int, str, str]]:
    vlines: list[tuple[int, str, str]] = []
    for condition in config.eval.eval_conditions:
        base, params = parse_condition(condition)
        color = _CONDITION_COLORS.get(base)
        if color is None or base not in {"blink", "stutter"} or len(params) != 2:
            continue
        s, e = params
        if s < num_steps:
            vlines.append((start_step + s, f"↓{base}", color))
        if e < num_steps:
            vlines.append((start_step + e, f"↑{base}", color))
    return vlines


def plot_training_curves(*, history: list[dict[str, float]], output_path: Path, config: ExperimentConfig) -> None:
    fig, ax = plt.subplots(figsize=(config.plot.plot_training_fig_width, config.plot.plot_training_fig_height))
    epochs = [row["epoch"] for row in history]
    for key, label in (
        ("total_loss", "total loss"),
        ("state_loss", "state loss"),
        ("control_loss", "control energy"),
        ("val_state_loss", "future state loss"),
    ):
        if key in history[0]:
            ax.plot(epochs, [row[key] for row in history], label=label, linewidth=config.plot.plot_series_linewidth)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.set_title("exp0526 training curves")
    ax.grid(True, alpha=config.plot.plot_grid_alpha)
    _maybe_add_legend(ax, ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot.plot_dpi)
    plt.close(fig)


def _build_timeline_panels(config: ExperimentConfig, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    total_start = int(records[0]["start_step"])
    total_end = int(records[-1]["end_step"])
    window = int(config.plot.plot_training_timeline_window_steps)
    panel_count = int(config.plot.plot_training_timeline_num_panels)
    max_start = max(total_end - window + 1, total_start)
    if panel_count <= 1 or max_start <= total_start:
        starts = [total_start]
    else:
        starts = [int(round(v)) for v in np.linspace(total_start, max_start, panel_count)]
    panels = [
        {
            "start_step": s,
            "end_step": s + window - 1,
            "steps": [],
            "x": [],
            "x_sq": [],
            "u": [],
            "updates": [],
        }
        for s in starts
    ]
    for record in records:
        rec_steps = np.asarray(record["steps"], dtype=int)
        rec_x = np.asarray(record["x"], dtype=float)
        rec_x_sq = np.asarray(record["x_sq"], dtype=float)
        rec_u = np.asarray(record["u"], dtype=float)
        for panel in panels:
            mask = (rec_steps >= panel["start_step"]) & (rec_steps <= panel["end_step"])
            if not np.any(mask):
                continue
            panel["steps"].extend(rec_steps[mask].tolist())
            panel["x"].extend(rec_x[mask].tolist())
            panel["x_sq"].extend(rec_x_sq[mask].tolist())
            panel["u"].extend(rec_u[mask].tolist())
            if panel["start_step"] <= record["end_step"] <= panel["end_step"]:
                panel["updates"].append(int(record["end_step"]))
    return panels


def plot_training_timeline(*, records: list[dict[str, Any]], output_path: Path, config: ExperimentConfig) -> None:
    panels = _build_timeline_panels(config, records)
    if not panels:
        return
    ncols = max(1, int(config.plot.plot_training_timeline_ncols))
    panel_rows = int(np.ceil(len(panels) / ncols))
    metric_rows = 3
    nrows = metric_rows * panel_rows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.plot.plot_training_timeline_fig_width, max(2.2 * nrows, 8.0)),
        squeeze=False,
    )
    metrics = [
        ("x", "x", "#1f77b4"),
        ("x_sq", "x^2", "#2ca02c"),
        ("u", "u", "#ff7f0e"),
    ]
    for panel_idx, panel in enumerate(panels):
        block_row = panel_idx // ncols
        col = panel_idx % ncols
        steps = np.asarray(panel["steps"], dtype=float)
        for metric_idx, (key, label, color) in enumerate(metrics):
            ax = axes[block_row * metric_rows + metric_idx, col]
            values = np.asarray(panel.get(key, []), dtype=float)
            if steps.size and values.size == steps.size:
                ax.plot(
                    steps,
                    values,
                    label=label,
                    color=color,
                    linewidth=config.plot.plot_series_linewidth if key != "u" else config.plot.plot_aux_linewidth,
                )
            for update in panel["updates"]:
                ax.axvline(update, color="#c7c7c7", linestyle="--", linewidth=0.9, alpha=0.9)
            if metric_idx == 0:
                ax.set_title(
                    f"t={panel['start_step']}..{panel['end_step']}",
                    fontsize=max(config.plot.plot_title_fontsize - 2, 10),
                )
            if metric_idx == metric_rows - 1:
                ax.set_xlabel("global step")
            else:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(label)
            ax.grid(True, alpha=config.plot.plot_grid_alpha)
            _maybe_add_legend(ax, ncol=1)

    used_cols_last_block = len(panels) % ncols
    if used_cols_last_block == 0:
        used_cols_last_block = ncols
    for row in range((panel_rows - 1) * metric_rows, panel_rows * metric_rows):
        for col in range(used_cols_last_block, ncols):
            axes[row, col].axis("off")
    fig.suptitle("exp0526 training timeline", fontsize=config.plot.plot_title_fontsize)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(output_path, dpi=config.plot.plot_dpi)
    plt.close(fig)


def plot_rollout_diagnostics(*, evals: dict[str, dict[str, Any]], output_path: Path, config: ExperimentConfig) -> None:
    ref = evals.get("full")
    if ref is None:
        return
    steps = np.asarray(ref["global_step"], dtype=float)
    num_steps = len(steps)
    start_step = int(ref["start_step"])
    panels: list[str] = ["state", "state_sq", "control"]
    has_messages = ref["messages"].numel() and ref["messages"].shape[1] > 0
    if has_messages and config.plot.plot_show_message_traces:
        panels.append("messages")
    if has_messages and config.plot.plot_show_message_norm:
        panels.append("message_norm")

    height = config.plot.plot_diag_fig_height + max(0.0, 1.6 * (len(panels) - 3))
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(config.plot.plot_diag_fig_width, height),
        gridspec_kw={"height_ratios": [1.25 if p in {"state", "state_sq", "control"} else 1.0 for p in panels]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    by_name = {name: ax for name, ax in zip(panels, axes)}

    for condition, result in evals.items():
        base, _ = parse_condition(condition)
        color = _CONDITION_COLORS.get(base)
        x = np.asarray(result["global_step"], dtype=float)
        by_name["state"].plot(
            x,
            result["x"],
            label=condition,
            color=color,
            linewidth=config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth,
        )
        by_name["state_sq"].plot(
            x,
            result["x_sq"],
            label=condition,
            color=color,
            linewidth=config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth,
        )
        by_name["control"].plot(
            x,
            result["u"],
            label=condition,
            color=color,
            linewidth=config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth,
        )

    by_name["state"].axhline(0.0, color="black", linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75)
    by_name["control"].axhline(0.0, color="black", linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75)

    for name in ("state", "state_sq", "control"):
        ax = by_name[name]
        for xpos, label, color in _transition_vlines(config, start_step, num_steps):
            ax.axvline(xpos, color=color, linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75, label=label)
        ax.grid(True, alpha=config.plot.plot_grid_alpha)
        _maybe_add_legend(ax, ncol=config.plot.plot_legend_ncols)
        ax.set_xlim(float(steps[0]), float(steps[-1]))
        ax.margins(x=0.0)

    by_name["state"].set_title("Future scalar state")
    by_name["state"].set_ylabel("x")
    by_name["state_sq"].set_title("Future state energy")
    by_name["state_sq"].set_ylabel("x^2")
    by_name["control"].set_title("Control signal")
    by_name["control"].set_ylabel("u")
    by_name["control"].set_xlabel("global step")

    if "messages" in by_name:
        ax = by_name["messages"]
        messages = np.asarray(ref["messages"], dtype=float)
        msg_steps = steps[: messages.shape[0]]
        for channel in range(messages.shape[1]):
            ax.plot(msg_steps, messages[:, channel], linewidth=1.1, alpha=0.85, label=f"m{channel}")
        ax.set_title("Language channel traces")
        ax.set_ylabel("message")
        ax.set_xlabel("global step")
        ax.grid(True, alpha=config.plot.plot_grid_alpha)
        ax.set_xlim(float(msg_steps[0]), float(msg_steps[-1]))
        ax.margins(x=0.0)
        _maybe_add_legend(ax, ncol=max(1, min(6, messages.shape[1])))

    if "message_norm" in by_name:
        ax = by_name["message_norm"]
        for condition, result in evals.items():
            base, _ = parse_condition(condition)
            color = _CONDITION_COLORS.get(base)
            x = np.asarray(result["global_step"], dtype=float)
            ax.plot(
                x,
                result["message_norm"],
                label=condition,
                color=color,
                linewidth=config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth,
            )
        ax.set_title("Message norm")
        ax.set_ylabel("||m||")
        ax.set_xlabel("global step")
        ax.grid(True, alpha=config.plot.plot_grid_alpha)
        ax.set_xlim(float(steps[0]), float(steps[-1]))
        ax.margins(x=0.0)
        _maybe_add_legend(ax, ncol=config.plot.plot_legend_ncols)

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot.plot_dpi)
    plt.close(fig)
