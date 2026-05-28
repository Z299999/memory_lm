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
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
import torch

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


def _blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    base = np.array(to_rgb(color))
    white = np.ones(3)
    return tuple((1.0 - blend) * base + blend * white)


def _draw_gradient_path(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, *, linewidth: float) -> None:
    points = np.column_stack([x, y])
    if len(points) < 2:
        return
    segments = np.stack([points[:-1], points[1:]], axis=1)
    start_rgb = _blend_with_white(color, 0.8)
    end_rgb = to_rgb(color)
    weights = np.linspace(0.0, 1.0, len(segments))
    colors = np.column_stack([
        start_rgb[0] + (end_rgb[0] - start_rgb[0]) * weights,
        start_rgb[1] + (end_rgb[1] - start_rgb[1]) * weights,
        start_rgb[2] + (end_rgb[2] - start_rgb[2]) * weights,
        np.linspace(0.22, 1.0, len(segments)),
    ])
    collection = LineCollection(segments, colors=colors, linewidths=linewidth, zorder=2)
    ax.add_collection(collection)


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
            "u": [],
            "updates": [],
        }
        for s in starts
    ]
    state_keys = sorted(key for key in records[0].keys() if key.startswith("state_") and key[6:].isdigit())
    has_state_norm_sq = "state_norm_sq" in records[0]
    for panel in panels:
        for key in state_keys:
            panel[key] = []
        if has_state_norm_sq:
            panel["state_norm_sq"] = []

    for record in records:
        rec_steps = np.asarray(record["steps"], dtype=int)
        rec_u = np.asarray(record["u"], dtype=float)
        for panel in panels:
            mask = (rec_steps >= panel["start_step"]) & (rec_steps <= panel["end_step"])
            if not np.any(mask):
                continue
            panel["steps"].extend(rec_steps[mask].tolist())
            panel["u"].extend(rec_u[mask].tolist())
            for key in state_keys:
                rec_values = np.asarray(record[key], dtype=float)
                panel[key].extend(rec_values[mask].tolist())
            if has_state_norm_sq:
                rec_norm_sq = np.asarray(record["state_norm_sq"], dtype=float)
                panel["state_norm_sq"].extend(rec_norm_sq[mask].tolist())
            if panel["start_step"] <= record["end_step"] <= panel["end_step"]:
                panel["updates"].append(int(record["end_step"]))
    return panels


def plot_training_timeline(*, records: list[dict[str, Any]], output_path: Path, config: ExperimentConfig) -> None:
    panels = _build_timeline_panels(config, records)
    if not panels:
        return
    ncols = max(1, int(config.plot.plot_training_timeline_ncols))
    panel_rows = int(np.ceil(len(panels) / ncols))
    state_keys = sorted(key for key in panels[0].keys() if key.startswith("state_") and key[6:].isdigit())
    if len(state_keys) == 1:
        metrics = [
            ("state_0", "x", "#1f77b4"),
            ("state_norm_sq", "||x||^2", "#2ca02c"),
            ("u", "u", "#ff7f0e"),
        ]
    else:
        metrics = [
            ("state_0", "x1", "#1f77b4"),
            ("state_1", "x2", "#2ca02c"),
            ("u", "u", "#ff7f0e"),
        ]
    metric_rows = len(metrics)
    nrows = metric_rows * panel_rows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.plot.plot_training_timeline_fig_width, max(2.2 * nrows, 8.0)),
        squeeze=False,
    )
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


def _plot_phase_portrait(ax: plt.Axes, *, evals: dict[str, dict[str, Any]], env: Any, config: ExperimentConfig) -> None:
    ref = evals.get("full")
    if ref is None:
        return
    x1_values: list[float] = []
    x2_values: list[float] = []
    for result in evals.values():
        if "state_0" in result:
            x1_values.extend(float(v) for v in result["state_0"])
        if "state_1" in result:
            x2_values.extend(float(v) for v in result["state_1"])
    equilibrium_points = env.equilibrium_points()
    for item in equilibrium_points:
        point = item["point"]
        x1_values.append(float(point[0]))
        x2_values.append(float(point[1]))
    max_abs = max([1.25] + [abs(v) for v in x1_values] + [abs(v) for v in x2_values]) * 1.15
    grid = np.linspace(-max_abs, max_abs, 21)
    gx1, gx2 = np.meshgrid(grid, grid)
    tx1 = torch.tensor(gx1, dtype=torch.float32)
    tx2 = torch.tensor(gx2, dtype=torch.float32)
    dx1, dx2 = env.phase_field(tx1, tx2)
    dx1_np = dx1.detach().cpu().numpy()
    dx2_np = dx2.detach().cpu().numpy()
    norm = np.sqrt(dx1_np ** 2 + dx2_np ** 2) + 1e-8
    ax.quiver(
        gx1,
        gx2,
        dx1_np / norm,
        dx2_np / norm,
        color="#bdbdbd",
        alpha=0.7,
        linewidth=0.6,
        angles="xy",
        scale_units="xy",
        scale=0.9 / max_abs,
    )
    for condition, result in evals.items():
        base, _ = parse_condition(condition)
        if base not in {"full", "blink", "stutter"}:
            continue
        color = _CONDITION_COLORS.get(base)
        x = np.asarray(result["state_0"], dtype=float)
        y = np.asarray(result["state_1"], dtype=float)
        linewidth = config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth
        _draw_gradient_path(ax, x, y, color, linewidth=linewidth)
        start_color = _blend_with_white(color, 0.8)
        ax.plot([], [], color=color, linewidth=linewidth, label=condition)
        if x.size:
            ax.scatter([x[0]], [y[0]], facecolors="none", edgecolors=[start_color], linewidths=1.15, marker="o", s=42, zorder=1)
            ax.scatter([x[-1]], [y[-1]], color=color, marker="X", s=52, linewidths=1.1, zorder=4)
    for item in equilibrium_points:
        x1, x2 = item["point"]
        stable = bool(item["stable"])
        if stable:
            ax.scatter([x1], [x2], color="black", s=28, zorder=4)
        else:
            ax.scatter([x1], [x2], facecolors="none", edgecolors="black", s=40, zorder=4)
    ax.set_title("Phase portrait")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.grid(True, alpha=config.plot.plot_grid_alpha)
    _maybe_add_legend(ax, ncol=3)


def plot_phase_portrait(*, evals: dict[str, dict[str, Any]], output_path: Path, config: ExperimentConfig, env: Any) -> None:
    if int(getattr(env, "state_dim", 1)) != 2:
        return
    ref = evals.get("full")
    if ref is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    _plot_phase_portrait(ax, evals=evals, env=env, config=config)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot.plot_dpi)
    plt.close(fig)


def plot_rollout_diagnostics(*, evals: dict[str, dict[str, Any]], output_path: Path, config: ExperimentConfig, env: Any) -> None:
    ref = evals.get("full")
    if ref is None:
        return
    steps = np.asarray(ref["global_step"], dtype=float)
    num_steps = len(steps)
    start_step = int(ref["start_step"])
    state_dim = int(getattr(env, "state_dim", 1))
    panels: list[str]
    if state_dim == 1:
        panels = ["state_0", "state_norm_sq", "control"]
    else:
        panels = ["state_0", "state_1", "state_norm_sq", "control"]
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
        gridspec_kw={
            "height_ratios": [
                1.25 if p in {"state_0", "state_1", "state_norm_sq", "control"} else 1.0
                for p in panels
            ]
        },
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    by_name = {name: ax for name, ax in zip(panels, axes)}

    series_panels = [name for name in panels if name not in {"messages", "message_norm"}]
    for condition, result in evals.items():
        base, _ = parse_condition(condition)
        color = _CONDITION_COLORS.get(base)
        x = np.asarray(result["global_step"], dtype=float)
        for panel_name in series_panels:
            series = result[panel_name] if panel_name != "control" else result["u"]
            by_name[panel_name].plot(
                x,
                series,
                label=condition,
                color=color,
                linewidth=config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth,
            )

    if state_dim == 1:
        by_name["state_0"].axhline(0.0, color="black", linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75)
    else:
        by_name["state_0"].axhline(0.0, color="black", linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75)
        by_name["state_1"].axhline(0.0, color="black", linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75)
    by_name["control"].axhline(0.0, color="black", linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75)

    for name in series_panels:
        ax = by_name[name]
        for xpos, label, color in _transition_vlines(config, start_step, num_steps):
            ax.axvline(xpos, color=color, linestyle="--", linewidth=config.plot.plot_zero_linewidth, alpha=0.75, label=label)
        ax.grid(True, alpha=config.plot.plot_grid_alpha)
        _maybe_add_legend(ax, ncol=config.plot.plot_legend_ncols)
        ax.set_xlim(float(steps[0]), float(steps[-1]))
        ax.margins(x=0.0)

    if state_dim == 1:
        by_name["state_0"].set_title("Future scalar state")
        by_name["state_0"].set_ylabel("x")
    else:
        by_name["state_0"].set_title("Future x1")
        by_name["state_0"].set_ylabel("x1")
        by_name["state_1"].set_title("Future x2")
        by_name["state_1"].set_ylabel("x2")
    by_name["state_norm_sq"].set_title("Future state energy")
    by_name["state_norm_sq"].set_ylabel("||x||^2")
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
