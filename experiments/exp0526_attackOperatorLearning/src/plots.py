"""Plotting utilities for exp0526 with exp0522-style layout."""

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
        ("eta_loss", "eta loss"),
        ("control_loss", "control energy"),
        ("val_eta_loss", "future eta loss"),
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
            "eta_norm_sq": [],
            "u": [],
            "population1": [],
            "population2": [],
            "updates": [],
        }
        for s in starts
    ]
    for record in records:
        rec_steps = np.asarray(record["steps"], dtype=int)
        rec_eta = np.asarray(record["eta_norm_sq"], dtype=float)
        rec_u = np.asarray(record["u"], dtype=float)
        rec_pop1 = np.asarray(record.get("population1", []), dtype=float)
        rec_pop2 = np.asarray(record.get("population2", []), dtype=float)
        for panel in panels:
            mask = (rec_steps >= panel["start_step"]) & (rec_steps <= panel["end_step"])
            if not np.any(mask):
                continue
            panel["steps"].extend(rec_steps[mask].tolist())
            panel["eta_norm_sq"].extend(rec_eta[mask].tolist())
            panel["u"].extend(rec_u[mask].tolist())
            if rec_pop1.size == rec_steps.size and rec_pop2.size == rec_steps.size:
                panel["population1"].extend(rec_pop1[mask].tolist())
                panel["population2"].extend(rec_pop2[mask].tolist())
            if panel["start_step"] <= record["end_step"] <= panel["end_step"]:
                panel["updates"].append(int(record["end_step"]))
    return panels


def plot_training_timeline(*, records: list[dict[str, Any]], output_path: Path, config: ExperimentConfig) -> None:
    panels = _build_timeline_panels(config, records)
    if not panels:
        return
    ncols = max(1, int(config.plot.plot_training_timeline_ncols))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(config.plot.plot_training_timeline_fig_width, max(3.4 * nrows, 6.8)),
        squeeze=False,
    )
    for ax, panel in zip(axes.flat, panels):
        steps = np.asarray(panel["steps"], dtype=float)
        ax_control = ax.twinx()
        if steps.size:
            if panel["population1"] and panel["population2"]:
                ax.plot(
                    steps,
                    panel["population1"],
                    label="N1",
                    color="#1f77b4",
                    linewidth=config.plot.plot_series_linewidth,
                )
                ax.plot(
                    steps,
                    panel["population2"],
                    label="N2",
                    color="#2ca02c",
                    linewidth=config.plot.plot_series_linewidth,
                )
            else:
                ax.plot(steps, panel["eta_norm_sq"], label="eta norm sq", linewidth=config.plot.plot_series_linewidth)
            ax_control.plot(
                steps,
                panel["u"],
                label="u",
                color="#ff7f0e",
                linewidth=config.plot.plot_aux_linewidth,
                alpha=0.88,
            )
        for update in panel["updates"]:
            ax.axvline(update, color="#c7c7c7", linestyle="--", linewidth=0.9, alpha=0.9)
            ax_control.axvline(update, color="#c7c7c7", linestyle="--", linewidth=0.9, alpha=0.0)
        ax.set_title(f"t={panel['start_step']}..{panel['end_step']}", fontsize=max(config.plot.plot_title_fontsize - 2, 10))
        ax.set_xlabel("global step")
        ax.set_ylabel("population")
        ax_control.set_ylabel("u")
        ax.grid(True, alpha=config.plot.plot_grid_alpha)
        handles, labels = ax.get_legend_handles_labels()
        control_handles, control_labels = ax_control.get_legend_handles_labels()
        if handles or control_handles:
            ax.legend(handles + control_handles, labels + control_labels, loc="upper right", ncol=3)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("exp0526 training timeline", fontsize=config.plot.plot_title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot.plot_dpi)
    plt.close(fig)


def plot_rollout_diagnostics(*, evals: dict[str, dict[str, Any]], output_path: Path, config: ExperimentConfig) -> None:
    ref = evals.get("full")
    if ref is None:
        return
    steps = np.asarray(ref["global_step"], dtype=float)
    num_steps = len(steps)
    start_step = int(ref["start_step"])
    panels: list[str] = ["population1", "population2", "control"]
    has_messages = ref["messages"].numel() and ref["messages"].shape[1] > 0
    if has_messages and config.plot.plot_show_message_traces:
        panels.append("messages")
    if has_messages and config.plot.plot_show_message_norm:
        panels.append("message_norm")

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(config.plot.plot_diag_fig_width, config.plot.plot_diag_fig_height),
        gridspec_kw={"height_ratios": [1.25 if p in {"population1", "population2", "control"} else 1.0 for p in panels]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    by_name = {name: ax for name, ax in zip(panels, axes)}

    for condition, result in evals.items():
        base, _ = parse_condition(condition)
        color = _CONDITION_COLORS.get(base)
        x = np.asarray(result["global_step"], dtype=float)
        by_name["population1"].plot(
            x,
            result["population1"],
            label=condition,
            color=color,
            linewidth=config.plot.plot_series_linewidth if base == "full" else config.plot.plot_aux_linewidth,
        )
        by_name["population2"].plot(
            x,
            result["population2"],
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

    by_name["population1"].plot(
        steps,
        ref["equilibrium_population1"],
        label="N1*",
        color="black",
        linestyle="--",
        linewidth=config.plot.plot_zero_linewidth,
        alpha=0.75,
    )
    by_name["population2"].plot(
        steps,
        ref["equilibrium_population2"],
        label="N2*",
        color="black",
        linestyle="--",
        linewidth=config.plot.plot_zero_linewidth,
        alpha=0.75,
    )

    for name in ("population1", "population2", "control"):
        ax = by_name[name]
        for xline, label, color in _transition_vlines(config, start_step, num_steps):
            ax.axvline(xline, color=color, linestyle="--", linewidth=1.0, alpha=0.55, label=label)
        ax.grid(True, alpha=config.plot.plot_grid_alpha)
        ax.set_xlim(steps[0], steps[-1])
        _maybe_add_legend(ax, ncol=config.plot.plot_legend_ncols)

    by_name["population1"].set_title("Continuous future species 1 population")
    by_name["population1"].set_ylabel("N1")
    by_name["population2"].set_title("Continuous future species 2 population")
    by_name["population2"].set_ylabel("N2")
    by_name["control"].set_title("Continuous future control")
    by_name["control"].set_ylabel("u")

    if "messages" in by_name:
        messages = ref["messages"].numpy()
        message_steps = steps[: min(len(steps), config.plot.plot_message_steps)]
        for idx in range(messages.shape[1]):
            by_name["messages"].plot(
                message_steps,
                messages[: len(message_steps), idx],
                linewidth=config.plot.plot_aux_linewidth,
                label=f"m{idx}",
            )
        by_name["messages"].set_title("Continuous language channel traces")
        by_name["messages"].set_ylabel("message")
        by_name["messages"].grid(True, alpha=config.plot.plot_grid_alpha)
        by_name["messages"].set_xlim(message_steps[0], message_steps[-1])
        _maybe_add_legend(by_name["messages"], ncol=config.plot.plot_legend_ncols)

    if "message_norm" in by_name:
        message_steps = steps[: min(len(steps), config.plot.plot_message_steps)]
        by_name["message_norm"].plot(
            message_steps,
            ref["message_norm"][: len(message_steps)],
            color="black",
            linewidth=config.plot.plot_series_linewidth,
        )
        by_name["message_norm"].set_title("Continuous message norm")
        by_name["message_norm"].set_ylabel("||m_t||")
        by_name["message_norm"].grid(True, alpha=config.plot.plot_grid_alpha)
        by_name["message_norm"].set_xlim(message_steps[0], message_steps[-1])

    axes[-1].set_xlabel("global step")
    fig.suptitle("exp0526 rollout diagnostics", fontsize=config.plot.plot_title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.plot.plot_dpi)
    plt.close(fig)
