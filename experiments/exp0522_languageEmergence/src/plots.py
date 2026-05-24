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
    "sole_eye":    "#ff7f0e",  # matplotlib C1 orange
    "sole_speech": "#2ca02c",  # matplotlib C2 green
    "neither":     "#d62728",  # matplotlib C3 red
    "late_blind":  "#c05030",  # muted red-orange
    "late_mute":   "#bcbd22",  # olive yellow
    "blink":       "#ff7f0e",  # vivid orange
    "stutter":     "#e377c2",  # pink
}

def _build_rollout_panels(config: ExperimentConfig, *, has_reset: bool, has_continuous: bool) -> list[tuple[str, float]]:
    panels: list[tuple[str, float]] = []
    if has_reset:
        panels.extend([("short", 1.35), ("long", 1.35)])
    if has_continuous:
        panels.extend([("continuous_short", 1.35), ("continuous_long", 1.35)])
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

    panels = _build_rollout_panels(config, has_reset=has_reset, has_continuous=has_continuous)
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
        for condition_str in series:
            base, params = parse_condition(condition_str)
            color = _CONDITION_COLORS.get(base)
            if color is None:
                continue
            if base in ("late_blind", "late_mute") and len(params) == 1:
                step = params[0]
                if step < num_steps:
                    vlines.append((step - 1, f"↓{base}", color))
            elif base in ("blink", "stutter") and len(params) == 2:
                s, e = params[0], params[1]
                if s < num_steps:
                    vlines.append((s - 1, f"↓{base}", color))
                if e < num_steps:
                    vlines.append((e - 1, f"↑{base}", color))
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
        for vline_step, vline_label, vline_color in _build_transition_vlines(config.eval_conditions, num_steps):
            ax.axvline(vline_step, color=vline_color, linestyle="--", linewidth=1.0, alpha=0.55, label=vline_label)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(ax, ncol=config.plot_prediction_legend_ncols)

    if "short" in axis_by_panel:
        _plot_rollout_panel(
            axis_by_panel["short"],
            reset_evals,
            config.plot_short_steps,
            "Reset short eval" if has_continuous else "Short rollout",
        )
    if "long" in axis_by_panel:
        _plot_rollout_panel(
            axis_by_panel["long"],
            reset_long_evals,
            config.plot_long_steps,
            "Reset long eval" if has_continuous else "Long rollout",
        )
    if "continuous_short" in axis_by_panel:
        _plot_rollout_panel(
            axis_by_panel["continuous_short"],
            continuous_evals,
            config.plot_short_steps,
            "Continuous short eval",
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
        for vline_step, vline_label, vline_color in _build_transition_vlines(config.eval_conditions, config.plot_error_steps):
            error_axis.axvline(vline_step, color=vline_color, linestyle="--", linewidth=1.0, alpha=0.55, label=vline_label)
        error_axis.set_title("Short rollout error")
        error_axis.set_ylabel("pred - target")
        error_axis.grid(True, alpha=config.plot_grid_alpha)
        _maybe_add_legend(error_axis, ncol=config.plot_error_legend_ncols)

    if "messages" in axis_by_panel:
        messages = message_source["messages"].numpy()
        num_agents = message_source.get("num_agents", 1)
        language_dim = messages.shape[1] // num_agents if num_agents > 1 else messages.shape[1]
        message_axis = axis_by_panel["messages"]
        if messages.shape[1] > 0:
            for idx in range(messages.shape[1]):
                if num_agents > 1:
                    label = f"a{idx // language_dim}:m{idx % language_dim}"
                else:
                    label = f"m{idx}"
                message_axis.plot(
                    message_steps,
                    messages[:, idx],
                    linewidth=config.plot_aux_linewidth,
                    label=label,
                )
            _maybe_add_legend(message_axis, ncol=config.plot_message_legend_ncols)
        message_axis.set_title("Full model language channel traces")
        message_axis.set_ylabel("message")
        message_axis.grid(True, alpha=config.plot_grid_alpha)

    if "message_norm" in axis_by_panel:
        norm_axis = axis_by_panel["message_norm"]
        norm = message_source["message_norm"].numpy()
        if norm.ndim == 2:  # (T, N) per-agent norms
            for i in range(norm.shape[1]):
                norm_axis.plot(
                    message_steps,
                    norm[:, i],
                    linewidth=config.plot_series_linewidth,
                    label=f"agent {i}",
                )
            _maybe_add_legend(norm_axis, ncol=norm.shape[1])
        else:
            norm_axis.plot(
                message_steps,
                norm,
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


def plot_agent_analysis(
    *,
    model: Any,
    rollout: dict[str, Any],
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    """Multi-agent analysis figure: D heatmap, w_in, per-agent norm, per-agent traces."""
    import torch

    N = model.num_agents
    language_dim = model.language_dim

    messages = rollout["messages"].numpy()   # (T, N*language_dim)
    message_norm = rollout["message_norm"].numpy()  # (T, N)
    T = messages.shape[0]
    steps = np.arange(T)

    # D matrix frobenius norms: (N, N)
    with torch.no_grad():
        DeltaD = model.DeltaD.detach().cpu()  # (N, N, d, d)
        d_norms = torch.linalg.norm(DeltaD.reshape(N, N, -1), dim=2).numpy()  # (N, N) Frobenius
        w_in = model.w_in.detach().cpu().numpy()  # (N,)

    # Layout: row 0 = D heatmap + w_in bar (side by side)
    #         row 1 = per-agent message norm
    #         rows 2..N+1 = per-agent message traces
    nrows = 2 + N
    height_ratios = [1.2, 0.9] + [1.0] * N
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(12.0, max(3.5 * nrows * 0.6, 6.0)),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
        constrained_layout=True,
    )

    # Row 0 left: D heatmap
    ax_d = axes[0, 0]
    im = ax_d.imshow(d_norms, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    ax_d.set_xticks(range(N))
    ax_d.set_yticks(range(N))
    ax_d.set_xticklabels([f"from a{j}" for j in range(N)])
    ax_d.set_yticklabels([f"agent {i}" for i in range(N)])
    ax_d.set_title("Inter-agent D matrix ‖D[i,j]‖_F")
    for i in range(N):
        for j in range(N):
            ax_d.text(j, i, f"{d_norms[i, j]:.2f}", ha="center", va="center", fontsize=9,
                      color="white" if d_norms[i, j] < d_norms.max() * 0.6 else "black")

    # Row 0 right: w_in bar chart
    ax_w = axes[0, 1]
    ax_w.bar(range(N), w_in, color=[f"C{i}" for i in range(N)])
    ax_w.set_xticks(range(N))
    ax_w.set_xticklabels([f"agent {i}" for i in range(N)])
    ax_w.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="init=1")
    ax_w.set_title("Error distribution weights w_in")
    ax_w.set_ylabel("w_in[i]")
    ax_w.legend(fontsize=8)
    ax_w.grid(axis="y", alpha=config.plot_grid_alpha)

    # Row 1 left: per-agent message norm over time
    ax_norm = axes[1, 0]
    if message_norm.ndim == 2:
        for i in range(N):
            ax_norm.plot(steps, message_norm[:, i], linewidth=config.plot_series_linewidth,
                         color=f"C{i}", label=f"agent {i}")
    ax_norm.set_title("Per-agent message norm")
    ax_norm.set_ylabel("‖m_i‖")
    ax_norm.grid(alpha=config.plot_grid_alpha)
    ax_norm.legend(ncol=N, fontsize=8)

    # Row 1 right: prediction vs target
    ax_pred = axes[1, 1]
    prediction = rollout["prediction"].numpy()
    target = rollout["target"].numpy()
    ax_pred.plot(steps, target, color=config.plot_target_color,
                 linestyle=config.plot_target_linestyle,
                 linewidth=config.plot_target_linewidth, label="target")
    ax_pred.plot(steps, prediction, linewidth=config.plot_series_linewidth,
                 color="#1f77b4", label="prediction")
    ax_pred.set_title("Reservoir output vs target")
    ax_pred.set_ylabel("value")
    ax_pred.grid(alpha=config.plot_grid_alpha)
    ax_pred.legend(fontsize=8)

    # Compute per-agent readout contributions if hidden states are available
    # W_out: (1, N*D_last), agent i's slice: W_out[0, i*D:(i+1)*D]
    hidden_all = rollout.get("hidden")  # (T, N, D_last) or None
    D_last = list(model.trunk_Ws[-1].shape)[1]  # trunk_Ws[-1]: (N, D_out, D_in) → D_out
    W_out = model.readout_head.weight.detach().cpu().numpy()  # (1, N*D_last)

    # Rows 2..N+1: per-agent message traces (left) + readout contribution (right)
    for i in range(N):
        ax_tr = axes[2 + i, 0]
        agent_msgs = messages[:, i * language_dim : (i + 1) * language_dim]
        for ch in range(language_dim):
            ax_tr.plot(steps, agent_msgs[:, ch], linewidth=config.plot_aux_linewidth,
                       label=f"m{ch}")
        ax_tr.set_title(f"Agent {i} message traces")
        ax_tr.set_ylabel("message")
        ax_tr.grid(alpha=config.plot_grid_alpha)
        _maybe_add_legend(ax_tr, ncol=config.plot_message_legend_ncols)

        ax_contrib = axes[2 + i, 1]
        if hidden_all is not None:
            h_i = hidden_all[:, i, :].numpy()          # (T, D_last)
            w_i = W_out[0, i * D_last : (i + 1) * D_last]  # (D_last,)
            contrib_i = h_i @ w_i                       # (T,)
            ax_contrib.plot(steps, contrib_i, linewidth=config.plot_series_linewidth,
                            color=f"C{i}", label=f"agent {i} contribution")
            ax_contrib.plot(steps, target, color=config.plot_target_color,
                            linestyle=config.plot_target_linestyle,
                            linewidth=config.plot_target_linewidth, label="target")
            ax_contrib.legend(fontsize=8)
        ax_contrib.set_title(f"Agent {i} readout contribution")
        ax_contrib.set_ylabel("W_out_i · h_i")
        ax_contrib.grid(alpha=config.plot_grid_alpha)

    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")

    fig.suptitle("exp0522 agent analysis", fontsize=config.plot_title_fontsize)
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)
