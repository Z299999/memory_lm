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
    """Fixed 4-row agent analysis figure (layout independent of N)."""
    import torch

    N = model.num_agents
    language_dim = model.language_dim
    messages = rollout["messages"].numpy()    # (T, N*language_dim)
    message_norm = rollout["message_norm"].numpy()  # (T, N)
    prediction = rollout["prediction"].numpy()
    target = rollout["target"].numpy()
    T = messages.shape[0]
    steps = np.arange(T)

    readout_mode = getattr(model, "readout_mode", "shared_linear")
    D_last = list(model.trunk_Ws[-1].shape)[1]  # (N, D_out, D_in) → D_out
    label_fs = max(6, 10 - N // 3)

    # --- Coupling matrix (N, N) ---
    with torch.no_grad():
        if model.DeltaD is not None:
            DeltaD = model.DeltaD.detach().cpu()
            if getattr(model, "message_carry_mode", "identity") == "learnable_diagonal":
                coupling_values = (1.0 + DeltaD).mean(dim=2).numpy()
            else:
                d = DeltaD.shape[-1]
                eye = torch.eye(d, dtype=DeltaD.dtype).unsqueeze(0).unsqueeze(0)
                coupling_values = ((eye + DeltaD).diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(d)).numpy()
        else:
            coupling_values = np.eye(N, dtype=float)
        w_in_eff = (
            (1.0 + model.delta_w_in).detach().cpu().numpy()
            if model.delta_w_in is not None
            else np.ones(N)
        )

    # --- Readout gains (N,) ---
    if readout_mode == "learnable":
        readout_gains = (1.0 + model.delta_pool_out).detach().cpu().numpy()
    elif readout_mode == "mean_pool":
        readout_gains = np.ones(N)
    else:
        W_out_np = model.readout_head.weight.detach().cpu().numpy()
        readout_gains = np.array([
            np.linalg.norm(W_out_np[0, i * D_last : (i + 1) * D_last])
            for i in range(N)
        ])

    # --- Per-agent readout contributions (T, N) ---
    hidden_all = rollout.get("hidden")  # (T, N, D_last) or None
    contrib_matrix = None
    if hidden_all is not None:
        if readout_mode in {"mean_pool", "learnable"}:
            W_agents = model.agent_head_W.detach().cpu().squeeze(1).numpy()  # (N, D_last)
            b_agents = model.agent_head_b.detach().cpu().squeeze(1).numpy()  # (N,)
            gains = (1.0 + model.delta_pool_out).detach().cpu().numpy() if readout_mode == "learnable" else np.ones(N)
            contribs = [gains[i] * (hidden_all[:, i, :].numpy() @ W_agents[i] + b_agents[i]) for i in range(N)]
        else:
            W_out = model.readout_head.weight.detach().cpu().numpy()
            contribs = [hidden_all[:, i, :].numpy() @ W_out[0, i * D_last : (i + 1) * D_last] for i in range(N)]
        contrib_matrix = np.stack(contribs, axis=1)  # (T, N)

    # --- Heatmap matrices: shape (N, T), rows = agents ---
    msg_norm_heat = message_norm.T if message_norm.ndim == 2 else np.zeros((N, T))  # (N, T)
    msg_val_heat = np.stack([messages[:, i * language_dim] for i in range(N)], axis=0)  # (N, T), ch 0
    contrib_heat = contrib_matrix.T if contrib_matrix is not None else None  # (N, T)

    # --- Layout: 4 rows × 2 cols, fixed height ---
    row0_h = max(1.1, N * 0.22 + 0.5)   # coupling heatmap scales with N
    height_ratios = [row0_h, 1.1, 0.85, 0.85]
    fig_h = sum(height_ratios) * 2.2 + 0.8
    fig, axes = plt.subplots(
        4, 2,
        figsize=(13.0, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
        constrained_layout=True,
    )
    agent_colors = [f"C{i % 10}" for i in range(N)]

    # --- Row 0 left: coupling heatmap ---
    ax_d = axes[0, 0]
    im0 = ax_d.imshow(coupling_values, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(im0, ax=ax_d, fraction=0.046, pad=0.04)
    ax_d.set_xticks(range(N))
    ax_d.set_yticks(range(N))
    ax_d.set_xticklabels([f"a{j}" for j in range(N)], fontsize=label_fs)
    ax_d.set_yticklabels([f"a{i}" for i in range(N)], fontsize=label_fs)
    ax_d.set_xlabel("from", fontsize=8)
    ax_d.set_ylabel("to", fontsize=8)
    ax_d.set_title("Effective coupling (1+Δ)", fontsize=10)
    if N <= 12:
        d_max = float(np.max(coupling_values)) if coupling_values.size else 0.0
        for i in range(N):
            for j in range(N):
                val = float(coupling_values[i, j])
                ax_d.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=max(5, 9 - N // 3),
                          color="white" if val < d_max * 0.6 else "black")

    # --- Row 0 right: per-agent weight bars ---
    ax_w = axes[0, 1]
    x = np.arange(N)
    width = 0.35
    ax_w.bar(x - width / 2, w_in_eff, width, color=agent_colors, alpha=0.9, label="1+Δw_in (error)")
    readout_label = "1+Δr_out (readout)" if readout_mode in {"mean_pool", "learnable"} else "readout ‖w_i‖"
    ax_w.bar(x + width / 2, readout_gains, width, color=agent_colors, alpha=0.45,
             hatch="//", label=readout_label)
    ax_w.set_xticks(x)
    ax_w.set_xticklabels([f"a{i}" for i in range(N)], fontsize=label_fs)
    ax_w.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="baseline=1")
    ax_w.set_title("Per-agent weights", fontsize=10)
    ax_w.legend(fontsize=8, loc="upper right")
    ax_w.grid(axis="y", alpha=config.plot_grid_alpha)

    # --- Row 1 left: output vs target ---
    ax_pred = axes[1, 0]
    ax_pred.plot(steps, target, color=config.plot_target_color,
                 linestyle=config.plot_target_linestyle,
                 linewidth=config.plot_target_linewidth, label="target")
    ax_pred.plot(steps, prediction, linewidth=config.plot_series_linewidth,
                 color="#1f77b4", label="output")
    ax_pred.set_title("Reservoir output vs target", fontsize=10)
    ax_pred.set_ylabel("value")
    ax_pred.grid(alpha=config.plot_grid_alpha)
    ax_pred.legend(fontsize=8)

    # --- Row 1 right: all contributions overlaid ---
    ax_ov = axes[1, 1]
    if contrib_matrix is not None:
        for i in range(N):
            ax_ov.plot(steps, contrib_matrix[:, i], linewidth=config.plot_aux_linewidth,
                       color=agent_colors[i], alpha=0.85, label=f"a{i}")
        ax_ov.plot(steps, target, color=config.plot_target_color,
                   linestyle=config.plot_target_linestyle,
                   linewidth=config.plot_target_linewidth, label="target")
    ax_ov.set_title("Readout contributions (overlaid)", fontsize=10)
    ax_ov.set_ylabel("v_i · h_i")
    ax_ov.grid(alpha=config.plot_grid_alpha)
    _maybe_add_legend(ax_ov, ncol=min(N + 1, 7))

    # --- Row 2 left: message norm heatmap (N × T) ---
    ax_mh = axes[2, 0]
    im2 = ax_mh.imshow(msg_norm_heat, aspect="auto", cmap="plasma",
                        interpolation="nearest", origin="lower")
    fig.colorbar(im2, ax=ax_mh, fraction=0.046, pad=0.04)
    ax_mh.set_yticks(range(N))
    ax_mh.set_yticklabels([f"a{i}" for i in range(N)], fontsize=label_fs)
    ax_mh.set_title("Message norm ‖m_i‖ over time", fontsize=10)
    ax_mh.set_xlabel("step")
    ax_mh.set_ylabel("agent")

    # --- Row 2 right: message channel-0 heatmap (N × T) ---
    ax_mv = axes[2, 1]
    vmax = float(np.abs(msg_val_heat).max()) + 1e-8
    im3 = ax_mv.imshow(msg_val_heat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax,
                        interpolation="nearest", origin="lower")
    fig.colorbar(im3, ax=ax_mv, fraction=0.046, pad=0.04)
    ax_mv.set_yticks(range(N))
    ax_mv.set_yticklabels([f"a{i}" for i in range(N)], fontsize=label_fs)
    ch_label = "m[0]" if language_dim > 1 else "m"
    ax_mv.set_title(f"Message {ch_label} over time", fontsize=10)
    ax_mv.set_xlabel("step")
    ax_mv.set_ylabel("agent")

    # --- Row 3 left: message norm lines overlaid ---
    ax_nl = axes[3, 0]
    if message_norm.ndim == 2:
        for i in range(N):
            ax_nl.plot(steps, message_norm[:, i], linewidth=config.plot_aux_linewidth,
                       color=agent_colors[i], label=f"a{i}")
    ax_nl.set_title("Message norm (overlaid)", fontsize=10)
    ax_nl.set_ylabel("‖m_i‖")
    ax_nl.set_xlabel("step")
    ax_nl.grid(alpha=config.plot_grid_alpha)
    _maybe_add_legend(ax_nl, ncol=min(N, 7))

    # --- Row 3 right: contribution heatmap (N × T) ---
    ax_ch = axes[3, 1]
    if contrib_heat is not None:
        cmax = float(np.abs(contrib_heat).max()) + 1e-8
        im4 = ax_ch.imshow(contrib_heat, aspect="auto", cmap="RdBu_r",
                            vmin=-cmax, vmax=cmax,
                            interpolation="nearest", origin="lower")
        fig.colorbar(im4, ax=ax_ch, fraction=0.046, pad=0.04)
        ax_ch.set_yticks(range(N))
        ax_ch.set_yticklabels([f"a{i}" for i in range(N)], fontsize=label_fs)
    ax_ch.set_title("Readout contribution heatmap", fontsize=10)
    ax_ch.set_xlabel("step")
    ax_ch.set_ylabel("agent")

    fig.suptitle("exp0522 agent analysis", fontsize=config.plot_title_fontsize)
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)
