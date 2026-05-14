"""Diagnostics collection and plotting for exp0513 V1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _safe_correlation_mean(q_batch: torch.Tensor) -> float:
    """Mean absolute off-diagonal correlation across q dimensions."""
    if q_batch.shape[0] < 2 or q_batch.shape[1] < 2:
        return 0.0
    centered = q_batch - q_batch.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    valid = std > 1e-12
    if valid.sum().item() < 2:
        return 0.0
    centered = centered[:, valid]
    cov = centered.T @ centered / max(centered.shape[0], 1)
    denom = std[valid].unsqueeze(0) * std[valid].unsqueeze(1)
    corr = cov / denom
    mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    return float(corr[mask].abs().mean().item())


def collect_batch_metrics(
    q_batch: torch.Tensor,
    q_mean: torch.Tensor,
    s: torch.Tensor,
    bp_flat: torch.Tensor,
    int_flat: torch.Tensor,
    total_flat: torch.Tensor,
    bp_mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize one batch for later epoch aggregation."""
    q_abs = q_batch.detach().abs()
    q_sat = (q_abs > 0.95).float().mean().item()
    bp_active = bp_flat[bp_mask]
    int_active = int_flat[bp_mask]
    bp_norm = float(bp_active.norm().item()) if bp_active.numel() else 0.0
    int_norm = float(int_flat.norm().item())
    total_norm = float(total_flat.norm().item())
    cosine = 0.0
    if bp_active.numel() and bp_active.norm().item() > 0 and int_active.norm().item() > 0:
        cosine = float(torch.nn.functional.cosine_similarity(bp_active, int_active, dim=0).item())

    return {
        "q_abs_mean": float(q_abs.mean().item()),
        "q_std_mean": float(q_batch.detach().std(dim=0, unbiased=False).mean().item()),
        "q_corr_abs_mean": _safe_correlation_mean(q_batch.detach()),
        "q_saturation_frac": float(q_sat),
        "q_mean_norm": float(q_mean.norm().item()),
        "s_abs_mean": float(s.detach().abs().mean().item()),
        "s_std": float(s.detach().std(unbiased=False).item()),
        "s_max_abs": float(s.detach().abs().max().item()),
        "bp_norm": bp_norm,
        "int_norm": int_norm,
        "total_norm": total_norm,
        "bp_int_cosine": cosine,
    }


def average_epoch_metrics(batch_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of batch metric dicts."""
    if not batch_metrics:
        return {}
    keys = batch_metrics[0].keys()
    return {key: float(sum(item[key] for item in batch_metrics) / len(batch_metrics)) for key in keys}


def plot_q_diagnostics(
    history: list[dict[str, float]],
    phase_a_epochs: int,
    output_path: Path,
) -> None:
    """Plot q- and s-related diagnostics over training."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("exp0513 q/s diagnostics")

    def _plot(ax, key: str, title: str) -> None:
        ax.plot(epochs, [row[key] for row in history], linewidth=1.6)
        ax.axvline(phase_a_epochs, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")

    _plot(axes[0, 0], "q_abs_mean", "Mean |q|")
    _plot(axes[0, 1], "q_corr_abs_mean", "Mean |corr(q_i, q_j)|")
    _plot(axes[1, 0], "q_saturation_frac", "q saturation fraction")
    _plot(axes[1, 1], "s_abs_mean", "Mean |s|")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_update_diagnostics(
    history: list[dict[str, float]],
    phase_a_epochs: int,
    output_path: Path,
) -> None:
    """Plot update norms and BP/internal alignment."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("exp0513 update diagnostics")

    def _plot(ax, key: str, title: str) -> None:
        ax.plot(epochs, [row[key] for row in history], linewidth=1.6)
        ax.axvline(phase_a_epochs, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")

    _plot(axes[0, 0], "bp_norm", "BP update norm")
    _plot(axes[0, 1], "int_norm", "Internal update norm")
    _plot(axes[1, 0], "total_norm", "Total controllable update norm")
    _plot(axes[1, 1], "bp_int_cosine", "Cosine(BP, internal)")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_history_json(history: list[dict[str, float]], output_path: Path) -> None:
    """Save history for later offline inspection."""
    output_path.write_text(json.dumps(history, indent=2))
