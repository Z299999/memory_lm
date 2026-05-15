"""Diagnostics collection and plotting for exp0513 dopamine experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _safe_correlation_mean(dopamine_batch: torch.Tensor) -> float:
    """Mean absolute off-diagonal correlation across dopamine dimensions."""
    if dopamine_batch.shape[0] < 2 or dopamine_batch.shape[1] < 2:
        return 0.0
    centered = dopamine_batch - dopamine_batch.mean(dim=0, keepdim=True)
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
    dopamine_batch: torch.Tensor,
    dopamine_mean: torch.Tensor,
    s: torch.Tensor,
    bp_flat: torch.Tensor,
    int_flat: torch.Tensor,
    total_flat: torch.Tensor,
    bp_mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize one batch for later epoch aggregation."""
    dopamine_abs = dopamine_batch.detach().abs()
    dopamine_sat = (dopamine_abs > 0.95).float().mean().item()
    bp_active = bp_flat[bp_mask]
    int_active = int_flat[bp_mask]
    bp_norm = float(bp_active.norm().item()) if bp_active.numel() else 0.0
    int_norm = float(int_flat.norm().item())
    total_norm = float(total_flat.norm().item())
    cosine = 0.0
    if bp_active.numel() and bp_active.norm().item() > 0 and int_active.norm().item() > 0:
        cosine = float(torch.nn.functional.cosine_similarity(bp_active, int_active, dim=0).item())

    return {
        "dopamine_abs_mean": float(dopamine_abs.mean().item()),
        "dopamine_std_mean": float(dopamine_batch.detach().std(dim=0, unbiased=False).mean().item()),
        "dopamine_corr_abs_mean": _safe_correlation_mean(dopamine_batch.detach()),
        "dopamine_saturation_frac": float(dopamine_sat),
        "dopamine_mean_norm": float(dopamine_mean.norm().item()),
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


def plot_dopamine_diagnostics(
    history: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot dopamine- and s-related diagnostics over training."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("exp0513 dopamine/s diagnostics")

    def _plot(ax, key: str, title: str) -> None:
        ax.plot(epochs, [row[key] for row in history], linewidth=1.6)
        ax.set_title(title)
        ax.set_xlabel("epoch")

    _plot(axes[0, 0], "dopamine_abs_mean", "Mean |dopamine|")
    _plot(axes[0, 1], "dopamine_corr_abs_mean", "Mean |corr(d_i, d_j)|")
    _plot(axes[1, 0], "dopamine_saturation_frac", "Dopamine saturation fraction")
    _plot(axes[1, 1], "s_abs_mean", "Mean |s|")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_update_diagnostics(
    history: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot update norms and BP/internal alignment."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("exp0513 update diagnostics")

    def _plot(ax, key: str, title: str) -> None:
        ax.plot(epochs, [row[key] for row in history], linewidth=1.6)
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
