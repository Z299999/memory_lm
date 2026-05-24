"""Deterministic rollout task for exp0522."""

from __future__ import annotations

import math

import torch


def omega_from_cycle_steps(cycle_steps: int) -> float:
    return (2.0 * math.pi) / float(cycle_steps)


def build_target_from_phase(
    phase: torch.Tensor,
    *,
    target_kind: str = "sine",
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5)),
) -> torch.Tensor:
    """Build one rollout target tensor from phase values."""
    if target_kind == "sine":
        target = torch.sin(phase)
    elif target_kind == "mixed_sin":
        raw = sum(amp * torch.sin(freq * phase) for freq, amp in mixed_sin_components)
        scale = sum(abs(amp) for _, amp in mixed_sin_components)
        target = raw / scale
    else:
        raise ValueError(f"Unsupported target_kind: {target_kind!r}")
    return target.unsqueeze(1)


def build_rollout_waveform(
    num_steps: int,
    cycle_steps: int,
    device: torch.device,
    *,
    start_step: int = 0,
    target_kind: str = "sine",
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5)),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phase and true y-target tensors for one rollout."""
    omega = omega_from_cycle_steps(cycle_steps)
    step_idx = torch.arange(start_step, start_step + num_steps, dtype=torch.float32, device=device)
    phase = step_idx * omega
    target = build_target_from_phase(
        phase,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
    )
    return phase, target


def build_rollout_targets(
    num_steps: int,
    cycle_steps: int,
    device: torch.device,
    *,
    start_step: int = 0,
    target_kind: str = "sine",
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5)),
    target_split: str = "train",
    ticker: str = "IBM",
    price_column: str = "Close",
    series_kind: str = "log_return",
    normalize: str = "train_zscore",
    test_days: int = 252,
    market_cache_dir: str = "data/yfinance",
) -> dict[str, torch.Tensor]:
    """Return true y targets for one rollout."""
    if target_kind == "yfinance_series":
        try:
            from .market_data import build_market_rollout_targets
        except ImportError:  # pragma: no cover - script mode
            from market_data import build_market_rollout_targets
        return build_market_rollout_targets(
            num_steps=num_steps,
            device=device,
            start_step=start_step,
            split=target_split,
            ticker=ticker,
            price_column=price_column,
            series_kind=series_kind,
            normalize=normalize,
            test_days=test_days,
            market_cache_dir=market_cache_dir,
        )

    phase, target_y = build_rollout_waveform(
        num_steps,
        cycle_steps,
        device,
        start_step=start_step,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
    )
    return {
        "phase": phase,
        "target_y": target_y,
        "train_target": target_y,
    }
