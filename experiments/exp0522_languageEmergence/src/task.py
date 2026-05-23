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


def build_rollout_targets(
    num_steps: int,
    cycle_steps: int,
    device: torch.device,
    *,
    start_step: int = 0,
    target_kind: str = "sine",
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5)),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phase and target tensors for one rollout."""
    omega = omega_from_cycle_steps(cycle_steps)
    step_idx = torch.arange(start_step, start_step + num_steps, dtype=torch.float32, device=device)
    phase = step_idx * omega
    target = build_target_from_phase(
        phase,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
    )
    return phase, target
