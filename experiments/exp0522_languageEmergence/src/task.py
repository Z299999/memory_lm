"""Deterministic rollout task for exp0522."""

from __future__ import annotations

import math

import torch


def omega_from_cycle_steps(cycle_steps: int) -> float:
    return (2.0 * math.pi) / float(cycle_steps)


def build_rollout_targets(num_steps: int, cycle_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phase and target tensors for one rollout."""
    omega = omega_from_cycle_steps(cycle_steps)
    step_idx = torch.arange(num_steps, dtype=torch.float32, device=device)
    phase = step_idx * omega
    target = torch.sin(phase).unsqueeze(1)
    return phase, target
