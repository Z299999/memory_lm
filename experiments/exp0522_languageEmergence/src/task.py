"""Deterministic rollout task for exp0522."""

from __future__ import annotations

import math

import torch


def prediction_target_history(prediction_target: str) -> int:
    if prediction_target == "y":
        return 0
    if prediction_target == "v":
        return 1
    if prediction_target == "a":
        return 2
    raise ValueError(f"Unsupported prediction_target: {prediction_target!r}")


def omega_from_cycle_steps(cycle_steps: int) -> float:
    return (2.0 * math.pi) / float(cycle_steps)


def build_target_from_phase(
    phase: torch.Tensor,
    *,
    target_kind: str = "sine",
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5)),
    phase_offset: float = 0.0,
) -> torch.Tensor:
    """Build one rollout target tensor from phase values."""
    if target_kind == "sine":
        target = torch.sin(phase + phase_offset)
    elif target_kind == "mixed_sin":
        raw = sum(amp * torch.sin(freq * phase + phase_offset) for freq, amp in mixed_sin_components)
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
    phase_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phase and true y-target tensors for one rollout."""
    omega = omega_from_cycle_steps(cycle_steps)
    step_idx = torch.arange(start_step, start_step + num_steps, dtype=torch.float32, device=device)
    phase = step_idx * omega
    target = build_target_from_phase(
        phase,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
        phase_offset=phase_offset,
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
    prediction_target: str = "y",
    phase_offset: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return true y targets, supervised targets, and reconstruction anchors."""
    history = prediction_target_history(prediction_target)
    full_start = start_step - history
    full_steps = num_steps + history
    phase, waveform = build_rollout_waveform(
        full_steps,
        cycle_steps,
        device,
        start_step=full_start,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
        phase_offset=phase_offset,
    )
    y_full = waveform.squeeze(1)
    phase_main = phase[history:]
    target_y = waveform[history:]

    if prediction_target == "y":
        train_target = target_y
        init_y_prev = torch.zeros(1, 1, device=device)
        init_v_prev = torch.zeros(1, 1, device=device)
    elif prediction_target == "v":
        velocity = y_full[1:] - y_full[:-1]
        train_target = velocity.unsqueeze(1)
        init_y_prev = y_full[0:1].unsqueeze(1)
        init_v_prev = torch.zeros(1, 1, device=device)
    elif prediction_target == "a":
        velocity = y_full[1:] - y_full[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        train_target = acceleration.unsqueeze(1)
        init_y_prev = y_full[1:2].unsqueeze(1)
        init_v_prev = velocity[0:1].unsqueeze(1)
    else:
        raise ValueError(f"Unsupported prediction_target: {prediction_target!r}")

    return {
        "phase": phase_main,
        "target_y": target_y,
        "train_target": train_target,
        "init_y_prev": init_y_prev,
        "init_v_prev": init_v_prev,
    }
