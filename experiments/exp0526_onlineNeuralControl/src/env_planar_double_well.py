"""Planar double-well environment for exp0526."""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    from .config import EnvConfig
except ImportError:  # pragma: no cover
    from config import EnvConfig


class PlanarDoubleWellEnv:
    """Differentiable 2D double-well system with one scalar control input."""

    def __init__(self, env_config: EnvConfig, *, device: torch.device) -> None:
        self.config = env_config
        self.device = device
        self.state_dim = 2

    def initial_state(self) -> torch.Tensor:
        x0 = self.config.x0
        if not isinstance(x0, (tuple, list)) or len(x0) != 2:
            raise ValueError("planar_double_well requires a 2D x0.")
        return torch.tensor([[float(x0[0]), float(x0[1])]], dtype=torch.float32, device=self.device)

    def eta(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        x1 = state[:, 0:1]
        x2 = state[:, 1:2]
        norm_sq = torch.sum(state ** 2)
        derived = {
            "x1": float(x1.detach().item()),
            "x2": float(x2.detach().item()),
            "state_0": float(x1.detach().item()),
            "state_1": float(x2.detach().item()),
            "state_norm_sq": float(norm_sq.detach().item()),
            "abs_state_max": float(torch.max(torch.abs(state.detach())).item()),
            "global_step": float(step),
        }
        return state, derived

    def step(self, state: torch.Tensor, u: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        x1 = state[:, 0:1]
        x2 = state[:, 1:2]
        x1_dot = x2
        x2_dot = (
            self.config.alpha * x1
            - self.config.beta * (x1 ** 3)
            - self.config.gamma * x2
            + self.config.control_gain * u
        )
        raw_next_state = state + self.config.dt * torch.cat([x1_dot, x2_dot], dim=1)
        next_state = self.config.state_limit * torch.tanh(raw_next_state / self.config.state_limit)
        next_eta, next_derived = self.eta(next_state, step + 1)
        _ = next_eta
        return next_state, next_derived

    def diagnostics(self, state: torch.Tensor, derived: dict[str, float]) -> dict[str, float]:
        return dict(derived)

    def equilibrium_points(self) -> list[dict[str, Any]]:
        if self.config.alpha <= 0.0 or self.config.beta <= 0.0:
            return [{"point": (0.0, 0.0), "stable": False}]
        outer = math.sqrt(self.config.alpha / self.config.beta)
        return [
            {"point": (-outer, 0.0), "stable": True},
            {"point": (0.0, 0.0), "stable": False},
            {"point": (outer, 0.0), "stable": True},
        ]

    def phase_field(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx1 = x2
        dx2 = self.config.alpha * x1 - self.config.beta * (x1 ** 3) - self.config.gamma * x2
        return dx1, dx2
