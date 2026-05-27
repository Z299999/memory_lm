"""Scalar cubic control environment for exp0526."""

from __future__ import annotations

import torch

try:
    from .config import EnvConfig
except ImportError:  # pragma: no cover
    from config import EnvConfig


class ScalarCubicEnv:
    """Simple differentiable scalar system x_dot = a x + b x^3 + c u."""

    def __init__(self, env_config: EnvConfig, *, device: torch.device) -> None:
        self.config = env_config
        self.device = device

    def initial_state(self) -> torch.Tensor:
        return torch.tensor([[float(self.config.x0)]], dtype=torch.float32, device=self.device)

    def eta(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        x = state
        derived = {
            "x": float(x.detach().item()),
            "x_sq": float((x.detach() ** 2).item()),
            "abs_x": float(torch.abs(x.detach()).item()),
            "global_step": float(step),
        }
        return x, derived

    def step(self, state: torch.Tensor, u: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]:
        drift = self.config.linear_coeff * state + self.config.cubic_coeff * (state ** 3)
        controlled = self.config.control_gain * u
        raw_next_state = state + self.config.dt * (drift + controlled)
        # Keep early random-policy exploration numerically finite without changing the small-signal dynamics.
        next_state = self.config.state_limit * torch.tanh(raw_next_state / self.config.state_limit)
        next_derived = {
            "x": float(next_state.detach().item()),
            "x_sq": float((next_state.detach() ** 2).item()),
            "abs_x": float(torch.abs(next_state.detach()).item()),
            "global_step": float(step + 1),
        }
        return next_state, next_derived

    def diagnostics(self, state: torch.Tensor, derived: dict[str, float]) -> dict[str, float]:
        return dict(derived)
