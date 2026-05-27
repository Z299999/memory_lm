"""Environment factory for exp0526 online neural control."""

from __future__ import annotations

from typing import Protocol

import torch

try:
    from .config import ExperimentConfig
    from .env_scalar_cubic import ScalarCubicEnv
except ImportError:  # pragma: no cover
    from config import ExperimentConfig
    from env_scalar_cubic import ScalarCubicEnv


class ControllerEnv(Protocol):
    def initial_state(self) -> torch.Tensor: ...
    def eta(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]: ...
    def step(self, state: torch.Tensor, u: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]: ...
    def diagnostics(self, state: torch.Tensor, derived: dict[str, float]) -> dict[str, float]: ...


def build_env(config: ExperimentConfig, *, device: torch.device) -> ControllerEnv:
    if config.env.env_kind == "scalar_cubic":
        return ScalarCubicEnv(config.env, device=device)
    raise ValueError(f"Unsupported env_kind: {config.env.env_kind!r}")
