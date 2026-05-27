"""Environment factory for exp0526 online neural control."""

from __future__ import annotations

from typing import Any, Protocol

import torch

try:
    from .config import ExperimentConfig
    from .env_planar_double_well import PlanarDoubleWellEnv
    from .env_scalar_control_affine import ScalarControlAffineEnv
    from .env_scalar_cubic import ScalarCubicEnv
except ImportError:  # pragma: no cover
    from config import ExperimentConfig
    from env_planar_double_well import PlanarDoubleWellEnv
    from env_scalar_control_affine import ScalarControlAffineEnv
    from env_scalar_cubic import ScalarCubicEnv


class ControllerEnv(Protocol):
    state_dim: int
    def initial_state(self) -> torch.Tensor: ...
    def eta(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]: ...
    def step(self, state: torch.Tensor, u: torch.Tensor, step: int) -> tuple[torch.Tensor, dict[str, float]]: ...
    def diagnostics(self, state: torch.Tensor, derived: dict[str, float]) -> dict[str, float]: ...
    def equilibrium_points(self) -> list[dict[str, Any]]: ...


def build_env(config: ExperimentConfig, *, device: torch.device) -> ControllerEnv:
    if config.env.env_kind == "scalar_control_affine":
        return ScalarControlAffineEnv(config.env, device=device)
    if config.env.env_kind == "scalar_cubic":
        return ScalarCubicEnv(config.env, device=device)
    if config.env.env_kind == "planar_double_well":
        return PlanarDoubleWellEnv(config.env, device=device)
    raise ValueError(f"Unsupported env_kind: {config.env.env_kind!r}")
