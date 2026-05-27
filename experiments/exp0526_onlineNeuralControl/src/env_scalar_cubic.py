"""Backward-compatible scalar cubic environment wrapper for exp0526."""

from __future__ import annotations

import torch

try:
    from .config import EnvConfig
    from .env_scalar_control_affine import ScalarControlAffineEnv
except ImportError:  # pragma: no cover
    from config import EnvConfig
    from env_scalar_control_affine import ScalarControlAffineEnv


class ScalarCubicEnv(ScalarControlAffineEnv):
    """Compatibility wrapper for x_dot = a x + b x^3 + c u."""

    def __init__(self, env_config: EnvConfig, *, device: torch.device) -> None:
        compat = ScalarControlAffineEnv.from_cubic(env_config, device=device)
        self.__dict__ = compat.__dict__
