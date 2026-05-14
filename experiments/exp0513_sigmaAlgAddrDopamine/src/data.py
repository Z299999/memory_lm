"""Data utilities for exp0513 V1 mixed-sin experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class RegressionDataset:
    """Train/validation/plot splits for a 1D regression task."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_plot: torch.Tensor
    y_plot: torch.Tensor


def sin_mix_target(x: torch.Tensor) -> torch.Tensor:
    """Target from exp0414: 0.5*sin(x) + 0.3*sin(2x) + 0.2*sin(3x)."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return 0.5 * torch.sin(x) + 0.3 * torch.sin(2.0 * x) + 0.2 * torch.sin(3.0 * x)


def build_sin_mix_dataset(
    num_train: int = 500,
    num_val: int = 200,
    num_plot: int = 500,
    x_min: float = -2.0 * math.pi,
    x_max: float = 2.0 * math.pi,
    seed: int = 42,
) -> RegressionDataset:
    """Build a shuffled train split plus deterministic validation/plot grids."""
    generator = torch.Generator().manual_seed(seed)

    x_train = torch.linspace(x_min, x_max, num_train).unsqueeze(-1)
    x_val = torch.linspace(x_min, x_max, num_val).unsqueeze(-1)
    x_plot = torch.linspace(x_min, x_max, num_plot).unsqueeze(-1)

    perm = torch.randperm(num_train, generator=generator)
    x_train = x_train[perm]

    y_train = sin_mix_target(x_train)
    y_val = sin_mix_target(x_val)
    y_plot = sin_mix_target(x_plot)

    return RegressionDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_plot=x_plot,
        y_plot=y_plot,
    )
