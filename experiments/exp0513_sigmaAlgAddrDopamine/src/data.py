"""Data utilities for exp0513 1D regression tasks."""

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


def sin_target(x: torch.Tensor) -> torch.Tensor:
    """Simple sine target from exp0414."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return torch.sin(x)


def sin_mix_target(x: torch.Tensor) -> torch.Tensor:
    """Target from exp0414: 0.5*sin(x) + 0.3*sin(2x) + 0.2*sin(3x)."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return 0.5 * torch.sin(x) + 0.3 * torch.sin(2.0 * x) + 0.2 * torch.sin(3.0 * x)


def poly_wave_target(x: torch.Tensor) -> torch.Tensor:
    """Polynomial-wave target from exp0414."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return 0.1 * x**2 * torch.sin(x) + 0.5 * torch.cos(2.0 * x)


def piecewise_target(x: torch.Tensor) -> torch.Tensor:
    """Piecewise linear target from exp0414."""
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    y = torch.zeros_like(x)
    mask1 = x < -3.0
    mask2 = (x >= -3.0) & (x < 0.0)
    mask3 = (x >= 0.0) & (x < 3.0)
    mask4 = x >= 3.0
    y[mask1] = -0.5 * x[mask1] - 1.5
    y[mask2] = 0.3 * x[mask2] + 0.5
    y[mask3] = -0.2 * x[mask3] + 0.5
    y[mask4] = 0.4 * x[mask4] - 1.2
    return y


TASK_REGISTRY: dict[str, callable] = {
    "sin": sin_target,
    "sin_mix": sin_mix_target,
    "poly_wave": poly_wave_target,
    "piecewise": piecewise_target,
}


def available_task_names() -> list[str]:
    """Return supported 1D task names."""
    return sorted(TASK_REGISTRY.keys())


def target_function_1d(x: torch.Tensor, task_name: str) -> torch.Tensor:
    """Dispatch 1D target function by name."""
    try:
        return TASK_REGISTRY[task_name](x)
    except KeyError as exc:
        raise ValueError(f"Unsupported task_name={task_name!r}. Choose from {available_task_names()}.") from exc


def build_dataset(
    task_name: str,
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

    y_train = target_function_1d(x_train, task_name)
    y_val = target_function_1d(x_val, task_name)
    y_plot = target_function_1d(x_plot, task_name)

    return RegressionDataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_plot=x_plot,
        y_plot=y_plot,
    )
