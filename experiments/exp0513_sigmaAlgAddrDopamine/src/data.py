"""Data utilities for exp0513 regression tasks, including fixed MIMO families."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RegressionTaskSpec:
    """Static metadata for one named regression task."""

    name: str
    input_dim: int
    output_dim: int
    family: str


@dataclass
class RegressionDataset:
    """Train/validation/plot splits for a regression task."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_plot: torch.Tensor
    y_plot: torch.Tensor


def _as_2d(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.dim() != 2 or x.shape[1] != input_dim:
        raise ValueError(f"Expected input tensor with shape [n, {input_dim}], got {tuple(x.shape)}.")
    return x


def sin_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 1)
    return torch.sin(x)


def sin_mix_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 1)
    return 0.5 * torch.sin(x) + 0.3 * torch.sin(2.0 * x) + 0.2 * torch.sin(3.0 * x)


def poly_wave_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 1)
    return 0.1 * x**2 * torch.sin(x) + 0.5 * torch.cos(2.0 * x)


def piecewise_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 1)
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


def coupled_trig_2d2d_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 2)
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    y0 = torch.tanh(0.7 * torch.sin(x0) + 0.35 * torch.cos(1.5 * x1) + 0.12 * x0 * x1)
    y1 = torch.tanh(0.65 * torch.cos(x1) - 0.25 * torch.sin(x0 - x1) + 0.08 * x0**2)
    return torch.cat([y0, y1], dim=1)


def cross_poly_2d2d_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 2)
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    y0 = torch.tanh(0.1 * x0**3 - 0.2 * x0 * x1 + 0.55 * torch.sin(x1))
    y1 = torch.tanh(0.12 * x1**3 + 0.18 * x0 * x1 - 0.5 * torch.cos(x0))
    return torch.cat([y0, y1], dim=1)


def piecewise_2d2d_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 2)
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]

    y0 = torch.where(x0 < 0.0, 0.25 * x0 + 0.35 * torch.sign(x1), -0.3 * x0 + 0.15 * x1)
    y1 = torch.where(x1 < 0.0, -0.2 * x1 + 0.18 * x0, 0.3 * x1 - 0.12 * x0)
    return torch.tanh(torch.cat([y0, y1], dim=1))


def mixed_field_3d3d_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 3)
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    y0 = torch.tanh(0.55 * torch.sin(x0) + 0.22 * x1 * x2)
    y1 = torch.tanh(0.45 * torch.cos(x1 - x2) - 0.15 * x0 * x2 + 0.08 * x1**2)
    y2 = torch.tanh(0.4 * torch.sin(x2 + x0) + 0.18 * x0 * x1 - 0.12 * x2**2)
    return torch.cat([y0, y1, y2], dim=1)


def coupled_trig_4d4d_target(x: torch.Tensor) -> torch.Tensor:
    x = _as_2d(x, 4)
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    x3 = x[:, 3:4]
    y0 = torch.tanh(0.45 * torch.sin(x0 + x1) + 0.14 * x2 * x3)
    y1 = torch.tanh(0.42 * torch.cos(x1 - x2) - 0.11 * x0 * x3 + 0.08 * x2)
    y2 = torch.tanh(0.38 * torch.sin(x2 + x3) + 0.13 * x0 * x1 - 0.09 * x3**2)
    y3 = torch.tanh(0.35 * torch.cos(x3 - x0) + 0.1 * x1 * x2 + 0.07 * x0)
    return torch.cat([y0, y1, y2, y3], dim=1)


TASK_SPECS: dict[str, RegressionTaskSpec] = {
    "sin": RegressionTaskSpec("sin", 1, 1, "1d"),
    "sin_mix": RegressionTaskSpec("sin_mix", 1, 1, "1d"),
    "poly_wave": RegressionTaskSpec("poly_wave", 1, 1, "1d"),
    "piecewise": RegressionTaskSpec("piecewise", 1, 1, "1d"),
    "coupled_trig_2d2d": RegressionTaskSpec("coupled_trig_2d2d", 2, 2, "mimo"),
    "cross_poly_2d2d": RegressionTaskSpec("cross_poly_2d2d", 2, 2, "mimo"),
    "piecewise_2d2d": RegressionTaskSpec("piecewise_2d2d", 2, 2, "mimo"),
    "mixed_field_3d3d": RegressionTaskSpec("mixed_field_3d3d", 3, 3, "mimo"),
    "coupled_trig_4d4d": RegressionTaskSpec("coupled_trig_4d4d", 4, 4, "mimo"),
}


TASK_REGISTRY: dict[str, callable] = {
    "sin": sin_target,
    "sin_mix": sin_mix_target,
    "poly_wave": poly_wave_target,
    "piecewise": piecewise_target,
    "coupled_trig_2d2d": coupled_trig_2d2d_target,
    "cross_poly_2d2d": cross_poly_2d2d_target,
    "piecewise_2d2d": piecewise_2d2d_target,
    "mixed_field_3d3d": mixed_field_3d3d_target,
    "coupled_trig_4d4d": coupled_trig_4d4d_target,
}


def available_task_names() -> list[str]:
    """Return supported task names."""
    return sorted(TASK_REGISTRY.keys())


def get_task_spec(task_name: str) -> RegressionTaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_name={task_name!r}. Choose from {available_task_names()}.") from exc


def validate_task_dimensions(task_name: str, input_dim: int, output_dim: int) -> RegressionTaskSpec:
    spec = get_task_spec(task_name)
    if spec.input_dim != input_dim or spec.output_dim != output_dim:
        raise ValueError(
            f"Task {task_name!r} expects input_dim={spec.input_dim}, output_dim={spec.output_dim}, "
            f"but got input_dim={input_dim}, output_dim={output_dim}."
        )
    return spec


def target_function(x: torch.Tensor, task_name: str) -> torch.Tensor:
    try:
        return TASK_REGISTRY[task_name](x)
    except KeyError as exc:
        raise ValueError(f"Unsupported task_name={task_name!r}. Choose from {available_task_names()}.") from exc


def _build_1d_dataset(
    task_name: str,
    num_train: int,
    num_val: int,
    num_plot: int,
    x_min: float,
    x_max: float,
    seed: int,
) -> RegressionDataset:
    generator = torch.Generator().manual_seed(seed)

    x_train = torch.linspace(x_min, x_max, num_train).unsqueeze(-1)
    x_val = torch.linspace(x_min, x_max, num_val).unsqueeze(-1)
    x_plot = torch.linspace(x_min, x_max, num_plot).unsqueeze(-1)

    perm = torch.randperm(num_train, generator=generator)
    x_train = x_train[perm]

    return RegressionDataset(
        x_train=x_train,
        y_train=target_function(x_train, task_name),
        x_val=x_val,
        y_val=target_function(x_val, task_name),
        x_plot=x_plot,
        y_plot=target_function(x_plot, task_name),
    )


def _sample_box(
    num_samples: int,
    input_dim: int,
    x_min: float,
    x_max: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return x_min + (x_max - x_min) * torch.rand((num_samples, input_dim), generator=generator)


def _build_mimo_dataset(
    task_name: str,
    input_dim: int,
    num_train: int,
    num_val: int,
    num_plot: int,
    x_min: float,
    x_max: float,
    seed: int,
) -> RegressionDataset:
    x_train = _sample_box(num_train, input_dim, x_min, x_max, seed)
    x_val = _sample_box(num_val, input_dim, x_min, x_max, seed + 1)
    x_plot = _sample_box(num_plot, input_dim, x_min, x_max, seed + 2)
    return RegressionDataset(
        x_train=x_train,
        y_train=target_function(x_train, task_name),
        x_val=x_val,
        y_val=target_function(x_val, task_name),
        x_plot=x_plot,
        y_plot=target_function(x_plot, task_name),
    )


def build_dataset(
    task_name: str,
    input_dim: int,
    output_dim: int,
    num_train: int = 500,
    num_val: int = 200,
    num_plot: int = 500,
    x_min: float = -2.0 * math.pi,
    x_max: float = 2.0 * math.pi,
    seed: int = 42,
) -> RegressionDataset:
    """Build train/validation/plot splits for 1D or fixed MIMO tasks."""
    spec = validate_task_dimensions(task_name, input_dim, output_dim)
    if spec.family == "1d":
        return _build_1d_dataset(task_name, num_train, num_val, num_plot, x_min, x_max, seed)
    return _build_mimo_dataset(task_name, input_dim, num_train, num_val, num_plot, x_min, x_max, seed)
