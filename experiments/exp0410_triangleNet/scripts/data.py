from __future__ import annotations

# This file defines the toy regression dataset, including the target function
# being fitted and the input normalization used by the training script.

from dataclasses import dataclass

import torch

from config import Config


@dataclass
class DatasetBundle:
    # Training, validation, and plotting tensors are prepared once here so the
    # train/eval scripts can stay simple.
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_plot: torch.Tensor
    y_plot: torch.Tensor


def target_function(x: torch.Tensor, task_name: str, n_out: int = 1) -> torch.Tensor:
    # This is the main place to read or edit built-in fitting targets.
    # x shape: (batch, n_in)
    # Returns: (batch, n_out) for n_out >= 1

    # === 1D → 1D functions (n_in = 1) ===
    # These functions expect scalar input. For n_in > 1, use only the first dimension.
    if task_name in ("sin", "sin_mix", "poly_wave", "piecewise"):
        x1 = x[:, 0:1] if x.dim() > 1 and x.shape[1] > 1 else x
        base = None

        if task_name == "sin":
            base = torch.sin(x1)
        elif task_name == "sin_mix":
            base = 0.6 * torch.sin(x1) + 0.3 * torch.sin(3 * x1 + 0.4) + 0.1 * torch.cos(5 * x1)
        elif task_name == "poly_wave":
            base = 0.15 * (x1 ** 3) / 10.0 + 0.6 * torch.sin(2 * x1)
        elif task_name == "piecewise":
            base = torch.where(
                x1 < -1.5,
                -0.8 + 0.15 * x1,
                torch.where(x1 < 1.5, 0.5 * torch.sin(3 * x1), 0.8 - 0.15 * x1),
            )

        if n_out == 1:
            return base
        else:
            y = torch.zeros(x.shape[0], n_out)
            y[:, 0:1] = base
            for i in range(1, n_out):
                y[:, i:i+1] = base * (0.5 ** i)
            return y

    # 2D → 1D functions (n_in = 2)
    if task_name == "sin2d":
        x1, x2 = x[:, 0:1], x[:, 1:2]
        if n_out == 1:
            return torch.sin(x1) * torch.cos(x2)
        else:
            # Multi-output: different frequency combinations
            y = torch.zeros(x.shape[0], n_out)
            for i in range(n_out):
                freq = i + 1
                y[:, i:i+1] = torch.sin(freq * x1) * torch.cos(freq * x2)
            return y

    if task_name == "distance2d":
        x1, x2 = x[:, 0:1], x[:, 1:2]
        r = torch.sqrt(x1 ** 2 + x2 ** 2)
        if n_out == 1:
            return torch.sin(r)
        else:
            y = torch.zeros(x.shape[0], n_out)
            for i in range(n_out):
                freq = i + 1
                y[:, i:i+1] = torch.sin(freq * r)
            return y

    if task_name == "saddle2d":
        x1, x2 = x[:, 0:1], x[:, 1:2]
        if n_out == 1:
            return (x1 ** 2 - x2 ** 2) / 10.0
        else:
            # Multi-output: different scales/frequencies
            y = torch.zeros(x.shape[0], n_out)
            for i in range(n_out):
                scale = i + 1
                y[:, i:i+1] = (x1 ** 2 - x2 ** 2) * scale / 10.0
            return y

    if task_name == "peakvalley2d":
        x1, x2 = x[:, 0:1], x[:, 1:2]
        if n_out == 1:
            return torch.exp(-(x1 ** 2 + x2 ** 2)) - 0.5 * torch.exp(-((x1 - 1) ** 2 + (x2 + 1) ** 2))
        else:
            # Multi-output: different relative positions of peaks
            y = torch.zeros(x.shape[0], n_out)
            for i in range(n_out):
                offset = i * 0.5
                y[:, i:i+1] = (
                    torch.exp(-((x1 - offset) ** 2 + (x2 + offset) ** 2)) -
                    0.5 * torch.exp(-((x1 - 1 + offset) ** 2 + (x2 + 1 - offset) ** 2))
                )
            return y

    if task_name == "mix2d":
        x1, x2 = x[:, 0:1], x[:, 1:2]
        if n_out == 1:
            return 0.5 * torch.sin(2 * x1) + 0.3 * torch.sin(5 * x2) + 0.2 * torch.cos(3 * (x1 + x2))
        else:
            # Multi-output: different frequency combinations
            y = torch.zeros(x.shape[0], n_out)
            for i in range(n_out):
                freq = i + 1
                y[:, i:i+1] = (
                    0.5 * torch.sin(freq * 2 * x1) +
                    0.3 * torch.sin(freq * 5 * x2) +
                    0.2 * torch.cos(freq * 3 * (x1 + x2))
                )
            return y

    raise ValueError(f"Unsupported built-in task_name: {task_name}")


def target_function_from_config(x: torch.Tensor, config: Config) -> torch.Tensor:
    # Simplest rule:
    # 1. If custom_function is provided, use it directly.
    # 2. Otherwise use one of the built-in task_name functions above.
    #
    # Example custom_function:
    #   torch.sin(2 * x) + 0.2 * torch.cos(7 * x)
    if not config.custom_function.strip():
        return target_function(x, config.task_name, config.n_out)

    allowed_globals = {"torch": torch, "__builtins__": {}}
    allowed_locals = {"x": x}
    y = eval(config.custom_function, allowed_globals, allowed_locals)
    if not isinstance(y, torch.Tensor):
        raise ValueError("custom_function must evaluate to a torch.Tensor")
    return y


def normalize_input(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    # We normalize x into a stable range before feeding it into the network.
    # For 2D input (x shape: batch, 2), normalize each dimension independently.
    scale = max(abs(x_min), abs(x_max))
    return x / scale


def build_dataset(config: Config) -> DatasetBundle:
    n_in = config.n_in

    if n_in == 1:
        # 1D input
        x_train = torch.linspace(config.x_min, config.x_max, steps=config.num_train).unsqueeze(-1)
        x_val = torch.linspace(config.x_min, config.x_max, steps=config.num_val).unsqueeze(-1)
        x_plot = torch.linspace(config.x_min, config.x_max, steps=config.num_plot).unsqueeze(-1)
    elif n_in == 2:
        # 2D input: generate grid
        import math
        n_side = int(math.sqrt(config.num_train))
        x1 = torch.linspace(config.x_min, config.x_max, steps=n_side)
        x2 = torch.linspace(config.x_min, config.x_max, steps=n_side)
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        x_train = torch.stack([X1.flatten(), X2.flatten()], dim=1)  # (n, 2)

        n_side_val = int(math.sqrt(config.num_val))
        x1_val = torch.linspace(config.x_min, config.x_max, steps=n_side_val)
        x2_val = torch.linspace(config.x_min, config.x_max, steps=n_side_val)
        X1_val, X2_val = torch.meshgrid(x1_val, x2_val, indexing='ij')
        x_val = torch.stack([X1_val.flatten(), X2_val.flatten()], dim=1)

        n_side_plot = int(math.sqrt(config.num_plot))
        x1_plot = torch.linspace(config.x_min, config.x_max, steps=n_side_plot)
        x2_plot = torch.linspace(config.x_min, config.x_max, steps=n_side_plot)
        X1_plot, X2_plot = torch.meshgrid(x1_plot, x2_plot, indexing='ij')
        x_plot = torch.stack([X1_plot.flatten(), X2_plot.flatten()], dim=1)

        # Store grid info for plotting
        x_plot_grid = (x1_plot, x2_plot, X1_plot, X2_plot)
    else:
        raise ValueError(f"n_in={n_in} not supported yet")

    # y is computed from the raw x values so the plotted target remains
    # the original mathematical function.
    y_train = target_function_from_config(x_train, config)
    y_val = target_function_from_config(x_val, config)
    y_plot = target_function_from_config(x_plot, config)

    bundle = DatasetBundle(
        # The model sees normalized x, but it still learns to predict the
        # original target y.
        x_train=normalize_input(x_train, config.x_min, config.x_max),
        y_train=y_train,
        x_val=normalize_input(x_val, config.x_min, config.x_max),
        y_val=y_val,
        x_plot=normalize_input(x_plot, config.x_min, config.x_max),
        y_plot=y_plot,
    )

    # Attach grid info for 2D plotting
    if n_in == 2:
        bundle.x_plot_grid = x_plot_grid  # type: ignore

    return bundle
