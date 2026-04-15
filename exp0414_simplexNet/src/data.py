"""Target functions for 1D and 2D regression tasks.

Supported tasks:
- sin: y = sin(x)
- sin_mix: y = 0.5*sin(x) + 0.3*sin(2x) + 0.2*sin(3x)
- poly_wave: polynomial + sinusoidal combination
- piecewise: piecewise linear function
- custom: user-defined function via custom_function config
"""

from __future__ import annotations

import torch

from config import Config


def target_function_1d(x: torch.Tensor, task_name: str, custom_fn: str = "") -> torch.Tensor:
    """Compute 1D target function.

    Args:
        x: Input tensor of shape (n, 1) or (n,)
        task_name: Name of the task
        custom_fn: Custom function expression (unused for built-in tasks)

    Returns:
        Target tensor of shape (n, 1)
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    if task_name == "sin":
        return torch.sin(x)
    elif task_name == "sin_mix":
        return 0.5 * torch.sin(x) + 0.3 * torch.sin(2 * x) + 0.2 * torch.sin(3 * x)
    elif task_name == "poly_wave":
        return 0.1 * x**2 * torch.sin(x) + 0.5 * torch.cos(2 * x)
    elif task_name == "piecewise":
        # Piecewise linear with smooth transitions
        y = torch.zeros_like(x)
        mask1 = x < -3
        mask2 = (x >= -3) & (x < 0)
        mask3 = (x >= 0) & (x < 3)
        mask4 = x >= 3
        y[mask1] = -0.5 * x[mask1] - 1.5
        y[mask2] = 0.3 * x[mask2] + 0.5
        y[mask3] = -0.2 * x[mask3] + 0.5
        y[mask4] = 0.4 * x[mask4] - 1.2
        return y
    elif task_name == "custom" and custom_fn:
        # Simple eval-based custom function (for research use)
        # Warning: eval is dangerous in production
        try:
            return eval(custom_fn)
        except Exception as e:
            raise ValueError(f"Failed to evaluate custom_fn: {e}")
    else:
        # Default: sin
        return torch.sin(x)


def target_function_2d(
    x1: torch.Tensor, x2: torch.Tensor, task_name: str, custom_fn: str = ""
) -> torch.Tensor:
    """Compute 2D target function.

    Args:
        x1: First coordinate (n,) or (n, 1)
        x2: Second coordinate (n,) or (n, 1)
        task_name: Name of the task
        custom_fn: Custom function expression

    Returns:
        Target tensor of shape (n, 1)
    """
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)

    if task_name == "sin_sum":
        return torch.sin(x1 + x2)
    elif task_name == "sin_product":
        return torch.sin(x1) * torch.cos(x2)
    elif task_name == "quadratic":
        return 0.1 * (x1**2 + x2**2) - 0.5 * torch.sin(x1 * x2)
    elif task_name == "custom" and custom_fn:
        try:
            return eval(custom_fn)
        except Exception as e:
            raise ValueError(f"Failed to evaluate custom_fn: {e}")
    else:
        # Default: sin_sum
        return torch.sin(x1 + x2)


def build_dataset(config: Config):
    """Build dataset for training, validation, and plotting.

    Args:
        config: Configuration object

    Returns:
        Dataset object with x_train, y_train, x_val, y_val, x_plot, y_plot
    """
    if config.n_in == 1:
        # 1D task
        x_train = torch.linspace(config.x_min, config.x_max, config.num_train).unsqueeze(-1)
        x_val = torch.linspace(config.x_min, config.x_max, config.num_val).unsqueeze(-1)
        x_plot = torch.linspace(config.x_min, config.x_max, config.num_plot).unsqueeze(-1)

        # Shuffle training data
        perm = torch.randperm(config.num_train)
        x_train = x_train[perm]

        y_train = target_function_1d(x_train, config.task_name, config.custom_function)
        y_val = target_function_1d(x_val, config.task_name, config.custom_function)
        y_plot = target_function_1d(x_plot, config.task_name, config.custom_function)

    elif config.n_in == 2:
        # 2D task
        # Generate grid for plotting
        n_side = int(config.num_plot ** 0.5)
        x1_range = torch.linspace(config.x_min, config.x_max, n_side)
        x2_range = torch.linspace(config.x_min, config.x_max, n_side)
        x1_grid, x2_grid = torch.meshgrid(x1_range, x2_range, indexing='ij')

        # Flatten for model input
        x_plot = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=-1)
        y_plot = target_function_2d(x1_grid.flatten(), x2_grid.flatten(), config.task_name, config.custom_function)

        # Random samples for training
        x_train = torch.rand(config.num_train, 2) * (config.x_max - config.x_min) + config.x_min
        x_val = torch.rand(config.num_val, 2) * (config.x_max - config.x_min) + config.x_min

        y_train = target_function_2d(x_train[:, 0], x_train[:, 1], config.task_name, config.custom_function)
        y_val = target_function_2d(x_val[:, 0], x_val[:, 1], config.task_name, config.custom_function)

    else:
        raise ValueError(f"Unsupported n_in: {config.n_in}")

    class Dataset:
        def __init__(self):
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val
            self.x_plot = x_plot
            self.y_plot = y_plot

    return Dataset()


# Quick test
if __name__ == "__main__":
    from config import Config
    config = Config()

    # Test 1D
    config.n_in = 1
    config.task_name = "sin_mix"
    ds = build_dataset(config)
    print(f"1D sin_mix: x_train={ds.x_train.shape}, y_train={ds.y_train.shape}")

    # Test 2D
    config.n_in = 2
    config.task_name = "sin_sum"
    ds2 = build_dataset(config)
    print(f"2D sin_sum: x_train={ds2.x_train.shape}, y_train={ds2.y_train.shape}")
    print(f"  x_plot grid: {ds2.x_plot.shape}")
