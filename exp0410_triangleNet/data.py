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


def target_function(x: torch.Tensor, task_name: str) -> torch.Tensor:
    # Change the fitted function here.
    # Example:
    #   return x ** 2
    #   return torch.sin(2 * x)
    #   return torch.where(x < 0, -x, x)
    if task_name != "sin":
        raise ValueError(f"Unsupported task_name: {task_name}")
    return torch.sin(x)


def normalize_input(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    # We normalize x into a stable range before feeding it into the network.
    scale = max(abs(x_min), abs(x_max))
    return x / scale


def build_dataset(config: Config) -> DatasetBundle:
    # These are the raw x-coordinates before normalization.
    x_train = torch.linspace(config.x_min, config.x_max, steps=config.num_train).unsqueeze(-1)
    x_val = torch.linspace(config.x_min, config.x_max, steps=config.num_val).unsqueeze(-1)
    x_plot = torch.linspace(config.x_min, config.x_max, steps=config.num_plot).unsqueeze(-1)

    # y is always computed from the raw x values so the plotted target remains
    # the original mathematical function.
    y_train = target_function(x_train, config.task_name)
    y_val = target_function(x_val, config.task_name)
    y_plot = target_function(x_plot, config.task_name)

    return DatasetBundle(
        # The model sees normalized x, but it still learns to predict the
        # original target y.
        x_train=normalize_input(x_train, config.x_min, config.x_max),
        y_train=y_train,
        x_val=normalize_input(x_val, config.x_min, config.x_max),
        y_val=y_val,
        x_plot=normalize_input(x_plot, config.x_min, config.x_max),
        y_plot=y_plot,
    )
