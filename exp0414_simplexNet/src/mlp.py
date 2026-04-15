"""MLP baseline for comparison with SMN.

Simple feedforward network with configurable hidden layers.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from config import Config


class MLPBaseline(nn.Module):
    """MLP baseline for comparison.

    Architecture:
    - Input layer (n_in dimensions)
    - Hidden layers (mlp_layers)
    - Output layer (n_out dimensions)
    - Output activation: tanh
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_in < 1 or config.n_out < 1:
            raise ValueError("MLPBaseline requires n_in >= 1 and n_out >= 1.")

        # Parse activation function
        activation = config.node_activation.lower()
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported node_activation: {config.node_activation}")

        # Build network
        layers = []
        in_dim = config.n_in
        for hidden_dim in config.mlp_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, config.n_out))

        self.network = nn.Sequential(*layers)
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.network(x))
