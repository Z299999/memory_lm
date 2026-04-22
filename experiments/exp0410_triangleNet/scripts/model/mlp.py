from __future__ import annotations

# This file implements the simple MLP baseline used to compare against the TMN
# on the same toy 1D function-fitting task.

import torch
from torch import nn

from config import Config


class MLPBaseline(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_in < 1 or config.n_out < 1:
            raise ValueError("MLPBaseline requires n_in >= 1 and n_out >= 1.")

        layers = []
        in_dim = config.n_in
        for hidden_dim in config.mlp_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, config.n_out))
        self.network = nn.Sequential(*layers)
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.network(x))
