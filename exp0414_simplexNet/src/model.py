"""Thin Config-aware wrapper around SMNModule.

Keeps backward compatibility with train.py (which builds models via Config).
All network logic lives in SMNModule; this file only bridges Config → SMNModule.
"""

from __future__ import annotations

import torch
from torch import nn

from config import Config       # type: ignore
from smn_module import SMNModule  # type: ignore


class SMNNetwork(nn.Module):
    """Config-aware wrapper around SMNModule.

    Instantiates an SMNModule from a Config object and delegates all
    forward-pass logic to it.  Exposes ``self.graph`` for architecture
    introspection (used by plot helpers).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_in < 1 or config.n_out < 1:
            raise ValueError("SMNNetwork requires n_in >= 1 and n_out >= 1.")

        bounds = config.resolved_x_bounds
        self._module = SMNModule(
            n=config.n,
            m=config.m,
            n_in=config.n_in,
            n_out=config.n_out,
            activation=config.node_activation,
            x_bounds=bounds,
        )
        # Expose graph for architecture text helpers in plot.py
        self.graph = self._module.graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._module(x)
