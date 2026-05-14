"""Model definition for exp0513 V1."""

from __future__ import annotations

from collections import OrderedDict
import math

import torch
import torch.nn as nn


def _init_linear_tanh(linear: nn.Linear) -> None:
    """Small Xavier init to delay tanh saturation in early training."""
    gain = nn.init.calculate_gain("tanh")
    nn.init.xavier_uniform_(linear.weight, gain=gain * 0.5)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class SelfModulatedMLP(nn.Module):
    """Shared trunk with task/output heads for self-modulated training."""

    def __init__(
        self,
        input_dim: int = 1,
        trunk_dims: tuple[int, int] = (16, 16),
        y_dim: int = 1,
        q_dim: int = 9,
    ) -> None:
        super().__init__()
        if trunk_dims != (16, 16):
            raise ValueError("V1 plan fixes trunk_dims to (16, 16).")
        if input_dim != 1:
            raise ValueError("V1 plan fixes input_dim to 1.")
        if y_dim != 1:
            raise ValueError("V1 plan fixes y_dim to 1.")

        self.input_dim = input_dim
        self.trunk_dims = trunk_dims
        self.y_dim = y_dim
        self.q_dim = q_dim

        self.trunk = nn.ModuleList([
            nn.Linear(input_dim, trunk_dims[0], bias=True),
            nn.Linear(trunk_dims[0], trunk_dims[1], bias=True),
        ])
        self.y_head = nn.Linear(trunk_dims[-1], y_dim, bias=True)
        self.q_head = nn.Linear(trunk_dims[-1], q_dim, bias=False)
        self.activation = torch.tanh

        for layer in self.trunk:
            _init_linear_tanh(layer)
        _init_linear_tanh(self.y_head)
        _init_linear_tanh(self.q_head)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return task output y and modulator output q."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = x
        for layer in self.trunk:
            h = self.activation(layer(h))
        y = self.activation(self.y_head(h))
        q = self.activation(self.q_head(h))
        return y, q

    def arch_dict(self) -> dict[str, object]:
        """Human-readable architecture metadata for summaries."""
        return {
            "input_dim": self.input_dim,
            "trunk_dims": list(self.trunk_dims),
            "y_dim": self.y_dim,
            "q_dim": self.q_dim,
            "activation": "tanh",
            "q_head_bias": False,
        }

    def controllable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return controllable weights in the fixed V1 flattening order."""
        ordered = []
        for idx, layer in enumerate(self.trunk, start=1):
            ordered.append((f"trunk.{idx}.weight", layer.weight))
        ordered.append(("y_head.weight", self.y_head.weight))
        ordered.append(("q_head.weight", self.q_head.weight))
        return ordered

    def bp_weight_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return controllable weights that still receive BP in V1."""
        ordered = []
        for idx, layer in enumerate(self.trunk, start=1):
            ordered.append((f"trunk.{idx}.weight", layer.weight))
        ordered.append(("y_head.weight", self.y_head.weight))
        return ordered

    def bp_only_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return non-controllable bias parameters updated by plain BP."""
        ordered = []
        for idx, layer in enumerate(self.trunk, start=1):
            ordered.append((f"trunk.{idx}.bias", layer.bias))
        ordered.append(("y_head.bias", self.y_head.bias))
        return ordered

    def named_parameter_dict(self) -> OrderedDict[str, nn.Parameter]:
        """Convenience lookup table used by update/assignment code."""
        out: OrderedDict[str, nn.Parameter] = OrderedDict()
        for name, param in self.controllable_named_parameters():
            out[name] = param
        for name, param in self.bp_only_named_parameters():
            out[name] = param
        return out

    def controllable_edge_count(self) -> int:
        """Total scalar weights in the controllable set."""
        return sum(param.numel() for _, param in self.controllable_named_parameters())


def solve_v1_q_dim() -> int:
    """Solve m = ceil(log2(288 + 16m + 1)) for the fixed V1 architecture."""
    m = 1
    while True:
        n_edges = 288 + 16 * m
        candidate = math.ceil(math.log2(n_edges + 1))
        if candidate == m:
            return m
        m = candidate
