"""Model definition for exp0513 hidden-dopamine experiments."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def _init_linear_tanh(linear: nn.Linear) -> None:
    """Small Xavier init to delay tanh saturation in early training."""
    gain = nn.init.calculate_gain("tanh")
    nn.init.xavier_uniform_(linear.weight, gain=gain * 0.5)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class SelfModulatedMLP(nn.Module):
    """Shared trunk whose hidden neurons can act as dopamine modulators."""

    def __init__(
        self,
        input_dim: int = 1,
        trunk_dims: tuple[int, int] = (16, 16),
        y_dim: int = 1,
    ) -> None:
        super().__init__()
        if trunk_dims != (16, 16):
            raise ValueError("The current exp0513 plan fixes trunk_dims to (16, 16).")
        if input_dim != 1:
            raise ValueError("The current exp0513 plan fixes input_dim to 1.")
        if y_dim != 1:
            raise ValueError("The current exp0513 plan fixes y_dim to 1.")

        self.input_dim = input_dim
        self.trunk_dims = trunk_dims
        self.y_dim = y_dim

        self.trunk = nn.ModuleList([
            nn.Linear(input_dim, trunk_dims[0], bias=True),
            nn.Linear(trunk_dims[0], trunk_dims[1], bias=True),
        ])
        self.y_head = nn.Linear(trunk_dims[-1], y_dim, bias=True)
        self.activation = torch.tanh

        for layer in self.trunk:
            _init_linear_tanh(layer)
        _init_linear_tanh(self.y_head)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return task output y and the hidden activations for dopamine extraction."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        h = x
        hidden_states: list[torch.Tensor] = []
        for layer in self.trunk:
            h = self.activation(layer(h))
            hidden_states.append(h)
        y = self.activation(self.y_head(h))
        return y, hidden_states

    def arch_dict(self) -> dict[str, object]:
        """Human-readable architecture metadata for summaries."""
        return {
            "input_dim": self.input_dim,
            "trunk_dims": list(self.trunk_dims),
            "y_dim": self.y_dim,
            "activation": "tanh",
            "hidden_pool_size": self.hidden_pool_size(),
            "dopamine_source": "hidden_activation",
        }

    def hidden_node_ids(self) -> list[str]:
        """Enumerate hidden neurons in a stable front-end friendly order."""
        node_ids: list[str] = []
        for layer_idx, layer_dim in enumerate(self.trunk_dims, start=1):
            for neuron_idx in range(layer_dim):
                node_ids.append(f"h{layer_idx}_{neuron_idx}")
        return node_ids

    def hidden_pool_size(self) -> int:
        """Total number of hidden neurons eligible for dopamine sampling."""
        return len(self.hidden_node_ids())

    @staticmethod
    def parse_hidden_node_id(node_id: str) -> tuple[int, int]:
        """Map an id like h2_7 to (layer_index, neuron_index)."""
        if not node_id.startswith("h") or "_" not in node_id:
            raise ValueError(f"Invalid hidden node id: {node_id}")
        layer_str, neuron_str = node_id[1:].split("_", maxsplit=1)
        layer_index = int(layer_str) - 1
        neuron_index = int(neuron_str)
        return layer_index, neuron_index

    def collect_dopamine_batch(
        self,
        hidden_states: list[torch.Tensor],
        dopamine_node_ids: list[str],
    ) -> torch.Tensor:
        """Gather the selected hidden activations into a batch x m tensor."""
        if not dopamine_node_ids:
            raise ValueError("dopamine_node_ids cannot be empty.")
        columns = []
        for node_id in dopamine_node_ids:
            layer_index, neuron_index = self.parse_hidden_node_id(node_id)
            columns.append(hidden_states[layer_index][:, neuron_index])
        return torch.stack(columns, dim=1)

    def controllable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Return controllable weights in the fixed flattening order."""
        ordered = []
        for idx, layer in enumerate(self.trunk, start=1):
            ordered.append((f"trunk.{idx}.weight", layer.weight))
        ordered.append(("y_head.weight", self.y_head.weight))
        return ordered

    def bp_weight_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """All controllable weights continue to receive BP in the current regime."""
        return self.controllable_named_parameters()

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
