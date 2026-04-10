from __future__ import annotations

# This file implements the Triangular Memory Network itself, including node
# blocks, per-edge transforms, and DAG-based forward propagation.

from typing import Dict

import torch
from torch import nn

from config import Config
from model.graph import NodeKey, TMNGraph


class NodeBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # First-stage node block: Linear + ReLU.
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class TMNNetwork(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_in != 1 or config.n_out != 1:
            raise ValueError("TMNNetwork currently supports scalar input and scalar output only.")

        self.config = config
        self.graph = TMNGraph(L=config.L, n_in=config.n_in, n_out=config.n_out)

        # Input heads turn the scalar x into the shared hidden dimension d_model.
        self.input_projections = nn.ModuleDict(
            {
                self._node_name(node): nn.Linear(1, config.d_model)
                for node in self.graph.input_nodes
            }
        )
        # Every directed edge has its own learnable transform.
        self.edge_transforms = nn.ModuleDict(
            {
                self._edge_name(src, dst): nn.Linear(config.d_model, config.d_model)
                for src, dst in self.graph.edges
            }
        )
        # Each core node owns one local computation block.
        self.core_blocks = nn.ModuleDict(
            {
                self._node_name(node): NodeBlock(config.d_model)
                for node in self.graph.core_nodes
            }
        )
        # Output heads map hidden states back to scalar regression outputs.
        self.output_heads = nn.ModuleDict(
            {
                self._node_name(node): nn.Linear(config.d_model, 1)
                for node in self.graph.output_nodes
            }
        )
        self.output_activation = nn.Tanh()

    def _node_name(self, node: NodeKey) -> str:
        return "_".join(str(part) for part in node)

    def _edge_name(self, src: NodeKey, dst: NodeKey) -> str:
        return f"{self._node_name(src)}__to__{self._node_name(dst)}"

    def _aggregate_parents(self, states: Dict[NodeKey, torch.Tensor], node: NodeKey) -> torch.Tensor:
        parent_states = []
        for parent in self.graph.preds[node]:
            transform = self.edge_transforms[self._edge_name(parent, node)]
            parent_states.append(transform(states[parent]))
        # Parent contributions are summed before entering the node block.
        return torch.stack(parent_states, dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states: Dict[NodeKey, torch.Tensor] = {}

        for node in self.graph.input_nodes:
            states[node] = self.input_projections[self._node_name(node)](x)

        # Traverse the triangular DAG level by level.
        for level, nodes in self.graph.topological_levels.items():
            if level == 1 or level == 2 * self.graph.L + 1:
                continue
            for node in nodes:
                aggregated = self._aggregate_parents(states, node)
                states[node] = self.core_blocks[self._node_name(node)](aggregated)

        outputs = []
        for node in self.graph.output_nodes:
            aggregated = self._aggregate_parents(states, node)
            outputs.append(self.output_heads[self._node_name(node)](aggregated))

        # Only the final output head applies tanh, matching the current toy task setup.
        output = torch.cat(outputs, dim=-1)
        return self.output_activation(output)
