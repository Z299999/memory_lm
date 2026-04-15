"""PyTorch implementation of simplex memory network.

Each node holds a scalar value. Each edge has one scalar weight.
Forward pass follows topological order, level by level.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from graph import SimplexMemoryGraph, NodeKey  # type: ignore


class SMNNetwork(nn.Module):
    """Simplex Memory Network as a PyTorch module.

    Forward pass:
    - Input nodes receive x[:, i]
    - Core nodes compute: h = σ(Σ w(u→v)*h(u) + b(v))
    - Output nodes compute: y = tanh(Σ w(u→out)*h(u) + b_out)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_in < 1 or config.n_out < 1:
            raise ValueError("SMNNetwork requires n_in >= 1 and n_out >= 1.")

        self.config = config
        self.graph = SimplexMemoryGraph(
            n=config.n,
            m=config.m,
            n_in=config.n_in,
            n_out=config.n_out,
        )

        n_edges = len(self.graph.edges)
        n_core = len(self.graph.core_nodes)

        # Parse activation function
        activation = config.node_activation.lower()
        if activation == "relu":
            self.activation_fn = lambda x: F.relu(x)
        elif activation == "leaky_relu":
            self.activation_fn = lambda x: F.leaky_relu(x, negative_slope=0.01)
        elif activation == "gelu":
            self.activation_fn = lambda x: F.gelu(x)
        elif activation == "tanh":
            self.activation_fn = lambda x: torch.tanh(x)
        else:
            raise ValueError(f"Unsupported node_activation: {config.node_activation}")

        # Kaiming init: std = sqrt(2 / n_parents) for ReLU nodes
        kaiming_stds = torch.tensor([
            (2.0 / len(self.graph.preds[dst])) ** 0.5
            for _, dst in self.graph.edges
        ])
        self.ew = nn.Parameter(torch.randn(n_edges) * kaiming_stds)
        self.nb = nn.Parameter(torch.zeros(n_core))
        self.output_bias = nn.Parameter(torch.zeros(config.n_out))

        # Build mappings for efficient forward pass
        node_to_idx = {node: i for i, node in enumerate(self.graph.nodes)}
        edge_to_idx = {edge: i for i, edge in enumerate(self.graph.edges)}
        core_to_bias = {node: i for i, node in enumerate(self.graph.core_nodes)}

        # Assign each node a column index in hist (topological order)
        hist_idx: dict[NodeKey, int] = {}
        h = 0
        for node in self.graph.input_nodes:
            hist_idx[node_to_idx[node]] = h
            h += 1
        for level, nodes in sorted(self.graph.topological_levels.items()):
            if nodes[0][0] in ("in", "out"):
                continue
            for node in nodes:
                hist_idx[node_to_idx[node]] = h
                h += 1

        # Level schedule for forward pass
        # Pre-compute total core nodes for hist pre-allocation
        n_core_nodes = len(self.graph.core_nodes)

        # Each entry: (src_t, dst_rows_0, ew_idx_t, bias_t, n_level, write_start)
        self._level_schedule: list[tuple] = []
        write_offset = config.n_in  # Start after input nodes
        for level, nodes in sorted(self.graph.topological_levels.items()):
            if nodes[0][0] in ("in", "out"):
                continue
            bias_list, src_list, dst_list, ew_list = [], [], [], []
            for i, node in enumerate(nodes):
                bias_list.append(core_to_bias[node])
                for pred in self.graph.preds[node]:
                    src_list.append(hist_idx[node_to_idx[pred]])
                    dst_list.append(i)
                    ew_list.append(edge_to_idx[(pred, node)])
                # Store write position for this node
                hist_idx[node_to_idx[node]] = write_offset + i
            self._level_schedule.append((
                torch.tensor(src_list, dtype=torch.long),
                torch.tensor(dst_list, dtype=torch.long).unsqueeze(0),
                torch.tensor(ew_list, dtype=torch.long),
                torch.tensor(bias_list, dtype=torch.long),
                len(nodes),
                write_offset,  # Where to write output
            ))
            write_offset += len(nodes)

        # Output mappings: one per output node
        # Store 1/sqrt(n_preds) for variance-preserving normalization
        self._output_mappings: list[tuple] = []
        self._output_norm_scales: list[float] = []
        for out_node in self.graph.output_nodes:
            preds = self.graph.preds[out_node]
            out_src_t = torch.tensor(
                [hist_idx[node_to_idx[p]] for p in preds], dtype=torch.long
            )
            out_ew_t = torch.tensor(
                [edge_to_idx[(p, out_node)] for p in preds], dtype=torch.long
            )
            self._output_mappings.append((out_src_t, out_ew_t))
            self._output_norm_scales.append(1.0 / (len(preds) ** 0.5))

        # Input node mapping
        self._input_node_indices = [
            node_to_idx[node] for node in self.graph.input_nodes
        ]

        # Pre-compile forward for faster training (optional)
        self._use_compiled = False

    def set_compiled(self, use_compiled: bool = True) -> None:
        """Enable or disable compiled forward pass."""
        self._use_compiled = use_compiled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_in)

        Returns:
            Output tensor of shape (batch, n_out)
        """
        if self._use_compiled:
            return _forward_compiled(
                x,
                self.config,
                self.activation_fn,
                self.ew,
                self.nb,
                self.output_bias,
                self._level_schedule,
                self._output_mappings,
                self._output_norm_scales,
                self._input_node_indices,
                len(self.graph.core_nodes),
            )

        # Use standard Python forward
        batch = x.shape[0]

        # Normalize input to [-1, 1] range for stable activation
        # This prevents ReLU/gelu saturation from large inputs
        if self.config.n_in == 1:
            # For 1D input: map [x_min, x_max] to [-1, 1]
            x_min = self.config.x_min
            x_max = self.config.x_max
            x = 2 * (x - x_min) / (x_max - x_min) - 1

        # Pre-allocate hist: [input nodes + all core nodes]
        total_hist_size = self.config.n_in + len(self.graph.core_nodes)
        hist = x.new_zeros(batch, total_hist_size)

        # Write inputs to their positions
        for i, node_idx in enumerate(self._input_node_indices):
            hist[:, i] = x[:, i]

        # Process core nodes level by level, writing directly to pre-allocated hist
        for src_t, dst_rows_0, ew_idx_t, bias_t, n_level, write_start in self._level_schedule:
            src_states = hist[:, src_t]                          # (batch, n_edges)
            weighted = src_states * self.ew[ew_idx_t]           # (batch, n_edges)
            agg = weighted.new_zeros(batch, n_level).scatter_add(
                1, dst_rows_0.expand(batch, -1), weighted
            )                                                    # (batch, n_level)
            out = self.activation_fn(agg + self.nb[bias_t])     # (batch, n_level)
            hist[:, write_start:write_start + n_level] = out    # Direct write, no cat

        # Compute outputs with variance-preserving normalization
        outputs = []
        for i, (out_src_t, out_ew_t) in enumerate(self._output_mappings):
            src_states = hist[:, out_src_t]                     # (batch, n_preds)
            out_weights = self.ew[out_ew_t]                     # Get actual edge weights
            out_val = (src_states * out_weights).sum(1, keepdim=True)
            out_val = out_val * self._output_norm_scales[i]     # Normalize by 1/sqrt(n_preds)
            outputs.append(out_val)

        output = torch.cat(outputs, dim=1)                      # (batch, n_out)
        return torch.tanh(output + self.output_bias)


# Compiled forward for faster training
@torch.compile(mode="reduce-overhead")
def _forward_compiled(
    x: torch.Tensor,
    config,
    activation_fn,
    ew: torch.Tensor,
    nb: torch.Tensor,
    output_bias: torch.Tensor,
    level_schedule: list,
    output_mappings: list,
    output_norm_scales: list,
    input_node_indices: list,
    n_core: int,
) -> torch.Tensor:
    """Compiled forward pass."""
    batch = x.shape[0]

    # Normalize input
    if config.n_in == 1:
        x_min = config.x_min
        x_max = config.x_max
        x = 2 * (x - x_min) / (x_max - x_min) - 1

    # Pre-allocate hist
    total_hist_size = config.n_in + n_core
    hist = x.new_zeros(batch, total_hist_size)

    # Write inputs
    for i, node_idx in enumerate(input_node_indices):
        hist[:, i] = x[:, i]

    # Process levels
    for src_t, dst_rows_0, ew_idx_t, bias_t, n_level, write_start in level_schedule:
        src_states = hist[:, src_t]
        weighted = src_states * ew[ew_idx_t]
        agg = weighted.new_zeros(batch, n_level).scatter_add(
            1, dst_rows_0.expand(batch, -1), weighted
        )
        out = activation_fn(agg + nb[bias_t])
        hist[:, write_start:write_start + n_level] = out

    # Output
    outputs = []
    for i, (out_src_t, out_ew_t) in enumerate(output_mappings):
        src_states = hist[:, out_src_t]
        out_weights = ew[out_ew_t]
        out_val = (src_states * out_weights).sum(1, keepdim=True)
        out_val = out_val * output_norm_scales[i]
        outputs.append(out_val)

    output = torch.cat(outputs, dim=1)
    return torch.tanh(output + output_bias)

    def _node_name(self, node: NodeKey) -> str:
        """Get string name for a node (for visualization)."""
        return "_".join(str(p) for p in node)

    def _edge_name(self, src: NodeKey, dst: NodeKey) -> str:
        """Get string name for an edge (for visualization)."""
        return f"{self._node_name(src)}__to__{self._node_name(dst)}"
