"""SMN: Simplex Memory Network as a PyTorch module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SimplexMemoryGraph import SimplexMemoryGraph


def _make_activation(name: str):
    """Return activation function by name."""
    name = name.lower()
    if name == "relu":
        return lambda x: F.relu(x)
    elif name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.01)
    elif name == "gelu":
        return lambda x: F.gelu(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    raise ValueError(f"Unsupported activation: {name!r}. Choose relu/leaky_relu/gelu/tanh.")


def _make_output_activation(name: str):
    """Return output activation function by name."""
    name = name.lower()
    if name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported output_activation: {name!r}. Choose tanh/identity.")


class SMN(nn.Module):
    """Simplex Memory Network as a pure PyTorch module.

    Args:
        n: Simplex dimension (>= 2). n=2 -> triangle, n=3 -> tetrahedron.
        m: Resolution (>= 2). Number of lattice points per simplex edge.
        n_in: Number of input dimensions.
        n_out: Number of output dimensions.
        activation: Hidden-node activation ('relu', 'leaky_relu', 'gelu', 'tanh').
        output_activation: Output activation ('identity' or 'tanh').
        scale_output: Whether to apply variance-preserving output scaling.
    """

    def __init__(
        self,
        n: int = 2,
        m: int = 3,
        n_in: int = 1,
        n_out: int = 1,
        activation: str = "relu",
        output_activation: str = "identity",
        scale_output: bool = True,
    ) -> None:
        super().__init__()
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if n_in < 1 or n_out < 1:
            raise ValueError("n_in and n_out must each be >= 1")

        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.output_activation = output_activation
        self.scale_output = scale_output

        self.graph = SimplexMemoryGraph(n=n, m=m, n_in=n_in, n_out=n_out)
        self.activation_fn = _make_activation(activation)
        self.output_activation_fn = _make_output_activation(output_activation)

        n_edges = len(self.graph.edges)
        n_core = len(self.graph.core_nodes)

        # Kaiming initialisation: std = sqrt(2 / fan_in) for each edge weight
        kaiming_stds = torch.tensor([
            (2.0 / len(self.graph.preds[dst])) ** 0.5
            for _, dst in self.graph.edges
        ])
        self.ew = nn.Parameter(torch.randn(n_edges) * kaiming_stds)
        self.nb = nn.Parameter(torch.zeros(n_core))
        self.output_bias = nn.Parameter(torch.zeros(n_out))

        # Pre-build index tensors for the forward pass (no Python loops at runtime)
        (
            self._level_schedule,
            self._output_mappings,
            self._output_norm_scales,
            self._input_node_indices,
        ) = self._build_schedule()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_schedule(self):
        """Pre-compute all index tensors needed for the forward pass."""
        node_to_idx = {node: i for i, node in enumerate(self.graph.nodes)}
        edge_to_idx = {edge: i for i, edge in enumerate(self.graph.edges)}
        core_to_bias = {node: i for i, node in enumerate(self.graph.core_nodes)}

        # Assign a column in the hist buffer to every node
        hist_idx: dict = {}
        col = 0
        for node in self.graph.input_nodes:
            hist_idx[node_to_idx[node]] = col
            col += 1
        for _level, nodes in sorted(self.graph.topological_levels.items()):
            if nodes[0][0] in ("in", "out"):
                continue
            for node in nodes:
                hist_idx[node_to_idx[node]] = col
                col += 1

        # Level schedule: one entry per topological level of core nodes
        level_schedule = []
        write_offset = self.n_in
        for _level, nodes in sorted(self.graph.topological_levels.items()):
            if nodes[0][0] in ("in", "out"):
                continue
            bias_list, src_list, dst_list, ew_list = [], [], [], []
            for i, node in enumerate(nodes):
                bias_list.append(core_to_bias[node])
                for pred in self.graph.preds[node]:
                    src_list.append(hist_idx[node_to_idx[pred]])
                    dst_list.append(i)
                    ew_list.append(edge_to_idx[(pred, node)])
                hist_idx[node_to_idx[node]] = write_offset + i
            level_schedule.append((
                torch.tensor(src_list,  dtype=torch.long),
                torch.tensor(dst_list,  dtype=torch.long).unsqueeze(0),
                torch.tensor(ew_list,   dtype=torch.long),
                torch.tensor(bias_list, dtype=torch.long),
                len(nodes),
                write_offset,
            ))
            write_offset += len(nodes)

        # Output mappings: one per output node
        output_mappings = []
        output_norm_scales = []
        for out_node in self.graph.output_nodes:
            preds = self.graph.preds[out_node]
            output_mappings.append((
                torch.tensor([hist_idx[node_to_idx[p]] for p in preds], dtype=torch.long),
                torch.tensor([edge_to_idx[(p, out_node)] for p in preds], dtype=torch.long),
            ))
            output_norm_scales.append(1.0 / (len(preds) ** 0.5) if self.scale_output else 1.0)

        input_node_indices = [node_to_idx[n] for n in self.graph.input_nodes]
        return level_schedule, output_mappings, output_norm_scales, input_node_indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, n_in)``.
               For n_in=1, also accepts shape ``(batch,)``.

        Returns:
            Output tensor of shape ``(batch, n_out)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.n_in:
            raise ValueError(f"Expected input with last dimension {self.n_in}, got {x.shape[-1]}")
        batch = x.shape[0]

        # Pre-allocated history buffer: [input cols | core cols]
        hist = x.new_zeros(batch, self.n_in + len(self.graph.core_nodes))
        for i in range(self.n_in):
            hist[:, i] = x[:, i]

        # Core nodes, level by level
        for src_t, dst_rows_0, ew_idx_t, bias_t, n_level, write_start in self._level_schedule:
            src_states = hist[:, src_t]                               # (B, n_edges)
            weighted   = src_states * self.ew[ew_idx_t]              # (B, n_edges)
            agg = weighted.new_zeros(batch, n_level).scatter_add(
                1, dst_rows_0.expand(batch, -1), weighted
            )                                                         # (B, n_level)
            out = self.activation_fn(agg + self.nb[bias_t])          # (B, n_level)
            hist[:, write_start:write_start + n_level] = out

        # Output nodes (variance-preserving aggregation + configurable output activation)
        outputs = []
        for i, (out_src_t, out_ew_t) in enumerate(self._output_mappings):
            val = (hist[:, out_src_t] * self.ew[out_ew_t]).sum(1, keepdim=True)
            outputs.append(val * self._output_norm_scales[i])

        return self.output_activation_fn(torch.cat(outputs, dim=1) + self.output_bias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def arch_str(self) -> str:
        """Human-readable architecture description."""
        n_e = self.graph.edge_count
        n_c = self.graph.core_node_count
        n_p = n_e + n_c + self.n_out        # edges + core biases + output biases
        return (
            f"SMN(n={self.n}, m={self.m}, {self.n_in}→{self.n_out}), "
            f"nodes={n_c}, edges={n_e}, params={n_p}"
        )

    def __repr__(self) -> str:
        return f"SMN({self.arch_str})"
