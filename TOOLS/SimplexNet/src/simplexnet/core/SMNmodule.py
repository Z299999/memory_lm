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
            self._output_schedule,
            self._hist_width,
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
        level_idx = 0
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
            unique_src = sorted(set(src_list))
            src_pos = {src: i for i, src in enumerate(unique_src)}
            flat_idx = [src_pos[src] * len(nodes) + dst for src, dst in zip(src_list, dst_list)]

            src_name = f"_level_{level_idx}_src"
            flat_name = f"_level_{level_idx}_flat"
            ew_name = f"_level_{level_idx}_ew"
            bias_name = f"_level_{level_idx}_bias"
            self.register_buffer(src_name, torch.tensor(unique_src, dtype=torch.long))
            self.register_buffer(flat_name, torch.tensor(flat_idx, dtype=torch.long))
            self.register_buffer(ew_name, torch.tensor(ew_list, dtype=torch.long))
            self.register_buffer(bias_name, torch.tensor(bias_list, dtype=torch.long))
            level_schedule.append((
                src_name,
                flat_name,
                ew_name,
                bias_name,
                len(unique_src),
                len(nodes),
                write_offset,
            ))
            write_offset += len(nodes)
            level_idx += 1

        # Output mappings: flatten all output fan-ins into one scatter schedule
        output_src_list = []
        output_ew_list = []
        output_dst_list = []
        output_norm_scales = []
        for out_idx, out_node in enumerate(self.graph.output_nodes):
            preds = self.graph.preds[out_node]
            output_src_list.extend(hist_idx[node_to_idx[p]] for p in preds)
            output_ew_list.extend(edge_to_idx[(p, out_node)] for p in preds)
            output_dst_list.extend([out_idx] * len(preds))
            output_norm_scales.append(1.0 / (len(preds) ** 0.5) if self.scale_output else 1.0)

        output_unique_src = sorted(set(output_src_list))
        output_src_pos = {src: i for i, src in enumerate(output_unique_src)}
        output_flat_idx = [
            output_src_pos[src] * self.n_out + dst
            for src, dst in zip(output_src_list, output_dst_list)
        ]

        self.register_buffer("_output_src_idx", torch.tensor(output_unique_src, dtype=torch.long))
        self.register_buffer("_output_ew_idx", torch.tensor(output_ew_list, dtype=torch.long))
        self.register_buffer("_output_flat_idx", torch.tensor(output_flat_idx, dtype=torch.long))
        self.register_buffer("_output_norm_scales", torch.tensor(output_norm_scales, dtype=torch.float32))

        output_schedule = (
            "_output_src_idx",
            "_output_ew_idx",
            "_output_flat_idx",
            "_output_norm_scales",
        )
        return level_schedule, output_schedule, write_offset

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
        hist = x.new_zeros(batch, self._hist_width)
        hist[:, :self.n_in] = x

        # Core nodes, level by level
        for src_name, flat_name, ew_name, bias_name, n_src, n_level, write_start in self._level_schedule:
            src_t = getattr(self, src_name)
            flat_t = getattr(self, flat_name)
            ew_idx_t = getattr(self, ew_name)
            bias_t = getattr(self, bias_name)
            src_states = hist.index_select(1, src_t)                 # (B, n_src)
            weight_flat = self.ew.new_zeros(n_src * n_level)
            weight_flat[flat_t] = self.ew.index_select(0, ew_idx_t)
            weight_matrix = weight_flat.view(n_src, n_level)
            agg = src_states.matmul(weight_matrix)                   # (B, n_level)
            out = self.activation_fn(agg + self.nb[bias_t])          # (B, n_level)
            hist[:, write_start:write_start + n_level] = out

        # Output nodes (variance-preserving aggregation + configurable output activation)
        out_src_name, out_ew_name, out_flat_name, out_scale_name = self._output_schedule
        out_src_t = getattr(self, out_src_name)
        out_ew_t = getattr(self, out_ew_name)
        out_flat_t = getattr(self, out_flat_name)
        out_scale_t = getattr(self, out_scale_name)

        output_src_states = hist.index_select(1, out_src_t)
        output_weight_flat = self.ew.new_zeros(out_src_t.numel() * self.n_out)
        output_weight_flat[out_flat_t] = self.ew.index_select(0, out_ew_t)
        output_weight_matrix = output_weight_flat.view(out_src_t.numel(), self.n_out)
        outputs = output_src_states.matmul(output_weight_matrix)
        outputs = outputs * out_scale_t.unsqueeze(0).to(outputs.dtype)
        return self.output_activation_fn(outputs + self.output_bias)

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
