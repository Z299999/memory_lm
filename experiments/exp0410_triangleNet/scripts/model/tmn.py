from __future__ import annotations

# Each node holds a scalar value. Each edge has one scalar weight.
# node(v) = ReLU( sum(w(u->v) * node(u) for u in parents(v)) + bias(v) )
# output  = Tanh( sum(w(u->out) * node(u)) + bias_out )
#
# hist tensor layout: (batch, n_processed_nodes).
# Per level: hist[:, src_t] selects parent states — one indexing op, no unsqueeze.
# self.ew[ew_idx_t] broadcasts naturally with (batch, n_edges) without unsqueeze.
# self.nb[bias_t]   broadcasts naturally with (batch, n_level)  without unsqueeze.
# dst_rows_0 = dst_rows.unsqueeze(0) is precomputed in __init__; only .expand in forward.

import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from model.graph import NodeKey, TMNGraph


class TMNNetwork(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_in < 1 or config.n_out < 1:
            raise ValueError("TMNNetwork requires n_in >= 1 and n_out >= 1.")

        self.config = config
        self.graph  = TMNGraph(
            L=config.L,
            n_in=config.n_in,
            n_out=config.n_out,
            depth=config.depth,
            cross_layer_mode=config.cross_layer_mode,
        )

        n_edges = len(self.graph.edges)
        n_core  = len(self.graph.core_nodes)

        # Parse node_activation from config
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

        # Kaiming init: std = sqrt(2 / n_parents) per edge, so ReLU nodes
        # maintain signal variance regardless of path length in the DAG.
        kaiming_stds = torch.tensor([
            (2.0 / len(self.graph.preds[dst])) ** 0.5
            for _, dst in self.graph.edges
        ])
        self.ew          = nn.Parameter(torch.randn(n_edges) * kaiming_stds)
        self.nb          = nn.Parameter(torch.zeros(n_core))
        self.output_bias = nn.Parameter(torch.zeros(config.n_out))

        node_to_idx  = {node: i for i, node in enumerate(self.graph.nodes)}
        edge_to_idx  = {edge: i for i, edge in enumerate(self.graph.edges)}
        core_to_bias = {node: i for i, node in enumerate(self.graph.core_nodes)}

        # Assign each node a column index in hist (topological order)
        hist_idx: dict[int, int] = {}
        h = 0
        for node in self.graph.input_nodes:
            hist_idx[node_to_idx[node]] = h;  h += 1
        for level in sorted(self.graph.topological_levels):
            # Skip input level (level 0) and output level (highest level)
            nodes_at_level = self.graph.topological_levels[level]
            if nodes_at_level[0][0] == 'in' or nodes_at_level[0][0] == 'out':
                continue
            for node in nodes_at_level:
                hist_idx[node_to_idx[node]] = h;  h += 1

        # Level schedule
        # Each entry: (src_t, dst_rows_0, ew_idx_t, bias_t, n_level)
        #   src_t     : LongTensor(n_edges)   — hist column index of each edge's source
        #   dst_rows_0: LongTensor(1, n_edges) — dst node within level, pre-unsqueezed
        #   ew_idx_t  : LongTensor(n_edges)   — index into self.ew
        #   bias_t    : LongTensor(n_level)   — index into self.nb
        #   n_level   : int
        self._level_schedule: list[tuple] = []
        for level in sorted(self.graph.topological_levels):
            nodes     = self.graph.topological_levels[level]
            # Skip input and output levels
            if nodes[0][0] == 'in' or nodes[0][0] == 'out':
                continue
            bias_list, src_list, dst_list, ew_list = [], [], [], []
            for i, node in enumerate(nodes):
                bias_list.append(core_to_bias[node])
                for p in self.graph.preds[node]:
                    src_list.append(hist_idx[node_to_idx[p]])
                    dst_list.append(i)
                    ew_list.append(edge_to_idx[(p, node)])
            self._level_schedule.append((
                torch.tensor(src_list,  dtype=torch.long),
                torch.tensor(dst_list,  dtype=torch.long).unsqueeze(0),  # (1, n_edges)
                torch.tensor(ew_idx_list := ew_list,  dtype=torch.long),
                torch.tensor(bias_list, dtype=torch.long),
                len(nodes),
            ))

        # Output mappings: one per output node
        self._output_mappings = []
        for out_node in self.graph.output_nodes:
            preds = self.graph.preds[out_node]
            out_src_t = torch.tensor([hist_idx[node_to_idx[p]] for p in preds], dtype=torch.long)
            out_ew_t = torch.tensor([edge_to_idx[(p, out_node)] for p in preds], dtype=torch.long)
            self._output_mappings.append((out_src_t, out_ew_t))

        # Input node mapping: input node i receives x[:, i]
        self._input_node_indices = [node_to_idx[node] for node in self.graph.input_nodes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_in)
        batch = x.shape[0]
        # Build initial hist from input nodes
        # Each input node gets one dimension of x
        hist_cols = []
        for i, node_idx in enumerate(self._input_node_indices):
            hist_cols.append(x[:, i:i+1])  # keep dim for concat
        hist = torch.cat(hist_cols, dim=1)  # (batch, n_in)

        for src_t, dst_rows_0, ew_idx_t, bias_t, n_level in self._level_schedule:
            src_states = hist[:, src_t]                         # (batch, n_edges)
            weighted   = src_states * self.ew[ew_idx_t]        # broadcast, no unsqueeze
            agg = weighted.new_zeros(batch, n_level).scatter_add(
                1, dst_rows_0.expand(batch, -1), weighted
            )                                                   # (batch, n_level)
            out  = self.activation_fn(agg + self.nb[bias_t])
            hist = torch.cat([hist, out], dim=1)                # (batch, n_hist + n_level)

        # Multi-output: compute each output dimension independently
        outputs = []
        for _, (out_src_t, out_ew_t) in enumerate(self._output_mappings):
            src_states = hist[:, out_src_t]                     # (batch, n_preds)
            out_val = (src_states * self.ew[out_ew_t]).sum(1, keepdim=True)
            outputs.append(out_val)

        output = torch.cat(outputs, dim=1)                      # (batch, n_out)
        return torch.tanh(output + self.output_bias)            # broadcast: (batch, n_out) + (n_out,)

    # kept for plot.py
    def _node_name(self, node: NodeKey) -> str:
        return "_".join(str(p) for p in node)

    def _edge_name(self, src: NodeKey, dst: NodeKey) -> str:
        return f"{self._node_name(src)}__to__{self._node_name(dst)}"
