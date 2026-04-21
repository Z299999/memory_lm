"""SMNmodule: Simplex Memory Network as a PyTorch module.

This module provides the core network implementation for TOOLS/SimplexNet.
It is self-contained — the SimplexMemoryGraph class is defined internally
with no external dependencies within the package.

Usage::

    from TOOLS.SimplexNet import SMNmodule

    # SISO function approximation
    module = SMNmodule(n=2, m=4, n_in=1, n_out=1)

    # DQN Q-network (CartPole)
    q_net = SMNmodule(n=2, m=4, n_in=4, n_out=2)

    # Forward pass
    output = module(input_tensor)
"""

from __future__ import annotations

from collections import defaultdict
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Type definitions
# =============================================================================

NodeKey: TypeAlias = tuple[str, ...] | tuple[str, int]  # ("core", α) or ("in"/"out", i)
Edge = tuple[NodeKey, NodeKey]


# =============================================================================
# Lattice generation (internal)
# =============================================================================

def _generate_lattice(n: int, m: int) -> list[tuple[int, ...]]:
    """Generate V_{n,m} = {α ∈ ℤ_≥0^{n+1} : Σαᵢ = m-1}.

    Uses recursive enumeration (stars-and-bars conceptually).

    Args:
        n: Simplex dimension (number of vertices is n+1)
        m: Resolution parameter (each edge has m lattice points)

    Returns:
        List of tuples α = (α₀, ..., αₙ) with Σαᵢ = m-1.
        The cardinality is C(m+n-1, n).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 2:
        raise ValueError(f"m must be >= 2, got {m}")

    s = m - 1  # Sum of coordinates
    result = []

    def helper(k: int, remaining: int, current: list[int]) -> None:
        """Enumerate (α₀, ..., αₖ) with sum = remaining."""
        if k == 0:
            current.append(remaining)
            result.append(tuple(current))
            current.pop()
            return

        for α_k in range(remaining + 1):
            current.append(α_k)
            helper(k - 1, remaining - α_k, current)
            current.pop()

    helper(n, s, [])
    return result


def _cardinality(n: int, m: int) -> int:
    """Return |V_{n,m}| = C(m+n-1, n)."""
    from math import comb
    return comb(m + n - 1, n)


# =============================================================================
# Potential functions and edge orientation (internal)
# =============================================================================

def _vertex_potentials(n: int) -> dict[int, int]:
    """Return vertex potentials β = {0: 0, 1: 2, 2..n: 1}."""
    beta = {0: 0, 1: 2}
    for k in range(2, n + 1):
        beta[k] = 1
    return beta


def _node_potential(alpha: tuple[int, ...], beta: dict[int, int]) -> int:
    """Compute H(α) = Σᵢ βᵢ αᵢ."""
    return sum(beta[i] * alpha[i] for i in range(len(alpha)))


def _move_mass(alpha: tuple[int, ...], i: int, j: int) -> tuple[int, ...]:
    """Compute α' = α + eᵢ - eⱼ."""
    alpha_list = list(alpha)
    alpha_list[i] += 1
    alpha_list[j] -= 1
    return tuple(alpha_list)


def _get_admissible_edges(
    alpha: tuple[int, ...],
    beta: dict[int, int]
) -> list[tuple[int, int]]:
    """Get all admissible outgoing edges from α."""
    n = len(alpha) - 1
    edges = []
    for j in range(n + 1):
        if alpha[j] < 1:
            continue
        for i in range(n + 1):
            if i == j:
                continue
            if beta[i] > beta[j]:
                edges.append((i, j))
    return edges


def _F_in(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return input facet F_in = {α : α₁ = 0}."""
    return [α for α in lattice if α[1] == 0]


def _F_out(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return output facet F_out = {α : α₀ = 0}."""
    return [α for α in lattice if α[0] == 0]


def _F_mid(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return shared face F_mid = F_in ∩ F_out = {α : α₀ = α₁ = 0}."""
    return [α for α in lattice if α[0] == 0 and α[1] == 0]


def _backbone(lattice: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Return backbone B_{n,m} = {α : α₂ = ... = αₙ = 0}."""
    return [α for α in lattice if all(α[k] == 0 for k in range(2, len(α)))]


# =============================================================================
# SimplexMemoryGraph class
# =============================================================================

class SimplexMemoryGraph:
    """Directed acyclic graph for simplex memory network SMN(n,m).

    Attributes:
        n: Simplex dimension
        m: Resolution parameter
        n_in: Number of input nodes
        n_out: Number of output nodes
        core_nodes: List of core (hidden) nodes
        input_nodes: List of input nodes
        output_nodes: List of output nodes
        nodes: All nodes (input + core + output)
        edges: List of directed edges (src, dst)
        preds: Predecessor map dst → [src₁, src₂, ...]
        succs: Successor map src → [dst₁, dst₂, ...]
        topological_levels: Dict level → [nodes at that level]
    """

    def __init__(self, n: int, m: int, n_in: int = 1, n_out: int = 1) -> None:
        """Initialize the simplex memory graph.

        Args:
            n: Simplex dimension (vertices: a, b, c₁, ..., c_{n-1})
            m: Resolution (each edge has m lattice points)
            n_in: Number of input nodes
            n_out: Number of output nodes
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if n_in < 1:
            raise ValueError(f"n_in must be >= 1, got {n_in}")
        if n_out < 1:
            raise ValueError(f"n_out must be >= 1, got {n_out}")

        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out

        # Generate lattice and create node keys
        self._lattice = _generate_lattice(n, m)
        self._beta = _vertex_potentials(n)

        # Node keys: ("core", α₀, ..., αₙ) for compact representation
        self.core_nodes = [("core",) + α for α in self._lattice]
        self.input_nodes = [("in", i) for i in range(n_in)]
        self.output_nodes = [("out", i) for i in range(n_out)]
        self.nodes = self.input_nodes + self.core_nodes + self.output_nodes

        # Build graph structure
        self.edges = self._build_edges()
        self.preds = self._build_adjacency(reverse=True)
        self.succs = self._build_adjacency(reverse=False)
        self.topological_levels = self._build_topological_levels()

        # Validate
        self._validate()

    def _node_key(self, alpha: tuple[int, ...]) -> NodeKey:
        """Convert lattice point to node key."""
        return ("core",) + alpha

    def _alpha(self, node_key: NodeKey) -> tuple[int, ...]:
        """Extract lattice point from node key."""
        return node_key[1:]

    def _build_edges(self) -> list[Edge]:
        """Build directed edges using potential orientation.

        For each α and each admissible (i,j), add edge:
        α → α + eᵢ - eⱼ

        Also add input→core and core→output edges.
        """
        edges: list[Edge] = []

        # Internal edges (core → core)
        for α in self._lattice:
            src = self._node_key(α)
            for i, j in _get_admissible_edges(α, self._beta):
                α_prime = _move_mass(α, i, j)
                dst = self._node_key(α_prime)
                edges.append((src, dst))

        # Input edges: connect each input to all nodes in F_in
        f_in_nodes = [self._node_key(α) for α in _F_in(self._lattice)]
        for inp in self.input_nodes:
            for core in f_in_nodes:
                edges.append((inp, core))

        # Output edges: connect all nodes in F_out to each output
        f_out_nodes = [self._node_key(α) for α in _F_out(self._lattice)]
        for out in self.output_nodes:
            for core in f_out_nodes:
                edges.append((core, out))

        return edges

    def _build_adjacency(self, reverse: bool) -> dict[NodeKey, list[NodeKey]]:
        """Build predecessor or successor map.

        Args:
            reverse: If True, build dst → [src]; else src → [dst]

        Returns:
            Adjacency dictionary
        """
        adj: dict[NodeKey, list[NodeKey]] = defaultdict(list)
        for node in self.nodes:
            adj[node] = []
        for src, dst in self.edges:
            if reverse:
                adj[dst].append(src)
            else:
                adj[src].append(dst)
        return dict(adj)

    def _build_topological_levels(self) -> dict[int, list[NodeKey]]:
        """Build topological levels using dynamic programming.

        level[node] = max(level[pred] for pred in preds) + 1
        Input nodes have level 0.
        """
        levels: dict[NodeKey, int] = {}

        # Input nodes: level 0
        for node in self.input_nodes:
            levels[node] = 0

        # Initialize core nodes with level -1 (unknown)
        for node in self.core_nodes:
            levels[node] = -1

        # Iteratively compute levels
        changed = True
        max_iterations = len(self.core_nodes) + 1
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for node in self.core_nodes:
                preds = self.preds[node]
                max_pred_level = -1
                all_preds_known = True
                for pred in preds:
                    if pred[0] == "in":
                        max_pred_level = max(max_pred_level, 0)
                    elif levels.get(pred, -1) == -1:
                        all_preds_known = False
                    else:
                        max_pred_level = max(max_pred_level, levels[pred])

                if all_preds_known and levels[node] == -1:
                    levels[node] = max_pred_level + 1
                    changed = True

        if iteration >= max_iterations:
            raise RuntimeError("Failed to compute topological levels (cycle detected?)")

        # Output nodes: level = max(core) + 1
        max_core_level = max(levels[n] for n in self.core_nodes)
        for node in self.output_nodes:
            levels[node] = max_core_level + 1

        # Group by level
        grouped: dict[int, list[NodeKey]] = defaultdict(list)
        for node, level in levels.items():
            grouped[level].append(node)

        return {level: sorted(nodes) for level, nodes in sorted(grouped.items())}

    def _validate(self) -> None:
        """Validate graph structure."""
        # Check: all edges go from lower to higher level
        for src, dst in self.edges:
            if self._level_of(src) >= self._level_of(dst):
                raise ValueError(f"Invalid edge {src} → {dst}: levels not increasing")

        # Check: all core + output nodes have at least one predecessor
        for node in self.core_nodes + self.output_nodes:
            if not self.preds[node]:
                raise ValueError(f"Node {node} has no predecessors")

        # Check: no duplicate edges
        seen = set()
        for edge in self.edges:
            if edge in seen:
                raise ValueError(f"Duplicate edge: {edge}")
            seen.add(edge)

    def _level_of(self, node: NodeKey) -> int:
        """Get topological level of a node."""
        for level, nodes in self.topological_levels.items():
            if node in nodes:
                return level
        raise ValueError(f"Node {node} not in topological_levels")

    @property
    def F_in(self) -> list[NodeKey]:
        """Return input facet nodes."""
        return [self._node_key(α) for α in _F_in(self._lattice)]

    @property
    def F_out(self) -> list[NodeKey]:
        """Return output facet nodes."""
        return [self._node_key(α) for α in _F_out(self._lattice)]

    @property
    def F_mid(self) -> list[NodeKey]:
        """Return shared face nodes."""
        return [self._node_key(α) for α in _F_mid(self._lattice)]

    @property
    def backbone(self) -> list[NodeKey]:
        """Return backbone nodes B_{n,m}."""
        return [self._node_key(α) for α in _backbone(self._lattice)]

    @property
    def core_node_count(self) -> int:
        """Return number of core nodes."""
        return len(self.core_nodes)

    @property
    def edge_count(self) -> int:
        """Return number of edges."""
        return len(self.edges)

    def node_potential(self, node: NodeKey) -> int:
        """Return H(α) for a core node."""
        if node[0] != "core":
            raise ValueError(f"node_potential only defined for core nodes, got {node}")
        return _node_potential(self._alpha(node), self._beta)

    def __repr__(self) -> str:
        return (
            f"SimplexMemoryGraph(n={self.n}, m={self.m}, "
            f"core_nodes={self.core_node_count}, edges={self.edge_count})"
        )


# =============================================================================
# SMNmodule — Core PyTorch module
# =============================================================================

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


class SMNmodule(nn.Module):
    """Simplex Memory Network as a pure PyTorch module.

    No dependency on Config or any experiment infrastructure — suitable for
    embedding in other projects.

    Args:
        n: Simplex dimension (>= 2).  n=2 → triangle, n=3 → tetrahedron.
        m: Resolution (>= 2).  Number of lattice points per simplex edge.
        n_in: Number of input dimensions.
        n_out: Number of output dimensions.
        activation: Hidden-node activation ('relu', 'leaky_relu', 'gelu', 'tanh').
        x_bounds: Per-channel input bounds as a list of (min, max) pairs.
            Inputs are linearly normalised to [-1, 1] per channel.
            If None, defaults to [(-1.0, 1.0)] * n_in (identity — caller is
            expected to pre-normalise their data).

    Example::

        module = SMNmodule(n=3, m=4, n_in=2, n_out=1,
                           x_bounds=[(-3.14, 3.14), (-3.14, 3.14)])
        y = module(x)   # x: (batch, 2)  →  y: (batch, 1)
    """

    def __init__(
        self,
        n: int = 2,
        m: int = 3,
        n_in: int = 1,
        n_out: int = 1,
        activation: str = "relu",
        x_bounds: list[tuple[float, float]] | None = None,
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

        # Resolve x_bounds and register as buffers so they move with the module
        # (device transfers, state_dict), avoiding torch.tensor() every forward.
        if x_bounds is None:
            x_bounds = [(-1.0, 1.0)] * n_in
        if len(x_bounds) != n_in:
            raise ValueError(f"x_bounds has {len(x_bounds)} entries but n_in={n_in}")
        self.register_buffer(
            "_x_min", torch.tensor([b[0] for b in x_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "_x_max", torch.tensor([b[1] for b in x_bounds], dtype=torch.float32)
        )

        self.graph = SimplexMemoryGraph(n=n, m=m, n_in=n_in, n_out=n_out)
        self.activation_fn = _make_activation(activation)

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
            output_norm_scales.append(1.0 / (len(preds) ** 0.5))

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
        batch = x.shape[0]

        # Per-channel normalisation to [-1, 1] using registered buffers
        x_min = self._x_min.to(dtype=x.dtype)
        x_max = self._x_max.to(dtype=x.dtype)
        x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

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

        # Output nodes (variance-preserving aggregation + tanh)
        outputs = []
        for i, (out_src_t, out_ew_t) in enumerate(self._output_mappings):
            val = (hist[:, out_src_t] * self.ew[out_ew_t]).sum(1, keepdim=True)
            outputs.append(val * self._output_norm_scales[i])

        return torch.tanh(torch.cat(outputs, dim=1) + self.output_bias)

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
        return f"SMNmodule({self.arch_str})"
