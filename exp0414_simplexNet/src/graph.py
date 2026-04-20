"""Graph structure for simplex memory networks.

The SimplexMemoryGraph class builds the directed acyclic graph
from the lattice V_{n,m} with potential-based edge orientation.

This module is self-contained — lattice generation and potential functions
are included as internal helpers (no external dependencies within the package).
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TypeAlias, Iterator

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


# Quick test
if __name__ == "__main__":
    graph = SimplexMemoryGraph(n=2, m=3, n_in=1, n_out=1)
    print(graph)
    print(f"\nTopological levels:")
    for level, nodes in graph.topological_levels.items():
        h_vals = [graph.node_potential(n) for n in nodes if n[0] == "core"]
        print(f"  Level {level}: {len(nodes)} nodes, H = {h_vals if h_vals else 'N/A'}")

    print(f"\nBackbone: {graph.backbone}")
    print(f"F_in: {graph.F_in}")
    print(f"F_out: {graph.F_out}")
    print(f"F_mid: {graph.F_mid}")

    # Verify F_mid has constant H = m-1
    h_mid = [graph.node_potential(n) for n in graph.F_mid]
    print(f"\nH values on F_mid: {set(h_mid)} (expected {{{3-1}}})")
