from __future__ import annotations

# This file builds the triangular DAG structure: nodes, directed edges,
# predecessor/successor lists, and the execution levels used in forward passes.

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

NodeKey = Tuple[str, int] | Tuple[str, int, int]
Edge = Tuple[NodeKey, NodeKey]


@dataclass
class TMNGraph:
    L: int
    n_in: int = 1
    n_out: int = 1

    def __post_init__(self) -> None:
        if self.L < 1:
            raise ValueError("L must be >= 1")
        if self.n_in < 1 or self.n_out < 1:
            raise ValueError("n_in and n_out must be >= 1")

        self.input_nodes = [("in", a) for a in range(1, self.n_in + 1)]
        self.core_nodes = [("core", r, c) for r in range(1, self.L + 1) for c in range(1, r + 1)]
        self.output_nodes = [("out", b) for b in range(1, self.n_out + 1)]
        self.nodes = self.input_nodes + self.core_nodes + self.output_nodes

        self.edges = self._build_edges()
        self.preds = self._build_adjacency(reverse=True)
        self.succs = self._build_adjacency(reverse=False)
        self.level = self._build_levels()
        self.topological_levels = self._build_topological_levels()
        self.validate()

    def _build_edges(self) -> List[Edge]:
        edges: List[Edge] = []

        for r in range(1, self.L):
            for c in range(1, r + 1):
                top = ("core", r, c)
                bottom_left = ("core", r + 1, c)
                bottom_right = ("core", r + 1, c + 1)
                edges.append((bottom_left, top))
                edges.append((bottom_left, bottom_right))
                edges.append((top, bottom_right))

        for input_node in self.input_nodes:
            for r in range(1, self.L + 1):
                edges.append((input_node, ("core", r, 1)))

        for b in range(1, self.n_out + 1):
            out_node = ("out", b)
            for r in range(1, self.L + 1):
                edges.append((("core", r, r), out_node))

        return edges

    def _build_adjacency(self, reverse: bool) -> Dict[NodeKey, List[NodeKey]]:
        adjacency: Dict[NodeKey, List[NodeKey]] = defaultdict(list)
        for node in self.nodes:
            adjacency[node] = []
        for src, dst in self.edges:
            if reverse:
                adjacency[dst].append(src)
            else:
                adjacency[src].append(dst)
        return {node: adjacency[node] for node in self.nodes}

    def _build_levels(self) -> Dict[NodeKey, int]:
        levels: Dict[NodeKey, int] = {}
        for node in self.input_nodes:
            levels[node] = 1
        for _, r, c in self.core_nodes:
            levels[("core", r, c)] = self.L - r + 2 * c
        for node in self.output_nodes:
            levels[node] = 2 * self.L + 1
        return levels

    def _build_topological_levels(self) -> Dict[int, List[NodeKey]]:
        grouped: Dict[int, List[NodeKey]] = defaultdict(list)
        for node, level in self.level.items():
            grouped[level].append(node)
        return {level: sorted(nodes) for level, nodes in sorted(grouped.items())}

    def validate(self) -> None:
        for src, dst in self.edges:
            if self.level[src] >= self.level[dst]:
                raise ValueError(f"Invalid edge {src} -> {dst}: levels must increase.")

        for node in self.core_nodes + self.output_nodes:
            if not self.preds[node]:
                raise ValueError(f"Node {node} has no predecessors.")

        seen = set()
        for edge in self.edges:
            if edge in seen:
                raise ValueError(f"Duplicate edge detected: {edge}")
            seen.add(edge)

    @property
    def semantic_node_count(self) -> int:
        return len(self.nodes)

    @property
    def core_node_count(self) -> int:
        return len(self.core_nodes)
