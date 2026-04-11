from __future__ import annotations

# This file builds the triangular DAG structure: nodes, directed edges,
# predecessor/successor lists, and the execution levels used in forward passes.

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

NodeKey = Tuple[str, int] | Tuple[str, int, int, int]
Edge = Tuple[NodeKey, NodeKey]


@dataclass
class TMNGraph:
    L: int
    n_in: int = 1
    n_out: int = 1
    depth: int = 1  # depth = neurons per position (same for all layers)
    cross_layer_mode: str = "shared_x"

    def __post_init__(self) -> None:
        if self.L < 1:
            raise ValueError("L must be >= 1")
        if self.n_in < 1 or self.n_out < 1:
            raise ValueError("n_in and n_out must be >= 1")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if self.cross_layer_mode not in {"shared_x", "full_x"}:
            raise ValueError("cross_layer_mode must be 'shared_x' or 'full_x'")

        self.input_nodes = [("in", a) for a in range(1, self.n_in + 1)]
        # 3D nodes: (x, y, z) = (depth, position, layer)
        # z from 1 (bottom) to L (top), layer z has (L-z+1) positions (zheng triangle)
        self.core_nodes = [
            ("core", x, y, z)
            for z in range(1, self.L + 1)           # layer (Z axis, bottom→top)
            for y in range(1, self.L - z + 2)        # position (Y axis, left→right), layer z has L-z+1 positions
            for x in range(1, self.depth + 1)       # depth (X axis, inner→outer)
        ]
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

        # Layer z has (L-z+1) positions: y from 1 to L-z+1
        # z=1 (bottom): L positions
        # z=L (top): 1 position
        d = self.depth  # same depth for all layers

        # === 1. Intra-layer connections (Y direction, left→right) ===
        # Full connect between adjacent positions within layer z
        for z in range(1, self.L + 1):
            n_pos = self.L - z + 1  # number of positions in layer z
            for y in range(1, n_pos):  # y from 1 to n_pos-1
                for x_out in range(1, d + 1):
                    for x_in in range(1, d + 1):
                        src = ("core", x_out, y, z)
                        dst = ("core", x_in, y + 1, z)
                        edges.append((src, dst))

        # === 2. Bottom→Top connections (Z+ direction, vertical up) ===
        # shared_x: (x, y, z) -> (x, y, z+1)
        # full_x:   (*, y, z) -> (*, y, z+1)
        # Target layer z+1 has L-(z+1)+1 = L-z positions
        for z in range(1, self.L):  # z from 1 to L-1
            n_pos_dst = self.L - z  # positions in layer z+1
            for y in range(1, n_pos_dst + 1):  # y must exist in both layers
                if self.cross_layer_mode == "shared_x":
                    for x in range(1, d + 1):
                        src = ("core", x, y, z)
                        dst = ("core", x, y, z + 1)
                        edges.append((src, dst))
                else:
                    for x_out in range(1, d + 1):
                        for x_in in range(1, d + 1):
                            src = ("core", x_out, y, z)
                            dst = ("core", x_in, y, z + 1)
                            edges.append((src, dst))

        # === 3. Top→Bottom connections (Z- direction, diagonal down-right) ===
        # shared_x: (x, y, z) -> (x, y+1, z-1)
        # full_x:   (*, y, z) -> (*, y+1, z-1)
        for z in range(2, self.L + 1):  # z from 2 to L
            n_pos_src = self.L - z + 1  # positions in layer z
            n_pos_dst = self.L - z + 2  # positions in layer z-1
            for y in range(1, n_pos_src + 1):  # y from 1 to n_pos_src (all positions)
                if y + 1 <= n_pos_dst:  # target position must exist
                    if self.cross_layer_mode == "shared_x":
                        for x in range(1, d + 1):
                            src = ("core", x, y, z)
                            dst = ("core", x, y + 1, z - 1)
                            edges.append((src, dst))
                    else:
                        for x_out in range(1, d + 1):
                            for x_in in range(1, d + 1):
                                src = ("core", x_out, y, z)
                                dst = ("core", x_in, y + 1, z - 1)
                                edges.append((src, dst))

        # === 4. Input head connections ===
        # Input connects to leftmost position (y=1) of each layer
        # Connect-in: to all depths of target position
        for input_node in self.input_nodes:
            for z in range(1, self.L + 1):
                for x in range(1, d + 1):
                    dst = ("core", x, 1, z)
                    edges.append((input_node, dst))

        # === 5. Output head connections ===
        # Rightmost position of each layer connects to output
        # Layer z's rightmost position is y = L-z+1
        # Connect-out: from all depths of source position
        for b in range(1, self.n_out + 1):
            out_node = ("out", b)
            for z in range(1, self.L + 1):
                y = self.L - z + 1  # rightmost position of layer z
                for x in range(1, d + 1):
                    src = ("core", x, y, z)
                    edges.append((src, out_node))

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
        # Use topological sort to assign levels
        # This handles the complex connection patterns where signals flow both up and down
        levels: Dict[NodeKey, int] = {}

        # Input nodes are level 0
        for node in self.input_nodes:
            levels[node] = 0

        # Process core nodes in topological order
        # We need to handle the fact that connections go both up (z→z+1) and down (z→z-1)
        # Use dynamic programming: level[node] = max(level[pred] for pred in preds) + 1

        # First, build a dependency graph for core nodes only
        # Initialize all core nodes with level -1 (unknown)
        for node in self.core_nodes:
            levels[node] = -1

        # Iteratively compute levels until all are known
        changed = True
        while changed:
            changed = False
            for node in self.core_nodes:
                preds = self.preds[node]
                # Skip input node preds (already have level 0)
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

        # Output nodes
        max_level = max(levels.values())
        for node in self.output_nodes:
            levels[node] = max_level + 1

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
