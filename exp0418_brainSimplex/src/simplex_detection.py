#!/usr/bin/env python3
"""Directed simplex detection for neural connectomes.

This module counts directed simplices (transitive tournaments) in a directed
graph. A directed k-simplex is a set of k+1 neurons where every pair is
connected by exactly one directed edge and the induced subgraph is acyclic.

The previous implementation relied on repeated NetworkX subgraph creation and
cached every simplex explicitly. That approach becomes prohibitively slow on the
Oxford central brain graph. This version remaps node IDs to a dense index space
and counts simplices by recursively intersecting successor sets, which is much
cheaper in Python.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dirs() -> None:
    """Ensure the results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_edge_list(edge_list_path: Path) -> pd.DataFrame:
    """Load an edge list CSV with columns [pre, post, weight]."""
    print(f"Loading graph from: {edge_list_path}")

    if not edge_list_path.exists():
        raise FileNotFoundError(f"Edge list not found: {edge_list_path}")

    edge_df = pd.read_csv(edge_list_path)
    required_columns = {"pre", "post"}
    if not required_columns.issubset(edge_df.columns):
        raise ValueError(f"Edge list must contain columns {sorted(required_columns)}")

    return edge_df


def build_dense_successor_sets(edge_df: pd.DataFrame) -> Tuple[List[Set[int]], int, int]:
    """Remap node IDs to 0..n-1 and build non-reciprocal successor sets."""
    node_ids = np.union1d(edge_df["pre"].unique(), edge_df["post"].unique())
    node_ids = np.sort(node_ids)
    node_to_idx = {int(node): idx for idx, node in enumerate(node_ids)}

    raw_successors: List[Set[int]] = [set() for _ in range(len(node_ids))]
    for pre, post in edge_df[["pre", "post"]].itertuples(index=False, name=None):
        raw_successors[node_to_idx[int(pre)]].add(node_to_idx[int(post)])

    successors: List[Set[int]] = [set() for _ in range(len(node_ids))]
    for pre_idx, posts in enumerate(raw_successors):
        exclusive_posts = {post_idx for post_idx in posts if pre_idx not in raw_successors[post_idx]}
        successors[pre_idx] = exclusive_posts

    return successors, len(node_ids), len(edge_df)


class DirectedSimplexCounter:
    """Count directed simplices without materializing them all."""

    def __init__(self, successors: Sequence[Set[int]]):
        self.successors = list(successors)
        self.num_nodes = len(self.successors)

    def _extend_counts(
        self,
        candidates: Set[int],
        next_dim: int,
        counts: List[int],
        max_dim: int,
    ) -> None:
        """Recursively count extensions of the current ordered simplex.

        `candidates` contains nodes that are successors of every node already in
        the simplex prefix. Every candidate therefore produces one simplex of
        dimension `next_dim`.
        """
        if not candidates or next_dim > max_dim:
            return

        counts[next_dim] += len(candidates)
        if next_dim == max_dim:
            return

        for node in candidates:
            child_candidates = candidates & self.successors[node]
            if child_candidates:
                self._extend_counts(child_candidates, next_dim + 1, counts, max_dim)

    def count_by_dimension(self, max_dim: int = 10) -> Dict[int, int]:
        """Count simplices by dimension from 0 through `max_dim`."""
        counts = [0] * (max_dim + 1)
        counts[0] = self.num_nodes

        for source, candidates in enumerate(self.successors):
            if self.num_nodes >= 5000 and source % 5000 == 0:
                print(f"  Processed sources: {source:,}/{self.num_nodes:,}")
            self._extend_counts(candidates, 1, counts, max_dim)

        return {dim: count for dim, count in enumerate(counts)}

    def get_simplex_distribution(self, max_dim: int = 10) -> pd.DataFrame:
        """Return simplex counts as a DataFrame."""
        counts = self.count_by_dimension(max_dim)
        return pd.DataFrame(
            [{"dimension": dim, "count": count} for dim, count in counts.items()]
        )


def compare_with_null_models(
    edge_df: pd.DataFrame,
    num_null_models: int = 10,
    max_dim: int = 6,
) -> pd.DataFrame:
    """Compare simplex counts with Erdős-Rényi null models."""
    print("Analyzing original graph...")
    successors, n_nodes, n_edges = build_dense_successor_sets(edge_df)
    detector = DirectedSimplexCounter(successors)
    original_counts = detector.count_by_dimension(max_dim)

    for dim, count in original_counts.items():
        print(f"  Dimension {dim}: {count:,}")

    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    print(f"\nGenerating {num_null_models} Erdős-Rényi null models...")
    print(f"  Parameters: n={n_nodes}, p={density:.6f}")

    null_counts: Dict[int, List[int]] = defaultdict(list)

    for i in range(num_null_models):
        null_G = nx.erdos_renyi_graph(n_nodes, density, directed=True)
        null_successors: List[Set[int]] = [set() for _ in range(n_nodes)]
        for pre, post in null_G.edges():
            null_successors[int(pre)].add(int(post))

        null_detector = DirectedSimplexCounter(null_successors)
        model_counts = null_detector.count_by_dimension(max_dim)

        for dim, count in model_counts.items():
            null_counts[dim].append(count)

        print(f"  Null model {i + 1}/{num_null_models} done")

    results = []
    for dim in range(max_dim + 1):
        original = original_counts[dim]
        null_mean = float(np.mean(null_counts[dim]))
        null_std = float(np.std(null_counts[dim]))
        null_max = int(max(null_counts[dim]))
        z_score = (original - null_mean) / null_std if null_std > 0 else 0.0

        results.append(
            {
                "dimension": dim,
                "original": original,
                "null_mean": null_mean,
                "null_std": null_std,
                "null_max": null_max,
                "z_score": z_score,
                "excess": original - null_mean,
                "excess_ratio": original / null_mean if null_mean > 0 else float("inf"),
            }
        )

    return pd.DataFrame(results)


def detect_simplices_from_file(
    edge_list_path: Optional[Path] = None,
    max_dim: int = 6,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a graph from disk, count simplices, and save the results."""
    ensure_dirs()

    if edge_list_path is None:
        edge_list_path = PROJECT_ROOT / "data" / "processed" / "edge_list.csv"

    edge_df = load_edge_list(edge_list_path)
    successors, num_nodes, num_edges = build_dense_successor_sets(edge_df)

    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    print()
    print("Detecting directed simplices...")

    detector = DirectedSimplexCounter(successors)
    counts = detector.count_by_dimension(max_dim)

    for dim, count in counts.items():
        print(f"  Dimension {dim}: {count:,} simplices")

    df = pd.DataFrame(
        [{"dimension": dim, "count": count} for dim, count in counts.items()]
    )

    if output_path is None:
        output_path = RESULTS_DIR / "simplex_counts.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results: {output_path}")

    return df


def main() -> None:
    """Main entry point."""
    import sys

    max_dim = 6
    compare_null = False
    num_null = 10

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--compare":
            compare_null = True
            if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                num_null = int(sys.argv[i + 1])
                i += 1
        elif arg.isdigit():
            max_dim = int(arg)
        i += 1

    print("=" * 60)
    print("Directed Simplex Detection")
    print("=" * 60)
    print()

    edge_list_path = PROJECT_ROOT / "data" / "processed" / "edge_list.csv"
    detect_simplices_from_file(edge_list_path=edge_list_path, max_dim=max_dim)

    if compare_null:
        print()
        print("=" * 60)
        print("Null Model Comparison")
        print("=" * 60)
        print()

        edge_df = load_edge_list(edge_list_path)
        comparison_df = compare_with_null_models(
            edge_df,
            num_null_models=num_null,
            max_dim=max_dim,
        )
        print("\nComparison results:")
        print(comparison_df.to_string(index=False))

        output_path = RESULTS_DIR / "null_model_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\nSaved comparison: {output_path}")

    print()
    print("=" * 60)
    print("Simplex detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
