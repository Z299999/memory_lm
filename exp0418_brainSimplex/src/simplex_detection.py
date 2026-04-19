#!/usr/bin/env python3
"""Directed simplex detection for neural connectomes.

This module implements algorithms to detect directed simplices (cliques) in
neural network graphs, following the methodology of Reimann et al. (2017).

A directed k-simplex is a set of k+1 neurons where:
- Every pair is connected (all-to-all)
- The subgraph is a directed acyclic graph (DAG)
- There is a total ordering consistent with edge directions

References:
    Reimann, M. W., et al. (2017). Cliques of Neurons Bound into Cavities
    Provide a Missing Link between Structure and Function.
    Frontiers in Computational Neuroscience, 11, 48.
"""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import networkx as nx
import pandas as pd

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dirs():
    """Ensure results directory exists."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class DirectedSimplexDetector:
    """Detector for directed simplices in graphs."""

    def __init__(self, G: nx.DiGraph):
        """Initialize detector.

        Args:
            G: Directed graph (should be acyclic for valid simplex detection)
        """
        self.G = G
        self._simplex_cache: Dict[int, List[Tuple[int, ...]]] = {0: [(n,) for n in G.nodes()]}

    def is_directed_clique(self, nodes: Set[int]) -> bool:
        """Check if a set of nodes forms a directed clique."""
        if len(nodes) <= 1:
            return True

        subgraph = self.G.subgraph(nodes)
        n = len(nodes)
        expected_edges = n * (n - 1) // 2

        if subgraph.number_of_edges() != expected_edges:
            return False

        if not nx.is_directed_acyclic_graph(subgraph):
            return False

        return True

    def find_1_simplices(self) -> List[Tuple[int, int]]:
        """Find all 1-simplices (directed edges)."""
        if 1 in self._simplex_cache:
            return self._simplex_cache[1]

        simplices = list(self.G.edges())
        self._simplex_cache[1] = simplices
        return simplices

    def find_2_simplices(self) -> List[Tuple[int, int, int]]:
        """Find all 2-simplices (directed triangles)."""
        if 2 in self._simplex_cache:
            return self._simplex_cache[2]

        simplices = []
        for a, b in self.G.edges():
            successors_b = set(self.G.successors(b))
            for c in self.G.successors(a):
                if c in successors_b and self.is_directed_clique({a, b, c}):
                    simplices.append((a, b, c))

        self._simplex_cache[2] = simplices
        return simplices

    def find_k_simplex_iterative(self, k: int) -> List[Tuple[int, ...]]:
        """Find all k-simplices using iterative approach (no recursion)."""
        if k in self._simplex_cache:
            return self._simplex_cache[k]

        # Start from 1-simplices and build up iteratively
        for dim in range(1, k + 1):
            if dim in self._simplex_cache:
                continue

            if dim == 1:
                self._simplex_cache[1] = list(self.G.edges())
                continue

            if dim == 2:
                self._simplex_cache[2] = self.find_2_simplices()
                continue

            # For dim >= 3, extend (dim-1)-simplices
            lower = self._simplex_cache.get(dim - 1, [])
            if not lower:
                self._simplex_cache[dim] = []
                continue

            simplices = set()
            for simplex in lower:
                simplex_nodes = set(simplex)
                common_successors = set.intersection(*[set(self.G.successors(n)) for n in simplex])
                common_preds = set.intersection(*[set(self.G.predecessors(n)) for n in simplex])

                for candidate in common_successors | common_preds:
                    if candidate not in simplex_nodes:
                        candidate_set = simplex_nodes | {candidate}
                        if self.is_directed_clique(candidate_set):
                            simplices.add(tuple(sorted(candidate_set)))

            self._simplex_cache[dim] = list(simplices)

        return self._simplex_cache[k]

    def count_by_dimension(self, max_dim: int = 10) -> Dict[int, int]:
        """Count simplices by dimension."""
        counts = {}
        for k in range(max_dim + 1):
            simplices = self.find_k_simplex_iterative(k)
            counts[k] = len(simplices)
            print(f"  Dimension {k}: {len(simplices):,} simplices")
            if len(simplices) == 0 and k > 0:
                break
        return counts

    def get_simplex_distribution(self, max_dim: int = 10) -> pd.DataFrame:
        """Get simplex distribution as DataFrame."""
        counts = self.count_by_dimension(max_dim)
        return pd.DataFrame([
            {'dimension': k, 'count': c}
            for k, c in counts.items()
        ])


def compare_with_null_models(
    G: nx.DiGraph,
    num_null_models: int = 10,
    max_dim: int = 6
) -> pd.DataFrame:
    """Compare simplex counts with null models."""
    print("Analyzing original graph...")
    detector = DirectedSimplexDetector(G)
    original_counts = {}

    for k in range(max_dim + 1):
        simplices = detector.find_k_simplex_iterative(k)
        original_counts[k] = len(simplices)
        print(f"  Dimension {k}: {len(simplices):,}")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    print(f"\nGenerating {num_null_models} Erdős-Rényi null models...")
    print(f"  Parameters: n={n_nodes}, p={density:.6f}")

    null_counts: Dict[int, List[int]] = defaultdict(list)

    for i in range(num_null_models):
        null_G = nx.erdos_renyi_graph(n_nodes, density, directed=True)
        null_detector = DirectedSimplexDetector(null_G)

        for k in range(max_dim + 1):
            count = len(null_detector.find_k_simplex_iterative(k))
            null_counts[k].append(count)

        print(f"  Null model {i + 1}/{num_null_models} done")

    results = []
    for k in range(max_dim + 1):
        original = original_counts[k]
        null_mean = np.mean(null_counts[k])
        null_std = np.std(null_counts[k])
        null_max = max(null_counts[k])

        z_score = (original - null_mean) / null_std if null_std > 0 else 0

        results.append({
            'dimension': k,
            'original': original,
            'null_mean': null_mean,
            'null_std': null_std,
            'null_max': null_max,
            'z_score': z_score,
            'excess': original - null_mean,
            'excess_ratio': original / null_mean if null_mean > 0 else float('inf'),
        })

    return pd.DataFrame(results)


def detect_simplices_from_file(
    edge_list_path: Optional[Path] = None,
    max_dim: int = 6,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Main entry point for simplex detection."""
    ensure_dirs()

    if edge_list_path is None:
        edge_list_path = PROJECT_ROOT / "data" / "processed" / "edge_list.csv"

    print(f"Loading graph from: {edge_list_path}")

    if not edge_list_path.exists():
        sample_path = PROJECT_ROOT / "data" / "raw" / "sample_edge_list.csv"
        if sample_path.exists():
            edge_list_path = sample_path
            print(f"Using sample data: {edge_list_path}")
        else:
            raise FileNotFoundError(f"Edge list not found: {edge_list_path}")

    edge_df = pd.read_csv(edge_list_path)
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(int(row['pre']), int(row['post']), weight=int(row['weight']))

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Check if DAG
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"Is DAG: {is_dag}")
    if not is_dag:
        print("Warning: Graph contains cycles. Directed simplices require acyclic subgraphs.")
        print("For biological graphs, consider using the transitive reduction or DAG approximation.")

    print()
    print("Detecting directed simplices...")
    detector = DirectedSimplexDetector(G)

    results = []
    for k in range(max_dim + 1):
        simplices = detector.find_k_simplex_iterative(k)
        count = len(simplices)
        results.append({'dimension': k, 'count': count})
        print(f"  Dimension {k}: {count:,} simplices")
        if count == 0 and k > 0:
            break

    df = pd.DataFrame(results)

    # Auto-save results
    output_path = RESULTS_DIR / "simplex_counts.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results: {output_path}")

    return df


def main():
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

    df = detect_simplices_from_file(max_dim=max_dim)

    if compare_null:
        print()
        print("=" * 60)
        print("Null Model Comparison")
        print("=" * 60)
        print()

        edge_list_path = PROJECT_ROOT / "data" / "raw" / "sample_edge_list.csv"
        edge_df = pd.read_csv(edge_list_path)
        G = nx.DiGraph()
        for _, row in edge_df.iterrows():
            G.add_edge(int(row['pre']), int(row['post']), weight=int(row['weight']))

        comparison_df = compare_with_null_models(G, num_null_models=num_null, max_dim=max_dim)
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
