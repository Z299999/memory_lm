#!/usr/bin/env python3
"""Cavity analysis for neural connectomes.

This module computes topological invariants of the directed flag complex
built from detected simplices.

Key concepts:
- Directed flag complex: Simplicial complex formed by all directed simplices
- Euler characteristic: χ = Σ(-1)^k · (# k-simplices)
- Betti numbers: β_k = dimension of k-th homology group (number of k-dimensional holes)

References:
    Reimann, M. W., et al. (2017). Cliques of Neurons Bound into Cavities
    Provide a Missing Link between Structure and Function.
    Frontiers in Computational Neuroscience, 11, 48.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import networkx as nx

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"


def ensure_dirs():
    """Ensure output directories exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_euler_characteristic(simplex_counts: Dict[int, int]) -> int:
    """Compute Euler characteristic from simplex counts.

    χ = Σ(-1)^k · (# k-simplices)

    Args:
        simplex_counts: Dictionary mapping dimension to simplex count

    Returns:
        Euler characteristic
    """
    euler = 0
    for k, count in simplex_counts.items():
        euler += ((-1) ** k) * count

    return euler


def load_simplex_counts(input_path: Optional[Path] = None) -> Dict[int, int]:
    """Load simplex counts from CSV file.

    Args:
        input_path: Path to simplex_counts.csv

    Returns:
        Dictionary mapping dimension to count
    """
    if input_path is None:
        input_path = RESULTS_DIR / "simplex_counts.csv"

    df = pd.read_csv(input_path)
    return dict(zip(df['dimension'].astype(int), df['count'].astype(int)))


def compute_betti_numbers_approx(
    G: nx.DiGraph,
    max_dim: int = 3
) -> Dict[int, int]:
    """Compute approximate Betti numbers using heuristic method.

    Exact Betti number computation requires sophisticated algebraic topology
    libraries (like Dionysus, GUDHI, or Ripser). This function provides
    heuristic approximations based on simplex counts.

    For a simplicial complex:
    - β₀ = number of connected components
    - β₁ = number of 1D holes (loops)
    - β₂ = number of 2D holes (voids/cavities)

    Args:
        G: Directed graph
        max_dim: Maximum dimension for Betti numbers

    Returns:
        Dictionary mapping dimension to Betti number
    """
    # β₀: Connected components (weakly connected for directed graphs)
    beta_0 = nx.number_weakly_connected_components(G)

    # For higher Betti numbers, we use the relation:
    # χ = Σ(-1)^k · β_k  (Euler-Poincaré formula)
    # This gives us a constraint, but not individual values

    # Heuristic: estimate β₁ from cycle structure
    # Count independent cycles using cycle basis
    undirected_G = G.to_undirected()
    try:
        cycle_basis = nx.cycle_basis(undirected_G)
        beta_1_approx = len(cycle_basis)
    except:
        beta_1_approx = 0

    # β₂ and higher are harder to estimate without proper homology computation
    # Set to 0 as placeholder
    betti = {
        0: beta_0,
        1: beta_1_approx,
    }

    for k in range(2, max_dim + 1):
        betti[k] = 0  # Placeholder

    return betti


def analyze_cavity_structure(
    simplex_counts: Dict[int, int],
    betti_numbers: Dict[int, int]
) -> Dict:
    """Analyze cavity structure from topological invariants.

    Args:
        simplex_counts: Simplex counts by dimension
        betti_numbers: Betti numbers by dimension

    Returns:
        Dictionary with analysis results
    """
    euler = compute_euler_characteristic(simplex_counts)

    # Check Euler-Poincaré consistency
    euler_from_betti = sum(((-1) ** k) * betti_numbers.get(k, 0) for k in range(max(betti_numbers.keys()) + 1))

    analysis = {
        'euler_characteristic': euler,
        'euler_from_betti': euler_from_betti,
        'consistency_check': euler == euler_from_betti,
        'simplex_counts': simplex_counts,
        'betti_numbers': betti_numbers,
        'total_simplices': sum(simplex_counts.values()),
        'max_simplex_dim': max(simplex_counts.keys()) if simplex_counts else 0,
    }

    return analysis


def detect_cavities_from_graph(
    G: nx.DiGraph,
    max_dim: int = 3
) -> Dict:
    """Full cavity analysis pipeline.

    Args:
        G: Directed graph
        max_dim: Maximum dimension for analysis

    Returns:
        Analysis results dictionary
    """
    ensure_dirs()

    print("Loading simplex detector...")
    from simplex_detection import DirectedSimplexDetector

    detector = DirectedSimplexDetector(G)

    # Count simplices
    print("Counting simplices...")
    simplex_counts = {}
    for k in range(max_dim + 1):
        simplices = detector.find_k_simplex_iterative(k)
        simplex_counts[k] = len(simplices)
        print(f"  Dimension {k}: {len(simplices):,} simplices")

    # Compute Euler characteristic
    print("\nComputing Euler characteristic...")
    euler = compute_euler_characteristic(simplex_counts)
    print(f"  Euler characteristic: χ = {euler:,}")

    # Compute Betti numbers
    print("\nEstimating Betti numbers...")
    betti = compute_betti_numbers_approx(G, max_dim)

    for k, b in betti.items():
        print(f"  β_{k} = {b:,}")

    # Analyze cavity structure
    analysis = analyze_cavity_structure(simplex_counts, betti)

    # Save results
    output_path = RESULTS_DIR / "cavity_analysis.json"
    with open(output_path, 'w') as f:
        # Convert to JSON-serializable format
        json_data = {
            'euler_characteristic': int(euler),
            'simplex_counts': {str(k): v for k, v in simplex_counts.items()},
            'betti_numbers': {str(k): v for k, v in betti.items()},
            'total_simplices': sum(simplex_counts.values()),
            'max_simplex_dim': max(simplex_counts.keys()) if simplex_counts else 0,
        }
        json.dump(json_data, f, indent=2)

    print(f"\nSaved results: {output_path}")

    return analysis


def visualize_cavities(
    G: nx.DiGraph,
    output_path: Optional[Path] = None
) -> Path:
    """Create visualization of cavity structure.

    Args:
        G: Directed graph
        output_path: Output image path

    Returns:
        Path to saved figure
    """
    import matplotlib.pyplot as plt

    ensure_dirs()

    if output_path is None:
        output_path = PLOTS_DIR / "cavity_structure.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Graph structure
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=50, node_color='steelblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', alpha=0.3, width=0.5)

    ax1.set_title("Graph Structure")
    ax1.axis('off')

    # Right: Simplex distribution
    ax2 = axes[1]

    # Load simplex counts
    simplex_path = RESULTS_DIR / "simplex_counts.csv"
    if simplex_path.exists():
        df = pd.read_csv(simplex_path)
        ax2.bar(df['dimension'], df['count'], color='steelblue', alpha=0.7)
        ax2.set_xlabel('Simplex Dimension')
        ax2.set_ylabel('Count')
        ax2.set_title('Simplex Distribution')
        ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {output_path}")

    return output_path


def main():
    """Main entry point."""
    import sys

    max_dim = 3
    if len(sys.argv) > 1:
        try:
            max_dim = int(sys.argv[1])
        except ValueError:
            pass

    print("=" * 60)
    print("Cavity Analysis")
    print("=" * 60)
    print()

    # Load graph
    edge_list_path = PROJECT_ROOT / "data" / "raw" / "sample_edge_list.csv"

    if not edge_list_path.exists():
        print(f"Edge list not found: {edge_list_path}")
        return

    edge_df = pd.read_csv(edge_list_path)
    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(int(row['pre']), int(row['post']), weight=int(row['weight']))

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    # Run analysis
    analysis = detect_cavities_from_graph(G, max_dim=max_dim)

    # Visualize
    print()
    visualize_cavities(G)

    print()
    print("=" * 60)
    print("Cavity analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
