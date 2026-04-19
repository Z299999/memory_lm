#!/usr/bin/env python3
"""Fractal analysis for neural connectomes.

This module computes fractal dimensions and tests for self-similarity
in the simplicial structure of neural graphs.

Key concepts:
- Box-counting dimension: How the detail in structure changes with scale
- Correlation dimension: Based on pairwise distances between nodes
- Power-law scaling: Testing for self-similarity across scales

References:
    Reimann, M. W., et al. (2017). Cliques of Neurons Bound into Cavities
    Provide a Missing Link between Structure and Function.
    Frontiers in Computational Neuroscience, 11, 48.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def compute_box_counting_dimension(
    G: nx.Graph,
    scales: Optional[List[int]] = None
) -> Tuple[float, List[Dict]]:
    """Compute box-counting dimension of a graph.

    The box-counting dimension measures how the number of "boxes" needed
    to cover the graph scales with box size.

    Args:
        G: Graph (should be undirected)
        scales: List of box sizes to try (default: powers of 2)

    Returns:
        Tuple of (dimension, data points)
    """
    if scales is None:
        # Default scales: powers of 2 up to graph diameter
        try:
            diameter = nx.diameter(G)
        except:
            diameter = 10
        scales = [2**i for i in range(1, int(np.log2(diameter)) + 2)]

    data = []

    for box_size in scales:
        # Count how many "boxes" of given size are needed
        # A box is defined as all nodes within distance < box_size from a center

        # Greedy box covering
        uncovered = set(G.nodes())
        boxes = 0

        while uncovered:
            # Pick a random uncovered node as box center
            center = next(iter(uncovered))
            # Find all nodes within distance < box_size
            try:
                lengths = nx.single_source_shortest_path_length(G, center, cutoff=box_size)
                box_nodes = set(lengths.keys())
            except:
                box_nodes = {center}

            uncovered -= box_nodes
            boxes += 1

        data.append({
            'box_size': box_size,
            'num_boxes': boxes,
            'log_size': np.log(box_size),
            'log_boxes': np.log(boxes),
        })

    # Fit dimension from slope of log-log plot
    df = pd.DataFrame(data)
    if len(df) > 1:
        # Dimension = -slope of log(N) vs log(size)
        coeffs = np.polyfit(df['log_size'], df['log_boxes'], 1)
        dimension = -coeffs[0]
    else:
        dimension = np.nan

    return dimension, data


def compute_correlation_dimension(
    G: nx.Graph,
    num_samples: int = 1000
) -> Tuple[float, List[Dict]]:
    """Compute correlation dimension of a graph.

    The correlation dimension measures how the number of node pairs
    within distance r scales with r.

    Args:
        G: Graph (should be undirected)
        num_samples: Number of node pairs to sample

    Returns:
        Tuple of (dimension, data points)
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    # Sample node pairs and compute distances
    distances = []

    for _ in range(num_samples):
        i, j = np.random.choice(n_nodes, 2, replace=False)
        try:
            d = nx.shortest_path_length(G, nodes[i], nodes[j])
            distances.append(d)
        except:
            pass  # No path between nodes

    if len(distances) < 10:
        return np.nan, []

    # Compute correlation integral C(r) = fraction of pairs with distance <= r
    max_dist = max(distances)
    data = []

    for r in range(1, int(max_dist) + 1):
        count = sum(1 for d in distances if d <= r)
        C_r = count / len(distances)
        if C_r > 0:
            data.append({
                'r': r,
                'C_r': C_r,
                'log_r': np.log(r),
                'log_C': np.log(C_r),
            })

    # Fit dimension from scaling region
    df = pd.DataFrame(data)
    if len(df) > 2:
        # Use linear region (exclude very small and very large r)
        mask = (df['log_C'] > np.percentile(df['log_C'], 10)) & \
               (df['log_C'] < np.percentile(df['log_C'], 90))
        if mask.sum() > 2:
            coeffs = np.polyfit(df.loc[mask, 'log_r'], df.loc[mask, 'log_C'], 1)
            dimension = coeffs[0]  # Correlation dimension = slope
        else:
            dimension = np.nan
    else:
        dimension = np.nan

    return dimension, data


def test_power_law_scaling(
    simplex_counts: Dict[int, int]
) -> Dict:
    """Test for power-law scaling in simplex distribution.

    Args:
        simplex_counts: Dictionary mapping dimension to count

    Returns:
        Dictionary with power-law fit results
    """
    dims = list(simplex_counts.keys())
    counts = list(simplex_counts.values())

    # Filter out zeros
    valid = [(d, c) for d, c in zip(dims, counts) if c > 0 and d > 0]

    if len(valid) < 3:
        return {
            'valid': False,
            'reason': 'Insufficient data points',
        }

    dims_valid, counts_valid = zip(*valid)

    # Fit log(N) = a - b * log(dimension)
    log_dims = np.log(dims_valid)
    log_counts = np.log(counts_valid)

    coeffs = np.polyfit(log_dims, log_counts, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Compute R²
    predictions = intercept + slope * log_dims
    ss_res = np.sum((log_counts - predictions) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'valid': True,
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'scaling_exponent': float(-slope),
        'interpretation': 'power-law' if r_squared > 0.9 else 'not power-law',
    }


def analyze_self_similarity(
    G: nx.DiGraph,
    scales: List[int] = None
) -> Dict:
    """Analyze self-similarity across scales.

    Args:
        G: Directed graph
        scales: Box sizes for box-counting

    Returns:
        Analysis results dictionary
    """
    # Convert to undirected for distance-based analysis
    G_undirected = G.to_undirected()

    # Box-counting dimension
    print("Computing box-counting dimension...")
    box_dim, box_data = compute_box_counting_dimension(G_undirected, scales)
    print(f"  Box-counting dimension: {box_dim:.3f}")

    # Correlation dimension
    print("Computing correlation dimension...")
    corr_dim, corr_data = compute_correlation_dimension(G_undirected)
    print(f"  Correlation dimension: {corr_dim:.3f}")

    # Load simplex counts and test power law
    simplex_path = RESULTS_DIR / "simplex_counts.csv"
    if simplex_path.exists():
        df = pd.read_csv(simplex_path)
        simplex_counts = dict(zip(df['dimension'].astype(int), df['count'].astype(int)))

        print("Testing power-law scaling...")
        power_law = test_power_law_scaling(simplex_counts)

        if power_law.get('valid'):
            print(f"  Scaling exponent: {power_law.get('scaling_exponent', np.nan):.3f}")
            print(f"  R²: {power_law.get('r_squared', np.nan):.3f}")
            print(f"  Interpretation: {power_law.get('interpretation', 'N/A')}")
    else:
        power_law = {'valid': False, 'reason': 'No simplex data'}

    results = {
        'box_counting_dimension': box_dim,
        'box_data': box_data,
        'correlation_dimension': corr_dim,
        'correlation_data': corr_data,
        'power_law_test': power_law,
    }

    return results


def visualize_fractal_analysis(
    results: Dict,
    output_path: Optional[Path] = None
) -> Path:
    """Create visualization of fractal analysis.

    Args:
        results: Analysis results dictionary
        output_path: Output image path

    Returns:
        Path to saved figure
    """
    import matplotlib.pyplot as plt

    ensure_dirs()

    if output_path is None:
        output_path = PLOTS_DIR / "fractal_analysis.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Box-counting analysis
    ax1 = axes[0]
    box_data = results.get('box_data', [])
    if box_data:
        df = pd.DataFrame(box_data)
        ax1.scatter(df['log_size'], df['log_boxes'], s=50, color='steelblue')
        if len(df) > 1:
            coeffs = np.polyfit(df['log_size'], df['log_boxes'], 1)
            ax1.plot(df['log_size'], coeffs[1] + coeffs[0] * df['log_size'],
                    'r--', alpha=0.5, label=f'slope={coeffs[0]:.2f}')
            ax1.legend()
        ax1.set_xlabel('log(box size)')
        ax1.set_ylabel('log(number of boxes)')
        ax1.set_title('Box-Counting Analysis')

    # Right: Correlation analysis
    ax2 = axes[1]
    corr_data = results.get('correlation_data', [])
    if corr_data:
        df = pd.DataFrame(corr_data)
        ax2.scatter(df['log_r'], df['log_C'], s=50, color='darkorange')
        ax2.set_xlabel('log(distance r)')
        ax2.set_ylabel('log(correlation C(r))')
        ax2.set_title('Correlation Analysis')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {output_path}")
    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Fractal Analysis")
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
    results = analyze_self_similarity(G)

    # Save results
    output_path = RESULTS_DIR / "fractal_analysis.json"

    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    json_results = convert(results)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nSaved results: {output_path}")

    # Visualize
    print()
    visualize_fractal_analysis(results)

    print()
    print("=" * 60)
    print("Fractal analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
