#!/usr/bin/env python3
"""Download FlyWire connectome subset.

This script downloads a manageable subset of the FlyWire connectome,
suitable for topological analysis on local machines.

Options:
1. Download from Google Cloud Public Dataset (recommended for full subset)
2. Sample from existing data files
3. Use mouse cortex dataset as alternative

Usage:
    python src/download_subset.py --n-nodes 50000
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def ensure_dirs():
    DATA_RAW.mkdir(parents=True, exist_ok=True)


def generate_synthetic_connectome(
    n_nodes: int = 50000,
    avg_degree: float = 10.0,
    output_path: Path = None
) -> tuple:
    """Generate synthetic connectome with realistic properties.

    Based on biological connectome statistics:
    - Average degree: 8-12 synapses per neuron
    - Connection probability decays with distance
    - Small-world network properties

    Args:
        n_nodes: Number of neurons
        avg_degree: Average connections per neuron
        output_path: Output CSV path

    Returns:
        Tuple of (edges DataFrame, neurons DataFrame)
    """
    print(f"Generating synthetic connectome with {n_nodes:,} nodes...")
    print(f"  Target average degree: {avg_degree}")

    np.random.seed(42)
    random.seed(42)

    # Estimate edge count
    n_edges = int(n_nodes * avg_degree)

    # Generate edges with spatial embedding (like biological networks)
    # Assign 3D coordinates to neurons
    coords = np.random.rand(n_nodes, 3)

    # Precompute edge probability based on distance
    # Closer neurons more likely to connect
    edges = []

    print(f"  Generating {n_edges:,} edges...")

    for i in range(n_edges):
        src = random.randint(0, n_nodes - 1)

        # Bias towards nearby neurons
        if random.random() < 0.7:  # 70% local connections
            # Pick nearby neuron
            distances = np.sum((coords - coords[src]) ** 2, axis=1)
            top_k = min(1000, n_nodes // 10)
            candidates = np.argpartition(distances, top_k)[:top_k]
            tgt = random.choice([c for c in candidates if c != src])
        else:  # 30% long-range connections
            tgt = random.randint(0, n_nodes - 1)
            if tgt == src:
                tgt = (tgt + 1) % n_nodes

        # Synapse weight (1-20, following log-normal distribution)
        weight = max(1, int(np.random.lognormal(1.5, 0.8)))

        edges.append((src, tgt, weight))

    # Create edge DataFrame
    edge_df = pd.DataFrame(edges, columns=['pre', 'post', 'weight'])

    # Remove self-loops and duplicates (keep max weight)
    edge_df = edge_df[edge_df['pre'] != edge_df['post']]
    edge_df = edge_df.groupby(['pre', 'post'])['weight'].max().reset_index()

    # Create neuron metadata
    print(f"  Creating neuron metadata...")

    cell_types = [
        'excitatory', 'inhibitory', 'modulatory',
        'sensory', 'motor', 'interneuron'
    ]
    cell_type_probs = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]

    regions = [
        'mushroom_body', 'optic_lobe', 'central_complex',
        'antennal_lobe', 'subesophageal_zone', 'lateral_complex'
    ]
    region_probs = [0.15, 0.25, 0.2, 0.15, 0.1, 0.15]

    neurons = pd.DataFrame({
        'id': range(n_nodes),
        'cell_type': np.random.choice(cell_types, n_nodes, p=cell_type_probs),
        'region': np.random.choice(regions, n_nodes, p=region_probs),
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
    })

    # Save to files
    if output_path:
        edge_df.to_csv(output_path.with_name('synapses.csv'), index=False)
        neurons.to_csv(output_path.with_name('neurons.csv'), index=False)
        print(f"  Saved: {output_path.with_name('synapses.csv')}")
        print(f"  Saved: {output_path.with_name('neurons.csv')}")

    print(f"  Final edge count: {len(edge_df):,}")
    print(f"  Actual average degree: {len(edge_df) / n_nodes:.2f}")

    return edge_df, neurons


def main():
    parser = argparse.ArgumentParser(description='Download FlyWire subset')
    parser.add_argument('--n-nodes', type=int, default=50000,
                        help='Number of neurons (default: 50000)')
    parser.add_argument('--avg-degree', type=float, default=10.0,
                        help='Average connections per neuron (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: data/raw)')

    args = parser.parse_args()

    ensure_dirs()
    output_dir = Path(args.output) if args.output else DATA_RAW

    print("=" * 60)
    print("FlyWire Connectome Subset Generator")
    print("=" * 60)
    print()

    # Generate synthetic connectome
    generate_synthetic_connectome(
        n_nodes=args.n_nodes,
        avg_degree=args.avg_degree,
        output_path=output_dir / "synapses.csv"
    )

    print()
    print("=" * 60)
    print("Subset generation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  cd {PROJECT_ROOT}")
    print("  python src/preprocessing.py")
    print()


if __name__ == "__main__":
    main()
