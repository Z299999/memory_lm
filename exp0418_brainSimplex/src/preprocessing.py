#!/usr/bin/env python3
"""Data preprocessing module for FlyWire connectome.

This module provides utilities to preprocess the FlyWire connectome data:
- Extract subgraphs (excitatory/inhibitory, brain regions)
- Filter by synapse weight thresholds
- Convert to various graph formats
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def ensure_dirs():
    """Ensure data directories exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load raw FlyWire or Oxford data from CSV files.

    Returns:
        Tuple of (synapses_df, neurons_df)
        neurons_df is None if not available

    Raises:
        FileNotFoundError: If edge list not found
    """
    # Try Oxford dataset first
    oxford_edges = DATA_RAW / "oxford_edge_list.csv"
    oxford_nodes = DATA_RAW / "oxford_nodes.csv"

    if oxford_edges.exists():
        print(f"Loading Oxford edges from: {oxford_edges}")
        edges_df = pd.read_csv(oxford_edges)

        nodes_df = None
        if oxford_nodes.exists():
            print(f"Loading Oxford nodes from: {oxford_nodes}")
            nodes_df = pd.read_csv(oxford_nodes)

        return edges_df, nodes_df

    # Fallback to FlyWire format
    synapses_path = DATA_RAW / "synapses.csv"
    neurons_path = DATA_RAW / "neurons.csv"

    if not synapses_path.exists():
        # Check for sample data
        sample_path = DATA_RAW / "sample_edge_list.csv"
        if sample_path.exists():
            print("Using sample data for testing...")
            sample_df = pd.read_csv(sample_path)
            return sample_df, None
        raise FileNotFoundError(
            f"Synapse data not found: {synapses_path}\n"
            "Run: python src/data_acquisition.py --full"
        )

    print(f"Loading synapses from: {synapses_path}")
    synapses_df = pd.read_csv(synapses_path)

    neurons_df = None
    if neurons_path.exists():
        print(f"Loading neuron metadata from: {neurons_path}")
        neurons_df = pd.read_csv(neurons_path)

    return synapses_df, neurons_df


def create_weighted_edge_list(
    synapses_df: pd.DataFrame,
    min_weight: int = 1,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Group synapses into weighted edges.

    Args:
        synapses_df: DataFrame with columns [pre, post] or [pre, post, size]
        min_weight: Minimum synapse count/weight to include edge
        output_path: Output file path

    Returns:
        DataFrame with columns [pre, post, weight]
    """
    print(f"Creating weighted edge list (min_weight={min_weight})...")

    # Determine weight column
    if 'size' in synapses_df.columns:
        weight_col = 'size'
    elif 'weight' in synapses_df.columns:
        weight_col = 'weight'
    else:
        # Count synapses per edge pair
        weight_col = None

    if weight_col:
        edge_list = synapses_df.groupby(['pre', 'post'])[weight_col].sum().reset_index()
        edge_list.columns = ['pre', 'post', 'weight']
    else:
        edge_list = synapses_df.groupby(['pre', 'post']).size().reset_index(name='weight')

    # Filter by minimum weight
    edge_list = edge_list[edge_list['weight'] >= min_weight]

    print(f"  Total edges: {len(edge_list):,}")
    print(f"  Edges after filtering (weight >= {min_weight}): {(edge_list['weight'] >= min_weight).sum():,}")

    if output_path:
        edge_list.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return edge_list


def extract_cell_type_subgraph(
    neurons_df: pd.DataFrame,
    edge_list: pd.DataFrame,
    cell_types: Union[str, List[str]],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Extract subgraph for specific cell types.

    Args:
        neurons_df: DataFrame with neuron metadata (columns: id, type)
        edge_list: DataFrame with columns [pre, post, weight]
        cell_types: Single cell type or list of types to extract
        output_path: Output file path

    Returns:
        Filtered edge DataFrame
    """
    if neurons_df is None:
        raise ValueError("neurons_df is required for cell type filtering")

    if isinstance(cell_types, str):
        cell_types = [cell_types]

    print(f"Extracting subgraph for cell types: {cell_types}")

    # Find neurons matching cell types
    if 'type' not in neurons_df.columns:
        raise ValueError("neurons_df must have 'type' column")

    neuron_ids = set(neurons_df[neurons_df['type'].isin(cell_types)]['id'])
    print(f"  Found {len(neuron_ids)} neurons matching cell types")

    # Filter edges where both pre and post are in the selected neurons
    mask = edge_list['pre'].isin(neuron_ids) & edge_list['post'].isin(neuron_ids)
    subgraph = edge_list[mask].copy()

    print(f"  Subgraph edges: {len(subgraph):,}")

    if output_path:
        subgraph.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return subgraph


def extract_region_subgraph(
    neurons_df: pd.DataFrame,
    edge_list: pd.DataFrame,
    region_column: str = 'region',
    regions: Optional[Union[str, List[str]]] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Extract subgraph for specific brain regions.

    Args:
        neurons_df: DataFrame with neuron metadata
        edge_list: DataFrame with columns [pre, post, weight]
        region_column: Column name for region annotation
        regions: Region(s) to extract (None = all)
        output_path: Output file path

    Returns:
        Filtered edge DataFrame
    """
    if neurons_df is None:
        raise ValueError("neurons_df is required for region filtering")

    if regions is None:
        print("No regions specified, returning full graph")
        return edge_list

    if isinstance(regions, str):
        regions = [regions]

    print(f"Extracting subgraph for regions: {regions}")

    if region_column not in neurons_df.columns:
        print(f"  Warning: {region_column} not found in neurons_df")
        print(f"  Available columns: {list(neurons_df.columns)}")
        return edge_list

    neuron_ids = set(neurons_df[neurons_df[region_column].isin(regions)]['id'])
    print(f"  Found {len(neuron_ids)} neurons in regions")

    mask = edge_list['pre'].isin(neuron_ids) & edge_list['post'].isin(neuron_ids)
    subgraph = edge_list[mask].copy()

    print(f"  Subgraph edges: {len(subgraph):,}")

    if output_path:
        subgraph.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    return subgraph


def create_networkx_graph(
    edge_list: pd.DataFrame,
    directed: bool = True
) -> nx.DiGraph:
    """Convert edge list to NetworkX graph.

    Args:
        edge_list: DataFrame with columns [pre, post, weight]
        directed: Whether to create directed graph

    Returns:
        NetworkX graph
    """
    print(f"Creating {'directed' if directed else 'undirected'} graph...")

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for _, row in edge_list.iterrows():
        G.add_edge(int(row['pre']), int(row['post']), weight=int(row['weight']))

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    return G


def create_adjacency_matrix(
    edge_list: pd.DataFrame,
    node_ids: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sparse adjacency matrix from edge list.

    Args:
        edge_list: DataFrame with columns [pre, post, weight]
        node_ids: Array of node IDs (None = infer from edge_list)
        output_path: Output .npy file path

    Returns:
        Tuple of (adjacency_matrix, node_id_mapping)
    """
    print("Creating adjacency matrix...")

    if node_ids is None:
        all_nodes = np.union1d(edge_list['pre'].unique(), edge_list['post'].unique())
        node_ids = np.sort(all_nodes)

    n_nodes = len(node_ids)
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}

    # Create sparse matrix
    from scipy import sparse
    row = edge_list['pre'].map(node_to_idx).values
    col = edge_list['post'].map(node_to_idx).values
    data = edge_list['weight'].values

    adj_matrix = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

    print(f"  Matrix shape: {adj_matrix.shape}")
    print(f"  Non-zero entries: {adj_matrix.nnz:,}")
    print(f"  Density: {adj_matrix.nnz / (n_nodes * n_nodes):.6f}")

    if output_path:
        np.savez(output_path, data=adj_matrix.data, indices=adj_matrix.indices,
                 indptr=adj_matrix.indptr, shape=adj_matrix.shape, node_ids=node_ids)
        print(f"  Saved to: {output_path}")

    return adj_matrix, node_ids


def compute_graph_statistics(G: nx.DiGraph) -> Dict:
    """Compute basic graph statistics.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of statistics
    """
    print("Computing graph statistics...")

    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G) if isinstance(G, nx.DiGraph) else nx.is_connected(G),
    }

    # Degree statistics
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    stats['in_degree'] = {
        'mean': float(np.mean(in_degrees)),
        'std': float(np.std(in_degrees)),
        'max': int(max(in_degrees)),
        'min': int(min(in_degrees)),
    }

    stats['out_degree'] = {
        'mean': float(np.mean(out_degrees)),
        'std': float(np.std(out_degrees)),
        'max': int(max(out_degrees)),
        'min': int(min(out_degrees)),
    }

    # Weight statistics
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    stats['edge_weight'] = {
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'max': int(max(weights)),
        'min': int(min(weights)),
        'median': float(np.median(weights)),
    }

    return stats


def preprocess_full(output_dir: Optional[Path] = None) -> Dict:
    """Run full preprocessing pipeline.

    Args:
        output_dir: Output directory (default: data/processed)

    Returns:
        Dictionary with paths to generated files
    """
    output_dir = output_dir or DATA_PROCESSED
    ensure_dirs()

    print("=" * 60)
    print("FlyWire Connectome Preprocessing Pipeline")
    print("=" * 60)
    print()

    # Load raw data
    synapses_df, neurons_df = load_raw_data()
    print()

    # Create weighted edge list
    edge_list = create_weighted_edge_list(
        synapses_df,
        min_weight=1,
        output_path=output_dir / "edge_list.csv"
    )
    print()

    # Create NetworkX graph
    G = create_networkx_graph(edge_list)
    print()

    # Compute statistics
    stats = compute_graph_statistics(G)
    stats_path = output_dir / "graph_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=lambda x: float(x) if isinstance(x, np.ndarray) else x)
    print(f"Saved statistics: {stats_path}")
    print()

    # Create adjacency matrix (for smaller graphs only)
    if stats['num_nodes'] < 100000:
        adj_matrix, node_ids = create_adjacency_matrix(
            edge_list,
            output_path=output_dir / "adjacency_matrix.npz"
        )
    else:
        print("Skipping adjacency matrix (graph too large)")
    print()

    # Save metadata
    metadata = {
        'num_neurons': stats['num_nodes'],
        'num_synapses': stats['num_edges'],
        'preprocessing_date': pd.Timestamp.now().isoformat(),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")
    print()

    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

    return {
        'edge_list': output_dir / "edge_list.csv",
        'adjacency_matrix': output_dir / "adjacency_matrix.npz",
        'statistics': stats_path,
        'metadata': metadata_path,
    }


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--sample":
            print("Processing sample data...")
            sample_edge = pd.read_csv(DATA_RAW / "sample_edge_list.csv")
            G = create_networkx_graph(sample_edge)
            stats = compute_graph_statistics(G)
            print(f"\nSample graph statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            return

    preprocess_full()


if __name__ == "__main__":
    main()
