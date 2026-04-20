#!/usr/bin/env python3
"""Import Oxford Drosophila connectome dataset.

This script imports the Oxford "Central" dataset:
- pete_fly_central_edges.csv: 849,980 directed synaptic connections
- pete_fly_central_nodes_metadata.csv: 32,272 neurons with 3D positions

Data source: Oxford C5.4 Networks mini-project (2025)
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def import_oxford_data():
    """Import Oxford dataset and create standardized formats."""

    print("=" * 60)
    print("Importing Oxford Drosophila Connectome Data")
    print("=" * 60)
    print()

    # Load raw data
    edges_path = DATA_RAW / "pete_fly_central_edges.csv"
    nodes_path = DATA_RAW / "pete_fly_central_nodes_metadata.csv"

    print(f"Loading edges from: {edges_path}")
    edges_df = pd.read_csv(edges_path)
    print(f"  Loaded {len(edges_df):,} edges")

    print(f"Loading nodes from: {nodes_path}")
    nodes_df = pd.read_csv(nodes_path)
    print(f"  Loaded {len(nodes_df):,} nodes")
    print()

    # Show data format
    print("Edge columns:", list(edges_df.columns))
    print("Node columns:", list(nodes_df.columns))
    print()

    # Create standardized edge list (pre, post, weight)
    # Original format: from, to
    # Add weight column (default = 1 for unweighted)
    if 'weight' not in edges_df.columns:
        edges_df['weight'] = 1

    # Rename to standard format
    standard_edges = edges_df.rename(columns={'from': 'pre', 'to': 'post'})
    standard_edges = standard_edges[['pre', 'post', 'weight']]

    # Save standardized edge list
    output_path = DATA_RAW / "oxford_edge_list.csv"
    standard_edges.to_csv(output_path, index=False)
    print(f"Saved standardized edge list: {output_path}")

    # Create node metadata with standard columns
    # Original: node_id, x, y, z, module, root_id
    standard_nodes = nodes_df.rename(columns={'node_id': 'id'})

    # Save standardized node metadata
    nodes_output = DATA_RAW / "oxford_nodes.csv"
    standard_nodes.to_csv(nodes_output, index=False)
    print(f"Saved node metadata: {nodes_output}")

    # Summary statistics
    print()
    print("=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"  Nodes: {len(nodes_df):,}")
    print(f"  Edges: {len(edges_df):,}")
    print(f"  Average degree: {len(edges_df) / len(nodes_df):.2f}")
    print(f"  Modules: {nodes_df['module'].nunique()}")
    print()

    return standard_edges, standard_nodes


if __name__ == "__main__":
    import_oxford_data()
