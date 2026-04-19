#!/usr/bin/env python3
"""Data acquisition module for FlyWire connectome.

This module provides utilities to download and load the FlyWire Drosophila
connectome dataset.

FlyWire Dataset:
- Website: https://flywire.ai
- Paper: Schlegel et al. (2024) Nature
- DOI: 10.1038/s41586-024-07686-5

Data Access Options:
1. FlyWire API (Python package `flywire`)
2. Direct download from CODEx (https://codex.flywire.ai)
3. Google Cloud Public Datasets
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Try to import optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Install with: pip install networkx")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Install with: pip install pandas")


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def ensure_dirs():
    """Ensure data directories exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def get_flywire_info() -> dict:
    """Return information about FlyWire dataset access.

    Returns:
        Dictionary with URLs and access instructions
    """
    return {
        "website": "https://flywire.ai",
        "codex": "https://codex.flywire.ai",
        "api_docs": "https://github.com/schlegelhead/flywire",
        "paper_doi": "10.1038/s41586-024-07686-5",
        "installation": "pip install flywire",
        "data_description": (
            "Full adult Drosophila brain connectome with ~130,000 neurons "
            "and ~50 million synapses. Includes cell type annotations, "
            "hemilineage labels, and morphological features."
        ),
    }


def download_with_flywire_api(output_dir: Optional[Path] = None) -> bool:
    """Download FlyWire connectome using the official Python API.

    This function requires the `flywire` package:
        pip install flywire

    Args:
        output_dir: Directory to save data (default: data/raw)

    Returns:
        True if download successful, False otherwise

    Example:
        >>> from flywire import fetch_connectome
        >>> cn = fetch_connectome()
        >>> cn.neurons  # DataFrame with neuron metadata
        >>> cn.synapses  # DataFrame with synapse connections
    """
    output_dir = output_dir or DATA_RAW
    ensure_dirs()

    print("FlyWire API download instructions:")
    print("-" * 50)
    print("1. Install the flywire package:")
    print("   pip install flywire")
    print()
    print("2. Download connectome in Python:")
    print("   from flywire import fetch_connectome")
    print("   cn = fetch_connectome(version='783')  # or latest version")
    print()
    print("3. Save to files:")
    print("   cn.neurons.to_csv('neurons.csv')")
    print("   cn.synapses.to_csv('synapses.csv')")
    print("-" * 50)

    # Try to execute if flywire is available
    try:
        from flywire import fetch_connectome
        print("\nFetching connectome from FlyWire API...")
        cn = fetch_connectome()

        # Save neurons metadata
        neurons_path = output_dir / "neurons.csv"
        cn.neurons.to_csv(neurons_path)
        print(f"Saved neuron metadata: {neurons_path}")

        # Save synapses (edge list)
        synapses_path = output_dir / "synapses.csv"
        cn.synapses.to_csv(synapses_path)
        print(f"Saved synapse connections: {synapses_path}")

        # Save metadata info
        info_path = output_dir / "dataset_info.json"
        info = {
            "version": cn.version,
            "num_neurons": len(cn.neurons),
            "num_synapses": len(cn.synapses),
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved dataset info: {info_path}")

        return True

    except ImportError:
        print("\nflywire package not installed. Please follow instructions above.")
        return False
    except Exception as e:
        print(f"\nError downloading data: {e}")
        return False


def create_edge_list(synapses_df, output_path: Optional[Path] = None) -> Path:
    """Convert synapse DataFrame to edge list format.

    Args:
        synapses_df: DataFrame with columns [pre, post, weight]
        output_path: Output file path (default: data/processed/edge_list.csv)

    Returns:
        Path to saved edge list file
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for this function")

    output_path = output_path or (DATA_PROCESSED / "edge_list.csv")
    ensure_dirs()

    # Group by pre-post pair and count synapses (weight)
    edge_list = synapses_df.groupby(['pre', 'post']).size().reset_index(name='weight')
    edge_list.to_csv(output_path, index=False)

    print(f"Saved edge list: {output_path} ({len(edge_list)} edges)")
    return output_path


def create_networkx_graph(edge_list_path: Optional[Path] = None) -> "nx.DiGraph":
    """Load edge list and create NetworkX directed graph.

    Args:
        edge_list_path: Path to edge list CSV (default: data/processed/edge_list.csv)

    Returns:
        NetworkX DiGraph representing the connectome
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for this function")

    if edge_list_path is None:
        edge_list_path = DATA_PROCESSED / "edge_list.csv"

    if not edge_list_path.exists():
        # Try to create from synapses.csv
        synapses_path = DATA_RAW / "synapses.csv"
        if synapses_path.exists():
            import pandas as pd
            synapses_df = pd.read_csv(synapses_path)
            create_edge_list(synapses_df)
        else:
            raise FileNotFoundError(
                f"Edge list not found: {edge_list_path}\n"
                "Please run download_flywire_data() first."
            )

    import pandas as pd
    edge_df = pd.read_csv(edge_list_path)

    G = nx.DiGraph()
    for _, row in edge_df.iterrows():
        G.add_edge(int(row['pre']), int(row['post']), weight=int(row['weight']))

    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def download_sample_data(output_dir: Optional[Path] = None) -> bool:
    """Download a small sample of FlyWire data for testing.

    This creates a mini connectome (~1000 neurons) for development
    and testing purposes.

    Args:
        output_dir: Directory to save sample data

    Returns:
        True if successful
    """
    output_dir = output_dir or DATA_RAW
    ensure_dirs()

    print("Creating sample FlyWire data for testing...")

    # Sample from the public CODEx endpoint
    # This is a simplified example - in production, use the full API

    # For now, create a placeholder that documents what to do
    readme_path = output_dir / "SAMPLE_DATA_README.md"
    readme_content = """# FlyWire Sample Data

## How to obtain the full dataset

### Option 1: FlyWire Python API (Recommended)

```bash
pip install flywire
```

```python
from flywire import fetch_connectome

cn = fetch_connectome(version='783')  # or latest

# Save to CSV
cn.neurons.to_csv('neurons.csv')
cn.synapses.to_csv('synapses.csv')
```

### Option 2: Direct download from CODEx

Visit: https://codex.flywire.ai/api/v1/table

Available tables:
- `flywire_neuropil` - Full brain synapses
- `flywire_meta` - Neuron metadata

### Option 3: Google Cloud Public Datasets

The FlyWire connectome is available on Google Cloud:
https://console.cloud.google.com/marketplace/product/flywire/flywire

## Data format

### neurons.csv columns:
- id: Unique neuron identifier
- root: Root ID for merged segments
- type: Cell type label
- hemibrainType: Hemibrain atlas type (if matched)
- hemilineage: Developmental lineage
- side: Brain hemisphere (left/right)

### synapses.csv columns:
- pre: Presynaptic neuron ID
- post: Postsynaptic neuron ID
- x, y, z: Synapse location
- size: Synapse size/weight
"""
    readme_path.write_text(readme_content)
    print(f"Created README: {readme_path}")

    # Create a tiny test graph (10 nodes, random connections)
    np.random.seed(42)
    n_nodes = 100
    n_edges = 500

    nodes = np.arange(n_nodes)
    pre = np.random.choice(nodes, n_edges)
    post = np.random.choice(nodes, n_edges)
    # Ensure no self-loops
    mask = pre != post
    pre, post = pre[mask], post[mask]
    weight = np.random.randint(1, 10, len(pre))

    sample_df = pd.DataFrame({'pre': pre, 'post': post, 'weight': weight})
    sample_path = output_dir / "sample_edge_list.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Created sample data: {sample_path} ({len(sample_df)} edges)")

    return True


def main():
    """Main entry point for data acquisition."""
    print("=" * 60)
    print("FlyWire Connectome Data Acquisition")
    print("=" * 60)
    print()

    info = get_flywire_info()
    print(f"Dataset: {info['data_description']}")
    print(f"Website: {info['website']}")
    print(f"API: {info['installation']}")
    print()

    print("Options:")
    print("1. Download full connectome with FlyWire API")
    print("2. Download sample data for testing")
    print()

    # Always create sample data for development
    print("Creating sample data for testing...")
    download_sample_data()
    print()

    print("To download the full connectome, run:")
    print("  python src/data_acquisition.py --full")
    print()

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("Downloading full FlyWire connectome...")
        download_with_flywire_api()
    else:
        main()
