#!/usr/bin/env python3
"""Visualize the Drosophila brain connectome and top simplices.

This module provides 3D visualizations of:
1. The full brain connectome (nodes and edges)
2. The largest directed cliques (simplex dimension >= 4)
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
PLOTS_DIR = PROJECT_ROOT / "plots"


def ensure_dirs() -> None:
    """Ensure output directories exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_nodes() -> pd.DataFrame:
    """Load node metadata with 3D positions."""
    # Try processed first, then raw sources
    for path in [
        PROCESSED_DIR / "nodes.csv",
        RAW_DIR / "oxford_nodes.csv",
        RAW_DIR / "pete_fly_central_nodes_metadata.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            # Normalize column names
            if "id" in df.columns and "node_id" not in df.columns:
                df = df.rename(columns={"id": "node_id"})
            return df

    raise FileNotFoundError(f"Node file not found in {PROCESSED_DIR} or {RAW_DIR}")


def load_edges() -> pd.DataFrame:
    """Load edge list."""
    edge_path = PROCESSED_DIR / "edge_list.csv"
    if not edge_path.exists():
        edge_path = RAW_DIR / "oxford_edge_list.csv"

    if not edge_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_path}")

    return pd.read_csv(edge_path)


def build_graph() -> Tuple[nx.DiGraph, pd.DataFrame]:
    """Build directed graph from edge list."""
    edge_df = load_edges()
    G = nx.DiGraph()

    for _, row in edge_df.iterrows():
        G.add_edge(int(row["pre"]), int(row["post"]), weight=row.get("weight", 1))

    nodes = load_nodes()
    positions = dict(zip(nodes["node_id"], nodes[["x", "y", "z"]].values))
    nx.set_node_attributes(G, positions, "pos")

    return G, nodes


def plot_brain_3d(
    nodes: pd.DataFrame,
    edges: Optional[pd.DataFrame] = None,
    sample_nodes: int = 5000,
    sample_edges: int = 10000,
    title: str = "Drosophila Central Brain Connectome",
    output_name: str = "brain_3d_overview.png",
) -> Path:
    """Plot 3D visualization of the brain.

    Args:
        nodes: DataFrame with node_id, x, y, z columns
        edges: Optional edge DataFrame with pre, post columns
        sample_nodes: Max nodes to plot (for performance)
        sample_edges: Max edges to plot (for performance)
        title: Plot title
        output_name: Output filename

    Returns:
        Path to saved figure
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Sample nodes if too many
    if len(nodes) > sample_nodes:
        plot_nodes = nodes.sample(n=sample_nodes, random_state=42)
    else:
        plot_nodes = nodes

    # Plot nodes
    scatter = ax.scatter(
        plot_nodes["x"],
        plot_nodes["y"],
        plot_nodes["z"],
        c=plot_nodes.get("module", [0] * len(plot_nodes)),
        cmap="tab20",
        s=3,
        alpha=0.6,
    )

    # Plot edges (subset for clarity)
    if edges is not None and sample_edges > 0:
        # Sample edges
        if len(edges) > sample_edges:
            plot_edges = edges.sample(n=sample_edges, random_state=42)
        else:
            plot_edges = edges

        # Create node position lookup
        pos_lookup = dict(zip(nodes["node_id"], nodes[["x", "y", "z"]].values))

        for _, edge in plot_edges.iterrows():
            pre, post = int(edge["pre"]), int(edge["post"])
            if pre in pos_lookup and post in pos_lookup:
                p1 = pos_lookup[pre]
                p2 = pos_lookup[post]
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    c="gray",
                    alpha=0.05,
                    linewidth=0.3,
                )

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_title(title)

    # Add colorbar for modules
    if "module" in nodes.columns:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Module / Brain Region", fontsize=10)

    output_path = PLOTS_DIR / output_name
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def find_largest_cliques(
    G: nx.DiGraph,
    max_order: int = 6,
    top_k: int = 10,
) -> Dict[int, List[List[int]]]:
    """Find the largest directed cliques (transitive tournaments).

    A directed clique of order k is a set of k nodes where every pair
    has exactly one directed edge, forming a transitive tournament.

    Args:
        G: Directed graph
        max_order: Maximum clique order to search for
        top_k: Number of top cliques to return per order

    Returns:
        Dict mapping order -> list of cliques (each clique is a node list)
    """
    print("Finding largest directed cliques...")

    # Build non-reciprocal successor sets
    successors: Dict[int, Set[int]] = {}
    for node in G.nodes():
        succ = set(G.successors(node))
        # Remove reciprocal edges
        exclusive = {s for s in succ if node not in G.successors(s)}
        successors[node] = exclusive

    cliques_by_order: Dict[int, List[List[int]]] = defaultdict(list)

    def extend_clique(clique: List[int], candidates: Set[int]) -> None:
        """Recursively extend a clique."""
        order = len(clique)

        if order >= 2:
            cliques_by_order[order].append(clique.copy())

        if order >= max_order:
            return

        for node in sorted(candidates):
            new_candidates = candidates & successors[node]
            if new_candidates:
                extend_clique(clique + [node], new_candidates)

    # Start from each node
    nodes = list(G.nodes())
    for i, source in enumerate(nodes):
        if len(nodes) >= 1000 and i % 1000 == 0:
            print(f"  Processed {i:,}/{len(nodes):,} nodes...")

        candidates = successors[source]
        if candidates:
            extend_clique([source], candidates)

    # Sort and keep top-k per order
    result = {}
    for order in sorted(cliques_by_order.keys()):
        cliques = cliques_by_order[order]
        # Sort by sum of node degrees (proxy for "importance")
        cliques_sorted = sorted(
            cliques,
            key=lambda c: sum(G.out_degree(n) + G.in_degree(n) for n in c),
            reverse=True,
        )[:top_k]
        result[order] = cliques_sorted

    return result


def plot_all_7cliques_together(
    cliques: List[List[int]],
    nodes: pd.DataFrame,
    G: nx.DiGraph,
    output_name: str = "all_7cliques.png",
    max_per_figure: int = 50,
) -> List[Path]:
    """Plot all 7-cliques (order 7, simplex dim 6) in multiple 3D figures.

    Args:
        cliques: List of cliques, each clique is a list of 7 node IDs
        nodes: DataFrame with node positions
        G: Directed graph
        output_name: Output filename base
        max_per_figure: Maximum cliques to show per figure

    Returns:
        List of paths to saved figures
    """
    if not cliques:
        raise ValueError("No 7-cliques provided")

    pos_lookup = dict(zip(nodes["node_id"], nodes[["x", "y", "z"]].values))
    output_paths = []

    # Split into batches
    num_figures = (len(cliques) + max_per_figure - 1) // max_per_figure

    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_per_figure
        end_idx = min(start_idx + max_per_figure, len(cliques))
        batch_cliques = cliques[start_idx:end_idx]

        # Use different colors for different cliques
        colors = plt.cm.tab20(np.linspace(0, 1, len(batch_cliques)))

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Collect all positions for setting bounds
        all_clique_nodes = set()
        for clique in batch_cliques:
            all_clique_nodes.update(clique)

        # Background: all brain neurons (very faint)
        ax.scatter(
            nodes["x"], nodes["y"], nodes["z"],
            c="lightgray", s=0.5, alpha=0.05
        )

        # Plot each 7-clique
        for clique_idx, clique in enumerate(batch_cliques):
            positions = []
            valid_nodes = []
            for node_id in clique:
                if node_id in pos_lookup:
                    positions.append(pos_lookup[node_id])
                    valid_nodes.append(node_id)

            if len(positions) != 7:
                continue

            positions = np.array(positions)
            color = colors[clique_idx % len(colors)]

            # Draw edges between all pairs (complete graph)
            for i in range(7):
                for j in range(i + 1, 7):
                    src, dst = valid_nodes[i], valid_nodes[j]
                    if G.has_edge(src, dst):
                        p1, p2 = positions[i], positions[j]
                    elif G.has_edge(dst, src):
                        p1, p2 = positions[j], positions[i]
                    else:
                        continue

                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c=color,
                        alpha=0.6,
                        linewidth=2,
                    )

            # Plot nodes for this clique - small spheres
            ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=[color],
                s=15,
                alpha=1.0,
                edgecolors="white",
                linewidths=0.3,
            )

        ax.set_xlabel("X (nm)", fontsize=10)
        ax.set_ylabel("Y (nm)", fontsize=10)
        ax.set_zlabel("Z (nm)", fontsize=10)

        if num_figures == 1:
            ax.set_title(f"All Directed 7-Cliques (Simplex Dimension 6) — {len(cliques)} total", fontsize=12)
        else:
            ax.set_title(
                f"7-Cliques {start_idx + 1}-{end_idx} of {len(cliques)}",
                fontsize=12
            )

        # Set bounds using full node dataset for context
        all_positions = np.array([pos_lookup[n] for n in all_clique_nodes if n in pos_lookup])
        if len(all_positions) > 0:
            # Use bounds from ALL neurons for consistent context
            x_bounds = (nodes["x"].min(), nodes["x"].max())
            y_bounds = (nodes["y"].min(), nodes["y"].max())
            z_bounds = (nodes["z"].min(), nodes["z"].max())

            mid = np.array([
                (x_bounds[0] + x_bounds[1]) / 2,
                (y_bounds[0] + y_bounds[1]) / 2,
                (z_bounds[0] + z_bounds[1]) / 2,
            ])
            max_range = max(x_bounds[1] - x_bounds[0],
                           y_bounds[1] - y_bounds[0],
                           z_bounds[1] - z_bounds[0]) * 0.6

            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        base_name = output_name.replace(".png", "")
        if num_figures == 1:
            output_path = PLOTS_DIR / output_name
        else:
            output_path = PLOTS_DIR / f"{base_name}_part{fig_idx + 1}.png"

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
        print(f"  Saved: {output_path} ({len(batch_cliques)} cliques)")

    return output_paths


def visualize_top_simplices(
    G: nx.DiGraph,
    nodes: pd.DataFrame,
    max_order: int = 6,
    top_k: int = 5,
) -> List[Path]:
    """Find and visualize the top simplices.

    Args:
        G: Directed graph
        nodes: Node DataFrame with positions
        max_order: Maximum simplex order to find
        top_k: Number of top simplices to visualize per order

    Returns:
        List of paths to saved figures
    """
    cliques = find_largest_cliques(G, max_order=max_order, top_k=top_k)

    output_paths = []

    for order, clique_list in cliques.items():
        print(f"\nOrder {order}: Found {len(clique_list)} cliques")

        for i, clique in enumerate(clique_list[:3]):  # Visualize top 3 per order
            try:
                output_name = f"clique_order{order}_idx{i}.png"
                path = plot_clique_3d(clique, nodes, G, order, output_name)
                output_paths.append(path)
                print(f"  Saved: {path}")
            except Exception as e:
                print(f"  Error plotting clique: {e}")

    return output_paths


def visualize_kcliques(
    G: nx.DiGraph,
    nodes: pd.DataFrame,
    k: int = 7,
    max_per_figure: int = 20,
) -> List[Path]:
    """Find and visualize ALL k-cliques (order k, simplex dim k-1) in one plot.

    Args:
        G: Directed graph
        nodes: Node DataFrame with positions
        k: Clique order to find (default 7)
        max_per_figure: Maximum cliques per figure

    Returns:
        List of paths to saved figures
    """
    print(f"\nFinding all {k}-cliques (order {k}, simplex dimension {k-1})...")

    # Build non-reciprocal successor sets
    successors: Dict[int, Set[int]] = {}
    for node in G.nodes():
        succ = set(G.successors(node))
        exclusive = {s for s in succ if node not in G.successors(s)}
        successors[node] = exclusive

    all_kcliques: List[List[int]] = []

    def extend_clique(clique: List[int], candidates: Set[int]) -> None:
        if len(clique) == k:
            all_kcliques.append(clique.copy())
            return

        for node in sorted(candidates):
            new_candidates = candidates & successors[node]
            if new_candidates:
                extend_clique(clique + [node], new_candidates)

    # Start from each node
    nodes_list = list(G.nodes())
    for i, source in enumerate(nodes_list):
        if i % 5000 == 0:
            print(f"  Searched from {i:,}/{len(nodes_list):,} nodes...")
        candidates = successors[source]
        if candidates:
            extend_clique([source], candidates)

    print(f"Found {len(all_kcliques)} {k}-cliques!")

    # Also print lower dimensional simplex counts for context
    print("\nSimplex summary (directed cliques):")
    all_cliques_by_order: Dict[int, int] = defaultdict(int)

    def count_cliques(clique: List[int], candidates: Set[int]) -> None:
        all_cliques_by_order[len(clique)] += 1
        if len(clique) >= 7:
            return
        for node in candidates:
            new_candidates = candidates & successors[node]
            if new_candidates:
                count_cliques(clique + [node], new_candidates)

    for i, source in enumerate(nodes_list):
        if i % 10000 == 0:
            print(f"  Counting from {i:,}/{len(nodes_list):,} nodes...")
        candidates = successors[source]
        if candidates:
            count_cliques([source], candidates)

    for order in sorted(all_cliques_by_order.keys()):
        print(f"  Order {order} (dim {order-1}): {all_cliques_by_order[order]:,}")

    output_paths = []

    if all_kcliques:
        print(f"\nPlotting all {k}-cliques together...")
        output_name = f"all_{k}cliques.png"
        paths = plot_all_kcliques_together(all_kcliques, nodes, G, output_name, max_per_figure=max_per_figure, k=k)
        output_paths.extend(paths)

    return output_paths


def plot_all_kcliques_together(
    cliques: List[List[int]],
    nodes: pd.DataFrame,
    G: nx.DiGraph,
    output_name: str = "all_kcliques.png",
    max_per_figure: int = 20,
    k: int = 7,
) -> List[Path]:
    """Plot all k-cliques in multiple 3D figures.

    Args:
        cliques: List of cliques, each clique is a list of k node IDs
        nodes: DataFrame with node positions
        G: Directed graph
        output_name: Output filename base
        max_per_figure: Maximum cliques to show per figure
        k: Clique order (number of nodes per clique)

    Returns:
        List of paths to saved figures
    """
    if not cliques:
        raise ValueError(f"No {k}-cliques provided")

    pos_lookup = dict(zip(nodes["node_id"], nodes[["x", "y", "z"]].values))
    output_paths = []
    dim = k - 1  # simplex dimension

    # Split into batches
    num_figures = (len(cliques) + max_per_figure - 1) // max_per_figure

    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_per_figure
        end_idx = min(start_idx + max_per_figure, len(cliques))
        batch_cliques = cliques[start_idx:end_idx]

        # Use different colors for different cliques
        colors = plt.cm.tab20(np.linspace(0, 1, len(batch_cliques)))

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Collect all positions for setting bounds
        all_clique_nodes = set()
        for clique in batch_cliques:
            all_clique_nodes.update(clique)

        # Background: all brain neurons (very faint)
        ax.scatter(
            nodes["x"], nodes["y"], nodes["z"],
            c="lightgray", s=0.5, alpha=0.05
        )

        # Plot each k-clique
        for clique_idx, clique in enumerate(batch_cliques):
            positions = []
            valid_nodes = []
            for node_id in clique:
                if node_id in pos_lookup:
                    positions.append(pos_lookup[node_id])
                    valid_nodes.append(node_id)

            if len(positions) != k:
                continue

            positions = np.array(positions)
            color = colors[clique_idx % len(colors)]

            # Draw edges between all pairs (complete graph)
            for i in range(k):
                for j in range(i + 1, k):
                    src, dst = valid_nodes[i], valid_nodes[j]
                    if G.has_edge(src, dst):
                        p1, p2 = positions[i], positions[j]
                    elif G.has_edge(dst, src):
                        p1, p2 = positions[j], positions[i]
                    else:
                        continue

                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        c=color,
                        alpha=0.6,
                        linewidth=2,
                    )

            # Plot nodes for this clique - small spheres
            ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=[color],
                s=15,
                alpha=1.0,
                edgecolors="white",
                linewidths=0.3,
            )

        ax.set_xlabel("X (nm)", fontsize=10)
        ax.set_ylabel("Y (nm)", fontsize=10)
        ax.set_zlabel("Z (nm)", fontsize=10)

        if num_figures == 1:
            ax.set_title(f"All Directed {k}-Cliques (Simplex Dimension {dim}) — {len(cliques)} total", fontsize=12)
        else:
            ax.set_title(
                f"{k}-Cliques {start_idx + 1}-{end_idx} of {len(cliques)}",
                fontsize=12
            )

        # Set bounds using full node dataset for context
        all_positions = np.array([pos_lookup[n] for n in all_clique_nodes if n in pos_lookup])
        if len(all_positions) > 0:
            x_bounds = (nodes["x"].min(), nodes["x"].max())
            y_bounds = (nodes["y"].min(), nodes["y"].max())
            z_bounds = (nodes["z"].min(), nodes["z"].max())

            mid = np.array([
                (x_bounds[0] + x_bounds[1]) / 2,
                (y_bounds[0] + y_bounds[1]) / 2,
                (z_bounds[0] + z_bounds[1]) / 2,
            ])
            max_range = max(x_bounds[1] - x_bounds[0],
                           y_bounds[1] - y_bounds[0],
                           z_bounds[1] - z_bounds[0]) * 0.6

            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        base_name = output_name.replace(".png", "")
        if num_figures == 1:
            output_path = PLOTS_DIR / output_name
        else:
            output_path = PLOTS_DIR / f"{base_name}_part{fig_idx + 1}.png"

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
        print(f"  Saved: {output_path} ({len(batch_cliques)} cliques)")

    return output_paths


def main() -> None:
    """Main entry point."""
    import sys

    print("=" * 60)
    print("Brain Connectome Visualization")
    print("=" * 60)

    ensure_dirs()

    # Load data
    print("\nLoading data...")
    nodes = load_nodes()
    edges = load_edges()

    print(f"  Nodes: {len(nodes):,}")
    print(f"  Edges: {len(edges):,}")

    # Plot brain overview
    print("\nPlotting brain overview...")
    brain_plot = plot_brain_3d(nodes, edges, title="Drosophila Central Brain", output_name="brain_3d_overview.png")
    print(f"Saved brain overview: {brain_plot}")

    # Build graph
    print("\nBuilding graph...")
    G, _ = build_graph()

    # Visualize 7-cliques and 8-cliques
    all_plots = [brain_plot]

    for k in [7, 8]:
        print("\n" + "=" * 60)
        print(f"Visualizing {k}-cliques (simplex dimension {k-1})")
        print("=" * 60)

        # Use smaller batch size for larger cliques (sparser)
        max_batch = 20 if k == 7 else 1
        plots = visualize_kcliques(G, nodes, k=k, max_per_figure=max_batch)
        all_plots.extend(plots)

    print("\n" + "=" * 60)
    print(f"Visualization complete! Generated {len(all_plots)} plots:")
    for p in all_plots:
        print(f"  - {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
