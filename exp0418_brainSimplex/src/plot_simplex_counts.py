#!/usr/bin/env python3
"""Plot simplex count histograms from results/simplex_counts.csv."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"


def plot_simplex_counts() -> Path:
    """Create a dimension-vs-count bar chart for simplex counts."""
    input_path = RESULTS_DIR / "simplex_counts.csv"
    output_path = PLOTS_DIR / "simplex_counts_dim0_15.png"

    if not input_path.exists():
        raise FileNotFoundError(f"Simplex counts file not found: {input_path}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df[(df["dimension"] >= 0) & (df["dimension"] <= 15)].copy()
    if df.empty:
        raise ValueError("No simplex counts found for dimensions 0-15")

    df["count_for_log"] = df["count"].clip(lower=1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.3]},
        constrained_layout=True,
    )

    colors = [
        "#1f4e79" if count > 0 else "#b0b8c2"
        for count in df["count"]
    ]

    axes[0].bar(df["dimension"], df["count"], color=colors, width=0.82, edgecolor="#0f172a")
    axes[0].set_title(
        "Directed Simplex Counts in the Drosophila Central Brain Connectome",
        fontsize=15,
        pad=12,
    )
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    axes[1].bar(df["dimension"], df["count_for_log"], color=colors, width=0.82, edgecolor="#0f172a")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Simplex Dimension")
    axes[1].set_ylabel("Count (log scale)")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    for dim, count in zip(df["dimension"], df["count"]):
        if count == 0:
            axes[1].text(dim, 1.15, "0", ha="center", va="bottom", fontsize=9, color="#475569")

    fig.suptitle("Dimensions 0-15", fontsize=17, y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    path = plot_simplex_counts()
    print(f"Saved plot: {path}")
