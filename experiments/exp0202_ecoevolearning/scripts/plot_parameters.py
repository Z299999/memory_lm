#!/usr/bin/env python3
"""
Parameter visualization utilities for exp2-2.

Currently supports plotting metabolism cost vs age for the unified
basal+aging model (MetabolismCalculator), without running a simulation.
"""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Add src to path so we can import config/dynamics when running from exp2-2/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from config import Config  # noqa: E402
from dynamics import MetabolismCalculator  # noqa: E402


def plot_metabolism_vs_age(
    config_path: Path,
    output_dir: Path,
    max_age: int = 1000,
    mass: float | None = None,
) -> Path:
    """
    Plot metabolism multiplier vs age for the sigmoid_age model.

    y(age) = 1 + c_age * s(age, m)
    where s(age, m) = sigmoid(k * (age - a0(m)))
    """
    cfg = Config.from_yaml(config_path)

    if not cfg.metabolism_enabled or cfg.metabolism_mode != "sigmoid_age":
        raise RuntimeError("Metabolism model is not enabled or not in 'sigmoid_age' mode.")

    # Use reference mass by default (model is defined around this)
    if mass is None:
        mass = cfg.metabolism_m_ref

    mc = MetabolismCalculator(
        e0=cfg.e0,
        k=cfg.metabolism_k,
        a0_ref=cfg.metabolism_a0_ref,
        m_ref=cfg.metabolism_m_ref,
        lifespan_exp=cfg.metabolism_lifespan_exp,
        c_age=cfg.metabolism_c_age,
        precompute_enabled=cfg.metabolism_precompute_enabled,
        precompute_max_age=min(cfg.metabolism_precompute_max_age, max_age),
    )

    ages = np.arange(0, max_age + 1, dtype=float)
    masses = np.full_like(ages, float(mass), dtype=float)

    multiplier = mc.get_effective_basal_multiplier(ages, masses)
    per_edge_cost = cfg.e0 * multiplier

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ages, multiplier, color="tab:blue", linewidth=1.5, label="Multiplier (1 + c_age * s)")
    ax1.set_xlabel("Age (days)")
    ax1.set_ylabel("Metabolism multiplier", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ages, per_edge_cost, color="tab:orange", linewidth=1.2, linestyle="--", label="Cost per edge")
    ax2.set_ylabel("Cost per edge per day", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    title_mass = f"{mass:.0f}" if mass >= 1 else f"{mass:.2f}"
    ax1.set_title(f"Metabolism vs Age (m={title_mass}, e0={cfg.e0})")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / "metabolism_vs_age.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize parameter curves for exp2-2 (e.g., metabolism vs age)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (relative to exp2-2/).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots/params",
        help="Output directory for parameter plots (relative to exp2-2/).",
    )
    parser.add_argument(
        "--max_age",
        type=int,
        default=500,
        help="Maximum age to plot (days).",
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=None,
        help="Agent mass to use for metabolism curve (default: metabolism_m_ref).",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent  # exp2-2/
    config_path = base_dir / args.config
    out_dir = base_dir / args.out_dir

    try:
        out_path = plot_metabolism_vs_age(
            config_path=config_path,
            output_dir=out_dir,
            max_age=args.max_age,
            mass=args.mass,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"Metabolism vs age plot saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
