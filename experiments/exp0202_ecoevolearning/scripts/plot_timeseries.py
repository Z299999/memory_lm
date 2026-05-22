#!/usr/bin/env python3
"""
Post-processing plotting script for exp2-2 simulation outputs.

Reads timeseries.csv from a run directory and generates publication-quality
plots saved to a plots/ subdirectory.

Time-series smoothing:
- All time-series plots use moving average smoothing (default: 10-day window)
- Full daily data is preserved; smoothing reduces visual noise
- Smoothing window can be adjusted via --smooth_window CLI argument

Usage:
    # Plot latest run (auto-detected)
    python scripts/plot_timeseries.py

    # Plot specific run
    python scripts/plot_timeseries.py --run_dir outputs/runs/20260202_203454_seed0

    # With options
    python scripts/plot_timeseries.py --window 100 --smooth_window 10 --dpi 150 --show
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use('Agg')  # Headless backend by default
import matplotlib.pyplot as plt
import numpy as np

# Optional pandas import (fallback to manual CSV parsing if not available)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# Column name mappings for robust inference
# ============================================================================

COLUMN_ALIASES = {
    # Time column
    'time': ['t', 'day', 'days', 'time', 'step'],
    # Population
    'population': ['N', 'pop', 'population', 'population_size', 'n_agents'],
    # Births/Deaths counts
    'births': ['B', 'births', 'birth_count', 'n_births'],
    'deaths': ['D', 'deaths', 'death_count', 'n_deaths'],
    # Births/Deaths rates
    'birth_rate': ['b', 'birth_rate', 'br'],
    'death_rate': ['d', 'death_rate', 'dr'],
    # Mass
    'mass': ['M', 'mass', 'mass_total', 'biomass', 'total_mass'],
    # Energy stats
    'E_mean': ['E_mean', 'energy_mean', 'mean_energy', 'avg_energy'],
    'E_min': ['E_min', 'energy_min', 'min_energy'],
    'E_max': ['E_max', 'energy_max', 'max_energy'],
    # Energy fluxes
    'E_in': ['injected_energy', 'E_in', 'energy_in', 'energy_injected'],
    'E_eat': ['energy_gained_hunt_total', 'E_eat', 'energy_eat', 'energy_eaten', 'hunt_energy'],
    'E_rot': ['energy_rotted', 'E_rot', 'rotted_energy', 'rot_energy'],
    # Efficiency metrics
    'eta_day': ['eta_day', 'eta_daily', 'daily_eta'],
    'eta_cum': ['eta_cum', 'eta_cumulative', 'cumulative_eta'],
    'eta_window': ['eta_window', 'eta100', 'eta_rolling'],
    'rot_window': ['rot_window', 'rot100', 'rot_rolling'],
    # Prey
    'prey_count': ['prey_count', 'n_prey', 'prey'],
    'prey_energy': ['prey_energy_total', 'prey_energy'],
    # Costs (for energy budget plot)
    'E_move': ['energy_cost_motion_total', 'E_move', 'motion_cost', 'cost_motion'],
    'E_basal': ['energy_cost_basal_total', 'E_basal', 'basal_cost', 'cost_basal'],
    'E_birth': ['energy_cost_birth_total', 'E_birth', 'birth_cost', 'cost_birth'],
    'E_age': ['energy_cost_aging_total', 'E_age', 'aging_cost', 'cost_aging'],
}


def find_column(df_columns: List[str], key: str) -> Optional[str]:
    """
    Find the actual column name in the DataFrame that matches the logical key.

    Args:
        df_columns: List of column names in the DataFrame
        key: Logical key to search for (e.g., 'time', 'population')

    Returns:
        Actual column name or None if not found
    """
    aliases = COLUMN_ALIASES.get(key, [key])
    for alias in aliases:
        if alias in df_columns:
            return alias
        # Case-insensitive fallback
        for col in df_columns:
            if col.lower() == alias.lower():
                return col
    return None


def load_csv_manual(csv_path: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load CSV without pandas (fallback method).

    Returns:
        Tuple of (data dict, column names)
    """
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty CSV file: {csv_path}")

    headers = lines[0].strip().split(',')
    data = {h: [] for h in headers}

    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.strip().split(',')
        for h, v in zip(headers, values):
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(np.nan)

    # Convert to numpy arrays
    data = {k: np.array(v) for k, v in data.items()}
    return data, headers


def load_timeseries(run_dir: Path) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, str]]:
    """
    Load timeseries.csv from a run directory.

    Returns:
        Tuple of (data dict, column names, column mapping)
    """
    csv_path = run_dir / 'timeseries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"timeseries.csv not found in {run_dir}")

    # Load data
    if HAS_PANDAS:
        df = pd.read_csv(csv_path)
        columns = list(df.columns)
        data = {col: df[col].values for col in columns}
    else:
        data, columns = load_csv_manual(csv_path)

    # Build column mapping
    col_map = {}
    for key in COLUMN_ALIASES.keys():
        actual = find_column(columns, key)
        if actual:
            col_map[key] = actual

    return data, columns, col_map


def get_data(data: Dict[str, np.ndarray], col_map: Dict[str, str], key: str) -> Optional[np.ndarray]:
    """Get data array for a logical column key."""
    actual = col_map.get(key)
    if actual and actual in data:
        return data[actual]
    return None


def compute_rolling(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with given window size."""
    if len(arr) < window:
        window = len(arr)
    if window <= 0:
        return arr

    result = np.zeros_like(arr)
    cumsum = np.cumsum(arr)

    for i in range(len(arr)):
        start = max(0, i - window + 1)
        if start == 0:
            result[i] = cumsum[i] / (i + 1)
        else:
            result[i] = (cumsum[i] - cumsum[start - 1]) / window

    return result


def compute_rolling_ratio(num: np.ndarray, denom: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling ratio of two arrays."""
    if len(num) < window:
        window = len(num)
    if window <= 0:
        return np.zeros_like(num)

    result = np.zeros_like(num, dtype=float)
    cum_num = np.cumsum(num)
    cum_denom = np.cumsum(denom)

    for i in range(len(num)):
        start = max(0, i - window + 1)
        if start == 0:
            sum_num = cum_num[i]
            sum_denom = cum_denom[i]
        else:
            sum_num = cum_num[i] - cum_num[start - 1]
            sum_denom = cum_denom[i] - cum_denom[start - 1]

        if sum_denom > 0:
            result[i] = sum_num / sum_denom
        else:
            result[i] = 0.0

    return result


def moving_average(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Apply moving average smoothing to a time series.

    Replaces coarse subsampling (every N-th day) with rolling mean smoothing
    for cleaner visualization while preserving the full time resolution.

    Args:
        arr: Input array to smooth
        window: Rolling window size in days (default: 10)

    Returns:
        Smoothed array of same length as input
    """
    if arr is None:
        return None
    if window <= 1 or len(arr) <= window:
        return arr

    # Use cumsum for efficient rolling mean calculation
    # This is equivalent to pandas .rolling(window, min_periods=1).mean()
    result = np.zeros_like(arr, dtype=float)
    cumsum = np.cumsum(arr)

    for i in range(len(arr)):
        start = max(0, i - window + 1)
        if start == 0:
            result[i] = cumsum[i] / (i + 1)
        else:
            result[i] = (cumsum[i] - cumsum[start - 1]) / window

    return result


def smooth_timeseries(
    *arrays: np.ndarray,
    window: int = 10,
) -> Tuple[np.ndarray, ...]:
    """
    Apply moving average smoothing to multiple time series.

    Replaces coarse time subsampling with rolling mean smoothing for
    cleaner visualization while preserving full time resolution.

    Args:
        *arrays: Arrays to smooth (first array typically time, passed through unchanged)
        window: Rolling window size in days (default: 10)

    Returns:
        Tuple of arrays (first unchanged, rest smoothed)
    """
    if len(arrays) == 0:
        return ()

    # First array is typically time - pass through unchanged
    result = [arrays[0]]

    # Apply moving average to remaining arrays
    for arr in arrays[1:]:
        if arr is not None and len(arr) == len(arrays[0]):
            result.append(moving_average(arr, window))
        else:
            result.append(arr)  # Return unchanged if None or wrong length

    return tuple(result)


# ============================================================================
# Plotting functions
# ============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 5),
        'figure.dpi': 100,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_eta(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
    window: int,
    output_path: Path,
    dpi: int,
    fmt: str,
    smooth_window: int = 10,
) -> bool:
    """
    Plot energy utilization metrics (eta and rot).

    Uses moving average smoothing for visual clarity while preserving full time resolution.

    Returns True if plot was created successfully.
    """
    t = get_data(data, col_map, 'time')
    if t is None:
        print("  [WARNING] No time column found, skipping eta plot")
        return False

    # Try to get pre-computed window metrics
    eta_w = get_data(data, col_map, 'eta_window')
    rot_w = get_data(data, col_map, 'rot_window')
    eta_cum = get_data(data, col_map, 'eta_cum')

    # If not available, try to compute from raw fluxes
    if eta_w is None or rot_w is None:
        E_in = get_data(data, col_map, 'E_in')
        E_eat = get_data(data, col_map, 'E_eat')
        E_rot = get_data(data, col_map, 'E_rot')

        if E_in is not None and E_eat is not None:
            eta_w = compute_rolling_ratio(E_eat, E_in, window)
        if E_in is not None and E_rot is not None:
            rot_w = compute_rolling_ratio(E_rot, E_in, window)

    if eta_w is None:
        print("  [WARNING] Cannot compute eta metrics, skipping eta plot")
        return False

    # Apply moving average smoothing (replaces coarse time subsampling)
    _, eta_w_sm, rot_w_sm, eta_cum_sm = smooth_timeseries(
        t, eta_w, rot_w, eta_cum, window=smooth_window
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot eta_window
    ax.plot(t, eta_w_sm, 'b-', linewidth=1.5, label=f'eta{window} (utilization)')

    # Plot rot_window if available
    if rot_w_sm is not None:
        ax.plot(t, rot_w_sm, 'r-', linewidth=1.5, label=f'rot{window} (rotting)')

    # Plot eta_cum in lighter color if available
    if eta_cum_sm is not None:
        ax.plot(t, eta_cum_sm, 'b--', linewidth=1.0, alpha=0.5, label='eta_cum')

    ax.set_xlabel('Day')
    ax.set_ylabel('Fraction')
    ax.set_title(f'Energy Utilization Over Time ({smooth_window}-day smoothing)')
    ax.set_ylim(-0.05, 1.2)
    ax.legend(loc='best')
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


def plot_mass(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    fmt: str,
    smooth_window: int = 10,
) -> bool:
    """
    Plot total mass over time.

    Uses moving average smoothing for visual clarity while preserving full time resolution.
    """
    t = get_data(data, col_map, 'time')
    M = get_data(data, col_map, 'mass')

    if t is None or M is None:
        print("  [WARNING] Missing time or mass column, skipping mass plot")
        return False

    # Apply moving average smoothing (replaces coarse time subsampling)
    _, M_sm = smooth_timeseries(t, M, window=smooth_window)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, M_sm, 'g-', linewidth=1.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Mass (neurons)')
    ax.set_title(f'Total Biomass Over Time ({smooth_window}-day smoothing)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


def plot_birth_death(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    fmt: str,
    smooth_window: int = 10,
) -> bool:
    """
    Plot birth and death rates (or counts) over time.

    Uses moving average smoothing for visual clarity while preserving full time resolution.
    """
    t = get_data(data, col_map, 'time')
    if t is None:
        print("  [WARNING] No time column found, skipping birth_death plot")
        return False

    # Prefer rates
    b = get_data(data, col_map, 'birth_rate')
    d = get_data(data, col_map, 'death_rate')
    use_rates = b is not None and d is not None

    # Fallback to counts
    if not use_rates:
        b = get_data(data, col_map, 'births')
        d = get_data(data, col_map, 'deaths')

    if b is None or d is None:
        print("  [WARNING] Missing birth/death data, skipping birth_death plot")
        return False

    # Apply moving average smoothing (replaces coarse time subsampling)
    _, b_sm, d_sm = smooth_timeseries(t, b, d, window=smooth_window)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, b_sm, 'b-', linewidth=1.5, label='Birth rate' if use_rates else 'Births')
    ax.plot(t, d_sm, 'r-', linewidth=1.5, label='Death rate' if use_rates else 'Deaths')

    ax.set_xlabel('Day')
    ax.set_ylabel('Rate' if use_rates else 'Count')
    ax.set_title(f'Birth and Death Dynamics ({smooth_window}-day smoothing)')
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


def plot_pop_energy(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    fmt: str,
    smooth_window: int = 10,
) -> bool:
    """
    Plot population and mean energy with dual y-axes.

    Uses moving average smoothing for visual clarity while preserving full time resolution.
    """
    t = get_data(data, col_map, 'time')
    N = get_data(data, col_map, 'population')

    if t is None or N is None:
        print("  [WARNING] Missing time or population data, skipping pop_energy plot")
        return False

    E_mean = get_data(data, col_map, 'E_mean')

    # Apply moving average smoothing (replaces coarse time subsampling)
    _, N_sm, E_mean_sm = smooth_timeseries(t, N, E_mean, window=smooth_window)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot population on left axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Population (N)', color=color1)
    ax1.plot(t, N_sm, color=color1, linewidth=1.5, label='Population')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)

    # Plot energy on right axis if available
    if E_mean_sm is not None:
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Mean Energy', color=color2)
        ax2.plot(t, E_mean_sm, color=color2, linewidth=1.5, linestyle='--', label='E_mean')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(bottom=0)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title(f'Population and Mean Energy Over Time ({smooth_window}-day smoothing)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


def plot_energy_flux(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    fmt: str,
    smooth_window: int = 10,
) -> bool:
    """
    Plot energy fluxes (E_in, E_eat, E_rot).

    Uses moving average smoothing for visual clarity while preserving full time resolution.
    """
    t = get_data(data, col_map, 'time')
    E_in = get_data(data, col_map, 'E_in')
    E_eat = get_data(data, col_map, 'E_eat')
    E_rot = get_data(data, col_map, 'E_rot')

    if t is None:
        print("  [WARNING] No time column found, skipping energy_flux plot")
        return False

    # Need at least one flux
    has_any = E_in is not None or E_eat is not None or E_rot is not None
    if not has_any:
        print("  [WARNING] No energy flux data found, skipping energy_flux plot")
        return False

    # Apply moving average smoothing (replaces coarse time subsampling)
    _, E_in_sm, E_eat_sm, E_rot_sm = smooth_timeseries(
        t, E_in, E_eat, E_rot, window=smooth_window
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if E_in_sm is not None:
        ax.plot(t, E_in_sm, 'g-', linewidth=1.5, label='E_in (injected)')
    if E_eat_sm is not None:
        ax.plot(t, E_eat_sm, 'b-', linewidth=1.5, label='E_eat (eaten)')
    if E_rot_sm is not None:
        ax.plot(t, E_rot_sm, 'r-', linewidth=1.5, label='E_rot (rotted)')

    ax.set_xlabel('Day')
    ax.set_ylabel('Energy per Day')
    ax.set_title(f'Energy Fluxes Over Time ({smooth_window}-day smoothing)')
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


def plot_energy_budget(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
    output_path: Path,
    dpi: int,
    fmt: str,
    smooth_window: int = 10,
) -> bool:
    """
    Plot energy budget (E_move, E_basal, E_birth only - no E_eat/E_rot/E_age).

    Uses moving average smoothing for visual clarity while preserving full time resolution.
    """
    t = get_data(data, col_map, 'time')
    E_move = get_data(data, col_map, 'E_move')
    E_basal = get_data(data, col_map, 'E_basal')
    E_birth = get_data(data, col_map, 'E_birth')

    if t is None:
        print("  [WARNING] No time column found, skipping energy_budget plot")
        return False

    # Need at least one cost metric
    has_any = E_move is not None or E_basal is not None or E_birth is not None
    if not has_any:
        print("  [WARNING] No energy budget data found, skipping energy_budget plot")
        return False

    # Apply moving average smoothing (replaces coarse time subsampling)
    _, E_move_sm, E_basal_sm, E_birth_sm = smooth_timeseries(
        t, E_move, E_basal, E_birth, window=smooth_window
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if E_move_sm is not None:
        ax.plot(t, E_move_sm, 'b-', linewidth=1.5, label='E_move (motion)')
    if E_basal_sm is not None:
        ax.plot(t, E_basal_sm, 'g-', linewidth=1.5, label='E_basal (metabolism)')
    if E_birth_sm is not None:
        ax.plot(t, E_birth_sm, 'm-', linewidth=1.5, label='E_birth (reproduction)')

    ax.set_xlabel('Day')
    ax.set_ylabel('Energy per Day')
    ax.set_title(f'Energy Budget Over Time ({smooth_window}-day smoothing)')
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


# ============================================================================
# Age-dependent rates plot
# ============================================================================

def load_age_events(run_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load age event data from CSV files.

    Returns:
        Tuple of (events_data, exposure_data) or (None, None) if files don't exist
    """
    events_path = run_dir / 'age_events.csv'
    exposure_path = run_dir / 'age_exposure.csv'

    events_data = None
    exposure_data = None

    if events_path.exists():
        if HAS_PANDAS:
            df = pd.read_csv(events_path)
            events_data = df
        else:
            # Manual loading
            with open(events_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 1:
                events_data = {'t': [], 'event_type': [], 'age': [], 'value': []}
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        events_data['t'].append(int(parts[0]))
                        events_data['event_type'].append(parts[1])
                        events_data['age'].append(int(parts[2]))
                        events_data['value'].append(float(parts[3]))

    if exposure_path.exists():
        if HAS_PANDAS:
            df = pd.read_csv(exposure_path)
            exposure_data = df
        else:
            # Manual loading
            with open(exposure_path, 'r') as f:
                lines = f.readlines()
            if len(lines) > 1:
                exposure_data = {'t': [], 'age': [], 'count': []}
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        exposure_data['t'].append(int(parts[0]))
                        exposure_data['age'].append(int(parts[1]))
                        exposure_data['count'].append(int(parts[2]))

    return events_data, exposure_data


def compute_age_shares(
    events_data,
    final_window_days: int = 1000,
    max_age: int = 2000,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute normalized age distributions (shares) for birth, death, and intake.

    Each curve represents a share that sums to 1 over age bins (when denominator > 0).
    - Birth share: fraction of birth events at each age
    - Death share: fraction of death events at each age
    - Intake share: fraction of total absorbed energy at each age

    Args:
        events_data: DataFrame or dict with age events (t, event_type, age, value)
        final_window_days: Use events from last N days (default: 1000)
        max_age: Maximum age to consider

    Returns:
        Dictionary with 'ages', 'birth_share', 'death_share', 'intake_share', plus totals
    """
    if events_data is None:
        return None

    # Convert to dict if pandas DataFrame
    if HAS_PANDAS and hasattr(events_data, 'to_dict'):
        events = events_data.to_dict('list')
    else:
        events = events_data

    if not events.get('t'):
        return None

    # Find last day
    last_day = max(events['t']) if events['t'] else 0
    window_start = max(0, last_day - final_window_days)

    # Use 1-day age bins
    age_bin_width = 1
    n_bins = max_age + 1
    ages = np.arange(n_bins)  # Integer ages: 0, 1, 2, ..., max_age

    # Count events in each age bin
    birth_counts = np.zeros(n_bins)
    death_counts = np.zeros(n_bins)
    intake_sum = np.zeros(n_bins)

    for i, t in enumerate(events['t']):
        if t < window_start:
            continue
        age = events['age'][i]
        event_type = events['event_type'][i]
        value = events['value'][i]

        bin_idx = min(age, max_age)

        if event_type == 'birth':
            birth_counts[bin_idx] += 1
        elif event_type == 'death':
            death_counts[bin_idx] += 1
        elif event_type == 'intake':
            intake_sum[bin_idx] += value

    # Compute totals
    total_births = birth_counts.sum()
    total_deaths = death_counts.sum()
    total_intake = intake_sum.sum()

    # Compute normalized shares
    if total_births > 0:
        birth_share = birth_counts / total_births
    else:
        birth_share = np.zeros(n_bins)

    if total_deaths > 0:
        death_share = death_counts / total_deaths
    else:
        death_share = np.zeros(n_bins)

    if total_intake > 0:
        intake_share = intake_sum / total_intake
    else:
        intake_share = np.zeros(n_bins)

    # Sanity check: verify sums approximately equal 1
    birth_sum = birth_share.sum()
    death_sum = death_share.sum()
    intake_sum_check = intake_share.sum()

    print(f"    Sanity check - share sums: birth={birth_sum:.6f}, death={death_sum:.6f}, intake={intake_sum_check:.6f}")

    # Find valid age range (where at least one curve has data)
    has_data = (birth_counts > 0) | (death_counts > 0) | (intake_sum > 0)
    if not np.any(has_data):
        return None

    # Find max age with data
    max_age_with_data = np.max(np.where(has_data)[0]) if np.any(has_data) else 0
    valid_range = slice(0, max_age_with_data + 1)

    return {
        'ages': ages[valid_range],
        'birth_share': birth_share[valid_range],
        'death_share': death_share[valid_range],
        'intake_share': intake_share[valid_range],
        'total_births': int(total_births),
        'total_deaths': int(total_deaths),
        'total_intake': float(total_intake),
        'window_start': window_start,
        'window_end': last_day,
        'age_bin_width': age_bin_width,
    }


def plot_rates_vs_age(
    run_dir: Path,
    output_path: Path,
    dpi: int,
    fmt: str,
    final_window_days: int = 1000,
) -> bool:
    """
    Plot age-dependent normalized shares: birth event fraction, death event fraction, intake energy fraction.

    Each curve represents a normalized distribution (sums to 1 over age):
    - Birth share: fraction of birth events at each age
    - Death share: fraction of death events at each age
    - Intake share: fraction of total absorbed energy at each age

    Args:
        run_dir: Path to run directory
        output_path: Path to save the plot
        dpi: DPI for output
        fmt: Output format
        final_window_days: Use events from last N days (default: 1000)

    Returns:
        True if plot was created successfully
    """
    # Load age events data (only need events, not exposure for shares)
    events_data, _ = load_age_events(run_dir)

    if events_data is None:
        print("  [WARNING] Age event data not found, skipping age_shares plot")
        return False

    # Compute normalized shares
    shares = compute_age_shares(
        events_data,
        final_window_days=final_window_days,
    )

    if shares is None:
        print("  [WARNING] Could not compute age shares, skipping age_shares plot")
        return False

    ages = shares['ages']
    birth_share = shares['birth_share']
    death_share = shares['death_share']
    intake_share = shares['intake_share']

    # Re-bin to 5-day age bins for smoother curves
    bin_width = 5
    max_age = int(np.max(ages))
    n_bins = max_age // bin_width + 1
    bin_indices = ages // bin_width

    def aggregate_shares(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bin_centers = []
        bin_vals = []
        for k in range(n_bins):
            mask = bin_indices == k
            if not np.any(mask):
                continue
            # Sum shares within bin to preserve normalization
            v = float(np.sum(values[mask]))
            if v <= 0.0:
                continue
            center = (k + 0.5) * bin_width
            bin_centers.append(center)
            bin_vals.append(v)
        return np.asarray(bin_centers, dtype=float), np.asarray(bin_vals, dtype=float)

    ages_b, birth_share_b = aggregate_shares(birth_share)
    ages_d, death_share_b = aggregate_shares(death_share)
    ages_i, intake_share_b = aggregate_shares(intake_share)

    # Apply 5-bin moving average smoothing (age dimension)
    if len(birth_share_b) > 0:
        birth_share_b = moving_average(birth_share_b, window=5)
    if len(death_share_b) > 0:
        death_share_b = moving_average(death_share_b, window=5)
    if len(intake_share_b) > 0:
        intake_share_b = moving_average(intake_share_b, window=5)

    # Check for missing data and log warnings
    if shares['total_births'] == 0:
        print("    [NOTE] No birth events in window, birth curve will be zero")
    if shares['total_deaths'] == 0:
        print("    [NOTE] No death events in window, death curve will be zero")
    if shares['total_intake'] == 0:
        print("    [NOTE] No intake events in window, intake curve will be zero")

    # Create single-axis plot (all curves on same scale: 0 to 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    color_birth = 'tab:blue'
    color_death = 'tab:red'
    color_intake = 'tab:green'

    # Plot all three curves (5-day bins)
    if shares['total_births'] > 0 and len(ages_b) > 0:
        ax.plot(ages_b, birth_share_b, color=color_birth, linewidth=1.5, alpha=0.8,
                label=f"Birth event fraction (n={shares['total_births']})")
    if shares['total_deaths'] > 0 and len(ages_d) > 0:
        ax.plot(ages_d, death_share_b, color=color_death, linewidth=1.5, alpha=0.8,
                label=f"Death event fraction (n={shares['total_deaths']})")
    if shares['total_intake'] > 0 and len(ages_i) > 0:
        ax.plot(ages_i, intake_share_b, color=color_intake, linewidth=1.5, alpha=0.8,
                label=f"Intake energy fraction (E={shares['total_intake']:.0f})")

    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Normalized share (sum over age = 1)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')

    window_info = f"days {shares['window_start']}-{shares['window_end']}"
    ax.set_title(f'Age Distribution of Events ({window_info}, {bin_width}-day bins)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


# ============================================================================
# Age-energy histogram (final-day snapshot)
# ============================================================================

def load_age_energy_snapshot(
    run_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load final-day age-energy snapshot from CSV.

    Expects a file age_energy_snapshot.csv with columns:
        t,age,count,mean_energy
    """
    csv_path = run_dir / 'age_energy_snapshot.csv'
    if not csv_path.exists():
        return None, None, None

    try:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception as e:
        print(f"  [WARNING] Failed to load age_energy_snapshot.csv: {e}")
        return None, None, None

    # Handle case with a single row
    if data.ndim == 1:
        if data.size < 4:
            return None, None, None
        data = data.reshape(1, -1)

    ages = data[:, 1].astype(int)
    counts = data[:, 2].astype(int)
    mean_energy = data[:, 3].astype(float)
    return ages, counts, mean_energy


def plot_age_energy_histogram(
    run_dir: Path,
    output_path: Path,
    dpi: int,
    fmt: str,
) -> bool:
    """
    Plot average energy per age as a bar chart.

    Uses the final-day age-energy snapshot saved by the simulation.
    """
    ages, counts, mean_energy = load_age_energy_snapshot(run_dir)

    if ages is None or mean_energy is None or len(ages) == 0:
        print("  [WARNING] No age_energy_snapshot.csv found, skipping age_energy plot")
        return False

    # Bin ages into groups of 5 (0–4, 5–9, ...)
    bin_width = 5
    max_age = int(np.max(ages))
    n_bins = max_age // bin_width + 1
    bin_indices = ages // bin_width

    bin_centers = []
    bin_mean_energies = []

    for k in range(n_bins):
        mask = bin_indices == k
        if not np.any(mask):
            continue

        total_count = np.sum(counts[mask])
        if total_count <= 0:
            continue

        # Weighted mean energy in this age bin
        total_energy = np.sum(mean_energy[mask] * counts[mask])
        bin_mean = total_energy / total_count

        center = (k + 0.5) * bin_width
        bin_centers.append(center)
        bin_mean_energies.append(bin_mean)

    if not bin_centers:
        print("  [WARNING] No non-empty age bins, skipping age_energy plot")
        return False

    bin_centers = np.asarray(bin_centers, dtype=float)
    bin_mean_energies = np.asarray(bin_mean_energies, dtype=float)

    # Apply 5-bin moving average smoothing over age bins
    if len(bin_mean_energies) > 0:
        bin_mean_energies = moving_average(bin_mean_energies, window=5)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Line plot: binned age vs mean energy (no markers)
    ax.plot(bin_centers, bin_mean_energies, color='tab:purple', linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Mean energy per individual')
    ax.set_title(f'Final-day Age–Energy Distribution (bin={bin_width} days)')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, format=fmt, bbox_inches='tight')
    plt.close(fig)

    return True


# ============================================================================
# Run directory discovery
# ============================================================================

def find_latest_run(base_dir: Path) -> Optional[Path]:
    """
    Find the most recent run directory.

    Looks for directories matching the timestamp pattern YYYYMMDD_HHMMSS_*
    or falls back to modification time.
    """
    runs_dir = base_dir / 'outputs' / 'runs'
    if not runs_dir.exists():
        return None

    # Get all subdirectories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not run_dirs:
        return None

    # Sort by name (timestamp pattern) descending
    run_dirs.sort(key=lambda d: d.name, reverse=True)

    return run_dirs[0]


def get_last_values(
    data: Dict[str, np.ndarray],
    col_map: Dict[str, str],
) -> Dict[str, Any]:
    """Extract last-day values for summary."""
    result = {}

    for key in ['population', 'mass', 'E_mean', 'eta_window', 'rot_window', 'eta_cum']:
        arr = get_data(data, col_map, key)
        if arr is not None and len(arr) > 0:
            result[key] = float(arr[-1])

    t = get_data(data, col_map, 'time')
    if t is not None and len(t) > 0:
        result['last_day'] = int(t[-1])

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from exp2-2 simulation timeseries data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot latest run
  python scripts/plot_timeseries.py

  # Plot specific run
  python scripts/plot_timeseries.py --run_dir outputs/runs/20260202_203454_seed0

  # With options
  python scripts/plot_timeseries.py --window 100 --dpi 300 --show
        """
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        default=None,
        help='Path to run directory (default: latest run)'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=100,
        help='Rolling window size for eta/rot metrics (default: 100)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures interactively (default: off)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for output figures (default: 150)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='png',
        choices=['png', 'pdf', 'svg'],
        help='Output format (default: png)'
    )
    parser.add_argument(
        '--smooth_window',
        type=int,
        default=10,
        help='Moving average smoothing window in days (default: 10)'
    )

    args = parser.parse_args()

    # Enable interactive display if requested
    if args.show:
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

    # Find script directory to determine base path
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent  # exp2-2/

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir
    else:
        run_dir = find_latest_run(base_dir)
        if run_dir is None:
            print("ERROR: No run directories found in outputs/runs/")
            sys.exit(1)

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    print(f"Plotting run: {run_dir.name}")
    print(f"  Window size: {args.window}")
    print(f"  Smoothing window: {args.smooth_window} days")
    print(f"  Output format: {args.format}")
    print(f"  DPI: {args.dpi}")
    print()

    # Load data
    try:
        data, columns, col_map = load_timeseries(run_dir)
    except Exception as e:
        print(f"ERROR: Failed to load timeseries.csv: {e}")
        sys.exit(1)

    print(f"Detected columns: {', '.join(columns)}")
    print(f"Mapped columns: {col_map}")
    print()

    # Create plots directory
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Setup plot style
    setup_plot_style()

    # Generate plots
    created = []
    skipped = []

    print("Generating plots...")

    # (A) Energy utilization
    if plot_eta(data, col_map, args.window, plots_dir / f'eta.{args.format}', args.dpi, args.format, args.smooth_window):
        created.append('eta')
        print(f"  [OK] eta.{args.format}")
    else:
        skipped.append('eta')

    # (B) Mass
    if plot_mass(data, col_map, plots_dir / f'mass.{args.format}', args.dpi, args.format, args.smooth_window):
        created.append('mass')
        print(f"  [OK] mass.{args.format}")
    else:
        skipped.append('mass')

    # (C) Birth/death
    if plot_birth_death(data, col_map, plots_dir / f'birth_death.{args.format}', args.dpi, args.format, args.smooth_window):
        created.append('birth_death')
        print(f"  [OK] birth_death.{args.format}")
    else:
        skipped.append('birth_death')

    # (D) Population and energy
    if plot_pop_energy(data, col_map, plots_dir / f'pop_energy.{args.format}', args.dpi, args.format, args.smooth_window):
        created.append('pop_energy')
        print(f"  [OK] pop_energy.{args.format}")
    else:
        skipped.append('pop_energy')

    # (E) Energy flux
    if plot_energy_flux(data, col_map, plots_dir / f'energy_flux.{args.format}', args.dpi, args.format, args.smooth_window):
        created.append('energy_flux')
        print(f"  [OK] energy_flux.{args.format}")
    else:
        skipped.append('energy_flux')

    # (F) Energy budget
    if plot_energy_budget(data, col_map, plots_dir / f'energy_budget.{args.format}', args.dpi, args.format, args.smooth_window):
        created.append('energy_budget')
        print(f"  [OK] energy_budget.{args.format}")
    else:
        skipped.append('energy_budget')

    # (G) Age-dependent rates (uses separate age_events.csv and age_exposure.csv)
    if plot_rates_vs_age(run_dir, plots_dir / f'rates_vs_age.{args.format}', args.dpi, args.format):
        created.append('rates_vs_age')
        print(f"  [OK] rates_vs_age.{args.format}")
    else:
        skipped.append('rates_vs_age')

    # (H) Age–energy histogram (final-day snapshot)
    if plot_age_energy_histogram(run_dir, plots_dir / f'age_energy.{args.format}', args.dpi, args.format):
        created.append('age_energy')
        print(f"  [OK] age_energy.{args.format}")
    else:
        skipped.append('age_energy')

    print()

    # Get summary values
    last_vals = get_last_values(data, col_map)

    # Write summary JSON
    summary = {
        'run_dir': str(run_dir),
        'window': args.window,
        'smooth_window': args.smooth_window,
        'dpi': args.dpi,
        'format': args.format,
        'detected_columns': columns,
        'column_mapping': col_map,
        'plots_created': created,
        'plots_skipped': skipped,
        'last_values': last_vals,
    }

    summary_path = plots_dir / 'plot_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Run: {run_dir.name}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Created: {', '.join(created) if created else 'none'}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    print()
    print("Last-day values:")
    for key, val in last_vals.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print()
    print(f"Summary written to: {summary_path}")

    # Show if requested
    if args.show:
        plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
