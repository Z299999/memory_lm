"""
Metrics tracking module.

Collects and manages simulation metrics.

Tracks two types of data:
1. Daily aggregates (timeseries.csv) - population, energy, births/deaths per day
2. Age-at-event data (age_events.csv) - for computing age-dependent rates

Age-dependent rates (computed over final window):
- k(a): birth rate vs age = births_in_bin / exposure_in_bin
- μ(a): death rate vs age = deaths_in_bin / exposure_in_bin
- intake(a): mean energy intake per day vs age
"""

import numpy as np
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


@dataclass
class DayMetrics:
    """
    Metrics for a single simulation day.
    """

    t: int  # Day number
    N: int  # Population size
    B: int  # Number of births
    D: int  # Number of deaths
    b: float  # Birth rate (B / N at start of day)
    d: float  # Death rate (D / N at start of day)
    M: int  # Total mass (neurons)
    E_mean: float  # Mean energy
    E_min: float  # Minimum energy
    E_max: float  # Maximum energy
    prey_count: int  # Number of preys

    # Optional detailed metrics
    prey_energy_total: float = 0.0
    energy_cost_motion_total: float = 0.0
    energy_cost_basal_total: float = 0.0
    energy_cost_birth_total: float = 0.0  # Total energy spent on reproduction
    energy_cost_aging_total: float = 0.0  # Total energy spent on aging metabolism
    energy_gained_hunt_total: float = 0.0
    injected_energy: float = 0.0

    # Energy utilization metrics
    energy_rotted: float = 0.0  # Energy lost to rotting (E_rot)
    eta_day: float = 0.0  # Daily utilization: E_eat / E_in
    eta_cum: float = 0.0  # Cumulative utilization
    eta_window: float = 0.0  # Rolling window utilization
    rot_window: float = 0.0  # Rolling window rot fraction


@dataclass
class AgeEvent:
    """
    A single age-related event (birth, death, or hunting intake).

    Used to compute age-dependent rates k(a), μ(a), and intake(a).
    """
    t: int              # Day of event
    event_type: str     # "birth", "death", or "intake"
    age: int            # Age of agent at event
    value: float = 0.0  # For intake: energy gained; for birth/death: 1.0


class MetricsTracker:
    """
    Tracks and stores simulation metrics.

    Tracks both daily aggregates and age-at-event data for computing
    age-dependent rates.
    """

    def __init__(self, eta_window_days: int = 100):
        """
        Initialize metrics tracker.

        Args:
            eta_window_days: Rolling window size for utilization metrics
        """
        self.history: List[DayMetrics] = []
        self._current_day_data: dict = {}
        self.eta_window_days = eta_window_days

        # Age-at-event tracking for computing k(a), μ(a), intake(a)
        self.age_events: List[AgeEvent] = []

        # Daily exposure tracking (for rate denominator): {day: {age: count}}
        # Tracks how many agent-days of exposure exist at each age
        self.daily_age_exposure: Dict[int, Dict[int, int]] = {}

    def start_day(self, t: int, N_start: int):
        """
        Start tracking a new day.

        Args:
            t: Day number
            N_start: Population at start of day
        """
        self._current_day_data = {
            "t": t,
            "N_start": N_start,
            "B": 0,
            "D": 0,
            "motion_cost": 0.0,
            "basal_cost": 0.0,
            "birth_cost": 0.0,
            "aging_cost": 0.0,
            "hunt_energy": 0.0,
            "injected": 0.0,
            "rotted": 0.0,
        }

    def record_births(self, n_births: int):
        """Record number of births."""
        self._current_day_data["B"] = n_births

    def record_deaths(self, n_deaths: int):
        """Record number of deaths."""
        self._current_day_data["D"] = n_deaths

    def record_energy_costs(self, motion: float, basal: float):
        """Record energy costs."""
        self._current_day_data["motion_cost"] = motion
        self._current_day_data["basal_cost"] = basal

    def record_birth_cost(self, total_birth_cost: float):
        """Record total energy spent on reproduction."""
        self._current_day_data["birth_cost"] = total_birth_cost

    def record_aging_cost(self, total_aging_cost: float):
        """Record total energy spent on aging metabolism."""
        self._current_day_data["aging_cost"] = total_aging_cost

    def record_hunting(self, energy_gained: float):
        """Record energy gained from hunting."""
        self._current_day_data["hunt_energy"] = energy_gained

    def record_prey_injection(self, energy_injected: float):
        """Record prey energy injected."""
        self._current_day_data["injected"] = energy_injected

    def record_rotted_energy(self, energy_rotted: float):
        """Record energy lost to rotting."""
        self._current_day_data["rotted"] = energy_rotted

    def record_death_ages(self, ages: List[int]):
        """
        Record ages of agents that died this day.

        Args:
            ages: List of ages at death
        """
        t = self._current_day_data.get("t", 0)
        for age in ages:
            self.age_events.append(AgeEvent(t=t, event_type="death", age=age, value=1.0))

    def record_birth_ages(self, parent_ages: List[int]):
        """
        Record ages of parent agents that gave birth this day.

        Args:
            parent_ages: List of parent ages at reproduction
        """
        t = self._current_day_data.get("t", 0)
        for age in parent_ages:
            self.age_events.append(AgeEvent(t=t, event_type="birth", age=age, value=1.0))

    def record_intake_by_age(self, age_intake_pairs: List[tuple]):
        """
        Record energy intake by agent age.

        Args:
            age_intake_pairs: List of (age, energy_gained) tuples
        """
        t = self._current_day_data.get("t", 0)
        for age, energy in age_intake_pairs:
            if energy > 0:
                self.age_events.append(AgeEvent(t=t, event_type="intake", age=age, value=energy))

    def record_age_exposure(self, ages: np.ndarray):
        """
        Record age exposure (agent-days) for the current day.

        This is used as the denominator when computing age-dependent rates.

        Args:
            ages: Array of agent ages at start of day
        """
        t = self._current_day_data.get("t", 0)
        if t not in self.daily_age_exposure:
            self.daily_age_exposure[t] = {}

        for age in ages:
            age = int(age)
            if age not in self.daily_age_exposure[t]:
                self.daily_age_exposure[t][age] = 0
            self.daily_age_exposure[t][age] += 1

    def end_day(
        self,
        N_end: int,
        M: int,
        energies: np.ndarray,
        prey_count: int,
        prey_energy_total: float,
    ):
        """
        Finalize metrics for the day.

        Args:
            N_end: Population at end of day
            M: Total mass
            energies: Array of agent energies
            prey_count: Number of preys
            prey_energy_total: Total prey energy
        """
        t = self._current_day_data["t"]
        N_start = self._current_day_data["N_start"]
        B = self._current_day_data["B"]
        D = self._current_day_data["D"]

        # Compute rates (relative to start population)
        b = B / N_start if N_start > 0 else 0.0
        d = D / N_start if N_start > 0 else 0.0

        # Energy statistics
        if len(energies) > 0:
            E_mean = float(np.mean(energies))
            E_min = float(np.min(energies))
            E_max = float(np.max(energies))
        else:
            E_mean = 0.0
            E_min = 0.0
            E_max = 0.0

        # Get current day energy values
        E_eat = self._current_day_data["hunt_energy"]
        E_in = self._current_day_data["injected"]
        E_rot = self._current_day_data["rotted"]

        # Daily utilization (guard division by zero)
        eta_day = E_eat / E_in if E_in > 0 else 0.0

        # Cumulative utilization
        cum_eat = sum(m.energy_gained_hunt_total for m in self.history) + E_eat
        cum_in = sum(m.injected_energy for m in self.history) + E_in
        eta_cum = cum_eat / cum_in if cum_in > 0 else 0.0

        # Rolling window utilization
        window_start = max(0, len(self.history) - self.eta_window_days + 1)
        window_history = self.history[window_start:]

        window_eat = sum(m.energy_gained_hunt_total for m in window_history) + E_eat
        window_in = sum(m.injected_energy for m in window_history) + E_in
        window_rot = sum(m.energy_rotted for m in window_history) + E_rot

        eta_window = window_eat / window_in if window_in > 0 else 0.0
        rot_window = window_rot / window_in if window_in > 0 else 0.0

        metrics = DayMetrics(
            t=t,
            N=N_end,
            B=B,
            D=D,
            b=round(b, 4),
            d=round(d, 4),
            M=M,
            E_mean=round(E_mean, 2),
            E_min=round(E_min, 2),
            E_max=round(E_max, 2),
            prey_count=prey_count,
            prey_energy_total=round(prey_energy_total, 2),
            energy_cost_motion_total=round(self._current_day_data["motion_cost"], 2),
            energy_cost_basal_total=round(self._current_day_data["basal_cost"], 2),
            energy_cost_birth_total=round(self._current_day_data["birth_cost"], 2),
            energy_cost_aging_total=round(self._current_day_data["aging_cost"], 2),
            energy_gained_hunt_total=round(self._current_day_data["hunt_energy"], 2),
            injected_energy=round(self._current_day_data["injected"], 2),
            energy_rotted=round(E_rot, 2),
            eta_day=round(eta_day, 4),
            eta_cum=round(eta_cum, 4),
            eta_window=round(eta_window, 4),
            rot_window=round(rot_window, 4),
        )

        self.history.append(metrics)
        return metrics

    def get_latest(self) -> Optional[DayMetrics]:
        """Get the most recent day's metrics."""
        if self.history:
            return self.history[-1]
        return None

    def save_csv(self, path: Path):
        """
        Save metrics history to CSV file.

        Args:
            path: Path to output CSV file
        """
        if not self.history:
            return

        fieldnames = [
            "t",
            "N",
            "B",
            "D",
            "b",
            "d",
            "M",
            "E_mean",
            "E_min",
            "E_max",
            "prey_count",
            "prey_energy_total",
            "energy_cost_motion_total",
            "energy_cost_basal_total",
            "energy_cost_birth_total",
            "energy_cost_aging_total",
            "energy_gained_hunt_total",
            "injected_energy",
            "energy_rotted",
            "eta_day",
            "eta_cum",
            "eta_window",
            "rot_window",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in self.history:
                writer.writerow(asdict(metrics))

    def save_age_events_csv(self, path: Path):
        """
        Save age-at-event data to CSV file.

        Args:
            path: Path to output CSV file
        """
        if not self.age_events:
            return

        fieldnames = ["t", "event_type", "age", "value"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.age_events:
                writer.writerow({
                    "t": event.t,
                    "event_type": event.event_type,
                    "age": event.age,
                    "value": event.value,
                })

    def save_age_exposure_csv(self, path: Path):
        """
        Save daily age exposure data to CSV file.

        Args:
            path: Path to output CSV file
        """
        if not self.daily_age_exposure:
            return

        fieldnames = ["t", "age", "count"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t, age_counts in sorted(self.daily_age_exposure.items()):
                for age, count in sorted(age_counts.items()):
                    writer.writerow({"t": t, "age": age, "count": count})

    def compute_age_dependent_rates(
        self,
        final_window_days: int = 500,
        age_bin_width: int = 10,
        max_age: int = 2000,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute age-dependent rates from event data.

        Args:
            final_window_days: Use events from last N days
            age_bin_width: Width of age bins
            max_age: Maximum age to consider

        Returns:
            Dictionary with 'ages' (bin centers), 'birth_rate', 'death_rate', 'intake_rate'
        """
        if not self.history:
            return {}

        last_day = self.history[-1].t
        window_start = max(0, last_day - final_window_days)

        # Create age bins
        n_bins = max_age // age_bin_width + 1
        age_bins = np.arange(0, n_bins * age_bin_width, age_bin_width)
        bin_centers = age_bins[:-1] + age_bin_width / 2

        # Count events in each bin
        birth_counts = np.zeros(n_bins - 1)
        death_counts = np.zeros(n_bins - 1)
        intake_sum = np.zeros(n_bins - 1)
        intake_counts = np.zeros(n_bins - 1)

        for event in self.age_events:
            if event.t < window_start:
                continue
            bin_idx = event.age // age_bin_width
            if bin_idx >= n_bins - 1:
                bin_idx = n_bins - 2  # Clamp to last bin

            if event.event_type == "birth":
                birth_counts[bin_idx] += 1
            elif event.event_type == "death":
                death_counts[bin_idx] += 1
            elif event.event_type == "intake":
                intake_sum[bin_idx] += event.value
                intake_counts[bin_idx] += 1

        # Compute exposure per age bin
        exposure = np.zeros(n_bins - 1)
        for t, age_counts in self.daily_age_exposure.items():
            if t < window_start:
                continue
            for age, count in age_counts.items():
                bin_idx = age // age_bin_width
                if bin_idx >= n_bins - 1:
                    bin_idx = n_bins - 2
                exposure[bin_idx] += count

        # Compute rates
        with np.errstate(divide='ignore', invalid='ignore'):
            birth_rate = np.where(exposure > 0, birth_counts / exposure, 0.0)
            death_rate = np.where(exposure > 0, death_counts / exposure, 0.0)
            # Intake rate: mean energy per agent-day
            intake_rate = np.where(exposure > 0, intake_sum / exposure, 0.0)

        return {
            "ages": bin_centers,
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "intake_rate": intake_rate,
            "exposure": exposure,
            "window_start": window_start,
            "window_end": last_day,
            "age_bin_width": age_bin_width,
        }


def format_log_line(metrics: DayMetrics, eta_window_days: int = 100) -> str:
    """
    Format metrics as a concise log line.

    Args:
        metrics: DayMetrics object
        eta_window_days: Window size for eta/rot labels (default 100)

    Returns:
        Formatted string for terminal/log output
    """
    return (
        f"day={metrics.t:5d} | N={metrics.N:4d} | "
        f"B={metrics.B:3d} D={metrics.D:3d} (b={metrics.b:.3f} d={metrics.d:.3f}) | "
        f"M={metrics.M:5d} | "
        f"E(mean/min/max)={metrics.E_mean:.1f}/{metrics.E_min:.1f}/{metrics.E_max:.1f} | "
        f"prey={metrics.prey_count:4d} | "
        f"eta{eta_window_days}={metrics.eta_window:.2f} rot{eta_window_days}={metrics.rot_window:.2f}"
    )
