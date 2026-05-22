"""
Dynamics module for agent motion.

Handles position updates and energy costs for motion.
"""

import numpy as np
import torch
from typing import Tuple

from world import World


def update_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    world: World,
    boundary_mode: str = "periodic",
) -> np.ndarray:
    """
    Update agent positions with boundary handling.

    Args:
        positions: (N, 2) array of current positions
        velocities: (N, 2) array of velocity commands
        dt: Time step
        world: World object for boundary handling
        boundary_mode: "periodic" for wrapping, "reflect" for elastic bouncing

    Returns:
        (N, 2) array of new positions after boundary handling
    """
    # Compute new positions
    new_positions = positions + velocities * dt

    # Apply boundary conditions based on mode
    new_positions = world.apply_boundary(new_positions, boundary_mode)

    return new_positions


def compute_motion_costs(
    velocities: np.ndarray,
    masses: np.ndarray,
    coef: float = 0.5,
) -> np.ndarray:
    """
    Compute motion energy costs for all agents.

    Motion cost = coef * 0.5 * mass * ||velocity||^2

    Args:
        velocities: (N, 2) array of velocities
        masses: (N,) array of agent masses
        coef: Cost coefficient (default 0.5 for standard kinetic energy)

    Returns:
        (N,) array of motion costs
    """
    vel_sq = np.sum(velocities ** 2, axis=1)  # (N,)
    costs = coef * 0.5 * masses * vel_sq
    return costs


def compute_basal_costs(
    masses: np.ndarray,
    e0: float,
) -> np.ndarray:
    """
    Compute basal metabolic costs for all agents (young/prime baseline).

    Basal cost = mass * e0 per day

    Args:
        masses: (N,) array of agent masses (edges/weights)
        e0: Cost per edge per day

    Returns:
        (N,) array of basal costs
    """
    return masses * e0


class MetabolismCalculator:
    """
    Computes unified metabolic costs (basal + aging) using sigmoid model.

    Model:
        - Sigmoid: σ(x) = 1 / (1 + exp(-x))
        - Aging onset: a0(m) = a0_ref * (m / m_ref)^lifespan_exp
        - Age modulation: s(age, m) = σ(k * (age - a0(m)))
        - Total daily cost: E(age, m) = m * e0 * (1 + c_age * s(age, m))

    Larger individuals age more slowly (lifespan ~ m^0.25).

    Optimization: Precomputes sigmoid values indexed by x = k*(age - a0).
    Uses interpolation for non-integer effective ages.
    """

    def __init__(
        self,
        e0: float,
        k: float,
        a0_ref: float,
        m_ref: float,
        lifespan_exp: float,
        c_age: float,
        precompute_enabled: bool = True,
        precompute_max_age: int = 2000,
    ):
        self.e0 = e0
        self.k = k
        self.a0_ref = a0_ref
        self.m_ref = m_ref
        self.lifespan_exp = lifespan_exp
        self.c_age = c_age
        self.precompute_enabled = precompute_enabled
        self.precompute_max_age = precompute_max_age

        # Precompute lookup table indexed by x-value (scaled age difference)
        # x ranges from -k*a0_ref (young) to k*(max_age - a0_ref) (old)
        self._x_min = -self.k * self.a0_ref * 2  # Allow for larger masses
        self._x_max = self.k * self.precompute_max_age
        self._x_table = None
        self._sigmoid_table = None

        # Cache for uniform mass (common case: all agents same architecture)
        self._cached_mass = None
        self._cached_a0 = None

        if precompute_enabled:
            self._precompute_sigmoid_table()

    def _precompute_sigmoid_table(self):
        """Precompute sigmoid values for a range of x = k*(age - a0)."""
        # Create lookup table with fine resolution (0.1 step in x)
        n_points = int((self._x_max - self._x_min) * 10) + 1
        self._x_table = np.linspace(self._x_min, self._x_max, n_points)
        self._sigmoid_table = 1.0 / (1.0 + np.exp(-self._x_table))

    def compute_a0(self, masses: np.ndarray) -> np.ndarray:
        """
        Compute aging onset age: a0(m) = a0_ref * (m / m_ref)^lifespan_exp
        """
        return self.a0_ref * np.power(masses / self.m_ref, self.lifespan_exp)

    def _get_a0_cached(self, masses: np.ndarray) -> np.ndarray:
        """Get a0 with caching for uniform mass case."""
        # Check if all masses are the same (common case)
        if len(masses) > 0 and np.all(masses == masses[0]):
            if self._cached_mass != masses[0]:
                self._cached_mass = masses[0]
                self._cached_a0 = self.a0_ref * (masses[0] / self.m_ref) ** self.lifespan_exp
            return np.full(len(masses), self._cached_a0)
        return self.compute_a0(masses)

    def compute_age_modulation(
        self,
        ages: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute s(age, m) = σ(k * (age - a0(m))) using lookup table.
        """
        if len(ages) == 0:
            return np.empty((0,))

        a0 = self._get_a0_cached(masses)
        x = self.k * (ages - a0)

        if self.precompute_enabled and self._sigmoid_table is not None:
            # Use interpolation from lookup table
            return np.interp(x, self._x_table, self._sigmoid_table)
        else:
            # Direct computation
            return 1.0 / (1.0 + np.exp(-x))

    def compute_metabolic_costs(
        self,
        ages: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute E(age, m) = m * e0 * (1 + c_age * s(age, m))
        """
        if len(ages) == 0:
            return np.empty((0,))

        s = self.compute_age_modulation(ages, masses)
        return masses * self.e0 * (1.0 + self.c_age * s)

    def get_effective_basal_multiplier(
        self,
        ages: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Get (1 + c_age * s) for diagnostic purposes."""
        s = self.compute_age_modulation(ages, masses)
        return 1.0 + self.c_age * s


def apply_energy_costs(
    energies: np.ndarray,
    motion_costs: np.ndarray,
    basal_costs: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Apply energy costs to agent energies.

    Args:
        energies: (N,) array of current energies
        motion_costs: (N,) array of motion costs
        basal_costs: (N,) array of basal costs

    Returns:
        Tuple of:
            - (N,) array of new energies
            - Total motion cost (scalar)
            - Total basal cost (scalar)
    """
    new_energies = energies - motion_costs - basal_costs
    return new_energies, float(np.sum(motion_costs)), float(np.sum(basal_costs))


def velocities_from_torch(velocities_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert velocity tensor to numpy array.

    Args:
        velocities_tensor: (N, 2) torch tensor

    Returns:
        (N, 2) numpy array
    """
    return velocities_tensor.numpy()
