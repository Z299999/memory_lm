"""
Prey class for stationary resources.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Prey:
    """
    A stationary prey resource.

    Attributes:
        id: Unique identifier
        position: 2D position (x, y)
        energy: Current energy content
        shelf_life: Remaining days before rotting (decrements each day)
    """

    id: int
    position: np.ndarray  # (2,)
    energy: float
    shelf_life: int

    def is_alive(self) -> bool:
        """Check if prey is still valid (has energy and not rotten)."""
        return self.energy > 0 and self.shelf_life > 0

    def decay(self):
        """Decrement shelf life by one day."""
        self.shelf_life -= 1


class PreyManager:
    """
    Manages the prey population.
    """

    def __init__(
        self,
        e_prey: float,
        T_prey: int,
        E_f: float,
        rng: np.random.Generator,
    ):
        """
        Initialize prey manager.

        Args:
            e_prey: Energy per prey
            T_prey: Shelf-life of prey (days)
            E_f: Total injected prey energy per day
            rng: NumPy random generator
        """
        self.e_prey = e_prey
        self.T_prey = T_prey
        self.E_f = E_f
        self.rng = rng
        self.preys: List[Prey] = []
        self._next_id = 0

    def inject_preys(self, x_bounds: tuple, y_bounds: tuple):
        """
        Inject new preys for the current day.

        Introduces new preys such that total injected energy equals E_f.
        N_new = floor(E_f / e_prey)

        Args:
            x_bounds: (x_min, x_max)
            y_bounds: (y_min, y_max)
        """
        n_new = int(self.E_f / self.e_prey)

        for _ in range(n_new):
            x = self.rng.uniform(x_bounds[0], x_bounds[1])
            y = self.rng.uniform(y_bounds[0], y_bounds[1])
            position = np.array([x, y])

            prey = Prey(
                id=self._next_id,
                position=position,
                energy=self.e_prey,
                shelf_life=self.T_prey,
            )
            self.preys.append(prey)
            self._next_id += 1

    def decay_preys(self) -> float:
        """
        Decrement shelf life of all preys and remove rotten ones.

        Returns:
            Total energy of rotted (removed) preys - energy lost to rotting
        """
        for prey in self.preys:
            prey.decay()

        # Calculate energy of rotten preys before removal
        # A prey is rotten if shelf_life <= 0 (after decay) but may still have energy
        rotted_energy = sum(
            p.energy for p in self.preys if not p.is_alive() and p.energy > 0
        )

        # Remove rotten preys
        self.preys = [p for p in self.preys if p.is_alive()]

        return rotted_energy

    def remove_exhausted(self):
        """Remove preys with no energy remaining."""
        self.preys = [p for p in self.preys if p.energy > 0]

    def get_positions(self) -> np.ndarray:
        """
        Get positions of all preys.

        Returns:
            (M, 2) array of prey positions
        """
        if len(self.preys) == 0:
            return np.empty((0, 2))
        return np.stack([p.position for p in self.preys], axis=0)

    def get_energies(self) -> np.ndarray:
        """
        Get energies of all preys.

        Returns:
            (M,) array of prey energies
        """
        if len(self.preys) == 0:
            return np.empty((0,))
        return np.array([p.energy for p in self.preys])

    def count(self) -> int:
        """Return number of preys."""
        return len(self.preys)

    def total_energy(self) -> float:
        """Return total energy across all preys."""
        return sum(p.energy for p in self.preys)
