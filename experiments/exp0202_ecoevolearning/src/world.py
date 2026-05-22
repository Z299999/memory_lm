"""
World/domain handling with reflecting and periodic boundaries.
"""

import numpy as np
from typing import Tuple


class World:
    """
    2D rectangular domain with configurable boundary conditions.

    Domain: Ω = [x_min, x_max] × [y_min, y_max]

    Supports:
    - Reflecting (elastic) boundaries
    - Periodic (wrap-around) boundaries for 2D torus topology
    """

    def __init__(
        self,
        omega_x: Tuple[float, float] = (0.0, 100.0),
        omega_y: Tuple[float, float] = (0.0, 100.0),
    ):
        """
        Initialize the world domain.

        Args:
            omega_x: (x_min, x_max) bounds
            omega_y: (y_min, y_max) bounds
        """
        self.x_min, self.x_max = omega_x
        self.y_min, self.y_max = omega_y
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

    def reflect_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions to positions.

        If a position crosses the boundary, it reflects elastically
        to remain inside the domain.

        Args:
            positions: (N, 2) array of positions

        Returns:
            (N, 2) array of reflected positions
        """
        pos = positions.copy()

        # Reflect x coordinates
        pos[:, 0] = self._reflect_1d(pos[:, 0], self.x_min, self.x_max)

        # Reflect y coordinates
        pos[:, 1] = self._reflect_1d(pos[:, 1], self.y_min, self.y_max)

        return pos

    def _reflect_1d(
        self, coords: np.ndarray, lo: float, hi: float
    ) -> np.ndarray:
        """
        Reflect coordinates in 1D to stay within [lo, hi].

        Uses modular arithmetic to handle multiple reflections.
        """
        span = hi - lo
        # Shift to [0, span]
        shifted = coords - lo
        # Number of times we cross the boundary
        n_periods = np.floor(shifted / span).astype(int)
        # Position within period
        remainder = shifted - n_periods * span
        # Odd periods mean we're going backwards (reflected)
        reflected = np.where(
            n_periods % 2 == 0,
            remainder,
            span - remainder,
        )
        return reflected + lo

    def wrap_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions (wrapping) to positions.

        Args:
            positions: (N, 2) array of positions

        Returns:
            (N, 2) array of wrapped positions within [x_min, x_max) x [y_min, y_max)
        """
        pos = positions.copy()
        # Wrap x coordinates using modulo arithmetic
        pos[:, 0] = self.x_min + (pos[:, 0] - self.x_min) % self.width
        # Wrap y coordinates
        pos[:, 1] = self.y_min + (pos[:, 1] - self.y_min) % self.height
        return pos

    def periodic_displacement(
        self,
        q_from: np.ndarray,
        q_to: np.ndarray,
    ) -> np.ndarray:
        """
        Compute displacement vector using minimum-image convention.

        For periodic boundaries, computes the shortest displacement
        considering all periodic images. This ensures that distances
        are calculated correctly across boundary wrapping.

        Args:
            q_from: (..., 2) array of source positions
            q_to: (..., 2) array of target positions

        Returns:
            (..., 2) array of displacement vectors (q_to - q_from with minimum image)
        """
        diff = q_to - q_from
        # Apply minimum image convention:
        # dx = dx - Lx * round(dx / Lx)
        diff = diff.copy()
        diff[..., 0] = diff[..., 0] - self.width * np.round(diff[..., 0] / self.width)
        diff[..., 1] = diff[..., 1] - self.height * np.round(diff[..., 1] / self.height)
        return diff

    def apply_boundary(
        self,
        positions: np.ndarray,
        mode: str = "periodic",
    ) -> np.ndarray:
        """
        Apply boundary conditions based on mode.

        Args:
            positions: (N, 2) array of positions
            mode: "periodic" for wrapping, "reflect" for elastic bouncing

        Returns:
            (N, 2) array of positions after boundary handling
        """
        if mode == "periodic":
            return self.wrap_positions(positions)
        elif mode == "reflect":
            return self.reflect_positions(positions)
        else:
            raise ValueError(f"Unknown boundary mode: {mode}")

    def random_positions(
        self, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate n random positions uniformly in the domain.

        Args:
            n: Number of positions to generate
            rng: NumPy random generator

        Returns:
            (n, 2) array of positions
        """
        x = rng.uniform(self.x_min, self.x_max, size=n)
        y = rng.uniform(self.y_min, self.y_max, size=n)
        return np.stack([x, y], axis=1)

    def is_inside(self, positions: np.ndarray) -> np.ndarray:
        """
        Check if positions are inside the domain.

        Args:
            positions: (N, 2) array of positions

        Returns:
            (N,) boolean array
        """
        in_x = (positions[:, 0] >= self.x_min) & (positions[:, 0] <= self.x_max)
        in_y = (positions[:, 1] >= self.y_min) & (positions[:, 1] <= self.y_max)
        return in_x & in_y
