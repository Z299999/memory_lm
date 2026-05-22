"""
Observer module for local perception (nose).

Computes sensory signals from prey (nose) for agent perception.

Supports two modes:
- "vector": Original 5D observation [S_vec_x, S_vec_y, S_scalar, energy, age]
- "stencil": 11D observation combining nose (9D) and internal state
             (energy, age):
             [nose_0..8, energy, age]

Stencil mode uses a 9-point nose stencil:
- Nose: 9 points (center + 8 directions on a circle of radius h)
"""

import numpy as np
import torch
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world import World

# Small epsilon to avoid singularity in inverse-square scent
EPS = 1e-6

# =============================================================================
# Stencil Offsets (precomputed)
# =============================================================================

# Nose: 9-point stencil (center + 8 directions on unit circle)
_NOSE_ANGLES = np.array([2 * np.pi * k / 8 for k in range(8)], dtype=np.float32)
NOSE_OFFSETS_CIRCLE = np.array(
    [[np.cos(theta), np.sin(theta)] for theta in _NOSE_ANGLES] + [[0.0, 0.0]],
    dtype=np.float32,
)


def compute_scent_intensity_at_points(
    query_positions: np.ndarray,
    prey_positions: np.ndarray,
    R: float,
    world: Optional["World"] = None,
    boundary_mode: str = "periodic",
) -> np.ndarray:
    """
    Compute scalar scent intensity at query positions.

    The scent intensity at position q is:
        I(q) = Σ_j 1 / (||q - q_j||² + eps)
    for all prey j within detection radius R.

    Args:
        query_positions: (K, 2) array of query positions
        prey_positions: (M, 2) array of prey positions
        R: Detection radius
        world: World object for periodic displacement calculation
        boundary_mode: "periodic" or "reflect"

    Returns:
        (K,) array of scent intensities at each query position
    """
    K = len(query_positions)
    M = len(prey_positions)

    if K == 0:
        return np.zeros(0, dtype=np.float32)

    if M == 0:
        return np.zeros(K, dtype=np.float32)

    # Compute pairwise displacements: (K, M, 2)
    # diff[i, j] = prey_positions[j] - query_positions[i]
    diff = prey_positions[np.newaxis, :, :] - query_positions[:, np.newaxis, :]

    # Apply minimum-image convention for periodic boundaries
    if world is not None and boundary_mode == "periodic":
        diff = diff.copy()
        diff[..., 0] = diff[..., 0] - world.width * np.round(diff[..., 0] / world.width)
        diff[..., 1] = diff[..., 1] - world.height * np.round(diff[..., 1] / world.height)

    # Squared distances: (K, M)
    dist_sq = np.sum(diff ** 2, axis=2)
    dist = np.sqrt(dist_sq)

    # Mask for preys within detection radius R
    within_radius = dist <= R  # (K, M)

    # Scent intensity: I = 1 / (r² + eps)
    intensity = 1.0 / (dist_sq + EPS)  # (K, M)

    # Apply mask and sum over all prey
    intensity_masked = intensity * within_radius
    total_intensity = np.sum(intensity_masked, axis=1)  # (K,)

    return total_intensity.astype(np.float32)


def compute_stencil_observations(
    agent_positions: np.ndarray,
    agent_energies: np.ndarray,
    agent_ages: np.ndarray,
    prey_positions: np.ndarray,
    R_nose: float,
    stencil_h: float,
    world: Optional["World"] = None,
    boundary_mode: str = "periodic",
    stencil_geometry: str = "square",
    R_eye: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute stencil-based nose observations for all agents (11D).

    Nose (9D, circle + center):
      - Samples prey scent intensity at 9 points:
        8 directions on a circle of radius h, plus center.

    Observation layout (11D):
      [nose_0..8,
       energy, age]

    Args:
        agent_positions: (N, 2) array of agent positions
        agent_energies: (N,) array of agent energies
        agent_ages: (N,) array of agent ages
        prey_positions: (M, 2) array of prey positions
        R_nose: Detection radius for nose (scent from prey)
        stencil_h: Sampling distance h for nose sampling points
        world: World object for periodic displacement calculation
        boundary_mode: "periodic" or "reflect"
        stencil_geometry: Ignored (kept for backward compatibility)
        R_eye: Unused (kept for backward compatibility).

    Returns:
        (N, 11) tensor of observations
    """
    N = len(agent_positions)

    if N == 0:
        return torch.zeros((0, 11), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Nose: prey intensity at 9 circle points (8 directions + center)
    # ------------------------------------------------------------------
    nose_offsets_scaled = NOSE_OFFSETS_CIRCLE * stencil_h  # (9, 2)

    # (N, 9, 2) query positions for nose
    nose_query_positions = agent_positions[:, np.newaxis, :] + nose_offsets_scaled[np.newaxis, :, :]
    nose_query_flat = nose_query_positions.reshape(-1, 2)

    # Apply boundary wrapping for nose queries if periodic
    if world is not None and boundary_mode == "periodic":
        nose_query_flat = nose_query_flat.copy()
        nose_query_flat[:, 0] = nose_query_flat[:, 0] % world.width
        nose_query_flat[:, 1] = nose_query_flat[:, 1] % world.height

    # Compute scent intensity at all nose query positions
    nose_intensities_flat = compute_scent_intensity_at_points(
        nose_query_flat, prey_positions, R_nose, world, boundary_mode
    )
    nose_intensities = nose_intensities_flat.reshape(N, 9)  # (N, 9)

    # ------------------------------------------------------------------
    # Build final observation tensor (N, 11)
    # ------------------------------------------------------------------
    observations = np.zeros((N, 11), dtype=np.float32)
    observations[:, 0:9] = nose_intensities
    observations[:, 9] = agent_energies.astype(np.float32)
    observations[:, 10] = agent_ages.astype(np.float32)

    return torch.from_numpy(observations)


def compute_observations(
    agent_positions: np.ndarray,
    agent_energies: np.ndarray,
    agent_ages: np.ndarray,
    prey_positions: np.ndarray,
    R: float,
    world: Optional["World"] = None,
    boundary_mode: str = "periodic",
    scent_mode: str = "vector",
    stencil_h: float = 2.0,
    stencil_geometry: str = "square",
    R_eye: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute observations for all agents.

    Supports two scent modes:

    **"vector" mode** (default, 5D):
    Each agent perceives prey within detection radius R via inverse-square
    scent signals. The observation consists of:
    - S_vec: Vector sum of scent contributions (2D)
    - S_scalar: Scalar sum of squared magnitudes (1D)
    - energy: Agent's current energy
    - age: Agent's current age

    Scent formula (inverse-square directional):
        d = q_prey - q_agent (with periodic minimum-image if boundary_mode=="periodic")
        r2 = ||d||^2 + eps
        s_i = d / r2

    **"stencil" mode** (11D: nose 9D + 2 internal):
    Each agent samples prey scent at 9 points (8 directions on circle + center),
    plus internal state (energy, age).

    Observation layout:
      [nose_0..8, energy, age]

    Nose intensity (prey) at position q:
        I_nose(q) = Σ_j 1 / (||q - q_prey_j||² + eps), for prey within R

    Args:
        agent_positions: (N, 2) array of agent positions
        agent_energies: (N,) array of agent energies
        agent_ages: (N,) array of agent ages
        prey_positions: (M, 2) array of prey positions
        R: Detection radius
        world: World object for periodic displacement calculation (optional)
        boundary_mode: "periodic" or "reflect" (default "periodic")
        scent_mode: "vector" (5D) or "stencil" (11D)
        stencil_h: Sampling distance h for nose sampling points
        stencil_geometry: Ignored (kept for backward compatibility)
        R_eye: Unused (kept for backward compatibility)

    Returns:
        (N, 5) tensor for vector mode or (N, 11) tensor for stencil mode
    """
    # Dispatch to stencil mode if requested
    if scent_mode == "stencil":
        return compute_stencil_observations(
            agent_positions, agent_energies, agent_ages, prey_positions,
            R, stencil_h, world, boundary_mode, stencil_geometry, R_eye
        )
    N = len(agent_positions)
    M = len(prey_positions)

    # Initialize observation arrays
    S_vec = np.zeros((N, 2), dtype=np.float32)
    S_scalar = np.zeros((N,), dtype=np.float32)

    if N > 0 and M > 0:
        # Compute pairwise displacements: (N, M, 2)
        # agent_positions: (N, 2), prey_positions: (M, 2)
        # diff[i, j] = prey_positions[j] - agent_positions[i]
        diff = prey_positions[np.newaxis, :, :] - agent_positions[:, np.newaxis, :]
        # diff shape: (N, M, 2)

        # Apply minimum-image convention for periodic boundaries
        if world is not None and boundary_mode == "periodic":
            diff = diff.copy()
            diff[..., 0] = diff[..., 0] - world.width * np.round(diff[..., 0] / world.width)
            diff[..., 1] = diff[..., 1] - world.height * np.round(diff[..., 1] / world.height)

        # Squared distances
        dist_sq = np.sum(diff ** 2, axis=2)  # (N, M)
        dist = np.sqrt(dist_sq)  # (N, M)

        # Mask for preys within detection radius R
        within_radius = dist <= R  # (N, M)

        # Compute scent contributions for each agent-prey pair
        # s_i = d / (r^2 + eps)
        r2_safe = dist_sq + EPS  # (N, M)

        # Scent vectors: s_i = diff / r2_safe
        # Shape: (N, M, 2)
        scent_vec = diff / r2_safe[:, :, np.newaxis]

        # Mask out preys outside radius
        scent_vec_masked = scent_vec * within_radius[:, :, np.newaxis]

        # Vector sum over all preys
        S_vec = np.sum(scent_vec_masked, axis=1)  # (N, 2)

        # Scalar: sum of |s_i|^2
        scent_mag_sq = np.sum(scent_vec ** 2, axis=2)  # (N, M)
        scent_mag_sq_masked = scent_mag_sq * within_radius
        S_scalar = np.sum(scent_mag_sq_masked, axis=1)  # (N,)

    # Build observation tensor
    # [S_vec_x, S_vec_y, S_scalar, energy, age]
    observations = np.zeros((N, 5), dtype=np.float32)
    if N > 0:
        observations[:, 0] = S_vec[:, 0]
        observations[:, 1] = S_vec[:, 1]
        observations[:, 2] = S_scalar
        observations[:, 3] = agent_energies.astype(np.float32)
        observations[:, 4] = agent_ages.astype(np.float32)

    return torch.from_numpy(observations)


def compute_single_observation(
    agent_position: np.ndarray,
    agent_energy: float,
    agent_age: int,
    prey_positions: np.ndarray,
    R: float,
) -> Tuple[np.ndarray, float]:
    """
    Compute observation for a single agent.

    Args:
        agent_position: (2,) array of agent position
        agent_energy: Agent's current energy
        agent_age: Agent's current age
        prey_positions: (M, 2) array of prey positions
        R: Detection radius

    Returns:
        Tuple of (S_vec (2,), S_scalar)
    """
    M = len(prey_positions)

    if M == 0:
        return np.zeros(2, dtype=np.float32), 0.0

    # Compute differences
    diff = prey_positions - agent_position[np.newaxis, :]  # (M, 2)

    # Distances
    dist_sq = np.sum(diff ** 2, axis=1)  # (M,)
    dist = np.sqrt(dist_sq)

    # Mask for preys within radius
    within_radius = dist <= R  # (M,)

    # Scent contributions
    r2_safe = dist_sq + EPS
    scent_vec = diff / r2_safe[:, np.newaxis]  # (M, 2)

    # Apply mask
    scent_vec_masked = scent_vec * within_radius[:, np.newaxis]

    # Aggregate
    S_vec = np.sum(scent_vec_masked, axis=0)  # (2,)
    scent_mag_sq = np.sum(scent_vec ** 2, axis=1)
    S_scalar = np.sum(scent_mag_sq * within_radius)

    return S_vec.astype(np.float32), float(S_scalar)
