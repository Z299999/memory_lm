"""
Hunting module for energy transfer between agents and prey.

Handles conflict resolution when multiple agents hunt the same prey.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from world import World

from agents import AgentManager
from prey import PreyManager


def compute_hunting_pairs(
    agent_positions: np.ndarray,
    prey_positions: np.ndarray,
    r_eat: float,
    world: Optional["World"] = None,
    boundary_mode: str = "periodic",
) -> Dict[int, List[int]]:
    """
    Compute which agents can hunt which preys.

    Returns a mapping from prey index to list of agent indices
    that are within hunting radius of that prey.

    Args:
        agent_positions: (N, 2) array of agent positions
        prey_positions: (M, 2) array of prey positions
        r_eat: Hunting radius
        world: World object for periodic displacement calculation (optional)
        boundary_mode: "periodic" or "reflect" (default "periodic")

    Returns:
        Dict mapping prey_idx -> [agent_idx, ...]
    """
    N = len(agent_positions)
    M = len(prey_positions)

    if N == 0 or M == 0:
        return {}

    # Compute pairwise displacements
    # diff[i, j] = prey_positions[j] - agent_positions[i]
    diff = prey_positions[np.newaxis, :, :] - agent_positions[:, np.newaxis, :]

    # Apply minimum-image convention for periodic boundaries
    if world is not None and boundary_mode == "periodic":
        diff = diff.copy()
        diff[..., 0] = diff[..., 0] - world.width * np.round(diff[..., 0] / world.width)
        diff[..., 1] = diff[..., 1] - world.height * np.round(diff[..., 1] / world.height)

    dist_sq = np.sum(diff ** 2, axis=2)  # (N, M)
    dist = np.sqrt(dist_sq)

    # Find agent-prey pairs within hunting radius
    within_radius = dist <= r_eat  # (N, M)

    # Build mapping from prey to hunters
    prey_to_hunters: Dict[int, List[int]] = defaultdict(list)
    for agent_idx in range(N):
        for prey_idx in range(M):
            if within_radius[agent_idx, prey_idx]:
                prey_to_hunters[prey_idx].append(agent_idx)

    return dict(prey_to_hunters)


def resolve_hunting(
    agent_manager: AgentManager,
    prey_manager: PreyManager,
    r_eat: float,
    world: Optional["World"] = None,
    boundary_mode: str = "periodic",
    intake_rate_per_neuron: Optional[float] = None,  # per-edge (legacy name kept for compatibility)
) -> Tuple[float, int, List[Tuple[int, float]]]:
    """
    Execute hunting with conflict resolution (equal sharing) and optional intake limit.

    When multiple agents hunt the same prey simultaneously,
    the prey's energy is shared equally among all hunters, up to each hunter's
    daily intake limit (if specified).

    Args:
        agent_manager: Manager for agents
        prey_manager: Manager for preys
        r_eat: Hunting radius
        world: World object for periodic displacement calculation (optional)
        boundary_mode: "periodic" or "reflect" (default "periodic")
        intake_rate_per_neuron: Max energy intake per edge per day (None = unlimited)
            Note: Despite name, this is per-edge (mass = edge count). Legacy name kept.

    Returns:
        Tuple of (total_energy_gained, num_hunts, intake_by_age)
        intake_by_age is a list of (age, energy_gained) tuples for each agent that ate
    """
    agent_positions = agent_manager.get_positions()
    prey_positions = prey_manager.get_positions()

    # Find hunting pairs
    prey_to_hunters = compute_hunting_pairs(
        agent_positions, prey_positions, r_eat, world, boundary_mode
    )

    if not prey_to_hunters:
        return 0.0, 0, []

    # Get current energies, masses, and ages
    agent_energies = agent_manager.get_energies()
    agent_masses = agent_manager.get_masses()
    agent_ages = agent_manager.get_ages()

    # Track energy gained per agent (for age-dependent intake rates)
    energy_per_agent: Dict[int, float] = {}

    # Compute daily intake limits per agent (if rate is specified)
    if intake_rate_per_neuron is not None and intake_rate_per_neuron > 0:
        intake_limits = agent_masses * intake_rate_per_neuron
        intake_remaining = intake_limits.copy().astype(float)
    else:
        intake_remaining = None

    total_energy_gained = 0.0
    num_hunts = 0

    # Process each prey that has hunters
    for prey_idx, hunter_indices in prey_to_hunters.items():
        if not hunter_indices:
            continue

        prey = prey_manager.preys[prey_idx]
        if prey.energy <= 0:
            continue

        # Calculate how much each hunter can take
        num_hunters = len(hunter_indices)
        available_energy = prey.energy

        if intake_remaining is not None:
            # Each hunter gets min(fair_share, remaining_intake_capacity)
            # We iterate to handle cases where some hunters are capped
            energy_to_extract = {}
            remaining_prey_energy = available_energy
            active_hunters = list(hunter_indices)

            while remaining_prey_energy > 1e-9 and active_hunters:
                fair_share = remaining_prey_energy / len(active_hunters)
                next_active = []

                for agent_idx in active_hunters:
                    capacity = intake_remaining[agent_idx]
                    if capacity <= 0:
                        continue  # Already at limit

                    extract = min(fair_share, capacity)
                    if extract > 0:
                        energy_to_extract[agent_idx] = energy_to_extract.get(agent_idx, 0) + extract
                        intake_remaining[agent_idx] -= extract
                        remaining_prey_energy -= extract

                        # Check if agent can take more in next round
                        if intake_remaining[agent_idx] > 1e-9:
                            next_active.append(agent_idx)

                # If no progress was made, break
                if len(next_active) == len(active_hunters):
                    break
                active_hunters = next_active

            # Transfer energy
            for agent_idx, extract in energy_to_extract.items():
                agent_energies[agent_idx] += extract
                total_energy_gained += extract
                energy_per_agent[agent_idx] = energy_per_agent.get(agent_idx, 0) + extract
                num_hunts += 1

            # Update prey energy (may still have some left)
            prey.energy = remaining_prey_energy
        else:
            # Unlimited intake: original behavior (instant depletion)
            energy_per_hunter = prey.energy / num_hunters

            for agent_idx in hunter_indices:
                agent_energies[agent_idx] += energy_per_hunter
                total_energy_gained += energy_per_hunter
                energy_per_agent[agent_idx] = energy_per_agent.get(agent_idx, 0) + energy_per_hunter
                num_hunts += 1

            prey.energy = 0.0

    # Update agent energies
    agent_manager.set_energies(agent_energies)

    # Remove exhausted preys (energy <= 0)
    prey_manager.remove_exhausted()

    # Build intake_by_age list: (age, energy_gained) for each agent that ate
    intake_by_age = [(int(agent_ages[idx]), energy) for idx, energy in energy_per_agent.items()]

    return total_energy_gained, num_hunts, intake_by_age


def simple_hunting(
    agent_manager: AgentManager,
    prey_manager: PreyManager,
    r_eat: float,
    world: Optional["World"] = None,
    boundary_mode: str = "periodic",
) -> Tuple[float, int]:
    """
    Alternative hunting: first-come-first-served (by agent index).

    Each prey can only be consumed by one agent per day.
    The agent with the lowest index gets priority.

    Args:
        agent_manager: Manager for agents
        prey_manager: Manager for preys
        r_eat: Hunting radius
        world: World object for periodic displacement calculation (optional)
        boundary_mode: "periodic" or "reflect" (default "periodic")

    Returns:
        Tuple of (total_energy_gained, num_hunts)
    """
    agent_positions = agent_manager.get_positions()
    prey_positions = prey_manager.get_positions()

    N = len(agent_positions)
    M = len(prey_positions)

    if N == 0 or M == 0:
        return 0.0, 0

    # Track which preys have been consumed
    prey_consumed = np.zeros(M, dtype=bool)
    agent_energies = agent_manager.get_energies()

    total_energy_gained = 0.0
    num_hunts = 0

    # Process agents in order (deterministic)
    for agent_idx in range(N):
        agent_pos = agent_positions[agent_idx]

        # Find preys within hunting radius
        for prey_idx in range(M):
            if prey_consumed[prey_idx]:
                continue

            prey = prey_manager.preys[prey_idx]
            if prey.energy <= 0:
                continue

            # Compute distance (with periodic boundary if applicable)
            diff = prey.position - agent_pos
            if world is not None and boundary_mode == "periodic":
                diff = diff.copy()
                diff[0] = diff[0] - world.width * np.round(diff[0] / world.width)
                diff[1] = diff[1] - world.height * np.round(diff[1] / world.height)
            dist = np.linalg.norm(diff)
            if dist <= r_eat:
                # Hunt this prey
                agent_energies[agent_idx] += prey.energy
                total_energy_gained += prey.energy
                prey.energy = 0.0
                prey_consumed[prey_idx] = True
                num_hunts += 1
                break  # Agent can only hunt one prey per day

    # Update agent energies
    agent_manager.set_energies(agent_energies)

    # Remove exhausted preys
    prey_manager.remove_exhausted()

    return total_energy_gained, num_hunts
