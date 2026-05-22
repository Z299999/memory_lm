"""
Natural selection module with self-decided reproduction.

Handles death, survival, and reproduction decisions based on agent birth intent.

New Flow (hard-threshold self-decided reproduction):
1. Agents output y_birth via controller; birth_intent = (y_birth > theta_birth)
2. For agents with birth_intent, deduct birth costs BEFORE death check
3. Death occurs if energy < 0 or age > A (minimum energy is 0)
4. Surviving agents with birth_intent spawn offspring
"""

import numpy as np
from typing import List, Tuple

from agents import Agent, AgentManager


def apply_selection_with_birth_intent(
    agent_manager: AgentManager,
    eps_birth_cost: float,
    eps_birth_energy: float,
    A: int,
) -> Tuple[int, List[Agent], List[int], int]:
    """
    Apply natural selection with self-decided reproduction.

    This implements "next-day accounting" where:
    1. Birth costs are deducted for agents with birth_intent
    2. Death check: energy < 0 or age > A
    3. Surviving agents with birth_intent spawn offspring

    Energy deduction for birth-intending agents:
    - birth_cost_x = m_x * eps_birth_cost (cost to parent)
    - E_birth_x = m_x * eps_birth_energy (offspring initial energy, paid by parent)
    - Total deduction = birth_cost_x + E_birth_x

    Args:
        agent_manager: Manager for agents
        eps_birth_cost: Energy cost per neuron for reproduction
        eps_birth_energy: Initial energy per neuron for offspring (paid by parent)
        A: Maximum age

    Returns:
        Tuple of:
            - Number of deaths
            - List of parent agents that successfully reproduce
            - List of ages at death
            - Number of dystocia deaths (died while attempting birth)
    """
    parents_to_reproduce: List[Agent] = []
    alive_agents: List[Agent] = []
    death_ages: List[int] = []
    dystocia_count = 0

    # Precompute total birth deduction per neuron
    total_birth_deduction = eps_birth_cost + eps_birth_energy

    # Single pass: deduct costs, check death, identify reproducers, filter alive
    for agent in agent_manager.agents:
        # Deduct birth costs if agent intends to reproduce
        if agent.birth_intent:
            agent.energy -= agent.mass * total_birth_deduction

        # Check survival (energy >= 0 and age <= A)
        if agent.energy >= 0 and agent.age <= A:
            # Agent survives
            if agent.birth_intent:
                parents_to_reproduce.append(agent)
            agent.birth_intent = False  # Clear intent inline
            alive_agents.append(agent)
        else:
            # Agent dies
            death_ages.append(agent.age)
            if agent.birth_intent:
                dystocia_count += 1

    # Update agent list
    n_dead = len(agent_manager.agents) - len(alive_agents)
    agent_manager.agents = alive_agents

    return n_dead, parents_to_reproduce, death_ages, dystocia_count


def get_reproducers(
    agents: List[Agent],
    A: int,
) -> List[Agent]:
    """
    Identify agents that can reproduce (have birth_intent and are alive).

    This is a utility function for external use.

    Args:
        agents: List of agents
        A: Maximum age

    Returns:
        List of agents with birth_intent that are alive
    """
    reproducers = []
    for agent in agents:
        # Must be alive
        if agent.energy < 0 or agent.age > A:
            continue
        # Must have birth intent
        if agent.birth_intent:
            reproducers.append(agent)
    return reproducers


# Keep old function for backwards compatibility but mark as deprecated
def apply_selection(
    agent_manager: AgentManager,
    eps_survival: float,
    eps_birth_threshold: float,
    A: int,
) -> Tuple[int, List[Agent], List[int]]:
    """
    DEPRECATED: Use apply_selection_with_birth_intent instead.

    This function is kept for backwards compatibility but should not be used
    with the new self-decided reproduction system.
    """
    raise NotImplementedError(
        "apply_selection is deprecated. Use apply_selection_with_birth_intent instead. "
        "The new system uses self-decided reproduction based on controller output."
    )
