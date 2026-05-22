"""
Reproduction and mutation module.

Handles offspring creation with parameter mutation.
"""

import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass

from agents import Agent, AgentManager
from controller import NeuralController


@dataclass
class BirthEvent:
    """
    A pending birth event.

    Attributes:
        parent: Parent agent
        position: Position for offspring (same as parent)
        offspring_energy: Energy for offspring
        mutated_controller: Mutated neural controller for offspring
    """

    parent: Agent
    position: np.ndarray
    offspring_energy: float
    mutated_controller: NeuralController


def mutate_controller(
    controller: NeuralController,
    delta: float,
    torch_rng: torch.Generator,
) -> NeuralController:
    """
    Create a mutated copy of a neural controller.

    Offspring parameters = parent parameters + Gaussian noise * delta

    Args:
        controller: Parent's controller
        delta: Mutation amplitude
        torch_rng: PyTorch random generator

    Returns:
        New controller with mutated parameters
    """
    # Clone the controller
    new_controller = controller.clone()

    # Get flat parameters
    params = new_controller.get_parameters_flat()

    # Add Gaussian noise
    noise = torch.randn(params.shape, generator=torch_rng) * delta
    new_params = params + noise

    # Set mutated parameters
    new_controller.set_parameters_flat(new_params)

    return new_controller


def prepare_births(
    parents: List[Agent],
    eps_birth_energy: float,
    delta: float,
    torch_rng: torch.Generator,
) -> Tuple[List[BirthEvent], int]:
    """
    Prepare birth events from agents that successfully reproduced.

    NOTE: In the new self-decided reproduction system, birth costs have already
    been deducted by the selector. This function only creates the offspring.

    Uses neuron-scaled energy:
    - E_birth_x = m_x * eps_birth_energy (initial energy for offspring)

    For each parent:
    - Create mutated offspring controller
    - Schedule birth

    Args:
        parents: List of parent agents that successfully reproduce
        eps_birth_energy: Initial energy per neuron for offspring
        delta: Mutation amplitude
        torch_rng: PyTorch random generator

    Returns:
        Tuple of:
            - List of BirthEvent objects
            - Number of births scheduled
    """
    births: List[BirthEvent] = []

    for parent in parents:
        # Compute neuron-scaled energy for offspring
        E_birth_x = parent.mass * eps_birth_energy

        # Create mutated controller
        mutated_ctrl = mutate_controller(parent.controller, delta, torch_rng)

        # Create birth event
        birth = BirthEvent(
            parent=parent,
            position=parent.position.copy(),
            offspring_energy=E_birth_x,
            mutated_controller=mutated_ctrl,
        )
        births.append(birth)

    return births, len(births)


def execute_births(
    agent_manager: AgentManager,
    births: List[BirthEvent],
) -> int:
    """
    Execute pending birth events.

    Args:
        agent_manager: Manager for agents
        births: List of BirthEvent objects

    Returns:
        Number of births executed
    """
    for birth in births:
        agent_manager.add_agent(
            position=birth.position,
            energy=birth.offspring_energy,
            controller=birth.mutated_controller,
        )

    return len(births)


class BirthQueue:
    """
    Queue for managing births across days.

    Births are scheduled one day and executed the next.
    """

    def __init__(self):
        """Initialize empty birth queue."""
        self.pending_births: List[BirthEvent] = []

    def schedule(self, births: List[BirthEvent]):
        """
        Schedule births for the next day.

        Args:
            births: List of BirthEvent objects
        """
        self.pending_births.extend(births)

    def execute(self, agent_manager: AgentManager) -> int:
        """
        Execute all pending births and clear the queue.

        Args:
            agent_manager: Manager for agents

        Returns:
            Number of births executed
        """
        n_births = execute_births(agent_manager, self.pending_births)
        self.pending_births = []
        return n_births

    def count_pending(self) -> int:
        """Return number of pending births."""
        return len(self.pending_births)
