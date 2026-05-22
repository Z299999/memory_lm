"""
Agent class for neural-network-controlled individuals.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from controller import NeuralController, create_controller


@dataclass
class Agent:
    """
    A neural-network-controlled agent.

    Attributes:
        id: Unique identifier
        position: 2D position (x, y)
        energy: Current energy level
        age: Age in days
        mass: Number of edges (trainable weights) in the neural controller
        controller: Neural network controller
        birth_intent: Whether agent intends to reproduce next selection phase
    """

    id: int
    position: np.ndarray  # (2,)
    energy: float
    age: int
    mass: int  # Number of edges (weights) in the controller
    controller: NeuralController
    birth_intent: bool = False  # Set by controller, processed by selector

    def is_alive(self, A: int) -> bool:
        """
        Check if agent is alive based on energy and age.

        Death occurs if energy < 0 or age > A.

        Args:
            A: Maximum age

        Returns:
            True if agent survives
        """
        return self.energy >= 0 and self.age <= A

    def increment_age(self):
        """Increment agent's age by one day."""
        self.age += 1


class AgentManager:
    """
    Manages the agent population.
    """

    def __init__(
        self,
        n: int,
        eps_birth_energy: float,
        hidden_sizes: List[int],
        max_speed: float,
        np_rng: np.random.Generator,
        torch_rng: torch.Generator,
        input_size: int = 5,
    ):
        """
        Initialize agent manager.

        Args:
            n: Initial number of agents
            eps_birth_energy: Initial energy per edge for agents
            hidden_sizes: Hidden layer sizes for neural controller
            max_speed: Maximum speed for velocity output
            np_rng: NumPy random generator
            torch_rng: PyTorch random generator
            input_size: Neural network input size (5 for vector, 11 for stencil)
        """
        self.eps_birth_energy = eps_birth_energy
        self.hidden_sizes = hidden_sizes
        self.max_speed = max_speed
        self.np_rng = np_rng
        self.torch_rng = torch_rng
        self.input_size = input_size
        self.agents: List[Agent] = []
        self._next_id = 0

        # Initialize agents will be called separately after world is available

    def initialize_agents(
        self,
        n: int,
        positions: np.ndarray,
    ):
        """
        Create initial population of agents.

        Args:
            n: Number of agents to create
            positions: (n, 2) array of initial positions
        """
        for i in range(n):
            controller = create_controller(
                self.hidden_sizes,
                self.max_speed,
                self.torch_rng,
                self.input_size,
            )

            # Mass is the number of edges (trainable weights) in the controller
            mass = controller.get_num_edges()
            # Compute initial energy from per-edge value
            E_birth = mass * self.eps_birth_energy

            agent = Agent(
                id=self._next_id,
                position=positions[i].copy(),
                energy=E_birth,
                age=0,
                mass=mass,
                controller=controller,
            )
            self.agents.append(agent)
            self._next_id += 1

    def add_agent(
        self,
        position: np.ndarray,
        energy: float,
        controller: NeuralController,
    ):
        """
        Add a new agent (e.g., from reproduction).

        Args:
            position: 2D position
            energy: Initial energy
            controller: Neural controller
        """
        # Mass is the number of edges (trainable weights) in the controller
        mass = controller.get_num_edges()
        agent = Agent(
            id=self._next_id,
            position=position.copy(),
            energy=energy,
            age=0,
            mass=mass,
            controller=controller,
        )
        self.agents.append(agent)
        self._next_id += 1

    def remove_dead(self, A: int) -> tuple:
        """
        Remove dead agents.

        Death occurs if energy < 0 or age > A.

        Args:
            A: Maximum age

        Returns:
            Tuple of (number_dead, list_of_death_ages, list_of_had_birth_intent)
        """
        alive = []
        death_ages = []
        death_had_intent = []
        for a in self.agents:
            if a.is_alive(A):
                alive.append(a)
            else:
                death_ages.append(a.age)
                death_had_intent.append(a.birth_intent)
        n_dead = len(self.agents) - len(alive)
        self.agents = alive
        return n_dead, death_ages, death_had_intent

    def get_positions(self) -> np.ndarray:
        """
        Get positions of all agents.

        Returns:
            (N, 2) array of agent positions
        """
        if len(self.agents) == 0:
            return np.empty((0, 2))
        return np.stack([a.position for a in self.agents], axis=0)

    def set_positions(self, positions: np.ndarray):
        """
        Set positions of all agents.

        Args:
            positions: (N, 2) array of new positions
        """
        for i, agent in enumerate(self.agents):
            agent.position = positions[i].copy()

    def get_energies(self) -> np.ndarray:
        """
        Get energies of all agents.

        Returns:
            (N,) array of agent energies
        """
        if len(self.agents) == 0:
            return np.empty((0,))
        return np.array([a.energy for a in self.agents])

    def set_energies(self, energies: np.ndarray):
        """
        Set energies of all agents.

        Args:
            energies: (N,) array of new energies
        """
        for i, agent in enumerate(self.agents):
            agent.energy = energies[i]

    def get_ages(self) -> np.ndarray:
        """
        Get ages of all agents.

        Returns:
            (N,) array of agent ages
        """
        if len(self.agents) == 0:
            return np.empty((0,), dtype=int)
        return np.array([a.age for a in self.agents])

    def get_masses(self) -> np.ndarray:
        """
        Get masses of all agents.

        Returns:
            (N,) array of agent masses
        """
        if len(self.agents) == 0:
            return np.empty((0,), dtype=int)
        return np.array([a.mass for a in self.agents])

    def get_controllers(self) -> List[NeuralController]:
        """
        Get controllers of all agents.

        Returns:
            List of neural controllers
        """
        return [a.controller for a in self.agents]

    def increment_ages(self):
        """Increment age of all agents by one day."""
        for agent in self.agents:
            agent.increment_age()

    def count(self) -> int:
        """Return number of agents."""
        return len(self.agents)

    def total_mass(self) -> int:
        """Return total mass (edges/weights) across all agents."""
        return sum(a.mass for a in self.agents)

    def get_birth_intents(self) -> np.ndarray:
        """
        Get birth intents of all agents.

        Returns:
            (N,) boolean array of agent birth intents
        """
        if len(self.agents) == 0:
            return np.empty((0,), dtype=bool)
        return np.array([a.birth_intent for a in self.agents])

    def set_birth_intents(self, intents: np.ndarray):
        """
        Set birth intents of all agents.

        Args:
            intents: (N,) boolean array of birth intents
        """
        for i, agent in enumerate(self.agents):
            agent.birth_intent = bool(intents[i])

    def clear_birth_intents(self):
        """Clear all birth intents (reset to False)."""
        for agent in self.agents:
            agent.birth_intent = False
