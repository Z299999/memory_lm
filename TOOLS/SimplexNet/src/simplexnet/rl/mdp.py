"""MDP definitions for TOOLS/SimplexNet.

This module provides the Markov Decision Process (MDP) abstraction:
- MDP: Abstract base class (user implements state_transition + reward_function)
- GymMDP: Wrapper for Gymnasium environments
- MDPTrajectory: Data class for storing SARSA sequences

Usage::

    # User-defined MDP
    class MyMDP(MDP):
        def state_transition(self, state, action):
            # s' = f(s, a)
            ...

        def reward_function(self, state, action, next_state):
            # r = g(s, a, s')
            ...

    mdp = MyMDP()
    state = mdp.reset()
    next_state, reward, done = mdp.step(action)

    # Or use Gym environments
    mdp = GymMDP('CartPole-v1')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import gymnasium as gym


@dataclass
class MDPTrajectory:
    """Trajectory data from one episode.

    Stores SARSA sequences for algorithm training.

    Attributes:
        states: List of states [s_0, s_1, ..., s_T]
        actions: List of actions [a_0, a_1, ..., a_{T-1}]
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
        next_states: List of next states [s'_0, s'_1, ..., s'_{T-1}]
        dones: List of done flags
    """
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray | int | float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)

    def to_tensors(self):
        """Convert to torch tensors (for algorithm training)."""
        import torch
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.as_tensor(np.array(self.actions)),
            'rewards': torch.FloatTensor(self.rewards),
            'next_states': torch.FloatTensor(np.array(self.next_states)),
            'dones': torch.FloatTensor(self.dones),
        }


@dataclass
class RolloutBatch:
    """Rollout data for on-policy algorithms such as PPO."""

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    old_log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)


class MDP(ABC):
    """Abstract base class for MDP definitions.

    User only needs to implement two functions:
    - state_transition(state, action) -> next_state
    - reward_function(state, action, next_state) -> reward

    All other logic (step, reset, etc.) is provided by the base class.
    """

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """State space definition."""

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        """Action space definition."""

    @abstractmethod
    def state_transition(self, state: np.ndarray, action: Any) -> np.ndarray:
        """State transition function: s' = f(s, a).

        User defines how the state changes given an action.
        Can be:
        - Analytic function
        - Neural network
        - Physics engine (MuJoCo, etc.)
        - Real robot API
        - Anything else
        """

    @abstractmethod
    def reward_function(self, state: np.ndarray, action: Any, next_state: np.ndarray) -> float:
        """Reward function: r = g(s, a, s').

        User defines the task objective.
        """

    @property
    def gamma(self) -> float:
        """Discount factor (default 0.99). Override if needed."""
        return 0.99

    def reset(self) -> np.ndarray:
        """Reset to initial state. Override for custom initialization."""
        self._current_step = 0
        return self._initial_state()

    @abstractmethod
    def _initial_state(self) -> np.ndarray:
        """Return initial state. User must implement this."""

    def _check_terminal(self, state: np.ndarray) -> bool:
        """Check if state is terminal. Override for custom termination."""
        return False

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool]:
        """Standard step interface.

        Args:
            action: Action to take

        Returns:
            (next_state, reward, done)

        This method calls user's state_transition and reward_function.
        """
        current_state = self._state

        # User's state transition
        next_state = self.state_transition(current_state, action)

        # User's reward calculation
        reward = self.reward_function(current_state, action, next_state)

        # Check termination
        done = self._check_terminal(next_state)

        # Update internal state
        self._state = next_state
        self._current_step += 1

        return next_state, reward, done

    @property
    def _state(self) -> np.ndarray:
        """Get current state."""
        return getattr(self, '_current_state', None)

    @_state.setter
    def _state(self, value: np.ndarray):
        """Set current state."""
        self._current_state = value


class GymMDP(MDP):
    """Gymnasium environment wrapper as MDP.

    This class adapts any Gymnasium environment to the MDP interface.
    The state transition and reward are handled by the Gym environment.

    Usage::

        mdp = GymMDP('CartPole-v1')
        state = mdp.reset()
        next_state, reward, done = mdp.step(action)
    """

    def __init__(self, env: str | gym.Env, gamma: float = 0.99):
        """Initialize Gym MDP.

        Args:
            env: Gymnasium environment name (e.g., 'CartPole-v1') or env instance
            gamma: Discount factor
        """
        self._gamma = gamma
        self._env_state = None

        if isinstance(env, str):
            # Create env from string ID
            self.env = gym.make(env)
            self.env_name = env
        else:
            # Use existing env instance
            self.env = env
            self.env_name = str(env.spec.id) if hasattr(env, 'spec') and env.spec else 'unknown'

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def gamma(self) -> float:
        return self._gamma

    def _initial_state(self) -> np.ndarray:
        """Return initial state from Gym environment."""
        return self._env_state

    def reset(self) -> np.ndarray:
        """Reset Gym environment."""
        self._env_state, _ = self.env.reset()
        return np.array(self._env_state, dtype=np.float32)

    def state_transition(self, state: np.ndarray, action: Any) -> np.ndarray:
        """State transition handled by Gym environment."""
        # Gym's step() actually does the transition
        # This is called after step() has been called
        # We return the state that was already computed
        return self._next_state

    def reward_function(self, state: np.ndarray, action: Any, next_state: np.ndarray) -> float:
        """Reward handled by Gym environment."""
        # Return the reward that was already computed
        return self._reward

    def _check_terminal(self, state: np.ndarray) -> bool:
        """Check termination from Gym environment."""
        return self._done

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool]:
        """Step through Gym environment.

        Overrides base class to use Gym's native step().
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self._done = terminated or truncated
        self._reward = float(reward)
        self._next_state = np.array(next_state, dtype=np.float32)
        self._env_state = self._next_state
        return self._next_state, self._reward, self._done

    def close(self):
        """Close Gym environment."""
        self.env.close()
