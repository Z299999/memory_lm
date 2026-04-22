"""Trajectory collector for TOOLS/SimplexNet.

This module provides utilities for collecting SARSA trajectories
from MDP-Agent interactions.

Usage::

    mdp = GymMDP('CartPole-v1')
    agent = DQN(...)
    collector = TrajectoryCollector(mdp)

    trajectory = collector.collect_episode(agent, max_steps=500)
    # trajectory contains: states, actions, rewards, next_states, dones
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mdp import MDPTrajectory, RolloutBatch

if TYPE_CHECKING:
    from .algorithms.dqn import DQN


class TrajectoryCollector:
    """Collects SARSA trajectories from MDP-Agent interactions.

    This class connects MDP (environment) with Agent (algorithm)
    and collects interaction data for training.

    Attributes:
        mdp: The MDP environment

    Usage::

        collector = TrajectoryCollector(mdp)
        trajectory = collector.collect_episode(agent)
    """

    def __init__(self, mdp):
        """Initialize collector.

        Args:
            mdp: MDP instance (must have reset() and step() methods)
        """
        self.mdp = mdp

    @staticmethod
    def _normalize_action(action):
        """Store scalars as ints when possible, arrays as float vectors."""
        if np.isscalar(action):
            if isinstance(action, (int, np.integer)):
                return int(action)
            return float(action)
        return np.array(action, dtype=np.float32)

    def collect_episode(self, agent, max_steps: int = 500, training: bool = True) -> MDPTrajectory:
        """Collect one episode of interaction data.

        Args:
            agent: Agent instance with select_action() method
            max_steps: Maximum steps per episode
            training: If True, agent uses exploration; else uses greedy

        Returns:
            MDPTrajectory containing the SARSA sequence

        The interaction loop:
            1. Reset MDP to initial state
            2. For each step:
               a. Agent selects action given current state
               b. MDP responds with (next_state, reward, done)
               c. Store (state, action, reward, next_state, done)
               d. If done, stop; else continue with next_state
        """
        trajectory = MDPTrajectory()

        state = self.mdp.reset()

        for _ in range(max_steps):
            # Agent decides action
            action = agent.select_action(state, training=training)

            # MDP responds
            next_state, reward, done = self.mdp.step(action)

            # Store transition
            trajectory.states.append(np.array(state, dtype=np.float32))
            trajectory.actions.append(self._normalize_action(action))
            trajectory.rewards.append(float(reward))
            trajectory.next_states.append(np.array(next_state, dtype=np.float32))
            trajectory.dones.append(bool(done))

            # Move to next state
            state = next_state

            # Stop if episode ended
            if done:
                break

        return trajectory

    def collect_episodes(self, agent, num_episodes: int, max_steps: int = 500,
                        training: bool = True) -> list[MDPTrajectory]:
        """Collect multiple episodes.

        Args:
            agent: Agent instance
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            training: If True, agent uses exploration

        Returns:
            List of MDPTrajectory objects
        """
        trajectories = []
        for _ in range(num_episodes):
            traj = self.collect_episode(agent, max_steps=max_steps, training=training)
            trajectories.append(traj)
        return trajectories

    def collect_rollout(
        self,
        agent,
        rollout_steps: int,
        max_episode_steps: int = 500,
        training: bool = True,
    ) -> RolloutBatch:
        """Collect a fixed-step rollout batch, potentially spanning multiple episodes."""
        rollout = RolloutBatch()
        total_steps = 0

        while total_steps < rollout_steps:
            state = self.mdp.reset()
            episode_return = 0.0
            episode_length = 0

            for _ in range(max_episode_steps):
                action = agent.select_action(state, training=training)
                aux = {}
                if hasattr(agent, 'get_last_transition_info'):
                    aux = agent.get_last_transition_info() or {}

                next_state, reward, done = self.mdp.step(action)

                rollout.states.append(np.array(state, dtype=np.float32))
                rollout.actions.append(np.array(action, dtype=np.float32))
                rollout.rewards.append(float(reward))
                rollout.next_states.append(np.array(next_state, dtype=np.float32))
                rollout.dones.append(bool(done))
                rollout.old_log_probs.append(float(aux.get('old_log_prob', 0.0)))
                rollout.values.append(float(aux.get('value', 0.0)))

                total_steps += 1
                episode_return += float(reward)
                episode_length += 1
                state = next_state

                if done or total_steps >= rollout_steps:
                    rollout.episode_returns.append(episode_return)
                    rollout.episode_lengths.append(episode_length)
                    break

        return rollout
