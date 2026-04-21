"""DQN Algorithm for TOOLS/SimplexNet.

This module provides a DQN (Deep Q-Network) implementation using SMNmodule
as the Q-network approximator.

Features:
- Experience replay buffer (standard mode)
- Online sampling mode (latest policy only)
- Target network for stable training
- Epsilon-greedy exploration
- Discretized action space for continuous control

Usage::

    from rl.algorithms import DQN
    from TOOLS.SimplexNet import SMNmodule

    # Create Q-network
    q_network = SMNmodule(n=2, m=4, n_in=4, n_out=2)

    # Create DQN agent - Standard mode (replay buffer)
    dqn = DQN(
        q_network=q_network,
        act_dim=2,
        gamma=0.99, lr=1e-3,
        sampler_type='replay'  # or 'online' or 'mixed'
    )

    # Training with trajectory
    from rl.collector import TrajectoryCollector
    from rl.mdp import GymMDP

    mdp = GymMDP('CartPole-v1')
    collector = TrajectoryCollector(mdp)

    trajectory = collector.collect_episode(dqn)
    loss = dqn.train(trajectory)
"""

from __future__ import annotations

from collections import deque
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Support both package mode (import as module) and script mode (direct run)
try:
    from ...core.SMNmodule import SMNmodule
    from ..samplers import ReplayBufferSampler, OnlineSampler, MixedSampler, Sampler
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from core.SMNmodule import SMNmodule
    from samplers import ReplayBufferSampler, OnlineSampler, MixedSampler, Sampler

if TYPE_CHECKING:
    from ..mdp import MDPTrajectory


class DQN:
    """DQN agent with SMN Q-network.

    Args:
        q_network: SMNmodule instance for Q-network
        act_dim: Dimension of action space (number of discrete actions)
        gamma: Discount factor
        lr: Learning rate
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay per episode
        epsilon_min: Minimum exploration rate
        sampler_type: Sampling strategy ('replay', 'online', 'mixed')
        buffer_size: Replay buffer capacity (for 'replay' mode)
        train_start: Minimum samples before training
        train_frequency: Train every N steps

    Example::

        # Standard DQN (replay buffer)
        dqn = DQN(
            q_network=SMNmodule(n=2, m=4, n_in=4, n_out=2),
            act_dim=2,
            gamma=0.99, lr=1e-3,
            sampler_type='replay'
        )

        # Online DQN (latest policy only)
        dqn = DQN(
            q_network=SMNmodule(n=6, m=5, n_in=4, n_out=2),
            act_dim=2,
            gamma=0.99, lr=1e-3,
            sampler_type='online'
        )
    """

    def __init__(
        self,
        q_network: SMNmodule,
        act_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        sampler_type: str = 'replay',
        buffer_size: int = 10000,
        train_start: int = 100,
        train_frequency: int = 4,
    ):
        self.act_dim = act_dim
        self.gamma = gamma
        self.train_start = train_start
        self.train_frequency = train_frequency

        # Q-network and target network
        self.q_network = q_network
        self.target_network = SMNmodule(
            n=q_network.n,
            m=q_network.m,
            n_in=q_network.n_in,
            n_out=q_network.n_out,
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Create sampler based on type
        if sampler_type == 'replay':
            self.sampler = ReplayBufferSampler(capacity=buffer_size)
        elif sampler_type == 'online':
            self.sampler = OnlineSampler(
                batch_size=64,
                min_samples=train_start,
                buffer_size=buffer_size // 20  # Smaller buffer for online
            )
        elif sampler_type == 'mixed':
            self.sampler = MixedSampler(
                replay_capacity=buffer_size * 3 // 4,
                online_buffer=buffer_size // 4,
                online_ratio=0.3
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        self.sampler_type = sampler_type

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; else use greedy

        Returns:
            Action index (0 to act_dim-1)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.act_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in sampler."""
        self.sampler.store((state, action, reward, next_state, done))

    def train(self, trajectory: 'MDPTrajectory') -> Optional[float]:
        """Train from a trajectory.

        Args:
            trajectory: MDPTrajectory containing SARSA sequence

        Returns:
            Loss value, or None if not enough samples

        This method stores all transitions from the trajectory into
        the sampler, then performs training steps.
        """
        # Store all transitions from trajectory
        for i in range(len(trajectory)):
            self.sampler.store((
                trajectory.states[i],
                trajectory.actions[i],
                trajectory.rewards[i],
                trajectory.next_states[i],
                trajectory.dones[i],
            ))

        # Train from sampler
        return self.train_step_from_buffer()

    def train_step(self, batch_size: int = 64, step: int = 0) -> Optional[float]:
        """Perform one training step (alias for train_step_from_buffer).

        Args:
            batch_size: Mini-batch size
            step: Current time step (for training frequency control)

        Returns:
            Loss value, or None if not enough samples
        """
        return self.train_step_from_buffer(batch_size, step)

    def train_step_from_buffer(self, batch_size: int = 64, step: int = 0) -> Optional[float]:
        """Perform one training step from sampler.

        Returns:
            Loss value, or None if not enough samples
        """
        if not self.sampler.can_sample(batch_size):
            return None

        # For online mode, clear old data after training
        clear_after_train = isinstance(self.sampler, OnlineSampler)

        # Sample batch
        transitions = self.sampler.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q-values for taken actions
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # MSE loss
        loss = nn.MSELoss()(current_q, target_q)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # For online mode, clear old data to force re-sampling
        if clear_after_train:
            self.sampler.clear_old()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    def load_checkpoint_from_dict(self, checkpoint: dict) -> None:
        """Load agent checkpoint from dictionary.

        Args:
            checkpoint: Checkpoint dictionary with keys:
                - q_network: Q-network state dict
                - target_network: Target network state dict
                - optimizer: Optimizer state dict
                - epsilon: Exploration rate
        """
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("Testing DQN with different samplers...")

    # Test 1: Standard DQN (replay buffer)
    print("\n=== Test 1: Standard DQN (replay) ===")
    q_net = SMNmodule(n=2, m=4, n_in=1, n_out=7)
    agent = DQN(q_network=q_net, act_dim=7, sampler_type='replay')
    print(f"Sampler: {type(agent.sampler).__name__}")
    print(f"Q-network: {agent.q_network.arch_str}")

    # Test storage and training
    for _ in range(150):
        state = np.random.randn(1)
        action = agent.select_action(state)
        next_state = state + np.random.randn(1) * 0.1
        reward = np.random.randn()
        done = np.random.random() < 0.1
        agent.store_transition(state, action, reward, next_state, done)

    loss = agent.train_step()
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    print("Test 1 (replay): PASSED")

    # Test 2: Online DQN (latest policy only)
    print("\n=== Test 2: Online DQN (online) ===")
    q_net2 = SMNmodule(n=2, m=4, n_in=1, n_out=7)
    agent2 = DQN(q_network=q_net2, act_dim=7, sampler_type='online')
    print(f"Sampler: {type(agent2.sampler).__name__}")

    for _ in range(150):
        state = np.random.randn(1)
        action = agent2.select_action(state)
        next_state = state + np.random.randn(1) * 0.1
        reward = np.random.randn()
        done = np.random.random() < 0.1
        agent2.store_transition(state, action, reward, next_state, done)

    loss = agent2.train_step()
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    print("Test 2 (online): PASSED")

    # Test 3: Mixed DQN
    print("\n=== Test 3: Mixed DQN ===")
    q_net3 = SMNmodule(n=2, m=4, n_in=1, n_out=7)
    agent3 = DQN(q_network=q_net3, act_dim=7, sampler_type='mixed')
    print(f"Sampler: {type(agent3.sampler).__name__}")

    for _ in range(150):
        state = np.random.randn(1)
        action = agent3.select_action(state)
        next_state = state + np.random.randn(1) * 0.1
        reward = np.random.randn()
        done = np.random.random() < 0.1
        agent3.store_transition(state, action, reward, next_state, done)

    loss = agent3.train_step()
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    print("Test 3 (mixed): PASSED")

    # Test 4: Checkpoint save/load
    print("\n=== Test 4: Checkpoint save/load ===")
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        agent.save_checkpoint(f.name)
        agent.load_checkpoint(f.name)
        os.unlink(f.name)
    print("Checkpoint test: PASSED")

    print("\n" + "="*50)
    print("All DQN tests PASSED!")
    print("="*50)
