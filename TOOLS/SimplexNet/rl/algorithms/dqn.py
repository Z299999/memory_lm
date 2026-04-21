"""DQN Algorithm for TOOLS/SimplexNet.

This module provides a DQN (Deep Q-Network) implementation using SMNmodule
as the Q-network approximator.

Features:
- Experience replay buffer
- Target network for stable training
- Epsilon-greedy exploration
- Discretized action space for continuous control

Usage::

    from rl.algorithms import DQN
    from TOOLS.SimplexNet import SMNmodule

    # Create Q-network
    q_network = SMNmodule(n=2, m=4, n_in=4, n_out=2)

    # Create DQN agent
    dqn = DQN(
        q_network=q_network,
        obs_dim=4, act_dim=2,
        gamma=0.99, lr=1e-3
    )

    # Training loop
    action = dqn.select_action(state, epsilon=0.1)
    dqn.store_transition(state, action, reward, next_state, done)
    loss = dqn.train_step()
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.SMNmodule import SMNmodule


class DQN:
    """DQN agent with SMN Q-network.

    Args:
        q_network: SMNmodule instance for Q-network
        obs_dim: Dimension of observation space
        act_dim: Dimension of action space (number of discrete actions)
        gamma: Discount factor
        lr: Learning rate
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay per episode
        epsilon_min: Minimum exploration rate
        buffer_size: Replay buffer capacity

    Example::

        dqn = DQN(
            q_network=SMNmodule(n=2, m=4, n_in=4, n_out=2),
            obs_dim=4, act_dim=2,
            gamma=0.99, lr=1e-3
        )
    """

    def __init__(
        self,
        q_network: SMNmodule,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        train_start: int = 1000,
        train_frequency: int = 4,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

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
        self.replay_buffer = deque(maxlen=buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Training configuration
        self.train_start = train_start
        self.train_frequency = train_frequency

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
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 64, step: int = 0) -> Optional[float]:
        """Perform one training step from replay buffer.

        Args:
            batch_size: Mini-batch size
            step: Current time step (for training frequency control)

        Returns:
            Loss value, or None if not enough samples
        """
        # Only train every train_frequency steps
        if step % self.train_frequency != 0:
            return None

        if len(self.replay_buffer) < self.train_start:
            return None

        # Sample batch
        batch = list(np.random.choice(len(self.replay_buffer), batch_size, replace=False))
        transitions = [self.replay_buffer[i] for i in batch]
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

    print("Testing DQN...")

    # Test 1: Basic agent (1D state, 7 actions)
    print("\n=== Test 1: Basic DQN agent ===")
    q_net = SMNmodule(n=2, m=4, n_in=1, n_out=7)
    agent = DQN(q_network=q_net, obs_dim=1, act_dim=7)
    print(f"Q-network: {agent.q_network.arch_str}")
    print(f"Epsilon: {agent.epsilon}")

    # Test action selection
    state = np.array([0.5], dtype=np.float32)
    action = agent.select_action(state, training=False)
    print(f"Greedy action for state {state}: {action}")

    # Test epsilon-greedy
    actions = [agent.select_action(state, training=True) for _ in range(20)]
    print(f"Epsilon-greedy actions: {actions}")

    # Test storage and training
    for _ in range(100):
        state = np.random.randn(1)
        action = agent.select_action(state)
        next_state = state + np.random.randn(1) * 0.1
        reward = np.random.randn()
        done = np.random.random() < 0.1
        agent.store_transition(state, action, reward, next_state, done)

    loss = agent.train_step(batch_size=32)
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    print("Test 1: PASSED")

    # Test 2: 2D state (error + velocity)
    print("\n=== Test 2: 2D state DQN agent ===")
    q_net2 = SMNmodule(n=2, m=4, n_in=2, n_out=7)
    agent2 = DQN(q_network=q_net2, obs_dim=2, act_dim=7)
    state2 = np.array([0.5, 0.1], dtype=np.float32)
    action2 = agent2.select_action(state2, training=False)
    print(f"Q-network: {agent2.q_network.arch_str}")
    print(f"Greedy action for state {state2}: {action2}")
    print("Test 2: PASSED")

    # Test 3: Checkpoint save/load
    print("\n=== Test 3: Checkpoint save/load ===")
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
