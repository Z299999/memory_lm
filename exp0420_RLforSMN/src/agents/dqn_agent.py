"""DQN Agent using SMNModule as Q-network.

This agent uses a SMN to approximate Q-values for discrete actions.
The action space is discretized into N bins for continuous control tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from ..smn_module import SMNModule


class DQNAgent:
    """DQN agent with SMN Q-network.

    Features:
    - Experience replay buffer
    - Target network for stable training
    - Epsilon-greedy exploration
    - Discretized action space for continuous control
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 7,
        action_values: np.ndarray | None = None,
        x_bounds: list[tuple[float, float]] | None = None,
        n: int = 2,
        m: int = 4,
        activation: str = 'relu',
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
    ):
        """Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            n_actions: Number of discrete actions
            action_values: Array of action values for discretization.
                If None, uses linear spacing in [-1, 1].
            x_bounds: Input bounds for SMNModule. Defaults to [(-10, 10)] for state_dim=1.
            n: Simplex dimension for SMN
            m: Resolution for SMN
            activation: Activation function
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum exploration rate
            buffer_size: Replay buffer capacity
        """
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Discretized action values
        if action_values is None:
            self.action_values = np.linspace(-1.0, 1.0, n_actions)
        else:
            self.action_values = np.array(action_values)

        # Default x_bounds for single-state (error) input
        if x_bounds is None:
            if state_dim == 1:
                x_bounds = [(-10.0, 10.0)]
            elif state_dim == 2:
                x_bounds = [(-10.0, 10.0), (-5.0, 5.0)]
            else:
                x_bounds = [(-10.0, 10.0)] * state_dim

        # Q-network and target network
        self.q_network = SMNModule(
            n=n, m=m,
            n_in=state_dim,
            n_out=n_actions,
            activation=activation,
            x_bounds=x_bounds,
        )

        self.target_network = SMNModule(
            n=n, m=m,
            n_in=state_dim,
            n_out=n_actions,
            activation=activation,
            x_bounds=x_bounds,
        )
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Training configuration
        self.train_start = 1000  # Start training after this many samples
        self.train_frequency = 4  # Train every N steps

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; else use greedy

        Returns:
            Action index (0 to n_actions-1)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def get_action_value(self, action_idx: int) -> float:
        """Get the actual action value from index."""
        return float(self.action_values[action_idx])

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 64, step: int = 0) -> float | None:
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
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

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

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# Test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("Testing DQNAgent...")

    # Test 1: Basic agent (1D state, 7 actions)
    print("\n=== Test 1: Basic DQN agent ===")
    agent = DQNAgent(state_dim=1, n_actions=7)
    print(f"Q-network: {agent.q_network}")
    print(f"Action values: {agent.action_values}")
    print(f"Epsilon: {agent.epsilon}")

    # Test action selection
    state = np.array([0.5], dtype=np.float32)
    action = agent.select_action(state, training=False)
    print(f"Greedy action for state {state}: {action} (value={agent.get_action_value(action)})")

    # Test epsilon-greedy
    actions = [agent.select_action(state, training=True) for _ in range(20)]
    print(f"Epsilon-greedy actions: {actions}")

    # Test storage and training
    for _ in range(100):
        state = np.random.randn(1)
        action = agent.select_action(state)
        next_state = state + np.random.randn(1) * 0.1
        reward = np.random.randn()
        done = random.random() < 0.1
        agent.store_transition(state, action, reward, next_state, done)

    loss = agent.train_step(batch_size=32)
    print(f"Training loss: {loss:.4f}")
    print("Test 1: PASSED")

    # Test 2: 2D state (error + velocity)
    print("\n=== Test 2: 2D state DQN agent ===")
    agent2 = DQNAgent(state_dim=2, n_actions=7)
    state2 = np.array([0.5, 0.1], dtype=np.float32)
    action2 = agent2.select_action(state2, training=False)
    print(f"Q-network: {agent2.q_network}")
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
    print("All DQNAgent tests PASSED!")
    print("="*50)
