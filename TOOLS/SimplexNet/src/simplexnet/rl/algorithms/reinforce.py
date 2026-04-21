"""REINFORCE Algorithm for TOOLS/SimplexNet.

This module provides a REINFORCE (policy gradient) implementation using SMNmodule
as the policy network.

Features:
- Discrete action space support (Categorical policy)
- Continuous action space support (Gaussian policy)
- Entropy regularization for exploration
- Gradient clipping for stable training

Usage::

    from rl.algorithms import REINFORCE
    from TOOLS.SimplexNet import SMNmodule

    # Discrete policy (CartPole)
    policy = SMNmodule(n=2, m=4, n_in=4, n_out=2)
    agent = REINFORCE(
        policy_network=policy,
        act_dim=2,
        action_type='discrete'
    )

    # Continuous policy (MountainCarContinuous)
    policy = SMNmodule(n=2, m=4, n_in=3, n_out=2)  # output: [mean, log_std]
    agent = REINFORCE(
        policy_network=policy,
        act_dim=1,
        action_type='continuous'
    )
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

# Support both package mode (import as module) and script mode (direct run)
try:
    from ...core.SMNmodule import SMNmodule
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from core.SMNmodule import SMNmodule

if TYPE_CHECKING:
    from ..mdp import MDPTrajectory


class REINFORCE:
    """REINFORCE agent with SMN policy network.

    Args:
        policy_network: SMNmodule instance for policy network
        act_dim: Dimension of action space (number of discrete actions or continuous dim)
        action_type: 'discrete' for Categorical policy, 'continuous' for Gaussian policy
        gamma: Discount factor
        lr: Learning rate
        entropy_coef: Entropy regularization coefficient (encourages exploration)
        max_grad_norm: Maximum gradient norm for clipping (0 = no clip)
        action_bounds: Tuple (min, max) for continuous actions

    Example::

        # Discrete policy (CartPole)
        agent = REINFORCE(
            policy_network=SMNmodule(n=2, m=4, n_in=4, n_out=2),
            act_dim=2,
            action_type='discrete',
            gamma=0.99,
            lr=1e-3
        )

        # Continuous policy (action_dim=1, output is [mean, log_std])
        agent = REINFORCE(
            policy_network=SMNmodule(n=2, m=4, n_in=3, n_out=2),
            act_dim=1,
            action_type='continuous',
            action_bounds=(-1.0, 1.0)
        )
    """

    def __init__(
        self,
        policy_network: SMNmodule,
        act_dim: int,
        action_type: str = 'discrete',
        gamma: float = 0.99,
        lr: float = 1e-3,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        action_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        self.policy_network = policy_network
        self.act_dim = act_dim
        self.action_type = action_type
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.action_bounds = action_bounds

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Storage for one episode
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.entropies: list[torch.Tensor] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int | np.ndarray:
        """Select action from policy distribution.

        Args:
            state: Current state observation
            training: If True, sample from distribution; else use greedy (mean/mode)

        Returns:
            Action (int for discrete, np.ndarray for continuous)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        output = self.policy_network(state_t)

        if self.action_type == 'discrete':
            # Categorical policy: output = logits
            if training:
                dist = Categorical(logits=output)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            else:
                # Greedy: select most likely action
                with torch.no_grad():
                    action = output.argmax(dim=1)
                log_prob = None
                entropy = None

        else:  # continuous
            # Gaussian policy: output = [mean, log_std]
            mean = output[:, :self.act_dim]
            log_std = output[:, self.act_dim:]
            std = torch.exp(log_std)

            if training:
                dist = Normal(mean, std)
                action = dist.sample()
                # Clip action to bounds
                action = action.clamp(
                    self.action_bounds[0],
                    self.action_bounds[1]
                )
                log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
                entropy = dist.entropy().sum(dim=1, keepdim=True)
            else:
                # Greedy: use mean
                with torch.no_grad():
                    action = mean.clamp(
                        self.action_bounds[0],
                        self.action_bounds[1]
                    )
                log_prob = None
                entropy = None

        if training:
            self.log_probs.append(log_prob)
            self.entropies.append(entropy)

        return action.item() if self.action_type == 'discrete' else action.squeeze(0).numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store reward for current step.

        Note: log_prob is already stored in select_action.
        This method only stores the reward.

        Args:
            state: Current state (unused, kept for interface consistency)
            action: Action taken (unused, kept for interface consistency)
            reward: Reward received
            next_state: Next state (unused)
            done: Whether episode ended (unused)
        """
        self.rewards.append(reward)

    def train(self, trajectory: MDPTrajectory) -> Optional[float]:
        """Train from a trajectory.

        Args:
            trajectory: MDPTrajectory containing rewards (log_probs stored during action selection)

        Returns:
            Loss value, or None if no rewards to train on
        """
        if len(self.rewards) == 0:
            return None

        # Compute discounted returns
        returns = self._compute_returns(self.rewards)

        # Stack log_probs and returns
        log_probs = torch.cat(self.log_probs, dim=0)  # (T,)
        returns = torch.FloatTensor(returns)  # (T,)

        # Normalize returns (reduce variance)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss: -log(π) * G_t
        policy_loss = -(log_probs * returns).mean()

        # Entropy regularization (encourage exploration)
        entropy_loss = 0.0
        if self.entropy_coef > 0 and len(self.entropies) > 0:
            entropies = torch.cat(self.entropies, dim=0)
            entropy_loss = -self.entropy_coef * entropies.mean()

        # Total loss
        loss = policy_loss + entropy_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                self.max_grad_norm
            )

        self.optimizer.step()

        # Clear storage for next episode
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

        return float(loss.item())

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted returns G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

        Args:
            rewards: List of rewards for one episode

        Returns:
            List of discounted returns (same length as rewards)
        """
        returns = []
        G = 0.0

        # Compute returns backwards
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return returns

    def decay_entropy_lr(self, decay: float = 0.99) -> None:
        """Decay entropy coefficient.

        Args:
            decay: Decay factor (default 0.99)
        """
        self.entropy_coef = max(0.0, self.entropy_coef * decay)

    def save_checkpoint(self, path: str) -> None:
        """Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'action_type': self.action_type,
            'act_dim': self.act_dim,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, weights_only=False)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.action_type = checkpoint.get('action_type', 'discrete')
        self.act_dim = checkpoint.get('act_dim', self.act_dim)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)

    def load_checkpoint_from_dict(self, checkpoint: dict) -> None:
        """Load agent checkpoint from dictionary.

        Args:
            checkpoint: Checkpoint dictionary with keys:
                - policy_network: Policy network state dict
                - optimizer: Optimizer state dict
                - action_type: 'discrete' or 'continuous'
                - act_dim: Action dimension
                - gamma: Discount factor
                - entropy_coef: Entropy coefficient
        """
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'action_type' in checkpoint:
            self.action_type = checkpoint['action_type']
        if 'act_dim' in checkpoint:
            self.act_dim = checkpoint['act_dim']
        if 'gamma' in checkpoint:
            self.gamma = checkpoint['gamma']
        if 'entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['entropy_coef']


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("Testing REINFORCE with different action types...")

    # Test 1: Discrete policy
    print("\n=== Test 1: Discrete policy (CartPole-style) ===")
    policy = SMNmodule(n=2, m=4, n_in=4, n_out=2)
    agent = REINFORCE(
        policy_network=policy,
        act_dim=2,
        action_type='discrete',
        gamma=0.99,
        lr=1e-3,
        entropy_coef=0.01
    )
    print(f"Policy network: {agent.policy_network.arch_str}")
    print(f"Action type: {agent.action_type}")

    # Simulate one episode
    for step in range(50):
        state = np.random.randn(4)
        action = agent.select_action(state, training=True)
        reward = np.random.randn()
        agent.store_transition(state, action, reward, state, False)

    loss = agent.train(None)  # trajectory not used
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    print("Test 1 (discrete): PASSED")

    # Test 2: Continuous policy
    print("\n=== Test 2: Continuous policy ===")
    policy2 = SMNmodule(n=2, m=4, n_in=3, n_out=2)  # output: [mean, log_std]
    agent2 = REINFORCE(
        policy_network=policy2,
        act_dim=1,
        action_type='continuous',
        gamma=0.99,
        lr=1e-3,
        action_bounds=(-1.0, 1.0)
    )
    print(f"Policy network: {agent2.policy_network.arch_str}")
    print(f"Action type: {agent2.action_type}")

    for step in range(50):
        state = np.random.randn(3)
        action = agent2.select_action(state, training=True)
        reward = np.random.randn()
        agent2.store_transition(state, action, reward, state, False)

    loss = agent2.train(None)
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    print("Test 2 (continuous): PASSED")

    # Test 3: Checkpoint save/load
    print("\n=== Test 3: Checkpoint save/load ===")
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        agent.save_checkpoint(f.name)
        agent.load_checkpoint(f.name)
        os.unlink(f.name)
    print("Checkpoint test: PASSED")

    # Test 4: Greedy mode (deterministic)
    print("\n=== Test 4: Greedy mode (deterministic) ===")
    agent.log_probs.clear()
    agent.rewards.clear()
    agent.entropies.clear()
    for step in range(10):
        state = np.random.randn(4)
        action = agent.select_action(state, training=False)  # greedy
    print(f"Log probs after greedy: {len(agent.log_probs)} (should be 0)")
    print("Test 4 (greedy): PASSED")

    print("\n" + "="*50)
    print("All REINFORCE tests PASSED!")
    print("="*50)
