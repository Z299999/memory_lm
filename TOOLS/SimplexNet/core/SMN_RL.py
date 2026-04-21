"""SMN_RL: High-level RL wrapper for Simplex Memory Networks.

This module provides a unified interface for training and testing SMN-based
RL agents. It integrates:
- SMNmodule for neural network architecture
- DQN algorithm for reinforcement learning
- CheckpointManager for persistence
- TrainingLogger for experiment tracking
- Plotting utilities for visualization

Usage::

    from SMN_RL import SMN_RL
    import gymnasium as gym

    # Create environment
    env = gym.make('CartPole-v1')

    # Create SMN_RL wrapper
    smn_rl = SMN_RL(
        env=env,
        n=2, m=4,           # Simplex parameters
        n_in=4, n_out=2,    # Network I/O dimensions
        gamma=0.99,
        lr=1e-3,
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        plot_dir='./plots'
    )

    # Train
    rewards = smn_rl.train(num_episodes=500)

    # Test
    test_reward = smn_rl.test(num_episodes=10)

    # Plot results
    smn_rl.plot_results()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .SMNmodule import SMNmodule
from rl.algorithms.dqn import DQN
from tools.checkpoint import CheckpointManager
from tools.logger import TrainingLogger
from tools.plot import plot_training_curves, plot_reward_curve


class SMN_RL:
    """High-level RL wrapper for SMN.

    This class provides a simple interface for training and testing SMN-based
    RL agents, with built-in checkpointing, logging, and visualization.

    Args:
        env: Gymnasium-style environment with observation_space and action_space
        n: Simplex dimension (order of simplices)
        m: Lattice parameter (size in each dimension)
        n_in: Input dimension (observation space)
        n_out: Output dimension (action space)
        gamma: Discount factor for RL
        lr: Learning rate
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay per episode
        epsilon_min: Minimum exploration rate
        buffer_size: Replay buffer capacity
        train_start: Minimum samples before training
        train_frequency: Train every N steps
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        plot_dir: Directory for plots

    Example::

        smn_rl = SMN_RL(
            env=gym.make('CartPole-v1'),
            n=2, m=4,
            n_in=4, n_out=2
        )
        rewards = smn_rl.train(num_episodes=500)
    """

    def __init__(
        self,
        env,
        n: int = 2,
        m: int = 4,
        n_in: int = 4,
        n_out: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        train_start: int = 100,  # Lower default for faster learning
        train_frequency: int = 4,
        checkpoint_dir: str | Path = './checkpoints',
        log_dir: str | Path = './logs',
        plot_dir: str | Path = './plots',
    ):
        self.env = env
        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out

        # Create Q-network
        self.q_network = SMNmodule(
            n=n, m=m, n_in=n_in, n_out=n_out, activation='relu'
        )

        # Create DQN agent
        self.agent = DQN(
            q_network=self.q_network,
            obs_dim=n_in,
            act_dim=n_out,
            gamma=gamma,
            lr=lr,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            buffer_size=buffer_size,
            train_start=train_start,
            train_frequency=train_frequency,
        )

        # Initialize managers
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.logger = TrainingLogger(log_dir)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.training_history: list[dict] = []

    def train(
        self,
        num_episodes: int = 500,
        max_steps: int = 500,
        update_target_every: int = 100,
        checkpoint_every: int = 50,
        verbose: bool = True,
        render: bool = False,
        reset: bool = False,
    ) -> list[float]:
        """Train the agent.

        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            update_target_every: Update target network every N episodes
            checkpoint_every: Save checkpoint every N episodes
            verbose: Print progress
            render: Render environment (requires gym to support it)
            reset: Reset training state (ignore existing checkpoint)

        Returns:
            List of episode rewards

        Example::

            rewards = smn_rl.train(num_episodes=500)
        """
        # Try to load checkpoint
        checkpoint = None
        start_episode = 0

        if not reset:
            checkpoint = self.checkpoint_mgr.load_latest()
            if checkpoint is not None:
                start_episode = checkpoint.get('episode', 0)
                # Convert CheckpointManager format to DQN format
                dqn_checkpoint = {
                    'q_network': checkpoint['state_dict'],
                    'target_network': checkpoint['state_dict'],  # Same as q_network
                    'optimizer': checkpoint.get('optimizer_state'),
                    'epsilon': self.agent.epsilon,  # Keep current epsilon
                }
                self.agent.load_checkpoint_from_dict(dqn_checkpoint)
                if verbose:
                    print(f"Loaded checkpoint from episode {start_episode}")

        # Log training start
        self.logger.log_init(
            config={
                'n': self.n,
                'm': self.m,
                'n_in': self.n_in,
                'n_out': self.n_out,
                'gamma': self.agent.gamma,
                'lr': self.agent.optimizer.param_groups[0]['lr'],
                'num_episodes': num_episodes,
            }
        )

        rewards_history = []
        losses_history = []
        total_steps = 0

        for episode in range(start_episode, num_episodes):
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            episode_loss = 0
            num_updates = 0

            for step in range(max_steps):
                if render:
                    self.env.render()

                # Select action
                action = self.agent.select_action(state, training=True)

                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                done = terminated or truncated

                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train
                loss = self.agent.train_step(batch_size=64, step=total_steps)
                if loss is not None:
                    episode_loss += loss
                    num_updates += 1

                state = next_state
                episode_reward += reward
                total_steps += 1

                if done:
                    break

            # Update target network
            if (episode + 1) % update_target_every == 0:
                self.agent.update_target_network()
                if verbose:
                    print(f"Episode {episode + 1}: Updated target network")

            # Decay epsilon
            self.agent.decay_epsilon()

            # Record history
            avg_loss = episode_loss / max(num_updates, 1)
            rewards_history.append(episode_reward)
            losses_history.append(avg_loss)

            self.training_history.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'loss': avg_loss,
                'epsilon': self.agent.epsilon,
            })

            # Log epoch
            self.logger.log_epoch(
                episode=episode + 1,
                reward=episode_reward,
                loss=avg_loss,
                epsilon=self.agent.epsilon,
            )

            # Save checkpoint
            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = self.checkpoint_mgr.save_checkpoint(
                    module=self.q_network,
                    optimizer=self.agent.optimizer,
                    episode=episode + 1,
                    reward=episode_reward,
                    loss=avg_loss,
                    metadata={
                        'rewards_history': rewards_history,
                        'losses_history': losses_history,
                    }
                )
                self.logger.log_checkpoint_saved(
                    str(ckpt_path), episode + 1, episode_reward
                )
                if verbose:
                    print(f"Episode {episode + 1}: Saved checkpoint")

            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

        return rewards_history

    def test(
        self,
        num_episodes: int = 10,
        max_steps: int = 500,
        render: bool = False,
        deterministic: bool = True,
    ) -> tuple[float, float, list[float]]:
        """Test the agent.

        Args:
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode
            render: Render environment
            deterministic: Use deterministic (greedy) actions

        Returns:
            Tuple of (mean_reward, std_reward, list_of_rewards)

        Example::

            mean, std, rewards = smn_rl.test(num_episodes=10)
            print(f"Test reward: {mean:.2f} +/- {std:.2f}")
        """
        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0

            for step in range(max_steps):
                if render:
                    self.env.render()

                # Select action (greedy for deterministic mode)
                action = self.agent.select_action(
                    state, training=not deterministic
                )

                # Take step
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                done = terminated or truncated

                state = next_state
                episode_reward += reward

                if done:
                    break

            rewards.append(episode_reward)

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        return float(mean_reward), float(std_reward), rewards

    def plot_results(
        self,
        window: int = 100,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """Plot training results.

        Args:
            window: Window size for moving average
            save_path: Path to save figure
            show: Display figure

        Example::

            smn_rl.plot_results(window=50)
        """
        if not self.training_history:
            print("No training history available. Run train() first.")
            return

        rewards = [h['reward'] for h in self.training_history]
        losses = [h['loss'] for h in self.training_history]

        if save_path is None:
            save_path = self.plot_dir / "training_curves.png"

        plot_training_curves(
            rewards, losses,
            save_path=save_path,
            title="SMN Training Results",
            show=show,
        )

        # Also plot reward curve with moving average
        reward_plot_path = self.plot_dir / "reward_curve.png"
        plot_reward_curve(
            rewards, window=window,
            save_path=reward_plot_path,
            title="SMN Reward Curve",
            show=show,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save agent checkpoint manually.

        Args:
            path: Path to save checkpoint
        """
        if self.training_history:
            last = self.training_history[-1]
            episode = last['episode']
            reward = last['reward']
        else:
            episode = 0
            reward = 0.0

        ckpt_path = self.checkpoint_mgr.save_checkpoint(
            module=self.q_network,
            optimizer=self.agent.optimizer,
            episode=episode,
            reward=reward,
            metadata={
                'training_history': self.training_history,
            }
        )
        print(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load agent checkpoint manually.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = self.checkpoint_mgr.load_checkpoint(Path(path))
        if checkpoint is not None:
            self.agent.load_checkpoint_from_dict(checkpoint)
            print(f"Loaded checkpoint from {path}")

    def launch_gui(self) -> None:
        """Launch GUI for interactive training (Phase 2).

        This is a stub for Phase 2 implementation.
        """
        print("GUI not yet implemented. Coming in Phase 2.")
        # TODO: Implement PySide-based GUI in tools/gui.py
        # from tools.gui import TrainingGUI
        # gui = TrainingGUI(self)
        # gui.run()
