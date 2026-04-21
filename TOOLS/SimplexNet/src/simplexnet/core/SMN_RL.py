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
from typing import Optional, Callable, TYPE_CHECKING

import numpy as np

# Support both package mode (import as module) and script mode (direct run)
try:
    from .SMNmodule import SMNmodule
    from ..rl.algorithms.dqn import DQN
    from ..rl.algorithms.reinforce import REINFORCE
    from ..rl.mdp import GymMDP, MDPTrajectory
    from ..rl.collector import TrajectoryCollector
    from ..tools.checkpoint import CheckpointManager
    from ..tools.logger import TrainingLogger
    from ..tools.plot import plot_training_curves, plot_reward_curve
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from .SMNmodule import SMNmodule
    from rl.algorithms.dqn import DQN
    from rl.algorithms.reinforce import REINFORCE
    from rl.mdp import GymMDP, MDPTrajectory
    from rl.collector import TrajectoryCollector
    from tools.checkpoint import CheckpointManager
    from tools.logger import TrainingLogger
    from tools.plot import plot_training_curves, plot_reward_curve

if TYPE_CHECKING:
    import gymnasium as gym


class SMN_RL:
    """High-level RL wrapper for SMN.

    This class provides a simple interface for training and testing SMN-based
    RL agents, with built-in checkpointing, logging, and visualization.

    Args:
        env: Gymnasium-style environment with observation_space and action_space
        algorithm: 'dqn' or 'reinforce'
        n: Simplex dimension (order of simplices)
        m: Lattice parameter (size in each dimension)
        n_in: Input dimension (observation space)
        n_out: Output dimension (action space)
        gamma: Discount factor for RL
        lr: Learning rate
        epsilon: Initial exploration rate (DQN only)
        epsilon_decay: Epsilon decay per episode (DQN only)
        epsilon_min: Minimum exploration rate (DQN only)
        buffer_size: Replay buffer capacity (DQN only)
        train_start: Minimum samples before training (DQN only)
        train_frequency: Train every N steps (DQN only)
        sampler_type: Sampling strategy (DQN only)
        entropy_coef: Entropy regularization coefficient (REINFORCE only)
        action_type: 'discrete' or 'continuous' (REINFORCE only)
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        plot_dir: Directory for plots

    Example::

        # DQN (value-based, discrete actions)
        smn_rl = SMN_RL(
            env=gym.make('CartPole-v1'),
            algorithm='dqn',
            n=2, m=4,
            n_in=4, n_out=2
        )

        # REINFORCE (policy-based, discrete or continuous)
        smn_rl = SMN_RL(
            env=gym.make('CartPole-v1'),
            algorithm='reinforce',
            n=2, m=4,
            n_in=4, n_out=2,
            entropy_coef=0.01
        )
    """

    def __init__(
        self,
        env: Optional['gym.Env'] = None,
        algorithm: str = 'dqn',
        n: int = 2,
        m: int = 4,
        n_in: int = 4,
        n_out: int = 2,
        gamma: float = 0.99,
        lr: float = 1e-3,
        # DQN-specific parameters
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        train_start: int = 100,
        train_frequency: int = 4,
        sampler_type: str = 'replay',
        # REINFORCE-specific parameters
        entropy_coef: float = 0.0,
        action_type: str = 'discrete',
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        max_grad_norm: float = 0.5,
        # Directories
        checkpoint_dir: str | Path = './runs/simplexnet/checkpoints',
        log_dir: str | Path = './runs/simplexnet/logs',
        plot_dir: str | Path = './runs/simplexnet/plots',
    ):
        # Store either env or mdp
        self._env = env
        self._mdp = None
        self.algorithm = algorithm

        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out

        # Create policy/Q-network
        self.network = SMNmodule(
            n=n, m=m, n_in=n_in, n_out=n_out, activation='relu'
        )

        # Create agent based on algorithm
        if algorithm == 'dqn':
            self.agent = DQN(
                q_network=self.network,
                act_dim=n_out,
                gamma=gamma,
                lr=lr,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                sampler_type=sampler_type,
                buffer_size=buffer_size,
                train_start=train_start,
                train_frequency=train_frequency,
            )
        elif algorithm == 'reinforce':
            # For continuous action type, n_out should be 2 * act_dim (mean + log_std)
            if action_type == 'continuous':
                # Network output is [mean, log_std], so n_out = 2 * act_dim
                self.network = SMNmodule(
                    n=n, m=m, n_in=n_in, n_out=2 * n_out, activation='relu'
                )
            self.agent = REINFORCE(
                policy_network=self.network,
                act_dim=n_out,
                action_type=action_type,
                gamma=gamma,
                lr=lr,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                action_bounds=action_bounds,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'dqn' or 'reinforce'.")

        # Initialize collectors and managers
        self._collector = None  # Created when train() is called
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
            update_target_every: Update target network every N episodes (DQN only)
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
                # Load checkpoint based on algorithm
                if self.algorithm == 'dqn':
                    dqn_checkpoint = {
                        'q_network': checkpoint['state_dict'],
                        'target_network': checkpoint['state_dict'],
                        'optimizer': checkpoint.get('optimizer_state'),
                        'epsilon': self.agent.epsilon,
                    }
                    self.agent.load_checkpoint_from_dict(dqn_checkpoint)
                else:  # reinforce
                    reinforce_checkpoint = {
                        'policy_network': checkpoint['state_dict'],
                        'optimizer': checkpoint.get('optimizer_state'),
                    }
                    self.agent.load_checkpoint_from_dict(reinforce_checkpoint)
                if verbose:
                    print(f"Loaded checkpoint from episode {start_episode}")

        # Create MDP and collector if not already created
        if self._mdp is None:
            if self._env is not None:
                self._mdp = GymMDP(self._env)
            else:
                raise ValueError("No env or mdp provided")

        if self._collector is None:
            self._collector = TrajectoryCollector(self._mdp)

        # Log training start
        self.logger.log_init(
            config={
                'algorithm': self.algorithm,
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

        for episode in range(start_episode, num_episodes):
            # Collect trajectory using new collector
            trajectory = self._collector.collect_episode(
                self.agent, max_steps=max_steps, training=True
            )

            # Train from trajectory
            loss = self.agent.train(trajectory)

            # DQN-specific: Update target network
            if self.algorithm == 'dqn':
                if (episode + 1) % update_target_every == 0:
                    self.agent.update_target_network()
                    if verbose:
                        print(f"Episode {episode + 1}: Updated target network")

                # DQN-specific: Decay epsilon
                self.agent.decay_epsilon()

            # Record history
            episode_reward = sum(trajectory.rewards)
            avg_loss = loss if loss is not None else 0.0

            rewards_history.append(episode_reward)
            losses_history.append(avg_loss)

            # Log epsilon for DQN, entropy_coef for REINFORCE
            if self.algorithm == 'dqn':
                log_extra = {'epsilon': self.agent.epsilon}
            else:
                log_extra = {'entropy_coef': self.agent.entropy_coef}

            self.training_history.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'loss': avg_loss,
                **log_extra,
            })

            # Log epoch
            self.logger.log_epoch(
                episode=episode + 1,
                reward=episode_reward,
                loss=avg_loss,
                **log_extra,
            )

            # Save checkpoint
            if (episode + 1) % checkpoint_every == 0:
                ckpt_path = self.checkpoint_mgr.save_checkpoint(
                    module=self.network,
                    optimizer=self.agent.optimizer,
                    episode=episode + 1,
                    reward=episode_reward,
                    loss=avg_loss,
                    metadata={
                        'rewards_history': rewards_history,
                        'losses_history': losses_history,
                        'algorithm': self.algorithm,
                    }
                )
                self.logger.log_checkpoint_saved(
                    str(ckpt_path), episode + 1, episode_reward
                )
                if verbose:
                    print(f"Episode {episode + 1}: Saved checkpoint")

            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                if self.algorithm == 'dqn':
                    print(
                        f"Episode {episode + 1}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Epsilon: {self.agent.epsilon:.3f}"
                    )
                else:
                    print(
                        f"Episode {episode + 1}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Loss: {avg_loss:.4f}"
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
        # Create MDP if not already created
        if self._mdp is None:
            if self._env is not None:
                self._mdp = GymMDP(self._env)
            else:
                raise ValueError("No env or mdp provided")

        rewards = []

        for episode in range(num_episodes):
            state = self._mdp.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select action (greedy for deterministic mode)
                action = self.agent.select_action(
                    state, training=not deterministic
                )

                # Take step
                next_state, reward, done = self._mdp.step(action)
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
            module=self.network,
            optimizer=self.agent.optimizer,
            episode=episode,
            reward=reward,
            metadata={
                'training_history': self.training_history,
                'algorithm': self.algorithm,
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
            if self.algorithm == 'dqn':
                dqn_checkpoint = {
                    'q_network': checkpoint['state_dict'],
                    'target_network': checkpoint['state_dict'],
                    'optimizer': checkpoint.get('optimizer_state'),
                    'epsilon': self.agent.epsilon,
                }
                self.agent.load_checkpoint_from_dict(dqn_checkpoint)
            else:  # reinforce
                reinforce_checkpoint = {
                    'policy_network': checkpoint['state_dict'],
                    'optimizer': checkpoint.get('optimizer_state'),
                }
                self.agent.load_checkpoint_from_dict(reinforce_checkpoint)
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
