"""SMN_RL: High-level RL wrapper for Simplex Memory Networks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - gymnasium is a runtime dependency
    gym = None  # type: ignore

# Support both package mode (import as module) and script mode (direct run)
try:
    from .SMNmodule import SMNmodule
    from ..rl.algorithms.dqn import DQN
    from ..rl.algorithms.ppo import PPO
    from ..rl.algorithms.reinforce import REINFORCE
    from ..rl.collector import TrajectoryCollector
    from ..rl.mdp import GymMDP
    from ..tools.checkpoint import CheckpointManager
    from ..tools.logger import TrainingLogger
    from ..tools.plot import plot_reward_curve, plot_training_curves
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from .SMNmodule import SMNmodule
    from rl.algorithms.dqn import DQN
    from rl.algorithms.ppo import PPO
    from rl.algorithms.reinforce import REINFORCE
    from rl.collector import TrajectoryCollector
    from rl.mdp import GymMDP
    from tools.checkpoint import CheckpointManager
    from tools.logger import TrainingLogger
    from tools.plot import plot_reward_curve, plot_training_curves

if TYPE_CHECKING:
    import gymnasium as gym


class SMN_RL:
    """High-level RL wrapper for SMN."""

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
        # DQN-specific
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        train_start: int = 100,
        train_frequency: int = 4,
        sampler_type: str = 'replay',
        # REINFORCE-specific
        entropy_coef: float = 0.0,
        action_type: str = 'discrete',
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        max_grad_norm: float = 0.5,
        # PPO-specific
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        rollout_steps: int = 2048,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        # Input normalization
        x_bounds: list[tuple[float, float]] | None = None,
        # Directories
        checkpoint_dir: str | Path = './runs/simplexnet/checkpoints',
        log_dir: str | Path = './runs/simplexnet/logs',
        plot_dir: str | Path = './runs/simplexnet/plots',
    ):
        self._env = env
        self._mdp = None
        self._collector = None
        self.algorithm = algorithm

        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out
        self.rollout_steps = rollout_steps

        self.value_network: SMNmodule | None = None

        if algorithm == 'dqn':
            self.network = SMNmodule(
                n=n, m=m, n_in=n_in, n_out=n_out, activation='tanh', x_bounds=x_bounds
            )
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
            reinforce_n_out = n_out
            if action_type == 'continuous':
                reinforce_n_out = 2 * n_out
            self.network = SMNmodule(
                n=n,
                m=m,
                n_in=n_in,
                n_out=reinforce_n_out,
                activation='relu',
                x_bounds=x_bounds,
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
        elif algorithm == 'ppo':
            if env is None:
                raise ValueError("PPO requires a Gymnasium environment.")
            if gym is None or not isinstance(env.action_space, gym.spaces.Box):
                raise ValueError("PPO currently only supports continuous Box action spaces.")

            action_low = np.asarray(env.action_space.low, dtype=np.float32)
            action_high = np.asarray(env.action_space.high, dtype=np.float32)
            act_dim = int(np.prod(env.action_space.shape))

            self.network = SMNmodule(
                n=n,
                m=m,
                n_in=n_in,
                n_out=2 * act_dim,
                activation='tanh',
                output_activation='identity',
                x_bounds=x_bounds,
            )
            self.value_network = SMNmodule(
                n=n,
                m=m,
                n_in=n_in,
                n_out=1,
                activation='tanh',
                output_activation='identity',
                x_bounds=x_bounds,
            )
            self.agent = PPO(
                actor_network=self.network,
                critic_network=self.value_network,
                act_dim=act_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                clip_eps=clip_eps,
                gae_lambda=gae_lambda,
                entropy_coef=entropy_coef,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
                max_grad_norm=max_grad_norm,
                action_low=action_low,
                action_high=action_high,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'dqn', 'reinforce', or 'ppo'.")

        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
        self.logger = TrainingLogger(log_dir)
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.training_history: list[dict] = []

    def _ensure_mdp(self) -> None:
        if self._mdp is None:
            if self._env is None:
                raise ValueError("No env provided.")
            self._mdp = GymMDP(self._env)
        if self._collector is None:
            self._collector = TrajectoryCollector(self._mdp)

    def _checkpoint_payload(self) -> dict:
        if self.algorithm == 'ppo':
            return {
                'extra_state': {
                    'actor_state_dict': self.network.state_dict(),
                    'critic_state_dict': self.value_network.state_dict() if self.value_network else {},
                    'actor_optimizer_state': self.agent.actor_optimizer.state_dict(),
                    'critic_optimizer_state': self.agent.critic_optimizer.state_dict(),
                    'algorithm': self.algorithm,
                },
                'config_override': {
                    'n': self.n,
                    'm': self.m,
                    'n_in': self.n_in,
                    'n_out': self.n_out,
                    'algorithm': self.algorithm,
                },
                'optimizer': None,
            }

        return {
            'extra_state': {'algorithm': self.algorithm},
            'config_override': {
                'n': self.n,
                'm': self.m,
                'n_in': self.n_in,
                'n_out': self.n_out,
                'algorithm': self.algorithm,
            },
            'optimizer': self.agent.optimizer,
        }

    def _load_checkpoint_dict(self, checkpoint: dict) -> int:
        self.training_history = checkpoint.get('metadata', {}).get('training_history', [])
        episode = int(checkpoint.get('episode', 0))

        if self.algorithm == 'dqn':
            dqn_checkpoint = {
                'q_network': checkpoint['state_dict'],
                'target_network': checkpoint.get('target_network_state_dict', checkpoint['state_dict']),
                'optimizer': checkpoint.get('optimizer_state'),
                'epsilon': checkpoint.get('metadata', {}).get('epsilon', self.agent.epsilon),
            }
            self.agent.load_checkpoint_from_dict(dqn_checkpoint)
        elif self.algorithm == 'reinforce':
            reinforce_checkpoint = {
                'policy_network': checkpoint['state_dict'],
                'optimizer': checkpoint.get('optimizer_state'),
                'action_type': checkpoint.get('metadata', {}).get('action_type', self.agent.action_type),
                'act_dim': checkpoint.get('metadata', {}).get('act_dim', self.agent.act_dim),
                'gamma': checkpoint.get('metadata', {}).get('gamma', self.agent.gamma),
                'entropy_coef': checkpoint.get('metadata', {}).get('entropy_coef', self.agent.entropy_coef),
            }
            self.agent.load_checkpoint_from_dict(reinforce_checkpoint)
        else:
            ppo_checkpoint = {
                'actor_state_dict': checkpoint['actor_state_dict'],
                'critic_state_dict': checkpoint['critic_state_dict'],
                'actor_optimizer_state': checkpoint.get('actor_optimizer_state'),
                'critic_optimizer_state': checkpoint.get('critic_optimizer_state'),
            }
            self.agent.load_checkpoint_from_dict(ppo_checkpoint)

        return episode

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
        """Train the agent."""
        del render

        self._ensure_mdp()
        start_episode = 0

        if not reset:
            checkpoint = self.checkpoint_mgr.load_latest()
            if checkpoint is not None:
                checkpoint_algorithm = checkpoint.get('metadata', {}).get('algorithm', checkpoint.get('algorithm'))
                if checkpoint_algorithm in (None, self.algorithm):
                    start_episode = self._load_checkpoint_dict(checkpoint)
                    self.logger.log_checkpoint_loaded("latest", start_episode)
                    if verbose:
                        print(f"Loaded checkpoint from episode {start_episode}")
                elif verbose:
                    print(
                        f"Latest checkpoint algorithm={checkpoint_algorithm} does not match "
                        f"current algorithm={self.algorithm}; starting fresh."
                    )

        self.logger.log_init(
            config={
                'algorithm': self.algorithm,
                'n': self.n,
                'm': self.m,
                'n_in': self.n_in,
                'n_out': self.n_out,
                'gamma': self.agent.gamma,
                'num_episodes': num_episodes,
            }
        )

        rewards_history = [float(h['reward']) for h in self.training_history]
        losses_history = [float(h.get('loss', 0.0)) for h in self.training_history]

        if self.algorithm == 'ppo':
            completed_episodes = start_episode
            while completed_episodes < num_episodes:
                rollout = self._collector.collect_rollout(
                    self.agent,
                    rollout_steps=self.rollout_steps,
                    max_episode_steps=max_steps,
                    training=True,
                )
                metrics = self.agent.train(rollout)
                if metrics is None:
                    break

                for reward in rollout.episode_returns:
                    if completed_episodes >= num_episodes:
                        break
                    completed_episodes += 1
                    entry = {
                        'episode': completed_episodes,
                        'reward': reward,
                        'loss': metrics['loss'],
                        'actor_loss': metrics['actor_loss'],
                        'critic_loss': metrics['critic_loss'],
                        'entropy': metrics['entropy'],
                        'approx_kl': metrics['approx_kl'],
                        'clip_fraction': metrics['clip_fraction'],
                        'value_mean': metrics['value_mean'],
                        'advantage_mean': metrics['advantage_mean'],
                    }
                    self.training_history.append(entry)
                    rewards_history.append(float(reward))
                    losses_history.append(float(metrics['loss']))
                    self.logger.log_epoch(**entry)

                    if completed_episodes % checkpoint_every == 0:
                        ckpt_args = self._checkpoint_payload()
                        ckpt_path = self.checkpoint_mgr.save_checkpoint(
                            module=self.network,
                            optimizer=ckpt_args['optimizer'],
                            episode=completed_episodes,
                            reward=float(reward),
                            loss=float(metrics['loss']),
                            metadata={
                                'training_history': self.training_history,
                                'algorithm': self.algorithm,
                            },
                            extra_state=ckpt_args['extra_state'],
                            config_override=ckpt_args['config_override'],
                        )
                        self.logger.log_checkpoint_saved(str(ckpt_path), completed_episodes, float(reward))

                    if verbose and completed_episodes % 10 == 0:
                        print(
                            f"Episode {completed_episodes}/{num_episodes} | "
                            f"Reward: {reward:.2f} | "
                            f"Actor: {metrics['actor_loss']:.4f} | "
                            f"Critic: {metrics['critic_loss']:.4f} | "
                            f"KL: {metrics['approx_kl']:.4f}"
                        )

            return rewards_history

        for episode in range(start_episode, num_episodes):
            trajectory = self._collector.collect_episode(
                self.agent, max_steps=max_steps, training=True
            )
            loss = self.agent.train(trajectory)

            if self.algorithm == 'dqn':
                if (episode + 1) % update_target_every == 0:
                    self.agent.update_target_network()
                    if verbose:
                        print(f"Episode {episode + 1}: Updated target network")
                self.agent.decay_epsilon()

            episode_reward = sum(trajectory.rewards)
            avg_loss = float(loss) if loss is not None else 0.0
            rewards_history.append(episode_reward)
            losses_history.append(avg_loss)

            if self.algorithm == 'dqn':
                log_extra = {'epsilon': self.agent.epsilon}
            else:
                log_extra = {'entropy_coef': self.agent.entropy_coef}

            entry = {
                'episode': episode + 1,
                'reward': episode_reward,
                'loss': avg_loss,
                **log_extra,
            }
            self.training_history.append(entry)
            self.logger.log_epoch(**entry)

            if (episode + 1) % checkpoint_every == 0:
                ckpt_args = self._checkpoint_payload()
                extra_state = dict(ckpt_args['extra_state'])
                if self.algorithm == 'dqn':
                    extra_state['target_network_state_dict'] = self.agent.target_network.state_dict()
                ckpt_path = self.checkpoint_mgr.save_checkpoint(
                    module=self.network,
                    optimizer=ckpt_args['optimizer'],
                    episode=episode + 1,
                    reward=episode_reward,
                    loss=avg_loss,
                    metadata={
                        'training_history': self.training_history,
                        'algorithm': self.algorithm,
                        **log_extra,
                    },
                    extra_state=extra_state,
                    config_override=ckpt_args['config_override'],
                )
                self.logger.log_checkpoint_saved(str(ckpt_path), episode + 1, episode_reward)
                if verbose:
                    print(f"Episode {episode + 1}: Saved checkpoint")

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
        """Test the agent."""
        del render
        self._ensure_mdp()

        rewards = []
        for _ in range(num_episodes):
            state = self._mdp.reset()
            episode_reward = 0.0
            for _ in range(max_steps):
                action = self.agent.select_action(state, training=not deterministic)
                next_state, reward, done = self._mdp.step(action)
                state = next_state
                episode_reward += reward
                if done:
                    break
            rewards.append(float(episode_reward))

        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        return float(mean_reward), float(std_reward), rewards

    def plot_results(
        self,
        window: int = 100,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """Plot training results."""
        if not self.training_history:
            print("No training history available. Run train() first.")
            return

        rewards = [h['reward'] for h in self.training_history]
        losses = [h.get('loss', 0.0) for h in self.training_history]

        if save_path is None:
            save_path = self.plot_dir / "training_curves.png"

        plot_training_curves(
            rewards, losses,
            save_path=save_path,
            title="SMN Training Results",
            show=show,
        )

        reward_plot_path = self.plot_dir / "reward_curve.png"
        plot_reward_curve(
            rewards,
            window=window,
            save_path=reward_plot_path,
            title="SMN Reward Curve",
            show=show,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save checkpoint manually."""
        del path
        if self.training_history:
            last = self.training_history[-1]
            episode = last['episode']
            reward = last['reward']
            loss = last.get('loss')
        else:
            episode = 0
            reward = 0.0
            loss = None

        ckpt_args = self._checkpoint_payload()
        ckpt_path = self.checkpoint_mgr.save_checkpoint(
            module=self.network,
            optimizer=ckpt_args['optimizer'],
            episode=episode,
            reward=reward,
            loss=loss,
            metadata={
                'training_history': self.training_history,
                'algorithm': self.algorithm,
            },
            extra_state=ckpt_args['extra_state'],
            config_override=ckpt_args['config_override'],
        )
        print(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load checkpoint manually."""
        checkpoint = self.checkpoint_mgr.load_full_checkpoint(Path(path))
        self._load_checkpoint_dict(checkpoint)
        print(f"Loaded checkpoint from {path}")

    def launch_gui(self) -> None:
        """Launch GUI for interactive training (Phase 2)."""
        print("GUI not yet implemented. Coming in Phase 2.")
