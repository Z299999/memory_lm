#!/usr/bin/env python3
"""Train SMN-based RL agent on Gymnasium environments.

This script demonstrates the full training pipeline using SMN_RL wrapper.

Usage::

    python3 examples/train_rl.py --env CartPole-v1 --episodes 1000
    python3 examples/train_rl.py --env Acrobot-v1 --episodes 1500 --n 6 --m 5
    python3 examples/train_rl.py --env CartPole-v1 --algorithm reinforce --entropy-coef 0.01
    python3 examples/train_rl.py --algorithm ppo --env Pendulum-v1

Options:
    --env: Gymnasium environment name (default: CartPole-v1, PPO uses Pendulum-v1)
    --episodes: Number of training episodes (default: 1000)
    --n: Simplex dimension (default: 6)
    --m: Lattice parameter (default: 5)
    --algorithm: RL algorithm (default: dqn)
    --sampler-type: Sampling strategy (default: replay, DQN only)
    --gamma: Discount factor (default: 0.99)
    --lr: Learning rate (default: 1e-3)
    --epsilon: Initial exploration rate (default: 1.0, DQN only)
    --epsilon-decay: Epsilon decay per episode (default: 0.995, DQN only)
    --epsilon-min: Minimum exploration rate (default: 0.01, DQN only)
    --entropy-coef: Entropy regularization (default: 0.0, REINFORCE only)
    --action-type: Action type (default: discrete, REINFORCE only)
    --checkpoint-dir: Checkpoint directory (default: ./checkpoints)
    --log-dir: Log directory (default: ./logs)
    --plot-dir: Plot directory (default: ./plots)
    --no-verbose: Disable verbose output
    --resume: Resume from existing checkpoint
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gymnasium as gym

from simplexnet.core.SMN_RL import SMN_RL


def get_env_obs_bounds(env: gym.Env) -> list[tuple[float, float]]:
    """Infer reasonable input bounds from environment observation space.

    For Box spaces with finite low/high, use those directly.
    For common environments, use known physical limits.

    Args:
        env: Gymnasium environment

    Returns:
        List of (min, max) tuples per observation dimension
    """
    obs_space = env.observation_space

    # Box space: use the low/high bounds directly
    if isinstance(obs_space, gym.spaces.Box):
        low = obs_space.low
        high = obs_space.high

        # Handle infinite bounds - replace with reasonable defaults
        bounds = []
        for i in range(len(low)):
            l, h = float(low[i]), float(high[i])
            if not np.isfinite(l) or not np.isfinite(h):
                # Use environment-specific defaults for unbounded observations
                if hasattr(env, 'spec') and env.spec:
                    env_name = env.spec.id
                    # CartPole-style: position/velocity typically ±2.4/±15
                    if 'CartPole' in env_name:
                        l, h = -20.0, 20.0
                    # MountainCar: position [-1.2, 0.6], velocity [-0.07, 0.07]
                    elif 'MountainCar' in env_name:
                        l, h = -1.5, 1.5
                    # Generic fallback
                    else:
                        l, h = -10.0, 10.0
                else:
                    l, h = -10.0, 10.0
            bounds.append((l, h))
        return bounds

    # Fallback: use [-1, 1] for unknown spaces
    return [(-1.0, 1.0)] * obs_space.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description='Train SMN RL agent')

    # Environment
    parser.add_argument('--env', type=str, default=None,
                        help='Gymnasium environment name')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')

    # SMN architecture - larger defaults for better capacity
    parser.add_argument('--n', type=int, default=6,
                        help='Simplex dimension (order of simplices)')
    parser.add_argument('--m', type=int, default=5,
                        help='Lattice parameter (size in each dimension)')

    # Algorithm selection
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'reinforce', 'ppo'],
                        help='RL algorithm: dqn, reinforce, or ppo')

    # Sampler type (DQN only)
    parser.add_argument('--sampler-type', type=str, default='replay',
                        choices=['replay', 'online', 'mixed'],
                        help='Sampling strategy: replay (standard), online (latest policy), mixed (DQN only)')

    # RL hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    # DQN-specific parameters
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Initial exploration rate (DQN only)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay per episode (DQN only)')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                        help='Minimum exploration rate (DQN only)')

    # REINFORCE-specific parameters
    parser.add_argument('--entropy-coef', type=float, default=0.0,
                        help='Entropy regularization coefficient (REINFORCE/PPO)')
    parser.add_argument('--action-type', type=str, default='discrete',
                        choices=['discrete', 'continuous'],
                        help='Action type for REINFORCE: discrete or continuous')
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                        help='Actor learning rate (PPO only)')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='Critic learning rate (PPO only)')
    parser.add_argument('--clip-eps', type=float, default=0.2,
                        help='PPO clipping epsilon')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda (PPO only)')
    parser.add_argument('--update-epochs', type=int, default=10,
                        help='PPO update epochs')
    parser.add_argument('--minibatch-size', type=int, default=64,
                        help='PPO minibatch size')
    parser.add_argument('--rollout-steps', type=int, default=2048,
                        help='PPO rollout steps')
    parser.add_argument('--log-std-min', type=float, default=-5.0,
                        help='Minimum log std clamp (PPO only)')
    parser.add_argument('--log-std-max', type=float, default=2.0,
                        help='Maximum log std clamp (PPO only)')

    # Directories (relative to script location)
    script_dir = Path(__file__).parent
    default_checkpoint_dir = script_dir / '../runs/simplexnet/checkpoints'
    default_log_dir = script_dir / '../runs/simplexnet/logs'
    default_plot_dir = script_dir / '../runs/simplexnet/plots'

    parser.add_argument('--checkpoint-dir', type=str, default=str(default_checkpoint_dir),
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default=str(default_log_dir),
                        help='Log directory')
    parser.add_argument('--plot-dir', type=str, default=str(default_plot_dir),
                        help='Plot directory')

    # Options
    parser.add_argument('--no-verbose', action='store_true',
                        help='Disable verbose output')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoint')
    parser.add_argument('--test-only', action='store_true',
                        help='Run test without training')
    parser.add_argument('--test-episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during test')

    return parser.parse_args()


def main():
    args = parse_args()
    env_name = args.env or ('Pendulum-v1' if args.algorithm == 'ppo' else 'CartPole-v1')

    # Create environment
    print(f"Creating environment: {env_name}")
    env = gym.make(env_name)

    # Infer observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        act_dim = int(np.prod(env.action_space.shape))
    else:
        act_dim = env.action_space.n

    # Infer observation bounds for input normalization
    obs_bounds = get_env_obs_bounds(env)
    print(f"Observation dimension: {obs_dim}")
    print(f"Observation bounds: {obs_bounds}")
    print(f"Action dimension: {act_dim}")

    # Create SMN_RL wrapper
    smn_rl = SMN_RL(
        env=env,
        algorithm=args.algorithm,
        n=args.n,
        m=args.m,
        n_in=obs_dim,
        n_out=act_dim,
        gamma=args.gamma,
        lr=args.lr,
        # DQN-specific
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        sampler_type=args.sampler_type,
        # REINFORCE-specific
        entropy_coef=args.entropy_coef,
        action_type=args.action_type,
        # PPO-specific
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        rollout_steps=args.rollout_steps,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        # Input normalization
        x_bounds=obs_bounds,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
    )

    print(f"Algorithm: {args.algorithm}")
    print(f"SMN architecture: n={args.n}, m={args.m}")
    if args.algorithm == 'dqn':
        print(f"Sampler type: {args.sampler_type}")
    elif args.algorithm == 'ppo':
        print(f"Actor lr: {args.actor_lr}, Critic lr: {args.critic_lr}, Rollout steps: {args.rollout_steps}")
    else:
        print(f"Action type: {args.action_type}, Entropy coef: {args.entropy_coef}")
    print(f"Network: {smn_rl.network.arch_str}")
    if args.algorithm == 'ppo' and smn_rl.value_network is not None:
        print(f"Value network: {smn_rl.value_network.arch_str}")

    if args.test_only:
        # Test only (no training)
        print("\n=== Testing ===")
        mean, std, rewards = smn_rl.test(
            num_episodes=args.test_episodes,
            render=args.render,
            deterministic=True
        )
        print(f"Test reward: {mean:.2f} +/- {std:.2f}")
        print(f"Episode rewards: {rewards}")
    else:
        # Train
        print(f"\n=== Training for {args.episodes} episodes ===")
        rewards = smn_rl.train(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=not args.no_verbose,
            reset=not args.resume
        )

        print(f"\nTraining completed!")
        print(f"Final reward: {rewards[-1]:.2f}")
        print(f"Best reward: {max(rewards):.2f}")
        print(f"Average (last 100): {sum(rewards[-100:])/min(100, len(rewards)):.2f}")

        # Test
        print(f"\n=== Testing ===")
        mean, std, test_rewards = smn_rl.test(
            num_episodes=args.test_episodes,
            render=args.render,
            deterministic=True
        )
        print(f"Test reward: {mean:.2f} +/- {std:.2f}")

        # Plot results
        print(f"\n=== Saving plots ===")
        smn_rl.plot_results()
        print("Done!")


if __name__ == '__main__':
    main()
