#!/usr/bin/env python3
"""Train SMN-based RL agent on Gymnasium environments.

This script demonstrates the full training pipeline using SMN_RL wrapper.

Usage::

    python3 examples/train_rl.py --env CartPole-v1 --episodes 1000
    python3 examples/train_rl.py --env Acrobot-v1 --episodes 1500 --n 6 --m 5
    python3 examples/train_rl.py --env CartPole-v1 --algorithm reinforce --entropy-coef 0.01

Options:
    --env: Gymnasium environment name (default: CartPole-v1)
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gymnasium as gym

from simplexnet.core.SMN_RL import SMN_RL


def parse_args():
    parser = argparse.ArgumentParser(description='Train SMN RL agent')

    # Environment
    parser.add_argument('--env', type=str, default='CartPole-v1',
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
                        choices=['dqn', 'reinforce'],
                        help='RL algorithm: dqn (value-based) or reinforce (policy gradient)')

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
                        help='Entropy regularization coefficient (REINFORCE only)')
    parser.add_argument('--action-type', type=str, default='discrete',
                        choices=['discrete', 'continuous'],
                        help='Action type for REINFORCE: discrete or continuous')

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

    # Create environment
    print(f"Creating environment: {args.env}")
    env = gym.make(args.env)

    # Infer observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f"Observation dimension: {obs_dim}")
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
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
    )

    print(f"Algorithm: {args.algorithm}")
    print(f"SMN architecture: n={args.n}, m={args.m}")
    if args.algorithm == 'dqn':
        print(f"Sampler type: {args.sampler_type}")
    else:
        print(f"Action type: {args.action_type}, Entropy coef: {args.entropy_coef}")
    print(f"Network: {smn_rl.network.arch_str}")

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
