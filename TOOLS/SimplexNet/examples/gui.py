#!/usr/bin/env python3
"""Launch SMN Training GUI.

This script launches the PySide6-based interactive training GUI.

Usage::

    python3 examples/gui.py
    python3 examples/gui.py --env CartPole-v1 --algorithm dqn
    python3 examples/gui.py --env Pendulum-v1 --algorithm ppo

Options::

    --env: Gymnasium environment (default: CartPole-v1, or Pendulum-v1 for PPO)
    --algorithm: dqn, reinforce, or ppo (default: dqn)
    --n: Simplex dimension (default: 2)
    --m: Lattice resolution (default: 4)
    --lr: Learning rate (default: 1e-3)
    --gamma: Discount factor (default: 0.99)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gymnasium as gym
from simplexnet.core.SMN_RL import SMN_RL


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SMN Training GUI')
    parser.add_argument('--env', type=str, default=None,
                        help='Gymnasium environment')
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'reinforce', 'ppo'],
                        help='RL algorithm')
    parser.add_argument('--n', type=int, default=2,
                        help='Simplex dimension')
    parser.add_argument('--m', type=int, default=4,
                        help='Lattice resolution')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')

    args = parser.parse_args()

    # Default environment based on algorithm
    env_name = args.env
    if env_name is None:
        if args.algorithm == 'ppo':
            env_name = 'Pendulum-v1'
        else:
            env_name = 'CartPole-v1'

    print(f"Creating environment: {env_name}")
    env = gym.make(env_name)

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        act_dim = int(np.prod(env.action_space.shape))
    else:
        act_dim = env.action_space.n

    print(f"Observation space: {obs_dim}")
    print(f"Action space: {act_dim}")
    print(f"Algorithm: {args.algorithm}")

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
    )

    print(f"\nNetwork: {smn_rl.network.arch_str}")
    print(f"Parameters: {smn_rl.network.param_count:,}")
    print("\nLaunching GUI...")

    # Launch GUI
    smn_rl.launch_gui()


if __name__ == '__main__':
    import numpy as np
    main()
