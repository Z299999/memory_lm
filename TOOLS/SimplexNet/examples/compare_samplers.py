#!/usr/bin/env python3
"""Compare different sampler types and SMN parameters.

This script runs training with different configurations and plots
comparison curves.

Usage::

    python3 examples/compare_samplers.py

This will:
1. Train with replay sampler (n=6, m=5)
2. Train with online sampler (n=6, m=5)
3. Train with replay sampler (n=2, m=4) - smaller network
4. Plot comparison curves
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import gymnasium as gym
from simplexnet.core.SMN_RL import SMN_RL


def run_training(env_name, episodes, sampler_type, n, m, run_name):
    """Run training and return rewards history."""
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Sampler: {sampler_type}, SMN: n={n}, m={m}")
    print(f"{'='*60}")

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Create unique output dirs for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    base_runs = script_dir / '../runs/simplexnet'

    smn_rl = SMN_RL(
        env=env,
        n=n,
        m=m,
        n_in=obs_dim,
        n_out=act_dim,
        gamma=0.99,
        lr=1e-3,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        sampler_type=sampler_type,
        checkpoint_dir=base_runs / f'compare_{timestamp}_{run_name}/checkpoints',
        log_dir=base_runs / f'compare_{timestamp}_{run_name}/logs',
        plot_dir=base_runs / f'compare_{timestamp}_{run_name}/plots',
    )

    print(f"Q-network: {smn_rl.q_network.arch_str}")

    # Train
    rewards = smn_rl.train(
        num_episodes=episodes,
        max_steps=500,
        verbose=False,
        update_target_every=100,
        checkpoint_every=100,
    )

    # Test
    mean, std, _ = smn_rl.test(num_episodes=5, deterministic=True)
    print(f"Test reward: {mean:.2f} +/- {std:.2f}")

    return rewards, mean, std


def plot_comparison(results, save_path=None, show=False):
    """Plot comparison curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'replay_n6m5': 'blue', 'online_n6m5': 'red', 'replay_n2m4': 'green'}
    labels = {
        'replay_n6m5': 'Replay (n=6, m=5)',
        'online_n6m5': 'Online (n=6, m=5)',
        'replay_n2m4': 'Replay (n=2, m=4)',
    }

    # Plot 1: Raw rewards
    ax = axes[0, 0]
    for name, (rewards, _, _) in results.items():
        ax.plot(rewards, alpha=0.3, color=colors.get(name, 'gray'))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Raw Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Moving average (window=50)
    ax = axes[0, 1]
    window = 50
    for name, (rewards, _, _) in results.items():
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(ma, label=labels.get(name, name), color=colors.get(name, 'gray'))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (50-ep)')
    ax.set_title('Training Rewards - Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Test comparison
    ax = axes[1, 0]
    names = list(results.keys())
    test_means = [results[n][1] for n in names]
    test_stds = [results[n][2] for n in names]
    ax.bar(names, test_means, yerr=test_stds, color=[colors.get(n, 'gray') for n in names])
    ax.set_ylabel('Test Reward')
    ax.set_title('Final Test Performance')
    ax.set_xticklabels([labels.get(n, n) for n in names], rotation=15)

    # Plot 4: Learning speed (episodes to reach reward threshold)
    ax = axes[1, 1]
    threshold = 50
    for name, (rewards, _, _) in results.items():
        # Find first episode where 50-ep MA exceeds threshold
        window = 50
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        exceeding = np.where(ma > threshold)[0]
        if len(exceeding) > 0:
            ax.scatter([name], [exceeding[0] + window], color=colors.get(name, 'gray'), s=100)
        else:
            ax.scatter([name], [len(rewards)], color=colors.get(name, 'gray'), marker='x', s=100,
                      label='Not reached')
    ax.set_ylabel('Episodes to Reach 50+ Reward')
    ax.set_title(f'Learning Speed (threshold={threshold})')
    ax.set_xticklabels([labels.get(n, n) for n in names], rotation=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare sampler types')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Environment')
    parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show plots')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / '../runs/simplexnet/compare'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run experiments
    results = {}

    # Experiment 1: Standard DQN with larger network
    rewards1, mean1, std1 = run_training(
        args.env, args.episodes, 'replay', n=6, m=5,
        run_name='replay_n6m5'
    )
    results['replay_n6m5'] = (rewards1, mean1, std1)

    # Experiment 2: Online DQN with larger network
    rewards2, mean2, std2 = run_training(
        args.env, args.episodes, 'online', n=6, m=5,
        run_name='online_n6m5'
    )
    results['online_n6m5'] = (rewards2, mean2, std2)

    # Experiment 3: Standard DQN with smaller network (for comparison)
    rewards3, mean3, std3 = run_training(
        args.env, args.episodes, 'replay', n=2, m=4,
        run_name='replay_n2m4'
    )
    results['replay_n2m4'] = (rewards3, mean3, std3)

    # Plot comparison
    save_path = output_dir / 'sampler_comparison.png'
    plot_comparison(results, save_path=save_path, show=args.show)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, (_, mean, std) in results.items():
        print(f"{name}: Test reward = {mean:.2f} +/- {std:.2f}")


if __name__ == '__main__':
    main()
