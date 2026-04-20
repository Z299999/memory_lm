#!/usr/bin/env python3
"""DQN experiment for SISO trajectory tracking.

This script trains a DQN agent using SMNModule as the Q-network
to track a sinusoidal trajectory.

Usage:
    cd exp0420_RLforSMN
    PYTHONPATH=. python3 experiments/run_dqn_siso.py

Output:
    results/<date>/training.png       - Training curve
    results/<date>/tracking.png       - Trajectory tracking visualization
    results/<date>/error_dist.png     - Error distribution
    results/<date>/checkpoint.pt      - Agent checkpoint
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.siso_tracker import SISOTrajectoryTracker
from src.agents.dqn_agent import DQNAgent
from src.utils.plot_utils import plot_training_results, plot_trajectory_tracking, plot_error_distribution


def train_dqn(
    num_episodes: int = 500,
    episode_length: int = 200,
    target_freq: float = 0.1,
    target_type: str = "sin",
    use_velocity: bool = False,
    n_actions: int = 7,
    batch_size: int = 64,
    save_results: bool = True,
) -> dict:
    """Train DQN agent on SISO trajectory tracking.

    Args:
        num_episodes: Number of training episodes
        episode_length: Time steps per episode
        target_freq: Target trajectory frequency
        target_type: Target type ('sin' or 'sin_mix')
        use_velocity: Whether to use velocity in state
        n_actions: Number of discrete actions
        batch_size: Training batch size
        save_results: Whether to save plots

    Returns:
        Dictionary with training history and final stats
    """
    # Create results directory
    if save_results:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent.parent / "results" / date_str
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {results_dir}")
    else:
        results_dir = None

    # Initialize environment and agent
    state_dim = 2 if use_velocity else 1
    env = SISOTrajectoryTracker(
        target_freq=target_freq,
        target_type=target_type,
        episode_length=episode_length,
        use_velocity=use_velocity,
    )

    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        x_bounds=[(-10, 10)] if state_dim == 1 else [(-10, 10), (-5, 5)],
        n=2, m=4,
        activation='relu',
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    print(f"\n{'='*60}")
    print(f"DQN Training for SISO Trajectory Tracking")
    print(f"{'='*60}")
    print(f"Environment:")
    print(f"  Target type: {target_type}")
    print(f"  Target freq: {target_freq}")
    print(f"  Episode length: {episode_length}")
    print(f"  State dim: {state_dim}")
    print(f"\nAgent:")
    print(f"  Q-network: {agent.q_network.arch_str}")
    print(f"  Action values: {agent.action_values}")
    print(f"\nTraining:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Training loop
    episode_rewards = []
    episode_losses = []
    best_avg_reward = -float('inf')
    best_episode = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        # Store trajectory for visualization
        if episode == num_episodes - 1:  # Last episode
            targets = []
            positions = []
            errors = []
            actions_list = []

        for t in range(episode_length):
            # Select action
            action_idx = agent.select_action(state, training=True)
            action_value = np.array([agent.get_action_value(action_idx)])

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action_value)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action_idx, reward, next_state, done)

            # Train (pass global step for training frequency control)
            global_step = episode * episode_length + t
            loss = agent.train_step(batch_size=batch_size, step=global_step)
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            state = next_state
            total_reward += reward

            # Store for visualization
            if episode == num_episodes - 1:
                targets.append(info['target'])
                positions.append(info['position'])
                errors.append(info['error'])
                actions_list.append(info['action'])

            if done:
                break

        # Update target network and decay epsilon
        agent.update_target_network()
        agent.decay_epsilon()

        # Record stats
        avg_loss = episode_loss / max(loss_count, 1)
        episode_rewards.append(total_reward)
        episode_losses.append(avg_loss)

        # Track best
        if episode >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_episode = episode

        # Progress logging
        if (episode + 1) % 50 == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards[-min(50, episode+1):])
            print(f"Episode {episode+1:4d}: reward={total_reward:7.2f}, "
                  f"avg(50)={avg_reward:7.2f}, loss={avg_loss:.4f}, "
                  f"eps={agent.epsilon:.3f}")

    # Final stats
    final_100_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    best_100_avg = max(np.mean(episode_rewards[i:i+100]) for i in range(len(episode_rewards) - 99)) if len(episode_rewards) >= 100 else np.mean(episode_rewards)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final 100-episode avg reward: {final_100_avg:.2f}")
    print(f"Best 100-episode avg reward:  {best_100_avg:.2f} (at episode {best_episode+1})")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"{'='*60}\n")

    # Save results
    if save_results and results_dir:
        print("Saving results...")

        # Training curve
        plot_training_results(
            episode_rewards, episode_losses,
            save_path=results_dir / "training.png",
            title=f"DQN {target_type} (ε={agent.epsilon:.2f})"
        )

        # Trajectory tracking (last episode)
        plot_trajectory_tracking(
            targets, positions, errors, actions_list,
            save_path=results_dir / "tracking.png",
            title=f"Last Episode (reward={total_reward:.1f})"
        )

        # Error distribution
        plot_error_distribution(
            np.array(errors),
            save_path=results_dir / "error_dist.png",
            title=f"Last Episode (MAE={np.mean(np.abs(errors)):.3f})"
        )

        # Save checkpoint
        checkpoint_path = results_dir / "checkpoint.pt"
        agent.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Save stats
        stats = {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'final_100_avg': final_100_avg,
            'best_100_avg': best_100_avg,
            'best_episode': best_episode,
            'final_epsilon': agent.epsilon,
            'config': {
                'num_episodes': num_episodes,
                'episode_length': episode_length,
                'target_type': target_type,
                'target_freq': target_freq,
                'n_actions': n_actions,
                'use_velocity': use_velocity,
            }
        }
        np.save(results_dir / "stats.npy", stats)
        print(f"Stats saved: {results_dir / 'stats.npy'}")

        print(f"\nAll results saved to: {results_dir}")

    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_100_avg': final_100_avg,
        'best_100_avg': best_100_avg,
        'best_episode': best_episode,
    }


if __name__ == "__main__":
    # Run default experiment with tuned hyperparameters
    results = train_dqn(
        num_episodes=300,           # Fewer episodes for testing
        episode_length=200,
        target_type="sin",
        target_freq=0.05,           # Lower frequency = easier to track
        use_velocity=True,          # Use velocity in state for better control
        n_actions=7,
        save_results=True,
    )
