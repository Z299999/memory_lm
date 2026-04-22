#!/usr/bin/env python3
"""DQN experiment for SISO trajectory tracking.

Usage:
    cd exp0420_RLforSMN
    python3 run.py

Output:
    results/<date>/training.png       - Training curve
    results/<date>/tracking.png       - Trajectory tracking visualization
    results/<date>/error_dist.png     - Error distribution
    results/<date>/checkpoint.pt      - Agent checkpoint
"""

from datetime import datetime
from pathlib import Path
import numpy as np

from src import SISOTrajectoryTracker, DQNAgent, plot_training_results, plot_trajectory_tracking, plot_error_distribution


def train_dqn(
    num_episodes: int = 300,
    episode_length: int = 200,
    target_freq: float = 0.05,
    target_type: str = "sin",
    use_velocity: bool = True,
    n_actions: int = 7,
    batch_size: int = 64,
    save_results: bool = True,
) -> dict:
    """Train DQN agent on SISO trajectory tracking."""

    # Create results directory
    if save_results:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent / "results" / date_str
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
    print(f"  Target type: {target_type}, freq: {target_freq}")
    print(f"  Episode length: {episode_length}, State dim: {state_dim}")
    print(f"\nAgent:")
    print(f"  Q-network: {agent.q_network.arch_str}")
    print(f"  Actions: {n_actions} discrete")
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

        # Store trajectory for visualization (last episode)
        if episode == num_episodes - 1:
            targets, positions, errors, actions_list = [], [], [], []

        for t in range(episode_length):
            # Select action
            action_idx = agent.select_action(state, training=True)
            action_value = np.array([agent.get_action_value(action_idx)])

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action_value)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action_idx, reward, next_state, done)

            # Train
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
    print(f"Final 100-ep avg reward: {final_100_avg:.2f}")
    print(f"Best 100-ep avg reward:  {best_100_avg:.2f} (episode {best_episode+1})")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"{'='*60}\n")

    # Save results
    if save_results and results_dir:
        print("Saving results...")

        plot_training_results(
            episode_rewards, episode_losses,
            save_path=results_dir / "training.png",
            title=f"DQN {target_type} (ε={agent.epsilon:.2f})"
        )

        plot_trajectory_tracking(
            targets, positions, errors, actions_list,
            save_path=results_dir / "tracking.png",
            title=f"Last Episode (reward={total_reward:.1f})"
        )

        plot_error_distribution(
            np.array(errors),
            save_path=results_dir / "error_dist.png",
            title=f"Last Episode (MAE={np.mean(np.abs(errors)):.3f})"
        )

        agent.save_checkpoint(results_dir / "checkpoint.pt")
        print(f"Checkpoint saved: {results_dir / 'checkpoint.pt'}")

        print(f"\nAll results saved to: {results_dir}")

    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_100_avg': final_100_avg,
        'best_100_avg': best_100_avg,
    }


if __name__ == "__main__":
    results = train_dqn()
