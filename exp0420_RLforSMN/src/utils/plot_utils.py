"""Plotting utilities for RL experiments."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_results(
    episode_rewards: list[float],
    episode_losses: list[float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Training Results",
) -> None:
    """Plot training results: rewards and optional losses.

    Args:
        episode_rewards: List of episode rewards
        episode_losses: List of training losses (optional)
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2 if episode_losses else 1, figsize=(12, 4))
    if episode_losses is None:
        axes = [axes]

    # Episode rewards
    ax = axes[0]
    ax.plot(episode_rewards, linewidth=1, alpha=0.7, label='Reward')

    # Moving average
    window = min(50, len(episode_rewards) // 10)
    if window > 1:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(ma, linewidth=2, label=f'MA({window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title(f'Episode Rewards\n{title}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training losses
    if episode_losses and len(episode_losses) > 0:
        ax = axes[1]
        ax.plot(episode_losses, linewidth=1, alpha=0.7, color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Losses')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_trajectory_tracking(
    targets: list[float],
    positions: list[float],
    errors: list[float],
    actions: list[float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Trajectory Tracking",
) -> None:
    """Plot trajectory tracking results.

    Args:
        targets: Target trajectory values
        positions: Actual positions
        errors: Tracking errors (target - position)
        actions: Control actions (optional)
        save_path: Path to save figure
        title: Plot title
    """
    time_steps = list(range(len(targets)))

    # Create figure
    has_actions = actions is not None and len(actions) > 0
    if has_actions:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Target vs Position
    ax = axes[0]
    ax.plot(time_steps, targets, 'b-', linewidth=1.5, label='Target', alpha=0.8)
    ax.plot(time_steps, positions, 'r-', linewidth=1.5, label='Position', alpha=0.8)
    ax.set_ylabel('Position')
    ax.set_title(f'Trajectory Tracking\n{title}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Middle: Error
    ax = axes[1]
    ax.plot(time_steps, errors, 'g-', linewidth=1, label='Error')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom: Actions
    if actions is not None and len(actions) > 0:
        ax = axes[2]
        ax.plot(time_steps, actions, 'purple', linewidth=1, label='Action (control force)')
        ax.set_ylabel('Action')
        ax.set_xlabel('Time Step')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_error_distribution(
    errors: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Error Distribution",
) -> None:
    """Plot error distribution histogram.

    Args:
        errors: Array of tracking errors
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram
    ax = axes[0]
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution\n{title}')
    ax.grid(True, alpha=0.3)

    # Cumulative
    ax = axes[1]
    sorted_errors = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, linewidth=2)
    ax.set_xlabel('|Error|')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Cumulative Error Distribution')
    ax.grid(True, alpha=0.3)

    # Mark percentiles
    for p, label in [(50, 'p50'), (90, 'p90'), (95, 'p95'), (99, 'p99')]:
        idx = int(len(sorted_errors) * p / 100)
        ax.axvline(sorted_errors[idx], linestyle='--', alpha=0.5, label=label)
    ax.legend()

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)


# Test
if __name__ == "__main__":
    import numpy as np

    # Test data
    episode_rewards = [np.random.randn() * 10 + i * 0.5 for i in range(100)]
    episode_losses = [np.exp(-i * 0.05) * 10 + np.random.randn() for i in range(100)]

    targets = np.sin(np.linspace(0, 20, 200))
    positions = np.sin(np.linspace(0, 20, 200)) + np.random.randn(200) * 0.1
    errors = targets - positions
    actions = np.random.randn(200) * 0.5

    plot_training_results(episode_rewards, episode_losses, save_path="results/test_training.png")
    plot_trajectory_tracking(targets, positions, errors, actions, save_path="results/test_tracking.png")
    plot_error_distribution(errors, save_path="results/test_error.png")

    print("All plot tests passed!")
