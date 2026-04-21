"""Visualization functions for SMN training.

This module provides functions for creating training plots including:
- Training curves (reward, loss)
- Trajectory tracking visualization
- Error distribution histograms

Usage::

    from tools.plot import plot_training_curves, plot_trajectory_tracking

    plot_training_curves(rewards, losses, save_path='training.png')
    plot_trajectory_tracking(targets, positions, errors, save_path='tracking.png')
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_curves(
    rewards: list[float],
    losses: list[float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Training Curves",
    show: bool = False,
) -> None:
    """Plot training curves (reward and loss over episodes).

    Args:
        rewards: List of episode rewards
        losses: List of training losses (optional)
        save_path: Path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    episodes = range(1, len(rewards) + 1)

    # Plot rewards on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(episodes, rewards, color=color, label='Reward', linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot losses on right y-axis (if provided)
    if losses is not None and len(losses) > 0:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Loss', color=color)
        # Shift losses to positive range for log scale if needed
        losses_arr = np.array(losses)
        if np.any(losses_arr <= 0):
            # Use linear scale for non-positive losses
            ax2.plot(episodes, losses, color=color, label='Loss', linewidth=1, alpha=0.7)
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        else:
            ax2.plot(episodes, losses, color=color, label='Loss', linewidth=1, alpha=0.7)
            ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_reward_curve(
    rewards: list[float],
    window: int = 100,
    save_path: str | Path | None = None,
    title: str = "Reward Curve",
    show: bool = False,
) -> None:
    """Plot reward curve with moving average.

    Args:
        rewards: List of episode rewards
        window: Window size for moving average
        save_path: Path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    episodes = range(1, len(rewards) + 1)

    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.3, label='Raw', linewidth=1)

    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        avg_episodes = range(window, len(rewards) + 1)
        ax.plot(avg_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-ep avg')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_tracking(
    targets: list[float],
    positions: list[float],
    errors: list[float],
    actions: list[float] | None = None,
    save_path: str | Path | None = None,
    title: str = "Trajectory Tracking",
    show: bool = False,
) -> None:
    """Plot trajectory tracking visualization.

    Args:
        targets: Target trajectory values
        positions: Actual positions
        errors: Tracking errors
        actions: Actions taken (optional)
        save_path: Path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    steps = range(len(targets))

    # Top: Target vs Position
    axes[0].plot(steps, targets, 'g-', label='Target', linewidth=1)
    axes[0].plot(steps, positions, 'b-', label='Position', linewidth=1, alpha=0.7)
    axes[0].set_ylabel('Value')
    axes[0].set_title('Target vs Position')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle: Error
    axes[1].plot(steps, errors, 'r-', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Error')
    axes[1].set_title('Tracking Error')
    axes[1].grid(True, alpha=0.3)

    # Bottom: Actions
    if actions is not None:
        axes[2].plot(steps, actions, 'orange', linewidth=1, label='Action')
        axes[2].set_ylabel('Action')
        axes[2].set_xlabel('Step')
        axes[2].set_title('Actions')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No action data', ha='center', va='center')
        axes[2].set_xlabel('Step')

    plt.suptitle(title)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_error_distribution(
    errors: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Error Distribution",
    show: bool = False,
) -> None:
    """Plot error distribution histogram.

    Args:
        errors: Array of errors
        save_path: Path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Zero error')
    ax.axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(errors):.4f}')

    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# Quick test
if __name__ == "__main__":
    import tempfile

    print("Testing plot functions...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Training curves
        print("\nTest 1: Training curves")
        rewards = [np.random.randn() * 10 + i * 0.5 for i in range(100)]
        losses = [np.exp(-i * 0.1) * np.random.rand() for i in range(100)]
        plot_training_curves(
            rewards, losses,
            save_path=Path(tmpdir) / "training.png",
            title="Test Training"
        )
        assert (Path(tmpdir) / "training.png").exists()
        print("  PASSED")

        # Test 2: Reward curve
        print("Test 2: Reward curve")
        plot_reward_curve(
            rewards, window=10,
            save_path=Path(tmpdir) / "reward.png",
            title="Test Reward"
        )
        assert (Path(tmpdir) / "reward.png").exists()
        print("  PASSED")

        # Test 3: Trajectory tracking
        print("Test 3: Trajectory tracking")
        steps = list(range(200))
        targets = [np.sin(i * 0.1) for i in steps]
        positions = [np.sin(i * 0.1) + np.random.randn() * 0.1 for i in steps]
        errors = [t - p for t, p in zip(targets, positions)]
        actions = [np.random.randn() for _ in steps]
        plot_trajectory_tracking(
            targets, positions, errors, actions,
            save_path=Path(tmpdir) / "tracking.png",
            title="Test Tracking"
        )
        assert (Path(tmpdir) / "tracking.png").exists()
        print("  PASSED")

        # Test 4: Error distribution
        print("Test 4: Error distribution")
        errors_np = np.random.randn(1000) * 0.5
        plot_error_distribution(
            errors_np,
            save_path=Path(tmpdir) / "error_dist.png",
            title="Test Error Distribution"
        )
        assert (Path(tmpdir) / "error_dist.png").exists()
        print("  PASSED")

    print("\n" + "="*50)
    print("All plot tests PASSED!")
    print("="*50)
