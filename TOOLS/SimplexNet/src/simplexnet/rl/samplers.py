"""Sampler abstractions for DQN training.

This module provides different sampling strategies for DQN:
- ReplayBufferSampler: Standard DQN, random sampling from history
- OnlineSampler: Online learning, only use latest policy data

Usage::

    # Standard DQN (replay buffer)
    dqn = DQN(..., sampler_type='replay')

    # Online DQN (latest policy only)
    dqn = DQN(..., sampler_type='online')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple, Optional
import random

import numpy as np


# Type alias for transition
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class Sampler(ABC):
    """Abstract base class for sampling strategies.

    Subclasses implement different ways of collecting and sampling
    experience data for DQN training.
    """

    @abstractmethod
    def store(self, transition: Transition) -> None:
        """Store a single transition."""

    @abstractmethod
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions for training."""

    @abstractmethod
    def should_train(self, step: int) -> bool:
        """Determine if it's time to train."""

    @abstractmethod
    def can_sample(self, batch_size: int) -> bool:
        """Check if we have enough data to sample."""


class ReplayBufferSampler(Sampler):
    """Standard DQN sampler with experience replay buffer.

    Stores all historical experiences and samples randomly.
    This breaks temporal correlations and provides stable training.

    Args:
        capacity: Maximum size of replay buffer

    Usage::

        sampler = ReplayBufferSampler(capacity=10000)
        sampler.store((state, action, reward, next_state, done))
        batch = sampler.sample(batch_size=64)
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def store(self, transition: Transition) -> None:
        """Store a transition in replay buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample transitions from buffer."""
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def should_train(self, step: int, train_frequency: int = 4) -> bool:
        """Train every train_frequency steps."""
        return step % train_frequency == 0

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineSampler(Sampler):
    """Online sampler that only uses latest policy data.

    Only keeps recent experiences collected with the current policy.
    After training, old data is discarded to force re-sampling with
    updated policy.

    This is what the user requested: training only based on data
    from the most recently updated policy.

    Args:
        batch_size: Batch size for training
        min_samples: Minimum samples before training starts
        buffer_size: Maximum recent samples to keep

    Usage::

        sampler = OnlineSampler(batch_size=64, min_samples=100)
        sampler.store((state, action, reward, next_state, done))
        if sampler.should_train():
            batch = sampler.sample(batch_size=64)
            # After training, data is cleared automatically
    """

    def __init__(self, batch_size: int = 64, min_samples: int = 100,
                 buffer_size: int = 500):
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.buffer: List[Transition] = []
        self.max_size = buffer_size
        self._total_collected = 0

    def store(self, transition: Transition) -> None:
        """Store a transition, keeping only recent data."""
        self.buffer.append(transition)
        self._total_collected += 1

        # Drop oldest if exceeding buffer size
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample from recent experiences."""
        available = len(self.buffer)
        if available <= batch_size:
            return self.buffer.copy()
        return random.sample(self.buffer, batch_size)

    def should_train(self, step: int = 0) -> bool:
        """Train when we have enough recent samples."""
        return len(self.buffer) >= self.min_samples

    def can_sample(self, batch_size: int) -> bool:
        """Check if we have enough samples."""
        return len(self.buffer) >= batch_size

    def clear_old(self) -> None:
        """Clear old data after training.

        This forces the agent to collect new samples with
        the updated policy before training again.
        """
        # Keep only the most recent samples (10% of batch)
        keep = max(1, self.batch_size // 10)
        if len(self.buffer) > keep:
            self.buffer = self.buffer[-keep:]

    def __len__(self) -> int:
        return len(self.buffer)


class MixedSampler(Sampler):
    """Mixed sampler combining replay buffer and online sampling.

    Uses both historical data and recent data for training.
    Good balance between stability (replay) and freshness (online).

    Args:
        replay_capacity: Size of replay buffer for historical data
        online_buffer: Size of online buffer for recent data
        online_ratio: Fraction of batch from online buffer (0-1)

    Usage::

        sampler = MixedSampler(
            replay_capacity=5000,
            online_buffer=200,
            online_ratio=0.3  # 30% from recent, 70% from history
        )
    """

    def __init__(self, replay_capacity: int = 5000,
                 online_buffer: int = 200,
                 online_ratio: float = 0.3):
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.online_buffer: List[Transition] = []
        self.online_max = online_buffer
        self.online_ratio = min(1.0, max(0.0, online_ratio))

    def store(self, transition: Transition) -> None:
        """Store in both buffers."""
        self.replay_buffer.append(transition)
        self.online_buffer.append(transition)

        if len(self.online_buffer) > self.online_max:
            self.online_buffer.pop(0)

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample mixed batch from both buffers."""
        online_count = int(batch_size * self.online_ratio)
        replay_count = batch_size - online_count

        transitions = []

        # Sample from online buffer (recent data)
        if self.online_buffer and online_count > 0:
            online_sample = min(online_count, len(self.online_buffer))
            transitions.extend(random.sample(self.online_buffer, online_sample))

        # Sample from replay buffer (historical data)
        if len(self.replay_buffer) >= replay_count:
            indices = np.random.choice(len(self.replay_buffer), replay_count, replace=False)
            transitions.extend([self.replay_buffer[i] for i in indices])

        return transitions

    def should_train(self, step: int, train_frequency: int = 4) -> bool:
        """Train when we have enough data."""
        return (step % train_frequency == 0 and
                len(self.replay_buffer) >= self.batch_size_for_train())

    def can_sample(self, batch_size: int) -> bool:
        """Check if combined buffers have enough samples."""
        return len(self.replay_buffer) + len(self.online_buffer) >= batch_size

    def batch_size_for_train(self, batch_size: int = 64) -> int:
        """Get effective batch size considering online ratio."""
        return int(batch_size * (1 - self.online_ratio))

    def __len__(self) -> int:
        return len(self.replay_buffer) + len(self.online_buffer)


def create_sampler(sampler_type: str, **kwargs) -> Sampler:
    """Factory function to create sampler by name.

    Args:
        sampler_type: 'replay', 'online', or 'mixed'
        **kwargs: Additional arguments passed to sampler constructor

    Returns:
        Configured sampler instance

    Usage::

        sampler = create_sampler('replay', capacity=10000)
        sampler = create_sampler('online', batch_size=64, min_samples=100)
        sampler = create_sampler('mixed', online_ratio=0.3)
    """
    if sampler_type == 'replay':
        return ReplayBufferSampler(**kwargs)
    elif sampler_type == 'online':
        return OnlineSampler(**kwargs)
    elif sampler_type == 'mixed':
        return MixedSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
