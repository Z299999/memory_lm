"""TrainingLogger: Append-only training log.

Log format (JSON lines in run.log):
    {"timestamp": "...", "event": "init", "config": {...}}
    {"timestamp": "...", "event": "load_checkpoint", "path": "...", "compatibility": "..."}
    {"timestamp": "...", "event": "epoch", "episode": 1, "reward": ..., "loss": ...}
    {"timestamp": "...", "event": "save_checkpoint", "path": "..."}

Usage::

    from tools.logger import TrainingLogger

    logger = TrainingLogger('./logs')
    logger.log('train_start', config={'n': 2, 'm': 3})
    logger.log('epoch', episode=1, reward=100, loss=0.5)
    logger.log('checkpoint_saved', path='checkpoints/xxx.pt')
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class TrainingLogger:
    """Append-only training log.

    Args:
        log_dir: Directory to store logs
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "run.log"

    def log(self, event: str, **kwargs: Any) -> None:
        """Append a log entry.

        Args:
            event: Event type (e.g., 'init', 'epoch', 'checkpoint_saved')
            **kwargs: Additional data to log
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            **kwargs
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")

    def log_init(self, config: dict) -> None:
        """Log training initialization.

        Args:
            config: Training configuration
        """
        self.log('init', config=config)

    def log_epoch(
        self,
        episode: int,
        reward: float,
        loss: float | None = None,
        epsilon: float | None = None,
        **kwargs: Any
    ) -> None:
        """Log epoch statistics.

        Args:
            episode: Episode number
            reward: Episode reward
            loss: Training loss (optional)
            epsilon: Exploration rate (optional)
            **kwargs: Additional metrics
        """
        self.log(
            'epoch',
            episode=episode,
            reward=reward,
            loss=loss,
            epsilon=epsilon,
            **kwargs
        )

    def log_checkpoint_saved(self, path: str, episode: int, reward: float) -> None:
        """Log checkpoint save event.

        Args:
            path: Checkpoint file path
            episode: Episode number
            reward: Episode reward
        """
        self.log(
            'checkpoint_saved',
            path=path,
            episode=episode,
            reward=reward
        )

    def log_checkpoint_loaded(self, path: str, episode: int) -> None:
        """Log checkpoint load event.

        Args:
            path: Checkpoint file path
            episode: Episode number from checkpoint
        """
        self.log(
            'checkpoint_loaded',
            path=path,
            episode=episode
        )

    def log_error(self, error: str, **kwargs: Any) -> None:
        """Log error event.

        Args:
            error: Error message
            **kwargs: Additional context
        """
        self.log('error', error=error, **kwargs)

    def log_custom(self, event: str, **kwargs: Any) -> None:
        """Log custom event.

        Args:
            event: Custom event name
            **kwargs: Event data
        """
        self.log(event, **kwargs)

    def get_logs(self) -> list[dict]:
        """Read all logs from file.

        Returns:
            List of log entries (dictionaries)
        """
        if not self.log_path.exists():
            return []

        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
        return logs

    def get_events(self, event_type: str) -> list[dict]:
        """Get logs of a specific event type.

        Args:
            event_type: Event type to filter

        Returns:
            List of matching log entries
        """
        logs = self.get_logs()
        return [log for log in logs if log.get('event') == event_type]

    def clear(self) -> None:
        """Clear all logs."""
        if self.log_path.exists():
            self.log_path.unlink()


# Quick test
if __name__ == "__main__":
    import tempfile

    print("Testing TrainingLogger...")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrainingLogger(tmpdir)

        # Test 1: Log init
        print("\nTest 1: Log init")
        logger.log_init(config={'n': 2, 'm': 3, 'lr': 1e-3})
        print("  PASSED")

        # Test 2: Log epoch
        print("Test 2: Log epoch")
        logger.log_epoch(episode=1, reward=100, loss=0.5, epsilon=0.9)
        logger.log_epoch(episode=2, reward=105, loss=0.45, epsilon=0.89)
        print("  PASSED")

        # Test 3: Log checkpoint
        print("Test 3: Log checkpoint")
        logger.log_checkpoint_saved('checkpoints/xxx.pt', episode=100, reward=95.5)
        print("  PASSED")

        # Test 4: Get logs
        print("Test 4: Get logs")
        logs = logger.get_logs()
        assert len(logs) == 4
        print(f"  Total logs: {len(logs)}")
        print("  PASSED")

        # Test 5: Get events
        print("Test 5: Get events (filter by type)")
        epoch_logs = logger.get_events('epoch')
        assert len(epoch_logs) == 2
        print(f"  Epoch logs: {len(epoch_logs)}")
        print("  PASSED")

        # Test 6: Log error
        print("Test 6: Log error")
        logger.log_error('Test error', context='testing')
        error_logs = logger.get_events('error')
        assert len(error_logs) == 1
        print("  PASSED")

        # Test 7: Clear logs
        print("Test 7: Clear logs")
        logger.clear()
        assert logger.get_logs() == []
        print("  PASSED")

    print("\n" + "="*50)
    print("All TrainingLogger tests PASSED!")
    print("="*50)
