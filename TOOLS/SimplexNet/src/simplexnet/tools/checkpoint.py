"""CheckpointManager: Manage checkpoint save/load for SMN training.

Features:
- Auto-discover existing checkpoints
- Save/load checkpoint with metadata
- Compatibility checking (Phase 2)
- User confirmation (Phase 2)

Usage::

    from tools.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager('./checkpoints')

    # Save checkpoint
    ckpt_mgr.save(
        module=smn,
        optimizer=optimizer,
        episode=100,
        rewards=rewards,
        metadata={'env': 'CartPole-v1'}
    )

    # Load latest checkpoint
    checkpoint = ckpt_mgr.load_latest()
    if checkpoint:
        smn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointInfo:
    """Metadata for a checkpoint file.

    Attributes:
        timestamp: ISO format timestamp
        config: Model configuration (n, m, n_in, n_out, etc.)
        episode: Current episode number
        reward: Episode reward
        loss: Training loss (if available)
        state_dict_keys: List of state dict keys
    """
    timestamp: str
    config: dict
    episode: int
    reward: float
    loss: float | None
    state_dict_keys: list[str]


class CheckpointManager:
    """Manage checkpoint save/load with metadata tracking.

    Features:
    - Auto-discover existing checkpoints
    - Save/load checkpoint with metadata
    - Conservative: prefer loading existing checkpoints

    Args:
        checkpoint_dir: Directory to store checkpoints
    """

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def find_latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint file.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            return None

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_files[0]

    def find_all_checkpoints(self) -> list[Path]:
        """Find all checkpoint files.

        Returns:
            List of checkpoint paths, sorted by modification time (newest first).
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_files

    def load_checkpoint(self, path: Path) -> tuple[dict, CheckpointInfo]:
        """Load checkpoint and return state_dict + metadata.

        Args:
            path: Path to checkpoint file

        Returns:
            Tuple of (state_dict, CheckpointInfo)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, weights_only=False)
        state_dict_keys = []
        if 'state_dict' in checkpoint:
            state_dict_keys = list(checkpoint['state_dict'].keys())
        elif 'actor_state_dict' in checkpoint:
            state_dict_keys = list(checkpoint['actor_state_dict'].keys())

        info = CheckpointInfo(
            timestamp=checkpoint['timestamp'],
            config=checkpoint['config'],
            episode=checkpoint['episode'],
            reward=checkpoint['reward'],
            loss=checkpoint.get('loss'),
            state_dict_keys=state_dict_keys,
        )
        state_dict = checkpoint.get('state_dict', checkpoint.get('actor_state_dict', {}))
        return state_dict, info

    def load_full_checkpoint(self, path: Path) -> dict:
        """Load the full raw checkpoint dictionary from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, weights_only=False)

    def load_latest(self) -> dict | None:
        """Load the latest checkpoint.

        Returns:
            Full checkpoint dict (including state_dict, optimizer_state, etc.),
            or None if no checkpoints exist.
        """
        latest = self.find_latest_checkpoint()
        if latest is None:
            return None
        return torch.load(latest, weights_only=False)

    def save_checkpoint(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        episode: int = 0,
        reward: float = 0.0,
        loss: float | None = None,
        metadata: dict | None = None,
        extra_state: dict | None = None,
        config_override: dict | None = None,
    ) -> Path:
        """Save checkpoint to disk.

        Args:
            module: The neural network module
            optimizer: Optimizer (optional)
            episode: Current episode number
            reward: Episode reward
            loss: Training loss (optional)
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}.pt"

        # Build config from module
        config = {}
        if hasattr(module, 'n'):
            config['n'] = module.n
        if hasattr(module, 'm'):
            config['m'] = module.m
        if hasattr(module, 'n_in'):
            config['n_in'] = module.n_in
        if hasattr(module, 'n_out'):
            config['n_out'] = module.n_out
        if hasattr(module, 'activation'):
            config['activation'] = module.activation

        checkpoint = {
            'timestamp': timestamp,
            'config': config_override or config,
            'episode': episode,
            'reward': reward,
            'loss': loss,
            'state_dict': module.state_dict(),
            'metadata': metadata or {},
        }

        if optimizer is not None:
            checkpoint['optimizer_state'] = optimizer.state_dict()

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def get_checkpoint_info(self, path: Path) -> CheckpointInfo:
        """Get metadata for a checkpoint without loading weights.

        Args:
            path: Path to checkpoint file

        Returns:
            CheckpointInfo object
        """
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        state_dict_keys = []
        if 'state_dict' in checkpoint:
            state_dict_keys = list(checkpoint['state_dict'].keys())
        elif 'actor_state_dict' in checkpoint:
            state_dict_keys = list(checkpoint['actor_state_dict'].keys())

        return CheckpointInfo(
            timestamp=checkpoint['timestamp'],
            config=checkpoint['config'],
            episode=checkpoint['episode'],
            reward=checkpoint['reward'],
            loss=checkpoint.get('loss'),
            state_dict_keys=state_dict_keys,
        )

    def delete_checkpoint(self, path: Path) -> bool:
        """Delete a checkpoint file.

        Args:
            path: Path to checkpoint file

        Returns:
            True if deleted, False if file didn't exist
        """
        if path.exists():
            path.unlink()
            return True
        return False


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from SMNmodule import SMNmodule
    import torch.optim as optim

    print("Testing CheckpointManager...")

    # Create test module
    module = SMNmodule(n=2, m=3, n_in=1, n_out=1)
    optimizer = optim.Adam(module.parameters(), lr=1e-3)

    # Create checkpoint manager
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(tmpdir)

        # Test 1: No checkpoints initially
        print("\nTest 1: No checkpoints initially")
        assert ckpt_mgr.find_latest_checkpoint() is None
        print("  PASSED")

        # Test 2: Save checkpoint
        print("Test 2: Save checkpoint")
        test_rewards = [1.0, 2.0, 3.0]
        path = ckpt_mgr.save_checkpoint(
            module=module,
            optimizer=optimizer,
            episode=100,
            reward=95.5,
            loss=0.5,
            metadata={'test': 'data'}
        )
        print(f"  Saved: {path}")
        assert path.exists()
        print("  PASSED")

        # Test 3: Find latest checkpoint
        print("Test 3: Find latest checkpoint")
        latest = ckpt_mgr.find_latest_checkpoint()
        assert latest == path
        print(f"  Found: {latest}")
        print("  PASSED")

        # Test 4: Load checkpoint
        print("Test 4: Load checkpoint")
        state_dict, info = ckpt_mgr.load_checkpoint(path)
        print(f"  Info: episode={info.episode}, reward={info.reward}")
        assert info.episode == 100
        assert info.reward == 95.5
        print("  PASSED")

        # Test 5: Load latest
        print("Test 5: Load latest")
        checkpoint = ckpt_mgr.load_latest()
        assert checkpoint is not None
        assert 'state_dict' in checkpoint
        assert 'optimizer_state' in checkpoint
        print("  PASSED")

        # Test 6: Get checkpoint info
        print("Test 6: Get checkpoint info (without loading)")
        info2 = ckpt_mgr.get_checkpoint_info(path)
        assert info2.episode == 100
        print("  PASSED")

        # Test 7: Delete checkpoint
        print("Test 7: Delete checkpoint")
        assert ckpt_mgr.delete_checkpoint(path)
        assert not path.exists()
        assert ckpt_mgr.find_latest_checkpoint() is None
        print("  PASSED")

    print("\n" + "="*50)
    print("All CheckpointManager tests PASSED!")
    print("="*50)
