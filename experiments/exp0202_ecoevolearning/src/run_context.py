"""
Run context management for organizing simulation outputs.

Each simulation run gets a unique run_id, and all outputs (plots, logs, checkpoints)
are organized under that run_id in both output/ and checkpoint/ directories.

When resuming from a checkpoint, the run_id is preserved to keep all artifacts
from the same simulation grouped together. Each segment (e.g., 0-1000, 1001-2000)
gets its own subfolder for outputs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class RunContext:
    """
    Manages run directory structure and run_id persistence.

    Directory structure:
        outputs/runs/<run_id>/
            - meta.json
            - <start>-<end>/           (segment folder, e.g., "0-1000")
                - config_resolved.yaml
                - console.log
                - timeseries.csv
                - plots/
                - video.mp4
        checkpoints/<run_id>/
            - ckpt_day_XXXXXXXX.pt
            - latest.json
            - meta.json

    When resuming, the run_id is extracted from the checkpoint path to ensure
    all outputs go to the same run folder, but in a new segment subfolder.
    """

    def __init__(
        self,
        base_dir: Path,
        run_id: Optional[str] = None,
        seed: int = 0,
        config_path: Optional[Path] = None,
        start_day: int = 0,
        end_day: int = 0,
    ):
        """
        Initialize run context.

        Args:
            base_dir: Base directory for the experiment (e.g., exp2-2/)
            run_id: Existing run_id (for resuming) or None to create new
            seed: Random seed (used in auto-generated run_id)
            config_path: Path to config file (for metadata)
            start_day: Starting day for this segment (0 for new runs)
            end_day: Target ending day for this segment
        """
        self.base_dir = Path(base_dir)
        self.seed = seed
        self.config_path = config_path
        self.start_day = start_day
        self.end_day = end_day

        # Generate or use existing run_id
        if run_id is None:
            self.run_id = self._generate_run_id(seed)
            self._is_new_run = True
        else:
            self.run_id = run_id
            self._is_new_run = False

        # Set up directory paths
        self.run_dir = self.base_dir / "outputs" / "runs" / self.run_id
        self.checkpoint_dir = self.base_dir / "checkpoints" / self.run_id

        # Segment folder for this run's outputs (e.g., "0-1000")
        segment_name = f"{start_day}-{end_day}"
        self.segment_dir = self.run_dir / segment_name

        # Legacy: output_dir points to segment_dir for compatibility
        self.output_dir = self.segment_dir

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata if this is a new run
        if self._is_new_run:
            self._save_metadata()

    @staticmethod
    def _generate_run_id(seed: int) -> str:
        """Generate a unique run_id based on timestamp and seed."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_seed{seed}"

    @classmethod
    def from_checkpoint_path(
        cls,
        checkpoint_path: Path,
        base_dir: Path,
        seed: int = 0,
        config_path: Optional[Path] = None,
    ) -> "RunContext":
        """
        Create RunContext by extracting run_id from a checkpoint path.

        The checkpoint path should be like:
            .../checkpoints/<run_id>/ckpt_day_XXXXXXXX.pt
        or:
            .../checkpoints/<run_id>/latest.json

        Args:
            checkpoint_path: Path to a checkpoint file
            base_dir: Base directory for the experiment
            seed: Random seed
            config_path: Path to config file

        Returns:
            RunContext with extracted run_id
        """
        checkpoint_path = Path(checkpoint_path)

        # Extract run_id from path
        # Expected: .../checkpoints/<run_id>/filename
        run_id = checkpoint_path.parent.name

        return cls(
            base_dir=base_dir,
            run_id=run_id,
            seed=seed,
            config_path=config_path,
        )

    @staticmethod
    def find_latest_checkpoint_info(
        base_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the latest checkpoint and return its info without creating a RunContext.

        Args:
            base_dir: Base directory for the experiment

        Returns:
            Dict with "run_id" and "day" if found, None otherwise
        """
        checkpoints_dir = Path(base_dir) / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        # Find all run directories with checkpoints
        latest_time = None
        latest_run_id = None
        latest_day = None

        for run_dir in checkpoints_dir.iterdir():
            if not run_dir.is_dir():
                continue

            latest_json = run_dir / "latest.json"
            if latest_json.exists():
                try:
                    with open(latest_json, "r") as f:
                        data = json.load(f)
                    filename = data.get("filename") or data.get("checkpoint_file")
                    day = data.get("day", 0)
                    if filename:
                        ckpt_path = run_dir / filename
                        if ckpt_path.exists():
                            mtime = ckpt_path.stat().st_mtime
                            if latest_time is None or mtime > latest_time:
                                latest_time = mtime
                                latest_run_id = run_dir.name
                                latest_day = day
                except (json.JSONDecodeError, KeyError):
                    continue

        if latest_run_id is None:
            return None

        return {"run_id": latest_run_id, "day": latest_day}

    @classmethod
    def from_latest_checkpoint(
        cls,
        base_dir: Path,
        seed: int = 0,
        config_path: Optional[Path] = None,
        start_day: int = 0,
        end_day: int = 0,
    ) -> Optional["RunContext"]:
        """
        Find and resume from the latest checkpoint across all runs.

        Looks for the most recent checkpoint in any run folder.

        Args:
            base_dir: Base directory for the experiment
            seed: Random seed
            config_path: Path to config file
            start_day: Starting day for this segment
            end_day: Target ending day for this segment

        Returns:
            RunContext if a checkpoint was found, None otherwise
        """
        info = cls.find_latest_checkpoint_info(base_dir)
        if info is None:
            return None

        return cls(
            base_dir=base_dir,
            run_id=info["run_id"],
            seed=seed,
            config_path=config_path,
            start_day=start_day,
            end_day=end_day,
        )

    def _save_metadata(self):
        """Save run metadata to both output and checkpoint directories."""
        meta = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "seed": self.seed,
            "config_path": str(self.config_path) if self.config_path else None,
        }

        # Save to output directory
        output_meta = self.output_dir / "meta.json"
        with open(output_meta, "w") as f:
            json.dump(meta, f, indent=2)

        # Save to checkpoint directory
        ckpt_meta = self.checkpoint_dir / "meta.json"
        with open(ckpt_meta, "w") as f:
            json.dump(meta, f, indent=2)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to the latest checkpoint file in this run's checkpoint directory.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        latest_json = self.checkpoint_dir / "latest.json"
        if not latest_json.exists():
            return None

        try:
            with open(latest_json, "r") as f:
                data = json.load(f)
            filename = data.get("filename")
            if filename:
                ckpt_path = self.checkpoint_dir / filename
                if ckpt_path.exists():
                    return ckpt_path
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    def __repr__(self) -> str:
        return f"RunContext(run_id={self.run_id}, output_dir={self.output_dir})"
