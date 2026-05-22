"""
Checkpoint system for saving and loading simulation state.

Supports two resume modes:
- "continue_population": Restore population + day, rebuild env from NEW config
- "exact_replay": Restore full state including prey and RNG for exact reproducibility
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch

# Checkpoint format version for compatibility checking
CHECKPOINT_VERSION = 1


class CheckpointError(Exception):
    """Exception raised for checkpoint-related errors."""
    pass


class IncompatibilityError(CheckpointError):
    """Exception raised when checkpoint is incompatible with current config."""
    pass


def get_hidden_sizes_from_controller(controller) -> List[int]:
    """Extract hidden sizes from a NeuralController."""
    import torch.nn as nn
    hidden_sizes = []
    for module in controller.mlp:
        if isinstance(module, nn.Linear):
            if module.out_features != 3:  # Not output layer (3 = vx, vy, y_birth)
                hidden_sizes.append(module.out_features)
    return hidden_sizes


def controller_to_state(controller) -> Dict[str, Any]:
    """
    Serialize a NeuralController to a state dict.

    Args:
        controller: NeuralController instance

    Returns:
        State dictionary with architecture and parameters
    """
    return {
        "input_size": controller.input_size,
        "hidden_sizes": get_hidden_sizes_from_controller(controller),
        "max_speed": controller.max_speed,
        "state_dict": controller.state_dict(),
    }


def controller_from_state(state: Dict[str, Any]):
    """
    Reconstruct a NeuralController from a state dict.

    Args:
        state: State dictionary from controller_to_state

    Returns:
        Reconstructed NeuralController
    """
    from controller import NeuralController

    controller = NeuralController(
        hidden_sizes=state["hidden_sizes"],
        max_speed=state["max_speed"],
        input_size=state["input_size"],
    )
    controller.load_state_dict(state["state_dict"])
    return controller


def agent_to_state(agent) -> Dict[str, Any]:
    """
    Serialize an Agent to a state dict.

    Args:
        agent: Agent instance

    Returns:
        State dictionary with agent attributes and controller
    """
    return {
        "id": agent.id,
        "position": agent.position.tolist(),
        "energy": float(agent.energy),
        "age": int(agent.age),
        "mass": int(agent.mass),
        "controller": controller_to_state(agent.controller),
    }


def agent_from_state(state: Dict[str, Any]):
    """
    Reconstruct an Agent from a state dict.

    Args:
        state: State dictionary from agent_to_state

    Returns:
        Reconstructed Agent
    """
    from agents import Agent

    return Agent(
        id=state["id"],
        position=np.array(state["position"], dtype=np.float32),
        energy=state["energy"],
        age=state["age"],
        mass=state["mass"],
        controller=controller_from_state(state["controller"]),
    )


def prey_to_state(prey) -> Dict[str, Any]:
    """
    Serialize a Prey to a state dict.

    Args:
        prey: Prey instance

    Returns:
        State dictionary
    """
    return {
        "id": prey.id,
        "position": prey.position.tolist(),
        "energy": float(prey.energy),
        "shelf_life": int(prey.shelf_life),
    }


def prey_from_state(state: Dict[str, Any]):
    """
    Reconstruct a Prey from a state dict.

    Args:
        state: State dictionary from prey_to_state

    Returns:
        Reconstructed Prey
    """
    from prey import Prey

    return Prey(
        id=state["id"],
        position=np.array(state["position"], dtype=np.float32),
        energy=state["energy"],
        shelf_life=state["shelf_life"],
    )


def build_checkpoint(
    day: int,
    agents: List,
    agent_manager_next_id: int,
    prey_manager: Optional[Any] = None,
    np_rng: Optional[np.random.Generator] = None,
    torch_rng: Optional[torch.Generator] = None,
    config_dict: Optional[Dict] = None,
    include_prey: bool = False,
    include_rng: bool = False,
) -> Dict[str, Any]:
    """
    Build a checkpoint dictionary from simulation state.

    Args:
        day: Current simulation day
        agents: List of Agent objects
        agent_manager_next_id: Next agent ID counter
        prey_manager: PreyManager instance (for exact_replay)
        np_rng: NumPy random generator (for exact_replay)
        torch_rng: PyTorch random generator (for exact_replay)
        config_dict: Configuration dictionary
        include_prey: Whether to include prey state
        include_rng: Whether to include RNG state

    Returns:
        Checkpoint dictionary ready for serialization
    """
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "saved_day": day,
        "timestamp": datetime.now().isoformat(),
        "population": {
            "agents": [agent_to_state(a) for a in agents],
            "next_id": agent_manager_next_id,
        },
    }

    if config_dict is not None:
        checkpoint["config"] = config_dict

    # For exact_replay mode
    if include_prey and prey_manager is not None:
        checkpoint["prey"] = {
            "preys": [prey_to_state(p) for p in prey_manager.preys],
            "next_id": prey_manager._next_id,
        }

    if include_rng:
        checkpoint["rng"] = {}
        if np_rng is not None:
            # NumPy RNG state (bit_generator state)
            checkpoint["rng"]["numpy"] = np_rng.bit_generator.state
        if torch_rng is not None:
            # PyTorch RNG state
            checkpoint["rng"]["torch"] = torch_rng.get_state().numpy().tolist()

    return checkpoint


def save_checkpoint(
    checkpoint: Dict[str, Any],
    filepath: Path,
    atomic: bool = True,
) -> None:
    """
    Save checkpoint to file with optional atomic write.

    Args:
        checkpoint: Checkpoint dictionary
        filepath: Destination file path
        atomic: Use atomic write (temp file + rename)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        # Write to temp file first, then rename
        fd, tmp_path = tempfile.mkstemp(
            suffix=".pt.tmp",
            dir=filepath.parent,
        )
        os.close(fd)
        try:
            torch.save(checkpoint, tmp_path)
            shutil.move(tmp_path, filepath)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    else:
        torch.save(checkpoint, filepath)


def load_checkpoint(filepath: Path) -> Dict[str, Any]:
    """
    Load checkpoint from file.

    Args:
        filepath: Path to checkpoint file

    Returns:
        Checkpoint dictionary

    Raises:
        CheckpointError: If file doesn't exist or is corrupted
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise CheckpointError(f"Checkpoint file not found: {filepath}")

    try:
        checkpoint = torch.load(filepath, weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint: {e}")

    # Version check
    version = checkpoint.get("checkpoint_version", 0)
    if version != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Checkpoint version mismatch: file={version}, expected={CHECKPOINT_VERSION}"
        )

    return checkpoint


def update_latest_json(
    checkpoint_dir: Path,
    checkpoint_filename: str,
    day: int,
) -> None:
    """
    Update latest.json to point to the most recent checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_filename: Name of the latest checkpoint file
        day: Day of the checkpoint
    """
    latest_path = checkpoint_dir / "latest.json"
    latest_info = {
        "filename": checkpoint_filename,  # Field name consistent with RunContext
        "day": day,
        "timestamp": datetime.now().isoformat(),
    }

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=checkpoint_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(latest_info, f, indent=2)
        shutil.move(tmp_path, latest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Get the path to the latest checkpoint file.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest_path = checkpoint_dir / "latest.json"

    if latest_path.exists():
        with open(latest_path, "r") as f:
            info = json.load(f)
        # Support both "filename" (new) and "checkpoint_file" (legacy) keys
        filename = info.get("filename") or info.get("checkpoint_file")
        if filename:
            checkpoint_path = checkpoint_dir / filename
            if checkpoint_path.exists():
                return checkpoint_path

    # Fallback: scan for checkpoint files
    checkpoints = sorted(checkpoint_dir.glob("ckpt_day_*.pt"))
    if checkpoints:
        return checkpoints[-1]

    return None


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> List[Path]:
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep (0 = keep all)

    Returns:
        List of deleted checkpoint paths
    """
    if keep_last <= 0:
        return []

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("ckpt_day_*.pt"))

    deleted = []
    if len(checkpoints) > keep_last:
        to_delete = checkpoints[:-keep_last]
        for ckpt in to_delete:
            ckpt.unlink()
            deleted.append(ckpt)

    return deleted


def check_compatibility(
    checkpoint: Dict[str, Any],
    config,
    strict: bool = True,
) -> Tuple[bool, str]:
    """
    Check if checkpoint is compatible with current config.

    Compatibility requires:
    - Same NN input size (scent_mode determines this)
    - Same hidden_sizes

    Note: Mass (edge count) is derived from architecture, so no separate check needed.

    Args:
        checkpoint: Loaded checkpoint dictionary
        config: Current Config object
        strict: If True, be strict about compatibility

    Returns:
        Tuple of (is_compatible, message)
    """
    # Check if there are any agents
    agents_data = checkpoint.get("population", {}).get("agents", [])
    if not agents_data:
        return True, "Empty population, compatible"

    # Get expected input size from current config
    # Vector mode: 5D, stencil mode: 11D (nose 9D + energy + age)
    expected_input_size = 11 if config.scent_mode == "stencil" else 5

    # Check first agent's controller architecture
    first_agent = agents_data[0]
    ctrl_state = first_agent.get("controller", {})

    ckpt_input_size = ctrl_state.get("input_size", 5)
    ckpt_hidden_sizes = ctrl_state.get("hidden_sizes", [])

    issues = []

    # Check input size (observation dimension)
    if ckpt_input_size != expected_input_size:
        issues.append(
            f"Input size mismatch: checkpoint={ckpt_input_size}, "
            f"config scent_mode={config.scent_mode} requires {expected_input_size}"
        )

    # Check hidden sizes
    if ckpt_hidden_sizes != config.hidden_sizes:
        issues.append(
            f"Hidden sizes mismatch: checkpoint={ckpt_hidden_sizes}, "
            f"config={config.hidden_sizes}"
        )

    # Note: Mass is now edge count, derived from architecture (input_size + hidden_sizes)
    # No separate mass check needed - architecture match implies mass match

    if issues:
        message = "Checkpoint incompatible: " + "; ".join(issues)
        return False, message

    return True, "Checkpoint compatible"


def restore_population(
    checkpoint: Dict[str, Any],
    agent_manager,
) -> int:
    """
    Restore agent population from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary
        agent_manager: AgentManager to populate

    Returns:
        Restored day number
    """
    pop_data = checkpoint["population"]

    # Clear existing agents
    agent_manager.agents = []
    agent_manager._next_id = pop_data["next_id"]

    # Restore agents
    for agent_state in pop_data["agents"]:
        agent = agent_from_state(agent_state)
        agent_manager.agents.append(agent)

    return checkpoint["saved_day"]


def restore_prey(
    checkpoint: Dict[str, Any],
    prey_manager,
) -> None:
    """
    Restore prey state from checkpoint (for exact_replay mode).

    Args:
        checkpoint: Loaded checkpoint dictionary
        prey_manager: PreyManager to populate
    """
    if "prey" not in checkpoint:
        return

    prey_data = checkpoint["prey"]

    # Clear existing preys
    prey_manager.preys = []
    prey_manager._next_id = prey_data["next_id"]

    # Restore preys
    for prey_state in prey_data["preys"]:
        prey = prey_from_state(prey_state)
        prey_manager.preys.append(prey)


def restore_rng(
    checkpoint: Dict[str, Any],
    np_rng: Optional[np.random.Generator] = None,
    torch_rng: Optional[torch.Generator] = None,
) -> None:
    """
    Restore RNG states from checkpoint (for exact_replay mode).

    Args:
        checkpoint: Loaded checkpoint dictionary
        np_rng: NumPy random generator to restore
        torch_rng: PyTorch random generator to restore
    """
    if "rng" not in checkpoint:
        return

    rng_data = checkpoint["rng"]

    if np_rng is not None and "numpy" in rng_data:
        np_rng.bit_generator.state = rng_data["numpy"]

    if torch_rng is not None and "torch" in rng_data:
        state_tensor = torch.tensor(rng_data["torch"], dtype=torch.uint8)
        torch_rng.set_state(state_tensor)


class CheckpointManager:
    """
    High-level checkpoint manager for the simulation.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        every_days: int = 0,
        keep_last: int = 3,
        logger=None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            every_days: Save every N days (0 = only at end)
            keep_last: Keep only last N checkpoints (0 = keep all)
            logger: Logger for messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_days = every_days
        self.keep_last = keep_last
        self.logger = logger
        self._last_save_day = -1

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        """Log a message if logger is available."""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def should_save(self, day: int) -> bool:
        """
        Check if checkpoint should be saved at this day.

        Args:
            day: Current simulation day

        Returns:
            True if checkpoint should be saved
        """
        if self.every_days <= 0:
            return False
        return day > 0 and day % self.every_days == 0 and day != self._last_save_day

    def save(
        self,
        day: int,
        agent_manager,
        prey_manager=None,
        np_rng=None,
        torch_rng=None,
        config_dict=None,
        resume_mode: str = "continue_population",
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            day: Current simulation day
            agent_manager: AgentManager with current population
            prey_manager: PreyManager (for exact_replay)
            np_rng: NumPy RNG (for exact_replay)
            torch_rng: PyTorch RNG (for exact_replay)
            config_dict: Current config as dict
            resume_mode: Checkpoint mode

        Returns:
            Path to saved checkpoint
        """
        include_prey = resume_mode == "exact_replay"
        include_rng = resume_mode == "exact_replay"

        checkpoint = build_checkpoint(
            day=day,
            agents=agent_manager.agents,
            agent_manager_next_id=agent_manager._next_id,
            prey_manager=prey_manager if include_prey else None,
            np_rng=np_rng if include_rng else None,
            torch_rng=torch_rng if include_rng else None,
            config_dict=config_dict,
            include_prey=include_prey,
            include_rng=include_rng,
        )

        # Generate filename
        filename = f"ckpt_day_{day:08d}.pt"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        save_checkpoint(checkpoint, filepath)
        self._last_save_day = day

        # Update latest.json
        update_latest_json(self.checkpoint_dir, filename, day)

        # Cleanup old checkpoints
        deleted = cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last)

        # Log
        n_agents = len(agent_manager.agents)
        self._log(f"Checkpoint saved: {filepath} (day={day}, agents={n_agents})")
        if deleted:
            self._log(f"  Removed {len(deleted)} old checkpoint(s)")

        return filepath

    def load(
        self,
        config,
        agent_manager,
        prey_manager=None,
        np_rng=None,
        torch_rng=None,
        strict: bool = True,
        resume_mode: str = "continue_population",
    ) -> Tuple[bool, int, str]:
        """
        Load the latest checkpoint.

        Args:
            config: Current Config object
            agent_manager: AgentManager to restore into
            prey_manager: PreyManager to restore (for exact_replay)
            np_rng: NumPy RNG to restore (for exact_replay)
            torch_rng: PyTorch RNG to restore (for exact_replay)
            strict: Error on incompatibility if True
            resume_mode: Resume mode

        Returns:
            Tuple of (success, restored_day, message)
        """
        # Find latest checkpoint
        checkpoint_path = get_latest_checkpoint(self.checkpoint_dir)
        if checkpoint_path is None:
            return False, 0, f"No checkpoint found in {self.checkpoint_dir}"

        self._log(f"Loading checkpoint: {checkpoint_path}")

        try:
            checkpoint = load_checkpoint(checkpoint_path)
        except CheckpointError as e:
            return False, 0, str(e)

        # Check compatibility
        is_compatible, message = check_compatibility(checkpoint, config, strict)
        if not is_compatible:
            if strict:
                return False, 0, message
            else:
                self._log(f"Warning: {message}")
                self._log("Starting from scratch due to incompatibility (resume_strict=false)")
                return False, 0, message

        # Restore population
        restored_day = restore_population(checkpoint, agent_manager)

        # For exact_replay, also restore prey and RNG
        if resume_mode == "exact_replay":
            if prey_manager is not None:
                restore_prey(checkpoint, prey_manager)
            restore_rng(checkpoint, np_rng, torch_rng)
            self._log(f"  Restored prey and RNG states (exact_replay mode)")

        n_agents = len(agent_manager.agents)
        saved_config = checkpoint.get("config", {})
        original_seed = saved_config.get("seed", "unknown")

        self._log(f"  Restored day={restored_day}, agents={n_agents}")
        self._log(f"  Original seed: {original_seed}, resume_mode: {resume_mode}")

        return True, restored_day, "Checkpoint loaded successfully"
