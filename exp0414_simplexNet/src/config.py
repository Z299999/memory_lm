"""Configuration for simplex memory network experiments.

This module provides a Config dataclass and YAML loading utilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


# Internal defaults
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_NUM_TRAIN = 10000
DEFAULT_NUM_VAL = 1000
DEFAULT_NUM_PLOT = 10000


@dataclass
class Config:
    """Configuration for SMN experiments.

    Attributes:
        model_type: "smn" or "mlp"
        run_name: Experiment name (default: "default")
        n: Simplex dimension (default: 2 for triangle)
        m: Resolution parameter (default: 3)
        n_in: Number of input dimensions
        n_out: Number of output dimensions
        mlp_layers: Hidden layer sizes for MLP baseline
        node_activation: Activation function ("relu", "leaky_relu", "gelu", "tanh")
        task_name: Target function name
        custom_function: Custom function expression
        lr: Learning rate
        batch_size: Batch size
        epochs: Number of training epochs
        x_min: Minimum x value for 1D tasks
        x_max: Maximum x value for 1D tasks
    """
    model_type: str = "smn"
    run_name: str = "default"

    # SMN architecture parameters
    n: int = 2          # Simplex dimension
    m: int = 3          # Resolution
    n_in: int = 1
    n_out: int = 1

    # MLP baseline architecture
    mlp_layers: list[int] = field(default_factory=lambda: [8, 8, 8])

    # Activation functions
    node_activation: str = "relu"
    output_activation: str = "tanh"

    # Task parameters
    task_name: str = "piecewise"
    custom_function: str = ""

    # Training parameters
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 300

    # Data sampling
    num_train: int = 10000
    num_val: int = 1000
    num_plot: int = 10000

    # Moving window training (for 1D tasks)
    # If window_width > 0, use moving window training instead of full-domain training
    # Window slides from left to right without cycling
    window_width: float = 0.0   # Window width as fraction of domain (0.0 = disabled, 0.5 = 50%)
    window_hold: int = 1        # Number of epochs per window position before sliding

    # Data sampling
    num_train: int = 10000
    num_val: int = 1000
    num_plot: int = 10000

    # Data range (1D tasks)
    x_min: float = -6.283185307179586
    x_max: float = 6.283185307179586

    @property
    def seed(self) -> int:
        return DEFAULT_SEED

    @property
    def weight_decay(self) -> float:
        return DEFAULT_WEIGHT_DECAY

    @property
    def use_windowed_training(self) -> bool:
        """Check if windowed training is enabled."""
        return self.window_width > 0 and self.window_width <= 1.0

    @property
    def domain_width(self) -> float:
        """Return the width of the input domain."""
        return self.x_max - self.x_min

    def to_dict(self) -> dict:
        """Convert to dictionary including derived properties."""
        data = asdict(self)
        data["seed"] = self.seed
        data["weight_decay"] = self.weight_decay
        data["num_train"] = self.num_train
        data["num_val"] = self.num_val
        data["num_plot"] = self.num_plot
        return data


def load_config_from_yaml(path: Path) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Config object with values from YAML

    Raises:
        ValueError: If YAML contains unsupported keys
    """
    config = Config()
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if not isinstance(raw_data, dict):
        raise ValueError("params.yaml must contain a top-level mapping.")

    editable_keys = set(Config.__dataclass_fields__.keys())
    for key, value in raw_data.items():
        if key not in editable_keys:
            raise ValueError(f"Unsupported config key in yaml: {key}")
        setattr(config, key, value)

    return config


# Quick test
if __name__ == "__main__":
    config = Config()
    print("Default config:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
