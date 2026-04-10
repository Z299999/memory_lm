from __future__ import annotations

# This file keeps only the basic experiment knobs visible at the top.
# Less important training defaults are hidden below as internal constants.

from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml


# Internal defaults. These still exist, but you usually do not need to edit them.
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_NUM_TRAIN = 512
DEFAULT_NUM_VAL = 256
DEFAULT_NUM_PLOT = 512


@dataclass
class Config:
    # These are the main parameters you are expected to edit.
    model_type: str = "tmn"
    run_name: str = "default"

    L: int = 3
    n_in: int = 1
    n_out: int = 1

    # Example: [8, 8, 8] means 3 hidden layers, each with 8 nodes.
    mlp_layers: list[int] = field(default_factory=lambda: [8,8,8])

    task_name: str = "piecewise"
    custom_function: str = ""
    node_activation: str = "relu"
    output_activation: str = "tanh"

    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 300

    x_min: float = -6.283185307179586
    x_max: float = 6.283185307179586

    @property
    def seed(self) -> int:
        return DEFAULT_SEED

    @property
    def weight_decay(self) -> float:
        return DEFAULT_WEIGHT_DECAY

    @property
    def num_train(self) -> int:
        return DEFAULT_NUM_TRAIN

    @property
    def num_val(self) -> int:
        return DEFAULT_NUM_VAL

    @property
    def num_plot(self) -> int:
        return DEFAULT_NUM_PLOT

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seed"] = self.seed
        data["weight_decay"] = self.weight_decay
        data["num_train"] = self.num_train
        data["num_val"] = self.num_val
        data["num_plot"] = self.num_plot
        return data


def load_config_from_yaml(path: Path) -> Config:
    # The yaml file is the user-facing parameter file at repo root.
    config = Config()
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    editable_keys = set(Config.__dataclass_fields__.keys())

    if not isinstance(raw_data, dict):
        raise ValueError("params.yaml must contain a top-level mapping.")

    for key, value in raw_data.items():
        if key not in editable_keys:
            raise ValueError(f"Unsupported config key in yaml: {key}")
        setattr(config, key, value)

    return config
