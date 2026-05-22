"""Configuration loading for exp0522."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import yaml


@dataclass
class ExperimentConfig:
    """User-facing configuration for exp0522 runs."""

    run_name: str = "exp0522_clock_v0"
    seed: int = 42
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    trunk_dims: tuple[int, ...] = (32,)
    language_dim: int = 4
    cycle_steps: int = 32
    train_steps: int = 128
    eval_steps: int = 128
    long_steps: int = 512
    pulse_value: float = 1.0
    log_every: int = 25
    output_root: str = "runs"
    plot_dpi: int = 160

    def to_user_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_resolved_dict(self) -> dict[str, object]:
        payload = self.to_user_dict()
        payload["output_root"] = str(payload["output_root"])
        payload["message_init"] = 0.0
        payload["target"] = "sin(phi_t)"
        payload["phase_init"] = 0.0
        payload["omega"] = omega_from_cycle_steps(self.cycle_steps)
        return payload


def omega_from_cycle_steps(cycle_steps: int) -> float:
    import math

    return (2.0 * math.pi) / float(cycle_steps)


def config_from_user_dict(raw: dict[str, object]) -> ExperimentConfig:
    defaults = ExperimentConfig()
    valid_keys = set(asdict(defaults).keys())
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    payload = asdict(defaults)
    payload.update(raw)

    trunk_dims = payload.get("trunk_dims")
    if not isinstance(trunk_dims, (list, tuple)) or not trunk_dims:
        raise ValueError("trunk_dims must be a non-empty list of positive integers.")
    payload["trunk_dims"] = tuple(int(dim) for dim in trunk_dims)
    if any(dim <= 0 for dim in payload["trunk_dims"]):
        raise ValueError("Each trunk_dims entry must be positive.")

    for key in ("seed", "epochs", "language_dim", "cycle_steps", "train_steps", "eval_steps", "long_steps", "log_every", "plot_dpi"):
        payload[key] = int(payload[key])
    for key in ("lr", "weight_decay", "grad_clip", "pulse_value"):
        payload[key] = float(payload[key])

    if payload["epochs"] <= 0:
        raise ValueError("epochs must be positive.")
    if payload["language_dim"] <= 0:
        raise ValueError("language_dim must be positive.")
    if payload["cycle_steps"] <= 1:
        raise ValueError("cycle_steps must be greater than 1.")
    if payload["train_steps"] <= 0 or payload["eval_steps"] <= 0 or payload["long_steps"] <= 0:
        raise ValueError("train_steps, eval_steps, and long_steps must be positive.")
    if payload["eval_steps"] < payload["cycle_steps"]:
        raise ValueError("eval_steps must be at least one cycle long.")
    if payload["long_steps"] < payload["eval_steps"]:
        raise ValueError("long_steps must be greater than or equal to eval_steps.")
    if payload["log_every"] <= 0:
        raise ValueError("log_every must be positive.")
    if payload["plot_dpi"] <= 0:
        raise ValueError("plot_dpi must be positive.")

    return ExperimentConfig(**payload)


def load_config_from_yaml(path: Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_user_dict(raw)


def copy_config_to_run_dir(source_path: Path, run_dir: Path) -> None:
    shutil.copy2(source_path, run_dir / "config.yaml")


def write_resolved_config(config: ExperimentConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(config.to_resolved_dict(), indent=2))
