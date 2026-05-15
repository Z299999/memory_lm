"""Configuration loading for exp0513."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import yaml


@dataclass
class ExperimentConfig:
    """User-facing configuration for exp0513 runs."""

    run_name: str = "exp0513_run"
    task_name: str = "sin_mix"
    seed: int = 42
    epochs: int = 1000
    lambda_value: float = 0.0
    resume_from: str = ""
    input_dim: int = 1
    output_dim: int = 1
    trunk_dims: tuple[int, ...] = (16, 16)
    batch_size: int = 64
    lr_bp: float = 1e-2
    eta_int: float = 1e-4
    gamma: float = 1.0
    coverage_c: int = 3
    dopamine_m_override: int | None = None
    num_train: int = 500
    num_val: int = 200
    num_plot: int = 500
    x_min: float = -6.283185307179586
    x_max: float = 6.283185307179586
    enable_diagnostics: bool = False

    def to_user_dict(self) -> dict[str, object]:
        """Serialize using the user-facing `lambda` key."""
        data = asdict(self)
        data["lambda"] = data.pop("lambda_value")
        return data

    def to_resolved_dict(self) -> dict[str, object]:
        """Serialize resolved values used by the trainer."""
        data = self.to_user_dict()
        if self.resume_from:
            data["resume_from"] = str(Path(self.resume_from).resolve())
        return data


def config_from_user_dict(raw: dict[str, object], base_dir: Path | None = None) -> ExperimentConfig:
    """Build an ExperimentConfig from user-facing keys."""
    defaults = ExperimentConfig()
    valid_keys = set(asdict(defaults).keys()) | {"lambda"}
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    payload = asdict(defaults)
    normalized = dict(raw)
    if "lambda" in normalized:
        payload["lambda_value"] = normalized.pop("lambda")
    payload.update(normalized)

    if payload["resume_from"]:
        resume_path = Path(str(payload["resume_from"])).expanduser()
        if not resume_path.is_absolute():
            anchor = base_dir or Path.cwd()
            resume_path = (anchor / resume_path).resolve()
        payload["resume_from"] = str(resume_path)

    payload["input_dim"] = int(payload["input_dim"])
    payload["output_dim"] = int(payload["output_dim"])
    if payload["input_dim"] <= 0 or payload["output_dim"] <= 0:
        raise ValueError("input_dim and output_dim must be positive integers.")

    trunk_dims = payload.get("trunk_dims")
    if not isinstance(trunk_dims, (list, tuple)) or not trunk_dims:
        raise ValueError("trunk_dims must be a non-empty list of positive integers.")
    payload["trunk_dims"] = tuple(int(dim) for dim in trunk_dims)
    if any(dim <= 0 for dim in payload["trunk_dims"]):
        raise ValueError("Every trunk_dims entry must be a positive integer.")

    payload["coverage_c"] = int(payload["coverage_c"])
    if payload["coverage_c"] <= 0:
        raise ValueError("coverage_c must be a positive integer.")

    dopamine_m_override = payload.get("dopamine_m_override")
    if dopamine_m_override in ("", None):
        payload["dopamine_m_override"] = None
    else:
        payload["dopamine_m_override"] = int(dopamine_m_override)
        if payload["dopamine_m_override"] <= 0:
            raise ValueError("dopamine_m_override must be a positive integer when provided.")

    return ExperimentConfig(**payload)


def load_config_from_yaml(path: Path) -> ExperimentConfig:
    """Load exp0513 config from a yaml file."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_user_dict(raw, base_dir=path.parent)


def dump_config_to_yaml(config: ExperimentConfig, path: Path) -> None:
    """Write a config back to yaml using user-facing keys."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_user_dict(), handle, sort_keys=False)


def copy_config_to_run_dir(source_path: Path, run_dir: Path) -> None:
    """Copy the user-facing config file into the run directory."""
    shutil.copy2(source_path, run_dir / "config.yaml")


def write_resolved_config(config: ExperimentConfig, extra: dict[str, object], output_path: Path) -> None:
    """Write the resolved training configuration as JSON."""
    payload = config.to_resolved_dict()
    payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2))
