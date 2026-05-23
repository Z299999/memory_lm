"""Configuration loading for exp0522."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import yaml


SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("run_name", "seed", "log_every", "output_root"),
    "model": ("trunk_dims", "language_dim", "language_readout_coverage"),
    "task": ("cycle_steps", "train_steps", "eval_steps", "long_steps", "pulse_value"),
    "train": ("epochs", "lr", "weight_decay", "grad_clip"),
    "plot": (
        "plot_dpi",
        "plot_training_fig_width",
        "plot_training_fig_height",
        "plot_diag_fig_width",
        "plot_diag_fig_height",
        "plot_short_steps",
        "plot_long_steps",
        "plot_error_steps",
        "plot_message_steps",
        "plot_grid_alpha",
        "plot_title_fontsize",
        "plot_target_linewidth",
        "plot_series_linewidth",
        "plot_aux_linewidth",
        "plot_zero_linewidth",
        "plot_prediction_legend_ncols",
        "plot_error_legend_ncols",
        "plot_message_legend_ncols",
    ),
}


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
    language_readout_coverage: int = 1
    cycle_steps: int = 32
    train_steps: int = 128
    eval_steps: int = 128
    long_steps: int = 512
    pulse_value: float = 1.0
    log_every: int = 25
    output_root: str = "runs"
    plot_dpi: int = 160
    plot_training_fig_width: float = 8.0
    plot_training_fig_height: float = 5.0
    plot_diag_fig_width: float = 12.0
    plot_diag_fig_height: float = 15.0
    plot_short_steps: int = 128
    plot_long_steps: int = 512
    plot_error_steps: int = 128
    plot_message_steps: int = 128
    plot_grid_alpha: float = 0.25
    plot_title_fontsize: int = 14
    plot_target_linewidth: float = 2.1
    plot_series_linewidth: float = 1.7
    plot_aux_linewidth: float = 1.5
    plot_zero_linewidth: float = 1.0
    plot_prediction_legend_ncols: int = 4
    plot_error_legend_ncols: int = 3
    plot_message_legend_ncols: int = 2

    def to_user_dict(self) -> dict[str, object]:
        flat = asdict(self)
        grouped: dict[str, dict[str, object]] = {}
        for section_name, section_keys in SECTION_KEYS.items():
            grouped[section_name] = {
                key: flat[key]
                for key in section_keys
            }
        return grouped

    def to_resolved_dict(self) -> dict[str, object]:
        payload = self.to_user_dict()
        payload["run"]["output_root"] = str(payload["run"]["output_root"])
        payload["resolved"] = {
            "message_init": 0.0,
            "target": "sin(phi_t)",
            "phase_init": 0.0,
            "omega": omega_from_cycle_steps(self.cycle_steps),
        }
        return payload


def omega_from_cycle_steps(cycle_steps: int) -> float:
    import math

    return (2.0 * math.pi) / float(cycle_steps)


def _flatten_user_config(raw: dict[str, object], defaults: ExperimentConfig) -> dict[str, object]:
    """Support grouped yaml while remaining backward compatible with flat keys."""
    valid_keys = set(asdict(defaults).keys())
    valid_sections = set(SECTION_KEYS.keys())
    unknown = set(raw.keys()) - valid_keys - valid_sections
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    flat_payload: dict[str, object] = {}
    for section_name, section_keys in SECTION_KEYS.items():
        section_payload = raw.get(section_name)
        if section_payload is None:
            continue
        if not isinstance(section_payload, dict):
            raise ValueError(f"Config section {section_name!r} must be a mapping.")
        unknown_section_keys = set(section_payload.keys()) - set(section_keys)
        if unknown_section_keys:
            raise ValueError(
                f"Unknown keys inside section {section_name!r}: {sorted(unknown_section_keys)}"
            )
        flat_payload.update(section_payload)

    # Allow flat keys as backward-compatible overrides.
    for key in valid_keys:
        if key in raw:
            flat_payload[key] = raw[key]
    return flat_payload


def config_from_user_dict(raw: dict[str, object]) -> ExperimentConfig:
    defaults = ExperimentConfig()
    payload = asdict(defaults)
    payload.update(_flatten_user_config(raw, defaults))

    trunk_dims = payload.get("trunk_dims")
    if not isinstance(trunk_dims, (list, tuple)) or not trunk_dims:
        raise ValueError("trunk_dims must be a non-empty list of positive integers.")
    payload["trunk_dims"] = tuple(int(dim) for dim in trunk_dims)
    if any(dim <= 0 for dim in payload["trunk_dims"]):
        raise ValueError("Each trunk_dims entry must be positive.")

    for key in (
        "seed",
        "epochs",
        "language_dim",
        "language_readout_coverage",
        "cycle_steps",
        "train_steps",
        "eval_steps",
        "long_steps",
        "log_every",
        "plot_dpi",
        "plot_short_steps",
        "plot_long_steps",
        "plot_error_steps",
        "plot_message_steps",
        "plot_title_fontsize",
        "plot_prediction_legend_ncols",
        "plot_error_legend_ncols",
        "plot_message_legend_ncols",
    ):
        payload[key] = int(payload[key])
    for key in (
        "lr",
        "weight_decay",
        "grad_clip",
        "pulse_value",
        "plot_training_fig_width",
        "plot_training_fig_height",
        "plot_diag_fig_width",
        "plot_diag_fig_height",
        "plot_grid_alpha",
        "plot_target_linewidth",
        "plot_series_linewidth",
        "plot_aux_linewidth",
        "plot_zero_linewidth",
    ):
        payload[key] = float(payload[key])

    if payload["epochs"] <= 0:
        raise ValueError("epochs must be positive.")
    if payload["language_dim"] <= 0:
        raise ValueError("language_dim must be positive.")
    if payload["language_readout_coverage"] <= 0:
        raise ValueError("language_readout_coverage must be positive.")
    if payload["language_readout_coverage"] > payload["language_dim"]:
        raise ValueError("language_readout_coverage must be <= language_dim.")
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
    if payload["plot_training_fig_width"] <= 0 or payload["plot_training_fig_height"] <= 0:
        raise ValueError("plot_training_fig_width and plot_training_fig_height must be positive.")
    if payload["plot_diag_fig_width"] <= 0 or payload["plot_diag_fig_height"] <= 0:
        raise ValueError("plot_diag_fig_width and plot_diag_fig_height must be positive.")
    if payload["plot_short_steps"] <= 0 or payload["plot_long_steps"] <= 0:
        raise ValueError("plot_short_steps and plot_long_steps must be positive.")
    if payload["plot_error_steps"] <= 0 or payload["plot_message_steps"] <= 0:
        raise ValueError("plot_error_steps and plot_message_steps must be positive.")
    if payload["plot_grid_alpha"] < 0.0 or payload["plot_grid_alpha"] > 1.0:
        raise ValueError("plot_grid_alpha must be in [0, 1].")
    if payload["plot_title_fontsize"] <= 0:
        raise ValueError("plot_title_fontsize must be positive.")
    if payload["plot_target_linewidth"] <= 0 or payload["plot_series_linewidth"] <= 0:
        raise ValueError("plot_target_linewidth and plot_series_linewidth must be positive.")
    if payload["plot_aux_linewidth"] <= 0 or payload["plot_zero_linewidth"] <= 0:
        raise ValueError("plot_aux_linewidth and plot_zero_linewidth must be positive.")
    if payload["plot_prediction_legend_ncols"] <= 0:
        raise ValueError("plot_prediction_legend_ncols must be positive.")
    if payload["plot_error_legend_ncols"] <= 0 or payload["plot_message_legend_ncols"] <= 0:
        raise ValueError("plot_error_legend_ncols and plot_message_legend_ncols must be positive.")
    if payload["plot_short_steps"] > payload["eval_steps"]:
        raise ValueError("plot_short_steps must be <= eval_steps.")
    if payload["plot_long_steps"] > payload["long_steps"]:
        raise ValueError("plot_long_steps must be <= long_steps.")
    if payload["plot_error_steps"] > payload["eval_steps"]:
        raise ValueError("plot_error_steps must be <= eval_steps.")
    if payload["plot_message_steps"] > payload["eval_steps"]:
        raise ValueError("plot_message_steps must be <= eval_steps.")

    return ExperimentConfig(**payload)


def load_config_from_yaml(path: Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_user_dict(raw)


def copy_config_to_run_dir(source_path: Path, run_dir: Path) -> None:
    shutil.copy2(source_path, run_dir / "config.yaml")


def write_resolved_config(config: ExperimentConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(config.to_resolved_dict(), indent=2))
