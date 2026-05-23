"""Configuration loading for exp0522."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil

import yaml


SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("run_name", "seed", "log_every", "output_root"),
    "model": ("trunk_dims", "activation", "language_dim", "language_readout_coverage", "use_error_input"),
    "task": ("cycle_steps", "pulse_value", "target_kind", "mixed_sin_components"),
    "train": (
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip",
        "sequence_mode",
        "fixed_train_steps",
        "train_phase_mode",
        "message_aux_loss_weight",
        "detach_error_input",
        "carry_error_between_windows",
        "force_zero_error_input",
    ),
    "eval": (
        "eval_steps",
        "long_steps",
        "continuous_eval_steps",
        "eval_phase_mode",
        "eval_conditions",
    ),
    "analysis": (
        "enable_continuous_collapse",
        "checkpoint_epochs",
    ),
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
        "plot_target_color",
        "plot_target_linestyle",
        "plot_target_linewidth",
        "plot_series_linewidth",
        "plot_aux_linewidth",
        "plot_zero_linewidth",
        "plot_prediction_legend_ncols",
        "plot_error_legend_ncols",
        "plot_message_legend_ncols",
        "plot_training_series",
        "plot_rollout_series",
        "plot_error_series",
        "plot_show_message_traces",
        "plot_show_message_norm",
        "plot_show_training_timeline",
        "plot_training_timeline_num_panels",
        "plot_training_timeline_window_steps",
    ),
}


LEGACY_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("train_baseline", "eval_mute_deaf"),
    "train": ("rollout_schedule", "train_steps", "message_refresh"),
    "task": (
        "train_steps",
        "eval_steps",
        "long_steps",
        "continuous_eval_steps",
        "train_phase_mode",
        "eval_phase_mode",
    ),
}

_VALID_EVAL_CONDITIONS = frozenset({"full", "sole_eye", "sole_speech", "neither"})


@dataclass
class ExperimentConfig:
    """User-facing configuration for exp0522 runs."""

    run_name: str = "exp0522_clock_v0"
    seed: int = 42
    epochs: int = 400
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    sequence_mode: str = "reset"
    fixed_train_steps: int = 128
    message_aux_loss_weight: float = 0.0
    detach_error_input: bool = True
    carry_error_between_windows: bool = True
    force_zero_error_input: bool = False
    trunk_dims: tuple[int, ...] = (32,)
    activation: str = "tanh"
    language_dim: int = 4
    language_readout_coverage: int = 1
    use_error_input: bool = False
    cycle_steps: int = 32
    target_kind: str = "sine"
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5))
    eval_steps: int = 128
    long_steps: int = 512
    continuous_eval_steps: int = 512
    eval_conditions: tuple[str, ...] = ("full", "sole_eye")
    enable_continuous_collapse: bool = True
    checkpoint_epochs: tuple[int, ...] = (1, 10, 50, 100, 500, 1000)
    pulse_value: float = 1.0
    train_phase_mode: str = "reset"
    eval_phase_mode: str = "both"
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
    plot_target_color: str = "#8a8a8a"
    plot_target_linestyle: str = "--"
    plot_target_linewidth: float = 1.0
    plot_series_linewidth: float = 1.7
    plot_aux_linewidth: float = 1.5
    plot_zero_linewidth: float = 1.0
    plot_prediction_legend_ncols: int = 4
    plot_error_legend_ncols: int = 3
    plot_message_legend_ncols: int = 2
    plot_training_series: tuple[str, ...] = (
        "full_train",
        "full_val",
    )
    plot_rollout_series: tuple[str, ...] = ("target", "full")
    plot_error_series: tuple[str, ...] = ("full",)
    plot_show_message_traces: bool = True
    plot_show_message_norm: bool = True
    plot_show_training_timeline: bool = True
    plot_training_timeline_num_panels: int = 6
    plot_training_timeline_window_steps: int = 200

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
            "target_kind": self.target_kind,
            "target": _resolved_target_description(self),
            "phase_init": 0.0,
            "omega": omega_from_cycle_steps(self.cycle_steps),
        }
        return payload


def omega_from_cycle_steps(cycle_steps: int) -> float:
    import math

    return (2.0 * math.pi) / float(cycle_steps)


def _resolved_target_description(config: ExperimentConfig) -> str:
    if config.target_kind == "sine":
        return "sin(phi_t)"
    terms = " + ".join(f"{amp}*sin({freq}*phi_t)" for freq, amp in config.mixed_sin_components)
    scale = sum(abs(amp) for _, amp in config.mixed_sin_components)
    return f"({terms}) / {scale}"


def _flatten_user_config(raw: dict[str, object], defaults: ExperimentConfig) -> dict[str, object]:
    """Support grouped yaml while remaining backward compatible with flat keys."""
    valid_keys = set(asdict(defaults).keys())
    valid_sections = set(SECTION_KEYS.keys())
    legacy_keys = {k for keys in LEGACY_SECTION_KEYS.values() for k in keys}
    unknown = set(raw.keys()) - valid_keys - valid_sections - legacy_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    flat_payload: dict[str, object] = {}
    for section_name, section_keys in SECTION_KEYS.items():
        section_payload = raw.get(section_name)
        if section_payload is None:
            continue
        if not isinstance(section_payload, dict):
            raise ValueError(f"Config section {section_name!r} must be a mapping.")
        allowed_keys = set(section_keys) | set(LEGACY_SECTION_KEYS.get(section_name, ()))
        unknown_section_keys = set(section_payload.keys()) - allowed_keys
        if unknown_section_keys:
            raise ValueError(
                f"Unknown keys inside section {section_name!r}: {sorted(unknown_section_keys)}"
            )
        if section_name in {"train", "task"} and "train_steps" in section_payload and "fixed_train_steps" not in section_payload:
            flat_payload["fixed_train_steps"] = section_payload["train_steps"]
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
        "eval_steps",
        "long_steps",
        "continuous_eval_steps",
        "log_every",
        "fixed_train_steps",
        "plot_dpi",
        "plot_short_steps",
        "plot_long_steps",
        "plot_error_steps",
        "plot_message_steps",
        "plot_title_fontsize",
        "plot_prediction_legend_ncols",
        "plot_error_legend_ncols",
        "plot_message_legend_ncols",
        "plot_training_timeline_num_panels",
        "plot_training_timeline_window_steps",
    ):
        payload[key] = int(payload[key])
    for key in (
        "lr",
        "weight_decay",
        "grad_clip",
        "pulse_value",
        "message_aux_loss_weight",
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

    # Drop legacy and removed keys
    payload.pop("rollout_schedule", None)
    payload.pop("train_steps", None)
    payload.pop("message_refresh", None)
    payload.pop("train_baseline", None)
    payload.pop("eval_mute_deaf", None)

    payload["sequence_mode"] = str(payload["sequence_mode"])
    payload["activation"] = str(payload["activation"])
    payload["train_phase_mode"] = str(payload["train_phase_mode"])
    payload["eval_phase_mode"] = str(payload["eval_phase_mode"])
    payload["target_kind"] = str(payload["target_kind"])
    payload["plot_target_color"] = str(payload["plot_target_color"])
    payload["plot_target_linestyle"] = str(payload["plot_target_linestyle"])
    for key in (
        "plot_training_series",
        "plot_rollout_series",
        "plot_error_series",
    ):
        value = payload[key]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a yaml list.")
        payload[key] = tuple(str(item) for item in value)
    checkpoint_epochs = payload["checkpoint_epochs"]
    if not isinstance(checkpoint_epochs, (list, tuple)):
        raise ValueError("checkpoint_epochs must be a yaml list.")
    payload["checkpoint_epochs"] = tuple(int(item) for item in checkpoint_epochs)

    eval_conditions_raw = payload.get("eval_conditions", defaults.eval_conditions)
    if not isinstance(eval_conditions_raw, (list, tuple)):
        raise ValueError("eval_conditions must be a yaml list.")
    payload["eval_conditions"] = tuple(str(item) for item in eval_conditions_raw)

    for key in (
        "use_error_input",
        "enable_continuous_collapse",
        "detach_error_input",
        "carry_error_between_windows",
        "force_zero_error_input",
        "plot_show_message_traces",
        "plot_show_message_norm",
        "plot_show_training_timeline",
    ):
        value = payload[key]
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false.")
        payload[key] = bool(value)

    if payload["epochs"] <= 0:
        raise ValueError("epochs must be positive.")
    if payload["message_aux_loss_weight"] < 0.0:
        raise ValueError("message_aux_loss_weight must be >= 0.")
    if payload["sequence_mode"] not in {"reset", "continuous_window"}:
        raise ValueError("sequence_mode must be either 'reset' or 'continuous_window'.")
    if payload["activation"] not in {"tanh", "relu", "leaky_relu"}:
        raise ValueError("activation must be 'tanh', 'relu', or 'leaky_relu'.")
    if payload["fixed_train_steps"] <= 0:
        raise ValueError("fixed_train_steps must be positive.")
    if payload["train_phase_mode"] not in {"reset", "continuous"}:
        raise ValueError("train_phase_mode must be either 'reset' or 'continuous'.")
    if payload["eval_phase_mode"] not in {"reset", "continuous", "both"}:
        raise ValueError("eval_phase_mode must be 'reset', 'continuous', or 'both'.")
    if payload["language_dim"] < 0:
        raise ValueError("language_dim must be >= 0.")
    if payload["language_dim"] > 0:
        if payload["language_readout_coverage"] <= 0:
            raise ValueError("language_readout_coverage must be positive.")
        if payload["language_readout_coverage"] > payload["language_dim"]:
            raise ValueError("language_readout_coverage must be <= language_dim.")
    if payload["cycle_steps"] <= 1:
        raise ValueError("cycle_steps must be greater than 1.")
    if payload["target_kind"] not in {"sine", "mixed_sin"}:
        raise ValueError("target_kind must be either 'sine' or 'mixed_sin'.")
    if payload["eval_steps"] <= 0 or payload["long_steps"] <= 0 or payload["continuous_eval_steps"] <= 0:
        raise ValueError("eval_steps, long_steps, and continuous_eval_steps must be positive.")
    if any(epoch <= 0 for epoch in payload["checkpoint_epochs"]):
        raise ValueError("checkpoint_epochs entries must all be positive.")

    invalid_conds = [c for c in payload["eval_conditions"] if c not in _VALID_EVAL_CONDITIONS]
    if invalid_conds:
        raise ValueError(f"eval_conditions contains unknown values: {invalid_conds}. Valid: {sorted(_VALID_EVAL_CONDITIONS)}")
    if "full" not in payload["eval_conditions"]:
        raise ValueError("eval_conditions must contain 'full'.")

    raw_components = payload["mixed_sin_components"]
    if not isinstance(raw_components, (list, tuple)) or not raw_components:
        raise ValueError("mixed_sin_components must be a non-empty list of [freq, amplitude] pairs.")
    parsed = []
    for item in raw_components:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each mixed_sin_components entry must be a [freq, amplitude] pair.")
        freq, amp = float(item[0]), float(item[1])
        if freq <= 0:
            raise ValueError("mixed_sin_components: each frequency must be positive.")
        if not math.isfinite(amp):
            raise ValueError("mixed_sin_components: each amplitude must be finite.")
        parsed.append((freq, amp))
    payload["mixed_sin_components"] = tuple(parsed)
    if payload["fixed_train_steps"] > payload["long_steps"]:
        raise ValueError("fixed_train_steps must be <= long_steps.")
    if payload["fixed_train_steps"] > payload["continuous_eval_steps"]:
        raise ValueError("fixed_train_steps must be <= continuous_eval_steps.")
    if payload["eval_steps"] < payload["cycle_steps"]:
        raise ValueError("eval_steps must be at least one cycle long.")
    if payload["long_steps"] < payload["eval_steps"]:
        raise ValueError("long_steps must be greater than or equal to eval_steps.")
    if payload["sequence_mode"] == "continuous_window" and payload["train_phase_mode"] != "continuous":
        raise ValueError("continuous_window mode requires train_phase_mode='continuous'.")
    if payload["sequence_mode"] == "reset" and payload["train_phase_mode"] != "reset":
        raise ValueError("reset mode requires train_phase_mode='reset'.")
    legacy_rollout_schedule = raw.get("rollout_schedule")
    if legacy_rollout_schedule is None and isinstance(raw.get("train"), dict):
        legacy_rollout_schedule = raw["train"].get("rollout_schedule")
    if legacy_rollout_schedule is not None and str(legacy_rollout_schedule) != "fixed":
        raise ValueError("curriculum training has been removed; only fixed_train_steps is supported now.")
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
    if not payload["plot_target_color"].strip():
        raise ValueError("plot_target_color must be a non-empty string.")
    if not payload["plot_target_linestyle"].strip():
        raise ValueError("plot_target_linestyle must be a non-empty string.")
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
    if payload["plot_long_steps"] > max(payload["long_steps"], payload["continuous_eval_steps"]):
        raise ValueError("plot_long_steps must be <= max(long_steps, continuous_eval_steps).")
    if payload["plot_error_steps"] > payload["eval_steps"]:
        raise ValueError("plot_error_steps must be <= eval_steps.")
    if payload["plot_message_steps"] > payload["eval_steps"]:
        raise ValueError("plot_message_steps must be <= eval_steps.")
    if payload["plot_training_timeline_num_panels"] <= 0:
        raise ValueError("plot_training_timeline_num_panels must be positive.")
    if payload["plot_training_timeline_window_steps"] <= 0:
        raise ValueError("plot_training_timeline_window_steps must be positive.")
    allowed_training_series = {
        "full_train",
        "full_val",
        "baseline_train",
        "baseline_val",
    }
    allowed_rollout_series = {"target"} | _VALID_EVAL_CONDITIONS
    allowed_error_series = _VALID_EVAL_CONDITIONS
    for key, allowed in (
        ("plot_training_series", allowed_training_series),
        ("plot_rollout_series", allowed_rollout_series),
        ("plot_error_series", allowed_error_series),
    ):
        invalid = [item for item in payload[key] if item not in allowed]
        if invalid:
            raise ValueError(f"{key} contains unknown entries: {invalid}")
    if len(payload["plot_training_series"]) == 0:
        raise ValueError("plot_training_series must contain at least one curve.")
    if len(payload["plot_rollout_series"]) == 0:
        raise ValueError("plot_rollout_series must contain at least one curve.")

    return ExperimentConfig(**payload)


def load_config_from_yaml(path: Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_user_dict(raw)


def copy_config_to_run_dir(source_path: Path, run_dir: Path) -> None:
    shutil.copy2(source_path, run_dir / "config.yaml")


def write_resolved_config(config: ExperimentConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(config.to_resolved_dict(), indent=2))
