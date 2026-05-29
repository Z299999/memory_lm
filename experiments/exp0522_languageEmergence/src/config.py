"""Configuration loading for exp0522."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import shutil

import yaml


SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("run_name", "seed", "log_every", "output_root"),
    "model": ("trunk_dims", "activation", "language_dim", "language_readout_coverage", "use_error_input", "use_residual", "use_dense", "language_readout_all_layers", "message_carry_mode", "language_readout_trainable", "readout_nonlinearity"),
    "task": ("cycle_steps", "pulse_value", "target_kind", "mixed_sin_components", "prediction_target"),
    "train": (
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip",
        "sequence_mode",
        "train_window_schedule",
        "train_phase_mode",
        "error_degrade",
        "message_aux_loss_weight",
        "detach_error_input",
        "carry_error_between_windows",
        "force_zero_error_input",
        "train_loss_tail_steps",
        "train_loss_space",
        "language_readout_norm_penalty",
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
        "plot_show_message_traces",
        "plot_show_message_norm",
        "plot_rollout_top_mode",
        "plot_aux_horizon",
        "plot_show_training_timeline",
        "plot_training_timeline_num_panels",
        "plot_training_timeline_ncols",
        "plot_training_timeline_window_steps",
        "plot_training_timeline_fig_width",
        "plot_training_timeline_shared_ylim",
        "plot_training_timeline_ylim",
    ),
}


LEGACY_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "run": ("train_baseline", "eval_mute_deaf"),
    "train": ("rollout_schedule", "train_steps", "fixed_train_steps", "message_refresh"),
    "task": (
        "train_steps",
        "eval_steps",
        "long_steps",
        "continuous_eval_steps",
        "train_phase_mode",
        "eval_phase_mode",
    ),
    "eval": (
        "eval_late_blind_step",
        "eval_late_mute_step",
        "eval_blink_blind_start",
        "eval_blink_blind_end",
        "eval_stutter_mute_start",
        "eval_stutter_mute_end",
    ),
}

_VALID_EVAL_CONDITIONS = frozenset({"full", "sole_eye", "sole_speech", "neither", "late_blind", "late_mute", "blink", "dim", "stutter"})

# Expected number of integer params per condition base name.
_CONDITION_PARAM_COUNTS: dict[str, int] = {
    "full": 0, "sole_eye": 0, "sole_speech": 0, "neither": 0,
    "late_blind": 1, "late_mute": 1,
    "blink": 2, "dim": 4, "stutter": 2,
}

_PREDICTION_TARGET_ALIASES = {
    "y": "y",
    "v": "v",
    "velocity": "v",
    "a": "a",
    "acceleration": "a",
}


def normalize_prediction_target(value: object) -> str:
    """Normalize public target names to the compact y/v/a API."""
    raw = str(value).strip()
    try:
        return _PREDICTION_TARGET_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "prediction_target must be 'y', 'v', or 'a' "
            "(legacy aliases 'velocity' and 'acceleration' are also accepted)."
        ) from exc


def parse_condition(s: str) -> tuple[str, tuple[int, ...]]:
    """Parse 'blink(40,100)' → ('blink', (40, 100)); 'full' → ('full', ())."""
    import re
    m = re.match(r"^(\w+)\(([^)]*)\)$", s.strip())
    if m:
        base = m.group(1)
        raw = m.group(2).strip()
        params = tuple(int(x.strip()) for x in raw.split(",") if x.strip()) if raw else ()
        return base, params
    return s.strip(), ()


@dataclass
class TrainWindowSchedule:
    mode: str
    min_steps: int
    max_steps: int
    threshold: float | None = None


@dataclass
class ErrorDegradeSchedule:
    mode: str
    rate: float = 0.0
    min_steps: int = 0
    max_steps: int = 0
    pct: int = 100
    ramp_steps: int = 0
    start_step: int | None = None
    end_step: int | None = None
    min_pct: int | None = None


def parse_train_window_schedule(spec: str) -> TrainWindowSchedule:
    raw = str(spec).strip()
    fixed_match = re.fullmatch(r"fixed\((\d+)\)", raw)
    if fixed_match:
        value = int(fixed_match.group(1))
        if value <= 0:
            raise ValueError("train_window_schedule fixed(L) requires L > 0.")
        return TrainWindowSchedule(mode="fixed", min_steps=value, max_steps=value)
    uniform_match = re.fullmatch(r"random_uniform\((\d+),\s*(\d+)\)", raw)
    if uniform_match:
        lower = int(uniform_match.group(1))
        upper = int(uniform_match.group(2))
        if lower <= 0 or upper <= 0:
            raise ValueError("train_window_schedule random_uniform(a,b) requires a,b > 0.")
        if lower > upper:
            raise ValueError("train_window_schedule random_uniform(a,b) requires a <= b.")
        return TrainWindowSchedule(mode="random_uniform", min_steps=lower, max_steps=upper)
    event_match = re.fullmatch(
        r"event_triggered\(([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?),\s*(\d+),\s*(\d+)\)",
        raw,
    )
    if event_match:
        threshold = float(event_match.group(1))
        min_steps = int(event_match.group(2))
        max_steps = int(event_match.group(3))
        if threshold <= 0.0:
            raise ValueError("train_window_schedule event_triggered(threshold,min,max) requires threshold > 0.")
        if min_steps <= 0 or max_steps <= 0:
            raise ValueError("train_window_schedule event_triggered(threshold,min,max) requires min,max > 0.")
        if min_steps > max_steps:
            raise ValueError("train_window_schedule event_triggered(threshold,min,max) requires min <= max.")
        return TrainWindowSchedule(
            mode="event_triggered",
            min_steps=min_steps,
            max_steps=max_steps,
            threshold=threshold,
        )
    raise ValueError(
        "train_window_schedule must be 'fixed(L)', 'random_uniform(a,b)', or "
        "'event_triggered(threshold,min_steps,max_steps)'."
    )


def train_window_bounds(spec: str) -> tuple[int, int]:
    schedule = parse_train_window_schedule(spec)
    return schedule.min_steps, schedule.max_steps


def train_window_reference_steps(spec: str) -> int:
    schedule = parse_train_window_schedule(spec)
    if schedule.mode == "fixed":
        return schedule.min_steps
    return int(round((schedule.min_steps + schedule.max_steps) / 2.0))


def parse_error_degrade(spec: str) -> ErrorDegradeSchedule:
    raw = str(spec).strip()
    if raw == "none":
        return ErrorDegradeSchedule(mode="none")
    dim_match = re.fullmatch(
        r"dim\(([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
        raw,
    )
    if dim_match:
        rate = float(dim_match.group(1))
        min_steps = int(dim_match.group(2))
        max_steps = int(dim_match.group(3))
        pct = int(dim_match.group(4))
        ramp_steps = int(dim_match.group(5))
        if rate < 0.0 or rate > 1.0:
            raise ValueError("error_degrade dim(rate,min,max,pct,ramp) requires 0 <= rate <= 1.")
        if min_steps <= 0 or max_steps <= 0:
            raise ValueError("error_degrade dim(rate,min,max,pct,ramp) requires min,max > 0.")
        if min_steps > max_steps:
            raise ValueError("error_degrade dim(rate,min,max,pct,ramp) requires min <= max.")
        if pct < 0 or pct > 100:
            raise ValueError("error_degrade dim(rate,min,max,pct,ramp) requires 0 <= pct <= 100.")
        if ramp_steps < 0:
            raise ValueError("error_degrade dim(rate,min,max,pct,ramp) requires ramp >= 0.")
        return ErrorDegradeSchedule(
            mode="dim",
            rate=rate,
            min_steps=min_steps,
            max_steps=max_steps,
            pct=pct,
            ramp_steps=ramp_steps,
        )
    tail_match = re.fullmatch(r"tail_dim\((\d+),\s*(\d+),\s*(\d+)\)", raw)
    if tail_match:
        start_step = int(tail_match.group(1))
        end_step = int(tail_match.group(2))
        min_pct = int(tail_match.group(3))
        if end_step <= start_step:
            raise ValueError("error_degrade tail_dim(start,end,min_pct) requires end > start.")
        if min_pct < 0 or min_pct > 100:
            raise ValueError("error_degrade tail_dim(start,end,min_pct) requires 0 <= min_pct <= 100.")
        return ErrorDegradeSchedule(
            mode="tail_dim",
            start_step=start_step,
            end_step=end_step,
            min_pct=min_pct,
        )
    raise ValueError(
        "error_degrade must be 'none', 'dim(rate,min_steps,max_steps,pct,ramp_steps)', "
        "or 'tail_dim(start_step,end_step,min_pct)'."
    )


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
    train_window_schedule: str = "fixed(128)"
    error_degrade: str = "none"
    message_aux_loss_weight: float = 0.0
    detach_error_input: bool = True
    carry_error_between_windows: bool = True
    force_zero_error_input: bool = False
    train_loss_tail_steps: int | None = None
    train_loss_space: str = "y"
    language_readout_norm_penalty: float = 0.0
    trunk_dims: tuple[int, ...] = (32,)
    activation: str = "tanh"
    language_dim: int = 4
    language_readout_coverage: int = 1
    use_error_input: bool = False
    use_residual: bool = True
    use_dense: bool = False
    language_readout_all_layers: bool = False
    language_readout_trainable: bool = False
    readout_nonlinearity: str = "none"
    message_carry_mode: str = "identity"
    cycle_steps: int = 32
    target_kind: str = "sine"
    mixed_sin_components: tuple[tuple[float, float], ...] = ((1.0, 1.0), (2.0, 0.5))
    prediction_target: str = "y"
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
    plot_show_message_traces: bool = True
    plot_show_message_norm: bool = True
    plot_rollout_top_mode: str = "match_train"
    plot_aux_horizon: str = "long"
    plot_show_training_timeline: bool = True
    plot_training_timeline_num_panels: int = 6
    plot_training_timeline_ncols: int = 2
    plot_training_timeline_window_steps: int = 200
    plot_training_timeline_fig_width: float = 14.0
    plot_training_timeline_shared_ylim: bool = False
    plot_training_timeline_ylim: tuple[float, float] | None = None

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
        train_schedule = parse_train_window_schedule(self.train_window_schedule)
        error_degrade = parse_error_degrade(self.error_degrade)
        payload["resolved"] = {
            "message_init": 0.0,
            "use_language_resolved": self.language_dim > 0,
            "target_kind": self.target_kind,
            "prediction_target": self.prediction_target,
            "raw_prediction_space": self.prediction_target,
            "reported_prediction_space": "y",
            "target": _resolved_target_description(self),
            "train_window_schedule": self.train_window_schedule,
            "resolved_train_window_mode": train_schedule.mode,
            "resolved_train_window_min": train_schedule.min_steps,
            "resolved_train_window_max": train_schedule.max_steps,
            "resolved_event_trigger_threshold": train_schedule.threshold,
            "resolved_train_window_reference_steps": train_window_reference_steps(self.train_window_schedule),
            "error_degrade": self.error_degrade,
            "resolved_error_degrade_mode": error_degrade.mode,
            "resolved_error_degrade_rate": error_degrade.rate,
            "resolved_error_degrade_min": error_degrade.min_steps,
            "resolved_error_degrade_max": error_degrade.max_steps,
            "resolved_error_degrade_pct": error_degrade.pct,
            "resolved_error_degrade_ramp_steps": error_degrade.ramp_steps,
            "resolved_error_degrade_start_step": error_degrade.start_step,
            "resolved_error_degrade_end_step": error_degrade.end_step,
            "resolved_error_degrade_min_pct": error_degrade.min_pct,
            "phase_init": 0.0,
            "omega": omega_from_cycle_steps(self.cycle_steps),
        }
        return payload


def omega_from_cycle_steps(cycle_steps: int) -> float:
    import math

    return (2.0 * math.pi) / float(cycle_steps)


def _resolved_target_description(config: ExperimentConfig) -> str:
    if config.target_kind == "sine":
        waveform = "sin(phi_t)"
    else:
        terms = " + ".join(f"{amp}*sin({freq}*phi_t)" for freq, amp in config.mixed_sin_components)
        scale = sum(abs(amp) for _, amp in config.mixed_sin_components)
        waveform = f"({terms}) / {scale}"
    if config.prediction_target == "y":
        return waveform
    if config.prediction_target == "v":
        return f"{waveform}_t - {waveform}_{{t-1}}"
    if config.prediction_target == "a":
        return f"({waveform}_t - {waveform}_{{t-1}}) - ({waveform}_{{t-1}} - {waveform}_{{t-2}})"
    raise ValueError(f"Unsupported prediction_target: {config.prediction_target!r}")


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
        if section_name in {"train", "task"} and "train_steps" in section_payload and "train_window_schedule" not in section_payload:
            flat_payload["train_window_schedule"] = f"fixed({int(section_payload['train_steps'])})"
        if section_name == "train" and "fixed_train_steps" in section_payload and "train_window_schedule" not in section_payload:
            flat_payload["train_window_schedule"] = f"fixed({int(section_payload['fixed_train_steps'])})"
        flat_payload.update(section_payload)

    # Allow flat keys as backward-compatible overrides.
    for key in valid_keys:
        if key in raw:
            flat_payload[key] = raw[key]
    if "fixed_train_steps" in raw and "train_window_schedule" not in flat_payload:
        flat_payload["train_window_schedule"] = f"fixed({int(raw['fixed_train_steps'])})"
    return flat_payload


def config_from_user_dict(raw: dict[str, object]) -> ExperimentConfig:
    defaults = ExperimentConfig()
    payload = asdict(defaults)
    payload.update(_flatten_user_config(raw, defaults))

    trunk_dims = payload.get("trunk_dims")
    if isinstance(trunk_dims, str):
        # shorthand: "16x8" or "[16]*8" → (16,)*8
        import re as _re
        _s = trunk_dims.strip()
        _m = _re.fullmatch(r"(\d+)x(\d+)", _s) or _re.fullmatch(r"\[(\d+)\]\*(\d+)", _s)
        if not _m:
            raise ValueError("trunk_dims string shorthand must be 'WxD' or '[W]*D' (e.g. '16x8' or '[16]*8').")
        trunk_dims = [int(_m.group(1))] * int(_m.group(2))
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
        "plot_training_timeline_ncols",
        "plot_training_timeline_window_steps",
    ):
        payload[key] = int(payload[key])
    for key in (
        "lr",
        "weight_decay",
        "grad_clip",
        "pulse_value",
        "message_aux_loss_weight",
        "language_readout_norm_penalty",
        "plot_training_fig_width",
        "plot_training_fig_height",
        "plot_diag_fig_width",
        "plot_diag_fig_height",
        "plot_grid_alpha",
        "plot_target_linewidth",
        "plot_series_linewidth",
        "plot_aux_linewidth",
        "plot_zero_linewidth",
        "plot_training_timeline_fig_width",
    ):
        payload[key] = float(payload[key])

    # Drop legacy and removed keys
    payload.pop("rollout_schedule", None)
    payload.pop("train_steps", None)
    payload.pop("fixed_train_steps", None)
    payload.pop("message_refresh", None)
    payload.pop("train_baseline", None)
    payload.pop("eval_mute_deaf", None)
    payload.pop("eval_late_blind_step", None)
    payload.pop("eval_late_mute_step", None)
    payload.pop("eval_blink_blind_start", None)
    payload.pop("eval_blink_blind_end", None)
    payload.pop("eval_stutter_mute_start", None)
    payload.pop("eval_stutter_mute_end", None)
    payload["sequence_mode"] = str(payload["sequence_mode"])
    payload["activation"] = str(payload["activation"])
    payload["message_carry_mode"] = str(payload["message_carry_mode"])
    payload["train_phase_mode"] = str(payload["train_phase_mode"])
    payload["eval_phase_mode"] = str(payload["eval_phase_mode"])
    payload["target_kind"] = str(payload["target_kind"])
    payload["prediction_target"] = normalize_prediction_target(payload["prediction_target"])
    payload["train_loss_space"] = str(payload["train_loss_space"])
    payload["train_window_schedule"] = str(payload["train_window_schedule"])
    payload["error_degrade"] = str(payload["error_degrade"])
    payload["plot_rollout_top_mode"] = str(payload["plot_rollout_top_mode"])
    payload["plot_aux_horizon"] = str(payload["plot_aux_horizon"])
    payload["plot_target_color"] = str(payload["plot_target_color"])
    payload["plot_target_linestyle"] = str(payload["plot_target_linestyle"])
    for key in ("plot_training_series",):
        value = payload[key]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} must be a yaml list.")
        payload[key] = tuple(str(item) for item in value)
    checkpoint_epochs = payload["checkpoint_epochs"]
    if not isinstance(checkpoint_epochs, (list, tuple)):
        raise ValueError("checkpoint_epochs must be a yaml list.")
    payload["checkpoint_epochs"] = tuple(int(item) for item in checkpoint_epochs)

    tail_raw = payload.get("train_loss_tail_steps", None)
    if tail_raw is None:
        payload["train_loss_tail_steps"] = None
    else:
        v = int(tail_raw)
        if v <= 0:
            raise ValueError("train_loss_tail_steps must be a positive integer or null.")
        payload["train_loss_tail_steps"] = v

    ylim_raw = payload.get("plot_training_timeline_ylim", None)
    if ylim_raw is None:
        payload["plot_training_timeline_ylim"] = None
    elif isinstance(ylim_raw, (list, tuple)) and len(ylim_raw) == 2:
        payload["plot_training_timeline_ylim"] = (float(ylim_raw[0]), float(ylim_raw[1]))
    else:
        raise ValueError("plot_training_timeline_ylim must be null or [ymin, ymax].")

    eval_conditions_raw = payload.get("eval_conditions", defaults.eval_conditions)
    if not isinstance(eval_conditions_raw, (list, tuple)):
        raise ValueError("eval_conditions must be a yaml list.")
    payload["eval_conditions"] = tuple(str(item) for item in eval_conditions_raw)

    for key in (
        "use_error_input",
        "use_residual",
        "use_dense",
        "language_readout_all_layers",
        "language_readout_trainable",
        "enable_continuous_collapse",
        "detach_error_input",
        "carry_error_between_windows",
        "force_zero_error_input",
        "plot_show_message_traces",
        "plot_show_message_norm",
        "plot_show_training_timeline",
        "plot_training_timeline_shared_ylim",
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
    if payload["train_loss_space"] not in {"raw", "y"}:
        raise ValueError("train_loss_space must be 'raw' or 'y'.")
    if payload["activation"] not in {"tanh", "relu", "leaky_relu"}:
        raise ValueError("activation must be 'tanh', 'relu', or 'leaky_relu'.")
    payload["readout_nonlinearity"] = str(payload["readout_nonlinearity"])
    if payload["readout_nonlinearity"] not in {"none", "tanh"}:
        raise ValueError("readout_nonlinearity must be 'none' or 'tanh'.")
    if payload["message_carry_mode"] not in {"identity", "learnable_diagonal", "learnable_matrix"}:
        raise ValueError("message_carry_mode must be 'identity', 'learnable_diagonal', or 'learnable_matrix'.")
    window_schedule = parse_train_window_schedule(payload["train_window_schedule"])
    parse_error_degrade(payload["error_degrade"])
    window_min = window_schedule.min_steps
    window_max = window_schedule.max_steps
    if payload["train_phase_mode"] not in {"reset", "continuous"}:
        raise ValueError("train_phase_mode must be either 'reset' or 'continuous'.")
    if payload["eval_phase_mode"] not in {"reset", "continuous", "both"}:
        raise ValueError("eval_phase_mode must be 'reset', 'continuous', or 'both'.")
    if payload["plot_rollout_top_mode"] not in {"match_train", "match_eval", "all_available"}:
        raise ValueError("plot_rollout_top_mode must be 'match_train', 'match_eval', or 'all_available'.")
    if payload["plot_aux_horizon"] not in {"short", "long", "both"}:
        raise ValueError("plot_aux_horizon must be 'short', 'long', or 'both'.")
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
    if payload["prediction_target"] not in {"y", "v", "a"}:
        raise ValueError("prediction_target must be 'y', 'v', or 'a'.")
    if payload["eval_steps"] <= 0 or payload["long_steps"] <= 0 or payload["continuous_eval_steps"] <= 0:
        raise ValueError("eval_steps, long_steps, and continuous_eval_steps must be positive.")
    if any(epoch <= 0 for epoch in payload["checkpoint_epochs"]):
        raise ValueError("checkpoint_epochs entries must all be positive.")

    _base_names_seen: set[str] = set()
    for _cond_str in payload["eval_conditions"]:
        _base, _params = parse_condition(_cond_str)
        if _base not in _VALID_EVAL_CONDITIONS:
            raise ValueError(
                f"eval_conditions contains unknown base '{_base}' in '{_cond_str}'. "
                f"Valid bases: {sorted(_VALID_EVAL_CONDITIONS)}"
            )
        _expected = _CONDITION_PARAM_COUNTS[_base]
        if len(_params) != _expected:
            raise ValueError(
                f"eval_conditions: '{_cond_str}' expects {_expected} param(s) for '{_base}', "
                f"got {len(_params)}."
            )
        if _base in ("late_blind", "late_mute") and _params[0] <= 0:
            raise ValueError(f"eval_conditions: transition step in '{_cond_str}' must be positive.")
        if _base in ("blink", "stutter"):
            if _params[0] < 0:
                raise ValueError(f"eval_conditions: loss_start in '{_cond_str}' must be >= 0.")
            if _params[1] <= _params[0]:
                raise ValueError(f"eval_conditions: loss_end must be > loss_start in '{_cond_str}'.")
        if _base == "dim":
            if _params[0] < 0:
                raise ValueError(f"eval_conditions: start in '{_cond_str}' must be >= 0.")
            if _params[1] <= _params[0]:
                raise ValueError(f"eval_conditions: ramp_end must be > start in '{_cond_str}'.")
            if _params[2] <= _params[1]:
                raise ValueError(f"eval_conditions: end must be > ramp_end in '{_cond_str}'.")
            if _params[3] < 0 or _params[3] > 100:
                raise ValueError(f"eval_conditions: pct in '{_cond_str}' must satisfy 0 <= pct <= 100.")
        _base_names_seen.add(_base)
    if "full" not in _base_names_seen:
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
        raise ValueError("curriculum training has been removed; use train_window_schedule instead.")
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
    if payload["plot_training_timeline_ncols"] <= 0:
        raise ValueError("plot_training_timeline_ncols must be positive.")
    if payload["plot_training_timeline_window_steps"] <= 0:
        raise ValueError("plot_training_timeline_window_steps must be positive.")
    allowed_training_series = {
        "full_train",
        "full_val",
        "baseline_train",
        "baseline_val",
    }
    for key, allowed in (("plot_training_series", allowed_training_series),):
        invalid = [item for item in payload[key] if item not in allowed]
        if invalid:
            raise ValueError(f"{key} contains unknown entries: {invalid}")
    if len(payload["plot_training_series"]) == 0:
        raise ValueError("plot_training_series must contain at least one curve.")

    return ExperimentConfig(**payload)


def load_config_from_yaml(path: Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_user_dict(raw)


def copy_config_to_run_dir(source_path: Path, run_dir: Path) -> None:
    shutil.copy2(source_path, run_dir / "config.yaml")


def write_resolved_config(config: ExperimentConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(config.to_resolved_dict(), indent=2))
