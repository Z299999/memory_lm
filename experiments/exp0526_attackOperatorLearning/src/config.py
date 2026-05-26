"""Configuration loading for exp0526 closed-loop controller experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml


def parse_train_window_schedule(spec: str) -> tuple[str, int, int]:
    raw = str(spec).strip()
    fixed_match = re.fullmatch(r"fixed\((\d+)\)", raw)
    if fixed_match:
        value = int(fixed_match.group(1))
        if value <= 0:
            raise ValueError("train_window_schedule fixed(L) requires L > 0.")
        return "fixed", value, value
    uniform_match = re.fullmatch(r"random_uniform\((\d+),\s*(\d+)\)", raw)
    if uniform_match:
        lower = int(uniform_match.group(1))
        upper = int(uniform_match.group(2))
        if lower <= 0 or upper <= 0 or lower > upper:
            raise ValueError("train_window_schedule random_uniform(a,b) requires 0 < a <= b.")
        return "random_uniform", lower, upper
    raise ValueError("train_window_schedule must be fixed(L) or random_uniform(a,b).")


def parse_condition(spec: str) -> tuple[str, tuple[int, ...]]:
    match = re.fullmatch(r"(\w+)\(([^)]*)\)", str(spec).strip())
    if match:
        raw = match.group(2).strip()
        params = tuple(int(part.strip()) for part in raw.split(",") if part.strip()) if raw else ()
        return match.group(1), params
    return str(spec).strip(), ()


@dataclass
class SpeciesProfileConfig:
    k_base: float
    k_amp: float
    k_center: float
    k_sigma: float
    mu_base: float
    mu_juv_amp: float
    mu_juv: float
    mu_sen_amp: float
    mu_sen: float
    g_base: float
    g_amp: float
    g_center: float
    g_sigma: float
    init_scale: float = 1.0


@dataclass
class RunConfig:
    run_name: str = "exp0526_closed_loop_v0"
    seed: int = 42
    log_every: int = 25
    output_root: str = "runs"


@dataclass
class ModelConfig:
    trunk_dims: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    language_dim: int = 17
    language_readout_coverage: int = 3
    use_error_view: bool = True
    use_residual: bool = True
    language_readout_all_layers: bool = True
    message_carry_mode: str = "learnable_matrix"


@dataclass
class EnvConfig:
    age_max: float = 1.0
    num_age: int = 64
    dt: float = 0.002
    pulse_value: float = 1.0
    positivity_eps: float = 1e-8
    root_bisect_iters: int = 64
    equilibrium_x1_multiplier: float = 2.0
    species: tuple[SpeciesProfileConfig, SpeciesProfileConfig] = field(default_factory=tuple)


@dataclass
class EcologyConfig:
    mixed_sin_components: tuple[tuple[float, float], ...] = ((0.003, 1.0), (0.011, 0.7), (0.037, 0.35))
    k_amp_strength: float = 0.12
    mu_base_strength: float = 0.10
    mu_sen_amp_strength: float = 0.10


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    train_window_schedule: str = "random_uniform(10,30)"
    control_loss_weight: float = 1e-4
    carry_error_between_windows: bool = True


@dataclass
class EvalConfig:
    eval_steps: int = 128
    future_steps: int = 256
    eval_conditions: tuple[str, ...] = ("full", "sole_eye", "sole_speech", "neither", "blink(40,80)", "stutter(100,160)")


@dataclass
class PlotConfig:
    plot_dpi: int = 160
    plot_training_fig_width: float = 10.0
    plot_training_fig_height: float = 6.0
    plot_diag_fig_width: float = 16.0
    plot_diag_fig_height: float = 15.0
    plot_short_steps: int = 128
    plot_long_steps: int = 256
    plot_message_steps: int = 128
    plot_grid_alpha: float = 0.25
    plot_title_fontsize: int = 14
    plot_series_linewidth: float = 1.7
    plot_aux_linewidth: float = 1.4
    plot_zero_linewidth: float = 1.0
    plot_legend_ncols: int = 4
    plot_show_message_traces: bool = True
    plot_show_message_norm: bool = True
    plot_show_training_timeline: bool = True
    plot_training_timeline_num_panels: int = 20
    plot_training_timeline_ncols: int = 5
    plot_training_timeline_window_steps: int = 120
    plot_training_timeline_fig_width: float = 24.0


@dataclass
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    env: EnvConfig
    ecology: EcologyConfig
    train: TrainConfig
    eval: EvalConfig
    plot: PlotConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_of_pairs(value: Any, *, name: str) -> tuple[tuple[float, float], ...]:
    pairs = tuple((float(row[0]), float(row[1])) for row in value)
    if not pairs:
        raise ValueError(f"{name} must contain at least one [frequency, amplitude] pair.")
    for freq, amp in pairs:
        if freq <= 0.0:
            raise ValueError(f"{name} frequencies must be positive.")
        if not (abs(amp) < float("inf")):
            raise ValueError(f"{name} amplitudes must be finite.")
    return pairs


def _build_species(raw: list[dict[str, Any]]) -> tuple[SpeciesProfileConfig, SpeciesProfileConfig]:
    if len(raw) != 2:
        raise ValueError("env.species must contain exactly two species profiles.")
    species = tuple(SpeciesProfileConfig(**row) for row in raw)
    for idx, profile in enumerate(species):
        values = asdict(profile)
        for key, value in values.items():
            if float(value) <= 0.0:
                raise ValueError(f"env.species[{idx}].{key} must be positive.")
    return species  # type: ignore[return-value]


def _dataclass_from(cls, raw: dict[str, Any]) -> Any:
    return cls(**raw)


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    run = _dataclass_from(RunConfig, raw.get("run", {}))

    model_raw = dict(raw.get("model", {}))
    if "trunk_dims" in model_raw:
        model_raw["trunk_dims"] = tuple(int(v) for v in model_raw["trunk_dims"])
    model = _dataclass_from(ModelConfig, model_raw)

    env_raw = dict(raw.get("env", {}))
    env_raw["species"] = _build_species(env_raw.get("species", []))
    env = _dataclass_from(EnvConfig, env_raw)

    ecology_raw = dict(raw.get("ecology", {}))
    if "mixed_sin_components" in ecology_raw:
        ecology_raw["mixed_sin_components"] = _tuple_of_pairs(
            ecology_raw["mixed_sin_components"],
            name="ecology.mixed_sin_components",
        )
    ecology = _dataclass_from(EcologyConfig, ecology_raw)

    train = _dataclass_from(TrainConfig, raw.get("train", {}))
    eval_raw = dict(raw.get("eval", {}))
    if "eval_conditions" in eval_raw:
        eval_raw["eval_conditions"] = tuple(str(v) for v in eval_raw["eval_conditions"])
    eval_cfg = _dataclass_from(EvalConfig, eval_raw)
    plot = _dataclass_from(PlotConfig, raw.get("plot", {}))

    config = ExperimentConfig(run=run, model=model, env=env, ecology=ecology, train=train, eval=eval_cfg, plot=plot)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.env.num_age < 8:
        raise ValueError("env.num_age must be >= 8.")
    if config.env.age_max <= 0.0 or config.env.dt <= 0.0:
        raise ValueError("env.age_max and env.dt must be positive.")
    da = config.env.age_max / float(config.env.num_age - 1)
    if config.env.dt / da > 1.0:
        raise ValueError("env.dt must satisfy dt / da <= 1 for the upwind transport step.")
    if config.model.language_dim < 0:
        raise ValueError("model.language_dim must be >= 0.")
    if config.model.language_dim > 0:
        if config.model.language_readout_coverage <= 0:
            raise ValueError("model.language_readout_coverage must be positive.")
        if config.model.language_readout_coverage > config.model.language_dim:
            raise ValueError("model.language_readout_coverage must be <= language_dim.")
    if config.model.message_carry_mode not in {"identity", "learnable_diagonal", "learnable_matrix"}:
        raise ValueError("model.message_carry_mode must be identity, learnable_diagonal, or learnable_matrix.")
    parse_train_window_schedule(config.train.train_window_schedule)
    if config.train.control_loss_weight < 0.0:
        raise ValueError("train.control_loss_weight must be >= 0.")
    valid_conditions = {"full", "sole_eye", "sole_speech", "neither", "blink", "stutter"}
    for condition in config.eval.eval_conditions:
        base, params = parse_condition(condition)
        if base not in valid_conditions:
            raise ValueError(f"Unknown eval condition: {condition!r}.")
        if base in {"blink", "stutter"} and len(params) != 2:
            raise ValueError(f"{base} conditions require two integer parameters.")
        if base not in {"blink", "stutter"} and params:
            raise ValueError(f"{base} does not take parameters.")


def copy_config_to_run_dir(config_path: Path, run_dir: Path) -> None:
    shutil.copy2(config_path, run_dir / "config.yaml")


def write_resolved_config(config: ExperimentConfig, output_path: Path) -> None:
    output_path.write_text(json.dumps(config.to_dict(), indent=2))
