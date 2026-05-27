"""Configuration loading for exp0526 online neural control experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def _normalize_eval_conditions(values: Any) -> tuple[str, ...]:
    items = [str(v).strip() for v in values]
    normalized: list[str] = []
    pending: list[str] = []
    balance = 0
    for item in items:
        if pending:
            pending.append(item)
            balance += item.count("(") - item.count(")")
            if balance <= 0:
                normalized.append(",".join(part.strip() for part in pending))
                pending = []
                balance = 0
            continue
        item_balance = item.count("(") - item.count(")")
        if item_balance > 0 and item.count(")") == 0:
            pending = [item]
            balance = item_balance
            continue
        normalized.append(item)
    if pending:
        normalized.append(",".join(part.strip() for part in pending))
    return tuple(normalized)


@dataclass
class RunConfig:
    run_name: str = "exp0526_scalar_cubic_v1"
    seed: int = 42
    log_every: int = 25
    output_root: str = "runs"


@dataclass
class ModelConfig:
    trunk_dims: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    language_dim: int = 17
    language_readout_coverage: int = 3
    use_residual: bool = True
    language_readout_all_layers: bool = True
    message_carry_mode: str = "learnable_matrix"


@dataclass
class EnvConfig:
    env_kind: str = "scalar_control_affine"
    dt: float = 0.005
    x0: Any = 0.5
    pulse_value: float = 1.0
    u_max: float = 2.0
    f_expr: str = "-x + x**3"
    g_expr: str = "1.0"
    linear_coeff: float = 1.0
    cubic_coeff: float = 1.0
    control_gain: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.4
    state_limit: float = 4.0


@dataclass
class TrainConfig:
    epochs: int = 1000
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    train_window_schedule: str = "random_uniform(10,30)"
    control_loss_weight: float = 1e-4


@dataclass
class EvalConfig:
    eval_steps: int = 64
    future_steps: int = 256
    eval_conditions: tuple[str, ...] = ("full", "sole_eye", "sole_speech", "neither", "blink(20,40)", "stutter(50,100)")


@dataclass
class PlotConfig:
    plot_dpi: int = 160
    plot_training_fig_width: float = 10.0
    plot_training_fig_height: float = 6.0
    plot_diag_fig_width: float = 16.0
    plot_diag_fig_height: float = 15.0
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
    train: TrainConfig
    eval: EvalConfig
    plot: PlotConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"]["use_language_resolved"] = bool(self.model.language_dim > 0)
        return payload


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
    if "x0" in env_raw and isinstance(env_raw["x0"], (list, tuple)):
        env_raw["x0"] = tuple(float(v) for v in env_raw["x0"])
    env = _dataclass_from(EnvConfig, env_raw)
    train = _dataclass_from(TrainConfig, raw.get("train", {}))

    eval_raw = dict(raw.get("eval", {}))
    if "eval_conditions" in eval_raw:
        eval_raw["eval_conditions"] = _normalize_eval_conditions(eval_raw["eval_conditions"])
    eval_cfg = _dataclass_from(EvalConfig, eval_raw)

    plot = _dataclass_from(PlotConfig, raw.get("plot", {}))
    config = ExperimentConfig(run=run, model=model, env=env, train=train, eval=eval_cfg, plot=plot)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.model.language_dim < 0:
        raise ValueError("model.language_dim must be >= 0.")
    if config.model.language_dim > 0:
        if config.model.language_readout_coverage <= 0:
            raise ValueError("model.language_readout_coverage must be positive when language_dim > 0.")
        if config.model.language_readout_coverage > config.model.language_dim:
            raise ValueError("model.language_readout_coverage must be <= language_dim.")
    if config.model.message_carry_mode not in {"identity", "learnable_diagonal", "learnable_matrix"}:
        raise ValueError("model.message_carry_mode must be identity, learnable_diagonal, or learnable_matrix.")
    if config.env.env_kind not in {"scalar_cubic", "scalar_control_affine", "planar_double_well"}:
        raise ValueError("env.env_kind must be scalar_cubic, scalar_control_affine, or planar_double_well.")
    if config.env.dt <= 0.0:
        raise ValueError("env.dt must be positive.")
    if config.env.u_max <= 0.0:
        raise ValueError("env.u_max must be positive.")
    if config.env.state_limit <= 0.0:
        raise ValueError("env.state_limit must be positive.")
    if config.env.env_kind == "scalar_control_affine":
        if not str(config.env.f_expr).strip():
            raise ValueError("env.f_expr must be non-empty for scalar_control_affine.")
        if not str(config.env.g_expr).strip():
            raise ValueError("env.g_expr must be non-empty for scalar_control_affine.")
        if isinstance(config.env.x0, (tuple, list)):
            raise ValueError("env.x0 must be a scalar for scalar_control_affine.")
    if config.env.env_kind == "scalar_cubic" and isinstance(config.env.x0, (tuple, list)):
        raise ValueError("env.x0 must be a scalar for scalar_cubic.")
    if config.env.env_kind == "planar_double_well":
        if not isinstance(config.env.x0, (tuple, list)) or len(config.env.x0) != 2:
            raise ValueError("env.x0 must be a length-2 list or tuple for planar_double_well.")
        if config.env.beta <= 0.0:
            raise ValueError("env.beta must be positive for planar_double_well.")
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
