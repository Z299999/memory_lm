"""Code for exp0513 self-modulated network prototypes."""

from .assignment import build_static_assignment, flatten_controllable_weights, unflatten_internal_signal
from .config import ExperimentConfig, copy_config_to_run_dir, dump_config_to_yaml, load_config_from_yaml
from .data import available_task_names, build_dataset, piecewise_target, poly_wave_target, sin_mix_target, sin_target
from .model import SelfModulatedMLP, solve_v1_q_dim
from .train import run_experiment

__all__ = [
    "ExperimentConfig",
    "SelfModulatedMLP",
    "available_task_names",
    "build_static_assignment",
    "build_dataset",
    "copy_config_to_run_dir",
    "dump_config_to_yaml",
    "flatten_controllable_weights",
    "load_config_from_yaml",
    "piecewise_target",
    "poly_wave_target",
    "run_experiment",
    "sin_mix_target",
    "sin_target",
    "solve_v1_q_dim",
    "unflatten_internal_signal",
]
