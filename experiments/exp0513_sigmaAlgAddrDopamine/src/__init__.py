"""Code for exp0513 hidden-dopamine network prototypes."""

from .assignment import (
    build_dopamine_assignment,
    build_forward_edge_records,
    build_graph_payload,
    flatten_controllable_weights,
    recommend_dopamine_m,
    resolve_dopamine_m,
    unflatten_internal_signal,
)
from .config import ExperimentConfig, copy_config_to_run_dir, dump_config_to_yaml, load_config_from_yaml
from .data import available_task_names, build_dataset, piecewise_target, poly_wave_target, sin_mix_target, sin_target
from .model import SelfModulatedMLP
from .train import load_experiment_checkpoint, run_experiment

__all__ = [
    "ExperimentConfig",
    "SelfModulatedMLP",
    "available_task_names",
    "build_dopamine_assignment",
    "build_dataset",
    "build_forward_edge_records",
    "build_graph_payload",
    "copy_config_to_run_dir",
    "dump_config_to_yaml",
    "flatten_controllable_weights",
    "load_experiment_checkpoint",
    "load_config_from_yaml",
    "piecewise_target",
    "poly_wave_target",
    "recommend_dopamine_m",
    "resolve_dopamine_m",
    "run_experiment",
    "sin_mix_target",
    "sin_target",
    "unflatten_internal_signal",
]
