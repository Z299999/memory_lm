"""Code for exp0513 self-modulated network prototypes."""

from .assignment import build_static_assignment, flatten_controllable_weights, unflatten_internal_signal
from .data import build_sin_mix_dataset, sin_mix_target
from .model import SelfModulatedMLP, solve_v1_q_dim
from .train import ExperimentConfig, run_experiment

__all__ = [
    "ExperimentConfig",
    "SelfModulatedMLP",
    "build_sin_mix_dataset",
    "build_static_assignment",
    "flatten_controllable_weights",
    "run_experiment",
    "sin_mix_target",
    "solve_v1_q_dim",
    "unflatten_internal_signal",
]
