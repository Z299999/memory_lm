"""Update-rule helpers for exp0513 V1."""

from __future__ import annotations

from collections import OrderedDict

import torch

try:
    from .assignment import ParameterSlice, unflatten_internal_signal
except ImportError:  # pragma: no cover - script mode
    from assignment import ParameterSlice, unflatten_internal_signal


def extract_flat_bp_update(
    model,
    index_map: list[ParameterSlice],
    lr_bp: float,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    """Extract plain-SGD BP updates for controllable weights in flatten order."""
    updates = []
    per_param: OrderedDict[str, torch.Tensor] = OrderedDict()
    param_lookup = dict(model.controllable_named_parameters())
    for entry in index_map:
        param = param_lookup[entry.name]
        if param.grad is None:
            update = torch.zeros_like(param)
        else:
            update = -lr_bp * param.grad.detach()
        per_param[entry.name] = update
        updates.append(update.reshape(-1))
    flat = torch.cat(updates) if updates else torch.empty(0)
    return flat, per_param


def extract_bias_bp_updates(
    model,
    lr_bp: float,
) -> OrderedDict[str, torch.Tensor]:
    """Collect plain-SGD BP updates for bias parameters only."""
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, param in model.bp_only_named_parameters():
        if param.grad is None:
            out[name] = torch.zeros_like(param)
        else:
            out[name] = -lr_bp * param.grad.detach()
    return out


def compute_internal_signal(q_batch: torch.Tensor, B_tilde: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate q over the batch and map it to edge-level modulation signals."""
    q_mean = q_batch.detach().mean(dim=0)
    s = B_tilde.to(q_batch.device, q_batch.dtype).matmul(q_mean)
    return s, q_mean


def compute_internal_update(
    q_batch: torch.Tensor,
    B_tilde: torch.Tensor,
    eta_int: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the batch-aggregated edge-level internal update."""
    s, q_mean = compute_internal_signal(q_batch, B_tilde)
    delta_int = eta_int * torch.tanh(gamma * s)
    return delta_int, s, q_mean


def mix_phase_b_updates(
    bp_flat: torch.Tensor,
    int_flat: torch.Tensor,
    index_map: list[ParameterSlice],
    lambda_value: float,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor], torch.Tensor]:
    """Mix BP and internal updates according to V1 Phase B rules."""
    total = torch.empty_like(bp_flat)
    bp_mask = torch.zeros_like(bp_flat, dtype=torch.bool)
    q_head_start = index_map[-1].start
    q_head_end = index_map[-1].end
    bp_mask[:q_head_start] = True

    total[bp_mask] = (1.0 - lambda_value) * bp_flat[bp_mask] + lambda_value * int_flat[bp_mask]
    total[q_head_start:q_head_end] = lambda_value * int_flat[q_head_start:q_head_end]
    per_param = unflatten_internal_signal(total, index_map)
    return total, per_param, bp_mask


def phase_a_controllable_updates(
    bp_flat: torch.Tensor,
    index_map: list[ParameterSlice],
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    """Phase A uses BP on trunk/y weights and zero update on q-head weights."""
    total = bp_flat.clone()
    q_head_start = index_map[-1].start
    q_head_end = index_map[-1].end
    total[q_head_start:q_head_end] = 0.0
    per_param = unflatten_internal_signal(total, index_map)
    return total, per_param


def apply_updates(
    model,
    controllable_updates: OrderedDict[str, torch.Tensor],
    bias_updates: OrderedDict[str, torch.Tensor],
) -> None:
    """Apply weight and bias updates in-place under no_grad."""
    name_to_param = model.named_parameter_dict()
    with torch.no_grad():
        for name, update in controllable_updates.items():
            name_to_param[name].add_(update)
        for name, update in bias_updates.items():
            name_to_param[name].add_(update)
