"""Static assignment construction for exp0513 V1."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import itertools

import torch


@dataclass
class ParameterSlice:
    """Flat-vector slice metadata for one controllable parameter tensor."""

    name: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def numel(self) -> int:
        return self.end - self.start


def build_static_assignment(edge_count: int, m: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a fixed binary assignment matrix and its energy-normalized version."""
    codes = []
    for code in itertools.product((0, 1), repeat=m):
        if sum(code) == 0:
            continue
        codes.append(tuple(code))

    if edge_count > len(codes):
        raise ValueError(f"Need {edge_count} unique nonzero addresses, but only have {len(codes)}.")

    midpoint = m / 2.0
    codes.sort(key=lambda code: (abs(sum(code) - midpoint), code))
    selected = codes[:edge_count]

    B = torch.tensor(selected, dtype=torch.float32)
    k = B.sum(dim=0)
    B_tilde = B / torch.sqrt(k).unsqueeze(0)

    hamming_weights = [int(sum(code)) for code in selected]
    metadata = {
        "edge_count": edge_count,
        "m": m,
        "all_rows_nonzero": bool((B.sum(dim=1) > 0).all().item()),
        "all_rows_unique": len({tuple(code) for code in selected}) == edge_count,
        "k_i": [float(v) for v in k.tolist()],
        "min_k": float(k.min().item()),
        "max_k": float(k.max().item()),
        "mean_k": float(k.mean().item()),
        "address_hamming_weights": hamming_weights,
        "min_hamming_weight": min(hamming_weights),
        "max_hamming_weight": max(hamming_weights),
        "selected_addresses_preview": [list(code) for code in selected[:16]],
    }
    return B, B_tilde, metadata


def flatten_controllable_weights(model) -> tuple[torch.Tensor, list[ParameterSlice]]:
    """Flatten controllable weights into the fixed V1 order."""
    flat_chunks = []
    index_map: list[ParameterSlice] = []
    offset = 0
    for name, param in model.controllable_named_parameters():
        chunk = param.detach().reshape(-1)
        flat_chunks.append(chunk)
        next_offset = offset + chunk.numel()
        index_map.append(
            ParameterSlice(
                name=name,
                shape=tuple(param.shape),
                start=offset,
                end=next_offset,
            )
        )
        offset = next_offset
    flat = torch.cat(flat_chunks) if flat_chunks else torch.empty(0)
    return flat, index_map


def unflatten_internal_signal(
    flat_signal: torch.Tensor,
    index_map: list[ParameterSlice],
) -> OrderedDict[str, torch.Tensor]:
    """Reshape a flat signal vector back into per-parameter tensors."""
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for entry in index_map:
        out[entry.name] = flat_signal[entry.start:entry.end].view(entry.shape)
    return out
