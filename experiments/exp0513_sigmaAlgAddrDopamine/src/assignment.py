"""Static dopamine assignment helpers for exp0513."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import random

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


def recommend_dopamine_m(coverage_c: int) -> int:
    """Recommend m so average edges per dopamine node stays at or below N/10."""
    return 10 * int(coverage_c)


def resolve_dopamine_m(
    coverage_c: int,
    hidden_pool_size: int,
    dopamine_m_override: int | None,
) -> tuple[int, int]:
    """Resolve the effective m and the recommended m under the current defaults."""
    coverage_c = int(coverage_c)
    if coverage_c <= 0:
        raise ValueError("coverage_c must be a positive integer.")

    recommended_m = recommend_dopamine_m(coverage_c)
    if dopamine_m_override is None:
        if recommended_m > hidden_pool_size:
            raise ValueError(
                "Current hidden pool is too small for the recommended dopamine m. "
                "Please reduce coverage_c or increase the hidden layer size."
            )
        dopamine_m = recommended_m
    else:
        dopamine_m = int(dopamine_m_override)

    if dopamine_m <= 0:
        raise ValueError("dopamine_m must be a positive integer.")
    if dopamine_m > hidden_pool_size:
        raise ValueError("dopamine_m cannot exceed the available hidden node count.")
    if dopamine_m < coverage_c:
        raise ValueError("dopamine_m must be at least coverage_c so average coverage is feasible.")

    return dopamine_m, recommended_m


def flatten_controllable_weights(model) -> tuple[torch.Tensor, list[ParameterSlice]]:
    """Flatten controllable weights into the fixed order."""
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


def build_forward_edge_records(model) -> list[dict[str, object]]:
    """Build forward edge metadata in the same order as flattened controllable weights."""
    edges: list[dict[str, object]] = []
    for layer_idx, layer in enumerate(model.trunk, start=1):
        in_dim = layer.in_features
        out_dim = layer.out_features
        layer_key = f"{'input' if layer_idx == 1 else f'hidden{layer_idx - 1}'}->hidden{layer_idx}"
        for out_idx in range(out_dim):
            for in_idx in range(in_dim):
                source = f"x{in_idx}" if layer_idx == 1 else f"h{layer_idx - 1}_{in_idx}"
                target = f"h{layer_idx}_{out_idx}"
                edges.append({
                    "id": f"trunk{layer_idx}[{out_idx},{in_idx}]",
                    "source": source,
                    "target": target,
                    "layerKey": layer_key,
                    "layerLabel": _layer_label_from_key(layer_key),
                    "orderIndex": len(edges),
                    "controllingDopamineIds": [],
                })

    final_hidden_layer = len(model.trunk_dims)
    for out_idx in range(model.y_head.out_features):
        for in_idx in range(model.y_head.in_features):
            edges.append({
                "id": f"y_head[{out_idx},{in_idx}]",
                "source": f"h{final_hidden_layer}_{in_idx}",
                "target": f"y{out_idx}",
                "layerKey": f"hidden{final_hidden_layer}->output",
                "layerLabel": _layer_label_from_key(f"hidden{final_hidden_layer}->output"),
                "orderIndex": len(edges),
                "controllingDopamineIds": [],
            })

    return edges


def build_graph_nodes(model, dopamine_node_ids: list[str]) -> list[dict[str, object]]:
    """Create front-end graph nodes with selected dopamine neurons marked."""
    dopamine_set = set(dopamine_node_ids)
    nodes: list[dict[str, object]] = []
    for idx in range(model.input_dim):
        node_id = f"x{idx}"
        nodes.append({
            "id": node_id,
            "label": node_id,
            "kind": "input",
            "column": 0,
            "row": idx,
            "rowCount": model.input_dim,
            "isDopamine": False,
        })

    for layer_idx, layer_dim in enumerate(model.trunk_dims, start=1):
        for neuron_idx in range(layer_dim):
            node_id = f"h{layer_idx}_{neuron_idx}"
            nodes.append({
                "id": node_id,
                "label": node_id,
                "kind": "hidden",
                "column": layer_idx,
                "row": neuron_idx,
                "rowCount": layer_dim,
                "isDopamine": node_id in dopamine_set,
            })

    output_column = len(model.trunk_dims) + 1
    for idx in range(model.y_dim):
        node_id = f"y{idx}"
        nodes.append({
            "id": node_id,
            "label": node_id,
            "kind": "output",
            "column": output_column,
            "row": idx,
            "rowCount": model.y_dim,
            "isDopamine": False,
        })
    return nodes


def _layer_label_from_key(layer_key: str) -> str:
    source_key, target_key = layer_key.split("->", maxsplit=1)

    def _format(part: str) -> str:
        if part == "input":
            return "Input"
        if part == "output":
            return "Output"
        if part.startswith("hidden"):
            return f"Hidden L{part.removeprefix('hidden')}"
        return part

    return f"{_format(source_key)} -> {_format(target_key)}"


def build_layer_metadata(model) -> list[dict[str, object]]:
    """Describe all forward layers for graph UIs and summaries."""
    metadata: list[dict[str, object]] = []
    for layer_idx in range(1, len(model.trunk_dims) + 1):
        key = f"{'input' if layer_idx == 1 else f'hidden{layer_idx - 1}'}->hidden{layer_idx}"
        metadata.append({"key": key, "label": _layer_label_from_key(key)})
    key = f"hidden{len(model.trunk_dims)}->output"
    metadata.append({"key": key, "label": _layer_label_from_key(key)})
    return metadata


def build_dopamine_assignment(
    edge_records: list[dict[str, object]],
    hidden_node_ids: list[str],
    coverage_c: int,
    dopamine_m: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build a static random hidden-dopamine to edge assignment."""
    if dopamine_m > len(hidden_node_ids):
        raise ValueError("dopamine_m cannot exceed the hidden node pool size.")

    rng = random.Random(int(seed))
    selected_dopamine_node_ids = rng.sample(hidden_node_ids, dopamine_m)
    edge_ids = [str(edge["id"]) for edge in edge_records]
    edge_count = len(edge_ids)
    total_slots = int(coverage_c) * edge_count
    if total_slots < edge_count:
        raise ValueError("coverage_c must be at least 1 so every edge can be covered.")
    if total_slots > edge_count * dopamine_m:
        raise ValueError("Requested global average coverage is impossible for the current dopamine_m.")

    B = torch.zeros((edge_count, dopamine_m), dtype=torch.float32)

    shuffled_edges = list(range(edge_count))
    rng.shuffle(shuffled_edges)
    shuffled_dopamine = list(range(dopamine_m))
    rng.shuffle(shuffled_dopamine)
    for idx, dopamine_idx in enumerate(shuffled_dopamine):
        if idx >= edge_count:
            break
        B[shuffled_edges[idx], dopamine_idx] = 1.0

    for edge_idx in range(edge_count):
        if B[edge_idx].sum().item() == 0:
            dopamine_idx = rng.randrange(dopamine_m)
            B[edge_idx, dopamine_idx] = 1.0

    remaining_slots = total_slots - int(B.sum().item())
    available_pairs = [
        (edge_idx, dopamine_idx)
        for edge_idx in range(edge_count)
        for dopamine_idx in range(dopamine_m)
        if B[edge_idx, dopamine_idx].item() == 0.0
    ]
    if remaining_slots > len(available_pairs):
        raise ValueError("Not enough unique edge-dopamine pairs to satisfy the requested global average coverage.")
    chosen_pairs = rng.sample(available_pairs, remaining_slots)
    for edge_idx, dopamine_idx in chosen_pairs:
        B[edge_idx, dopamine_idx] = 1.0

    k = B.sum(dim=0)
    B_tilde = B / torch.sqrt(k).unsqueeze(0)

    edge_to_dopamine_ids: dict[str, list[str]] = {}
    dopamine_to_edge_ids: dict[str, list[str]] = {node_id: [] for node_id in selected_dopamine_node_ids}
    edge_coverage_counts: dict[str, int] = {}
    for edge_idx, edge_id in enumerate(edge_ids):
        controlling_ids = [
            selected_dopamine_node_ids[dopamine_idx]
            for dopamine_idx, value in enumerate(B[edge_idx].tolist())
            if value > 0.0
        ]
        edge_to_dopamine_ids[edge_id] = controlling_ids
        edge_coverage_counts[edge_id] = len(controlling_ids)
        for node_id in controlling_ids:
            dopamine_to_edge_ids[node_id].append(edge_id)

    dopamine_stats_unsorted = []
    for node_id in selected_dopamine_node_ids:
        edge_count_for_node = len(dopamine_to_edge_ids[node_id])
        dopamine_stats_unsorted.append({
            "node_id": node_id,
            "edge_count": edge_count_for_node,
            "coverage_ratio": edge_count_for_node / edge_count,
        })
    ranked_stats = sorted(
        dopamine_stats_unsorted,
        key=lambda item: (-item["edge_count"], item["node_id"]),
    )
    for rank, stat in enumerate(ranked_stats, start=1):
        stat["rank"] = rank
        stat["c_r"] = stat["coverage_ratio"]

    stat_lookup = {stat["node_id"]: stat for stat in ranked_stats}
    metadata: dict[str, object] = {
        "coverage_c": int(coverage_c),
        "dopamine_m": dopamine_m,
        "selected_dopamine_node_ids": selected_dopamine_node_ids,
        "edge_count": edge_count,
        "edge_ids": edge_ids,
        "edge_to_dopamine_ids": edge_to_dopamine_ids,
        "dopamine_to_edge_ids": dopamine_to_edge_ids,
        "edge_coverage_counts": edge_coverage_counts,
        "average_edge_coverage": float(B.sum(dim=1).mean().item()),
        "average_edges_per_dopamine": float(k.mean().item()),
        "min_edges_per_dopamine": int(k.min().item()),
        "max_edges_per_dopamine": int(k.max().item()),
        "dopamine_stats": ranked_stats,
        "c_r": [float(stat["c_r"]) for stat in ranked_stats],
        "ranked_dopamine_node_ids": [stat["node_id"] for stat in ranked_stats],
        "column_loads": {node_id: int(stat_lookup[node_id]["edge_count"]) for node_id in selected_dopamine_node_ids},
    }
    return B, B_tilde, metadata


def rebuild_assignment_tensors(
    edge_records: list[dict[str, object]],
    assignment_metadata: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild B and B_tilde from saved dopamine assignment metadata."""
    selected_dopamine_node_ids = list(assignment_metadata["selected_dopamine_node_ids"])
    edge_to_dopamine_ids = {
        str(edge_id): list(node_ids)
        for edge_id, node_ids in dict(assignment_metadata["edge_to_dopamine_ids"]).items()
    }
    node_to_index = {node_id: idx for idx, node_id in enumerate(selected_dopamine_node_ids)}

    B = torch.zeros((len(edge_records), len(selected_dopamine_node_ids)), dtype=torch.float32)
    for edge_idx, edge in enumerate(edge_records):
        edge_id = str(edge["id"])
        for node_id in edge_to_dopamine_ids[edge_id]:
            B[edge_idx, node_to_index[node_id]] = 1.0

    k = B.sum(dim=0)
    B_tilde = B / torch.sqrt(k).unsqueeze(0)
    return B, B_tilde


def build_graph_payload(model, assignment_metadata: dict[str, object]) -> dict[str, object]:
    """Build the front-end graph payload for a fixed dopamine assignment."""
    selected_dopamine_node_ids = list(assignment_metadata["selected_dopamine_node_ids"])
    edge_to_dopamine_ids = {
        str(edge_id): list(node_ids)
        for edge_id, node_ids in dict(assignment_metadata["edge_to_dopamine_ids"]).items()
    }
    edge_records = build_forward_edge_records(model)
    for edge in edge_records:
        edge["controllingDopamineIds"] = edge_to_dopamine_ids[str(edge["id"])]

    return {
        "nodes": build_graph_nodes(model, selected_dopamine_node_ids),
        "edges": edge_records,
        "architecture": model.arch_dict(),
        "layerMetadata": build_layer_metadata(model),
        "totalNodes": model.input_dim + model.hidden_pool_size() + model.y_dim,
        "totalEdges": len(edge_records),
        "totalLayers": len(model.trunk_dims) + 2,
        "dopamineNodeIds": selected_dopamine_node_ids,
        "dopamineStats": assignment_metadata["dopamine_stats"],
        "coverageByRank": assignment_metadata["c_r"],
        "coverageC": assignment_metadata["coverage_c"],
        "dopamineM": assignment_metadata["dopamine_m"],
    }
