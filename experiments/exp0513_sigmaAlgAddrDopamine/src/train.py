"""Training loop for exp0513 hidden-dopamine experiments."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

# Keep matplotlib/font caches inside the experiment workspace.
_RUNTIME_CACHE = Path(__file__).resolve().parents[1] / ".runtime_cache"
_MPL_CACHE = _RUNTIME_CACHE / "mpl"
_XDG_CACHE = _RUNTIME_CACHE / "xdg"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .assignment import (
        build_dopamine_assignment,
        build_forward_edge_records,
        build_graph_payload,
        flatten_controllable_weights,
        rebuild_assignment_tensors,
        recommend_dopamine_m,
        resolve_dopamine_m,
    )
    from .config import ExperimentConfig, copy_config_to_run_dir, dump_config_to_yaml, write_resolved_config
    from .data import build_dataset
    from .diagnostics import (
        average_epoch_metrics,
        collect_batch_metrics,
        plot_dopamine_diagnostics,
        plot_update_diagnostics,
        save_history_json,
    )
    from .model import SelfModulatedMLP
    from .update_rule import (
        apply_updates,
        compute_internal_update,
        extract_bias_bp_updates,
        extract_flat_bp_update,
        mix_controllable_updates,
    )
except ImportError:  # pragma: no cover - script mode
    from assignment import (
        build_dopamine_assignment,
        build_forward_edge_records,
        build_graph_payload,
        flatten_controllable_weights,
        rebuild_assignment_tensors,
        recommend_dopamine_m,
        resolve_dopamine_m,
    )
    from config import ExperimentConfig, copy_config_to_run_dir, dump_config_to_yaml, write_resolved_config
    from data import build_dataset
    from diagnostics import (
        average_epoch_metrics,
        collect_batch_metrics,
        plot_dopamine_diagnostics,
        plot_update_diagnostics,
        save_history_json,
    )
    from model import SelfModulatedMLP
    from update_rule import (
        apply_updates,
        compute_internal_update,
        extract_bias_bp_updates,
        extract_flat_bp_update,
        mix_controllable_updates,
    )


def make_run_dir(base_dir: Path, run_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json_atomic(output_path: Path, payload: dict[str, object]) -> None:
    """Atomically write a JSON payload to disk."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(output_path)


def load_experiment_checkpoint(resume_from: str) -> dict[str, object]:
    """Load a modern exp0513 checkpoint for continuation."""
    payload = torch.load(resume_from, map_location="cpu")
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint at {resume_from} does not contain model_state_dict.")
    if "dopamine_assignment_metadata" not in payload:
        raise ValueError(
            f"Checkpoint at {resume_from} is missing dopamine_assignment_metadata and is not compatible with the current architecture."
        )
    return payload


def _build_live_status(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    edge_count: int,
    dopamine_m: int,
    recommended_dopamine_m: int,
    coverage_c: int,
    graph_payload: dict[str, object],
    status: str,
    local_epoch: int,
    global_epoch: int,
    global_epoch_start: int,
    global_epoch_end: int,
    train_loss: float | None,
    val_loss: float | None,
    best_val_loss: float | None,
    architecture: dict[str, object],
    local_loss_history: list[dict[str, float]],
    global_loss_history: list[dict[str, float]],
    node_activation_snapshot: dict[str, float | None],
    edge_weight_snapshot: dict[str, float],
    error: str | None = None,
) -> dict[str, object]:
    """Create the live dashboard payload for the current run state."""
    payload: dict[str, object] = {
        "run_name": config.run_name,
        "task_name": config.task_name,
        "status": status,
        "epoch": local_epoch,
        "epochs_total": config.epochs,
        "local_epoch": local_epoch,
        "local_epochs_total": config.epochs,
        "global_epoch": global_epoch,
        "global_epoch_start": global_epoch_start,
        "global_epoch_end": global_epoch_end,
        "lambda": config.lambda_value,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "seed": config.seed,
        "m": dopamine_m,
        "dopamine_m": dopamine_m,
        "recommended_dopamine_m": recommended_dopamine_m,
        "coverage_c": coverage_c,
        "N": edge_count,
        "resume_from": config.resume_from,
        "run_dir": str(run_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "graph_payload": graph_payload,
        "architecture": architecture,
        "local_loss_history": local_loss_history,
        "global_loss_history": global_loss_history,
        "loss_history": local_loss_history,
        "node_activation_snapshot": node_activation_snapshot,
        "edge_weight_snapshot": edge_weight_snapshot,
    }
    if error:
        payload["error"] = error
    return payload


def write_live_status(
    live_status_path: Path,
    payload: dict[str, object],
    status_callback: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Persist and optionally emit the current live dashboard state."""
    _write_json_atomic(live_status_path, payload)
    if status_callback is not None:
        status_callback(payload)


def _evaluate(model: SelfModulatedMLP, x: torch.Tensor, y_true: torch.Tensor) -> float:
    """Evaluate full-batch MSE on a split."""
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(x)
        mse = torch.mean((y_pred - y_true) ** 2).item()
    return float(mse)


def _build_node_activation_snapshot(
    model: SelfModulatedMLP,
    x: torch.Tensor,
) -> dict[str, float]:
    """Mean node activations from the latest validation forward pass."""
    model.eval()
    with torch.no_grad():
        y_pred, hidden_states = model(x)

    snapshot: dict[str, float] = {}
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    for idx in range(x.shape[1]):
        snapshot[f"x{idx}"] = float(x[:, idx].mean().item())
    for layer_idx, hidden in enumerate(hidden_states, start=1):
        for neuron_idx in range(hidden.shape[1]):
            snapshot[f"h{layer_idx}_{neuron_idx}"] = float(hidden[:, neuron_idx].mean().item())
    for idx in range(y_pred.shape[1]):
        snapshot[f"y{idx}"] = float(y_pred[:, idx].mean().item())
    return snapshot


def _build_edge_weight_snapshot(
    model: SelfModulatedMLP,
    edge_records: list[dict[str, object]],
) -> dict[str, float]:
    """Current scalar weight for every forward edge in graph order."""
    flat_weights, _ = flatten_controllable_weights(model)
    return {
        str(edge["id"]): float(flat_weights[idx].item())
        for idx, edge in enumerate(edge_records)
    }


def _architecture_signature_dict(config: ExperimentConfig) -> dict[str, object]:
    """Minimal architecture signature used for build and resume checks."""
    return {
        "input_dim": config.input_dim,
        "output_dim": config.output_dim,
        "trunk_dims": list(config.trunk_dims),
        "activation": "tanh",
        "dopamine_source": "hidden_activation",
    }


def _assert_resume_architecture(config: ExperimentConfig, checkpoint: dict[str, object]) -> None:
    """Reject resume attempts whose checkpoint structure does not match the current config."""
    expected = _architecture_signature_dict(config)
    actual = dict(checkpoint.get("architecture") or {})
    comparable = {
        "input_dim": actual.get("input_dim"),
        "output_dim": actual.get("output_dim", actual.get("y_dim")),
        "trunk_dims": list(actual.get("trunk_dims", [])),
        "activation": actual.get("activation"),
        "dopamine_source": actual.get("dopamine_source"),
    }
    if comparable != expected:
        raise ValueError(
            "resume_from architecture does not match the current config. "
            f"Expected {expected}, got {comparable}."
        )


def _epoch_train(
    model: SelfModulatedMLP,
    loader: DataLoader,
    B_tilde: torch.Tensor,
    index_map,
    dopamine_node_ids: list[str],
    config: ExperimentConfig,
) -> tuple[float, dict[str, float]]:
    """Run one epoch for the current fixed-lambda configuration."""
    model.train()
    batch_losses: list[float] = []
    batch_metric_rows: list[dict[str, float]] = []

    for bx, by in loader:
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        y_pred, hidden_states = model(bx)
        dopamine_batch = model.collect_dopamine_batch(hidden_states, dopamine_node_ids)
        loss = torch.mean((y_pred - by) ** 2)
        loss.backward()

        bp_flat, _ = extract_flat_bp_update(model, index_map, config.lr_bp)
        bias_updates = extract_bias_bp_updates(model, config.lr_bp)
        int_flat, s, dopamine_mean = compute_internal_update(dopamine_batch, B_tilde, config.eta_int, config.gamma)
        total_flat, controllable_updates, total_mask = mix_controllable_updates(
            bp_flat=bp_flat,
            int_flat=int_flat,
            index_map=index_map,
            lambda_value=config.lambda_value,
        )

        apply_updates(model, controllable_updates, bias_updates)
        batch_losses.append(float(loss.item()))
        batch_metric_rows.append(
            collect_batch_metrics(
                dopamine_batch=dopamine_batch.detach(),
                dopamine_mean=dopamine_mean.detach(),
                s=s.detach(),
                bp_flat=bp_flat.detach(),
                int_flat=int_flat.detach(),
                total_flat=total_flat.detach(),
                bp_mask=total_mask,
            )
        )

    return float(sum(batch_losses) / len(batch_losses)), average_epoch_metrics(batch_metric_rows)


def _format_resume_label(resume_from: str) -> str:
    """Short text for the plot config box."""
    if not resume_from:
        return "none"
    return Path(resume_from).name


def plot_loss_curve(
    history: list[dict[str, float]],
    config: ExperimentConfig,
    dopamine_m: int,
    recommended_dopamine_m: int,
    coverage_c: int,
    edge_count: int,
    output_path: Path,
) -> None:
    """Plot train/validation MSE with a compact config text box."""
    global_epochs = [row["global_epoch"] for row in history]
    train_losses = [max(row["train_loss"], 1e-12) for row in history]
    val_losses = [max(row["val_loss"], 1e-12) for row in history]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(global_epochs, train_losses, label="train", linewidth=1.8)
    ax.plot(global_epochs, val_losses, label="val", linewidth=1.8)
    ax.set_xlabel("global epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"exp0513 loss curve | task={config.task_name}")
    ax.grid(alpha=0.3)
    ax.legend()

    text = "\n".join([
        f"task={config.task_name}",
        f"in={config.input_dim}, out={config.output_dim}",
        f"trunk={list(config.trunk_dims)}",
        f"seed={config.seed}",
        f"lr_bp={config.lr_bp}",
        f"eta_int={config.eta_int}",
        f"gamma={config.gamma}",
        f"lambda={config.lambda_value}",
        f"epochs(+)= {config.epochs}",
        f"batch={config.batch_size}",
        f"c={coverage_c}",
        f"m={dopamine_m} (rec {recommended_dopamine_m})",
        f"N={edge_count}",
        f"resume={_format_resume_label(config.resume_from)}",
    ])
    ax.text(
        1.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.83, 1.0))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _checkpoint_payload(
    model: SelfModulatedMLP,
    config: ExperimentConfig,
    assignment_metadata: dict[str, object],
    graph_payload: dict[str, object],
    edge_count: int,
    summary: dict[str, object],
    source_checkpoint: dict[str, object] | None,
    global_epoch_completed: int,
    dopamine_m: int,
    recommended_dopamine_m: int,
    coverage_c: int,
    global_loss_history: list[dict[str, float]],
) -> dict[str, object]:
    """Final checkpoint payload for resume support."""
    return {
        "format_version": "exp0513_v3",
        "config": config.to_resolved_dict(),
        "epochs_added": config.epochs,
        "lambda": config.lambda_value,
        "coverage_c": coverage_c,
        "effective_dopamine_m": dopamine_m,
        "recommended_dopamine_m": recommended_dopamine_m,
        "dopamine_node_ids": assignment_metadata["selected_dopamine_node_ids"],
        "dopamine_assignment_metadata": assignment_metadata,
        "graph_payload": graph_payload,
        "edge_count": edge_count,
        "global_epoch_completed": global_epoch_completed,
        "global_loss_history": global_loss_history,
        "architecture": model.arch_dict(),
        "model_state_dict": model.state_dict(),
        "summary": summary,
        "source_checkpoint": source_checkpoint,
    }


def _make_summary(
    config: ExperimentConfig,
    train_history: list[float],
    val_history: list[float],
    local_loss_history: list[dict[str, float]],
    global_loss_history: list[dict[str, float]],
    dopamine_m: int,
    recommended_dopamine_m: int,
    coverage_c: int,
    edge_count: int,
    resume_info: dict[str, object] | None,
    assignment_metadata: dict[str, object],
    global_epoch_start: int,
    global_epoch_end: int,
    global_epoch_completed: int,
) -> dict[str, object]:
    """Compact JSON summary for the current run."""
    return {
        "run_name": config.run_name,
        "task_name": config.task_name,
        "architecture": _architecture_signature_dict(config),
        "epochs_added": len(train_history),
        "lambda": config.lambda_value,
        "local_loss_history": local_loss_history,
        "global_loss_history": global_loss_history,
        "final_train_loss": float(train_history[-1]),
        "final_val_loss": float(val_history[-1]),
        "best_val_loss": float(min(val_history)),
        "resume_from": config.resume_from,
        "coverage_c": coverage_c,
        "dopamine_m": dopamine_m,
        "recommended_dopamine_m": recommended_dopamine_m,
        "N": edge_count,
        "average_edge_coverage": assignment_metadata["average_edge_coverage"],
        "average_edges_per_dopamine": assignment_metadata["average_edges_per_dopamine"],
        "dopamine_node_ids": assignment_metadata["selected_dopamine_node_ids"],
        "c_r": assignment_metadata["c_r"],
        "resume_info": resume_info,
        "global_epoch_start": global_epoch_start,
        "global_epoch_end": global_epoch_end,
        "global_epoch_completed": global_epoch_completed,
    }


def run_experiment(
    config: ExperimentConfig | None = None,
    config_path: Path | None = None,
    run_dir: Path | None = None,
    live_status_path: Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    status_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run one exp0513 training job using hidden dopamine modulators."""
    config = config or ExperimentConfig()
    torch.manual_seed(config.seed)

    model = SelfModulatedMLP(
        input_dim=config.input_dim,
        trunk_dims=config.trunk_dims,
        y_dim=config.output_dim,
    )
    flat_weights, index_map = flatten_controllable_weights(model)
    edge_count = flat_weights.numel()
    edge_records = build_forward_edge_records(model)
    hidden_node_ids = model.hidden_node_ids()
    hidden_pool_size = len(hidden_node_ids)

    base_dir = Path(__file__).resolve().parents[1] / "runs"
    run_dir = run_dir or make_run_dir(base_dir, config.run_name)
    live_status_path = live_status_path or (run_dir / "live_viewer.json")
    if live_status_path.parent != run_dir:
        live_status_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        copy_config_to_run_dir(config_path, run_dir)
    else:
        dump_config_to_yaml(config, run_dir / "config.yaml")

    resume_info: dict[str, object] | None = None
    source_checkpoint: dict[str, object] | None = None
    if config.resume_from:
        source_checkpoint = load_experiment_checkpoint(config.resume_from)
        _assert_resume_architecture(config, source_checkpoint)
        model.load_state_dict(source_checkpoint["model_state_dict"])
        assignment_metadata = dict(source_checkpoint["dopamine_assignment_metadata"])
        B, B_tilde = rebuild_assignment_tensors(edge_records, assignment_metadata)
        dopamine_m = int(source_checkpoint["effective_dopamine_m"])
        recommended_dopamine_m = int(source_checkpoint.get("recommended_dopamine_m", recommend_dopamine_m(int(source_checkpoint["coverage_c"]))))
        coverage_c = int(source_checkpoint["coverage_c"])
        global_epoch_completed_before = int(source_checkpoint.get("global_epoch_completed", 0))
        graph_payload = dict(source_checkpoint.get("graph_payload") or build_graph_payload(model, assignment_metadata))
        inherited_global_loss_history = list(source_checkpoint.get("global_loss_history") or [])
        resume_info = {
            "path": str(Path(config.resume_from).resolve()),
            "source_lambda": source_checkpoint.get("lambda"),
            "source_epochs_added": source_checkpoint.get("epochs_added"),
            "source_global_epoch_completed": global_epoch_completed_before,
            "source_summary": source_checkpoint.get("summary"),
            "assignment_locked_from_checkpoint": True,
        }
    else:
        dopamine_m, recommended_dopamine_m = resolve_dopamine_m(
            coverage_c=config.coverage_c,
            hidden_pool_size=hidden_pool_size,
            dopamine_m_override=config.dopamine_m_override,
        )
        coverage_c = config.coverage_c
        B, B_tilde, assignment_metadata = build_dopamine_assignment(
            edge_records=edge_records,
            hidden_node_ids=hidden_node_ids,
            coverage_c=coverage_c,
            dopamine_m=dopamine_m,
            seed=config.seed,
        )
        graph_payload = build_graph_payload(model, assignment_metadata)
        global_epoch_completed_before = 0
        inherited_global_loss_history = []

    dopamine_node_ids = list(assignment_metadata["selected_dopamine_node_ids"])
    global_epoch_start = global_epoch_completed_before + 1
    global_epoch_end = global_epoch_completed_before + config.epochs

    write_resolved_config(
        config,
        extra={
            "architecture": model.arch_dict(),
            "N": edge_count,
            "hidden_pool_size": hidden_pool_size,
            "coverage_c_effective": coverage_c,
            "dopamine_m_effective": dopamine_m,
            "dopamine_m_recommended": recommended_dopamine_m,
            "global_epoch_start": global_epoch_start,
            "global_epoch_end": global_epoch_end,
            "assignment_locked_from_checkpoint": bool(config.resume_from),
        },
        output_path=run_dir / "resolved_config.json",
    )

    dataset = build_dataset(
        task_name=config.task_name,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_train=config.num_train,
        num_val=config.num_val,
        num_plot=config.num_plot,
        x_min=config.x_min,
        x_max=config.x_max,
        seed=config.seed,
    )
    loader = DataLoader(
        TensorDataset(dataset.x_train, dataset.y_train),
        batch_size=config.batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    local_loss_history: list[dict[str, float]] = []
    global_loss_history: list[dict[str, float]] = list(inherited_global_loss_history)
    edge_weight_snapshot = _build_edge_weight_snapshot(model, edge_records)
    node_activation_snapshot = {
        node["id"]: None
        for node in graph_payload["nodes"]
    }
    stopped_early = False

    write_live_status(
        live_status_path,
        _build_live_status(
            config=config,
            run_dir=run_dir,
            edge_count=edge_count,
            dopamine_m=dopamine_m,
            recommended_dopamine_m=recommended_dopamine_m,
            coverage_c=coverage_c,
            graph_payload=graph_payload,
            status="running",
            local_epoch=0,
            global_epoch=global_epoch_completed_before,
            global_epoch_start=global_epoch_start,
            global_epoch_end=global_epoch_end,
            train_loss=None,
            val_loss=None,
            best_val_loss=None,
            architecture=model.arch_dict(),
            local_loss_history=local_loss_history,
            global_loss_history=global_loss_history,
            node_activation_snapshot=node_activation_snapshot,
            edge_weight_snapshot=edge_weight_snapshot,
        ),
        status_callback=status_callback,
    )

    try:
        for local_epoch in range(1, config.epochs + 1):
            train_loss, metrics = _epoch_train(model, loader, B_tilde, index_map, dopamine_node_ids, config)
            val_loss = _evaluate(model, dataset.x_val, dataset.y_val)
            global_epoch = global_epoch_completed_before + local_epoch
            metrics.update({
                "epoch": local_epoch,
                "local_epoch": local_epoch,
                "global_epoch": global_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            history.append(metrics)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            history_point = {
                "local_epoch": float(local_epoch),
                "global_epoch": float(global_epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            local_loss_history.append(history_point)
            global_loss_history.append(history_point)
            node_activation_snapshot = _build_node_activation_snapshot(model, dataset.x_val)
            edge_weight_snapshot = _build_edge_weight_snapshot(model, edge_records)

            write_live_status(
                live_status_path,
                _build_live_status(
                    config=config,
                    run_dir=run_dir,
                    edge_count=edge_count,
                    dopamine_m=dopamine_m,
                    recommended_dopamine_m=recommended_dopamine_m,
                    coverage_c=coverage_c,
                    graph_payload=graph_payload,
                    status="running",
                    local_epoch=local_epoch,
                    global_epoch=global_epoch,
                    global_epoch_start=global_epoch_start,
                    global_epoch_end=global_epoch_end,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=min(val_losses),
                    architecture=model.arch_dict(),
                    local_loss_history=local_loss_history,
                    global_loss_history=global_loss_history,
                    node_activation_snapshot=node_activation_snapshot,
                    edge_weight_snapshot=edge_weight_snapshot,
                ),
                status_callback=status_callback,
            )

            if stop_requested is not None and stop_requested():
                stopped_early = True
                break
    except Exception as exc:
        current_global_epoch = global_epoch_completed_before + len(train_losses)
        write_live_status(
            live_status_path,
            _build_live_status(
                config=config,
                run_dir=run_dir,
                edge_count=edge_count,
                dopamine_m=dopamine_m,
                recommended_dopamine_m=recommended_dopamine_m,
                coverage_c=coverage_c,
                graph_payload=graph_payload,
                status="failed",
                local_epoch=len(train_losses),
                global_epoch=current_global_epoch,
                global_epoch_start=global_epoch_start,
                global_epoch_end=global_epoch_end,
                train_loss=train_losses[-1] if train_losses else None,
                val_loss=val_losses[-1] if val_losses else None,
                best_val_loss=min(val_losses) if val_losses else None,
                architecture=model.arch_dict(),
                local_loss_history=local_loss_history,
                global_loss_history=global_loss_history,
                node_activation_snapshot=node_activation_snapshot,
                edge_weight_snapshot=edge_weight_snapshot,
                error=str(exc),
            ),
            status_callback=status_callback,
        )
        raise

    global_epoch_completed = global_epoch_completed_before + len(train_losses)
    summary = _make_summary(
        config=config,
        train_history=train_losses,
        val_history=val_losses,
        local_loss_history=local_loss_history,
        global_loss_history=global_loss_history,
        dopamine_m=dopamine_m,
        recommended_dopamine_m=recommended_dopamine_m,
        coverage_c=coverage_c,
        edge_count=edge_count,
        resume_info=resume_info,
        assignment_metadata=assignment_metadata,
        global_epoch_start=global_epoch_start,
        global_epoch_end=global_epoch_end,
        global_epoch_completed=global_epoch_completed,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        _checkpoint_payload(
            model=model,
            config=config,
            assignment_metadata=assignment_metadata,
            graph_payload=graph_payload,
            edge_count=edge_count,
            summary=summary,
            source_checkpoint=source_checkpoint,
            global_epoch_completed=global_epoch_completed,
            dopamine_m=dopamine_m,
            recommended_dopamine_m=recommended_dopamine_m,
            coverage_c=coverage_c,
            global_loss_history=global_loss_history,
        ),
        run_dir / "model.pt",
    )

    plot_loss_curve(
        history,
        config,
        dopamine_m,
        recommended_dopamine_m,
        coverage_c,
        edge_count,
        run_dir / "loss_curve.png",
    )
    if config.enable_diagnostics:
        plot_dopamine_diagnostics(history, run_dir / "dopamine_diagnostics.png")
        plot_update_diagnostics(history, run_dir / "update_diagnostics.png")
        save_history_json(history, run_dir / "diagnostics_history.json")

    final_status = "stopped" if stopped_early else "completed"
    write_live_status(
        live_status_path,
        _build_live_status(
            config=config,
            run_dir=run_dir,
            edge_count=edge_count,
            dopamine_m=dopamine_m,
            recommended_dopamine_m=recommended_dopamine_m,
            coverage_c=coverage_c,
            graph_payload=graph_payload,
            status=final_status,
            local_epoch=len(train_losses),
            global_epoch=global_epoch_completed,
            global_epoch_start=global_epoch_start,
            global_epoch_end=global_epoch_end,
            train_loss=train_losses[-1],
            val_loss=val_losses[-1],
            best_val_loss=min(val_losses),
            architecture=model.arch_dict(),
            local_loss_history=local_loss_history,
            global_loss_history=global_loss_history,
            node_activation_snapshot=node_activation_snapshot,
            edge_weight_snapshot=edge_weight_snapshot,
        ),
        status_callback=status_callback,
    )

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "assignment_metadata": assignment_metadata,
        "status": final_status,
        "live_status_path": str(live_status_path),
    }
