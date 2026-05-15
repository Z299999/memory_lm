"""Training loop for exp0513 with single-run lambda control."""

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
    from .assignment import build_static_assignment, flatten_controllable_weights
    from .config import ExperimentConfig, copy_config_to_run_dir, dump_config_to_yaml, write_resolved_config
    from .data import build_dataset
    from .diagnostics import (
        average_epoch_metrics,
        collect_batch_metrics,
        plot_q_diagnostics,
        plot_update_diagnostics,
        save_history_json,
    )
    from .model import SelfModulatedMLP, solve_v1_q_dim
    from .update_rule import (
        apply_updates,
        compute_internal_update,
        extract_bias_bp_updates,
        extract_flat_bp_update,
        mix_controllable_updates,
    )
except ImportError:  # pragma: no cover - script mode
    from assignment import build_static_assignment, flatten_controllable_weights
    from config import ExperimentConfig, copy_config_to_run_dir, dump_config_to_yaml, write_resolved_config
    from data import build_dataset
    from diagnostics import (
        average_epoch_metrics,
        collect_batch_metrics,
        plot_q_diagnostics,
        plot_update_diagnostics,
        save_history_json,
    )
    from model import SelfModulatedMLP, solve_v1_q_dim
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


def _build_live_status(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    q_dim: int,
    edge_count: int,
    status: str,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
    best_val_loss: float | None,
    error: str | None = None,
) -> dict[str, object]:
    """Create the live dashboard payload for the current run state."""
    payload: dict[str, object] = {
        "run_name": config.run_name,
        "task_name": config.task_name,
        "status": status,
        "epoch": epoch,
        "epochs_total": config.epochs,
        "lambda": config.lambda_value,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "seed": config.seed,
        "m": q_dim,
        "N": edge_count,
        "resume_from": config.resume_from,
        "run_dir": str(run_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
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


def _evaluate(model: SelfModulatedMLP, x: torch.Tensor, y_true: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Evaluate full-batch MSE on a split."""
    model.eval()
    with torch.no_grad():
        y_pred, q = model(x)
        mse = torch.mean((y_pred - y_true) ** 2).item()
    return float(mse), y_pred, q


def _epoch_train(
    model: SelfModulatedMLP,
    loader: DataLoader,
    B_tilde: torch.Tensor,
    index_map,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float]]:
    """Run one epoch for the current fixed-lambda configuration."""
    model.train()
    batch_losses: list[float] = []
    batch_metric_rows: list[dict[str, float]] = []
    q_head_start = index_map[-1].start

    for bx, by in loader:
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        y_pred, q_batch = model(bx)
        loss = torch.mean((y_pred - by) ** 2)
        loss.backward()

        bp_flat, _ = extract_flat_bp_update(model, index_map, config.lr_bp)
        bias_updates = extract_bias_bp_updates(model, config.lr_bp)
        int_flat, s, q_mean = compute_internal_update(q_batch, B_tilde, config.eta_int, config.gamma)
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
                q_batch=q_batch.detach(),
                q_mean=q_mean.detach(),
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
    q_dim: int,
    edge_count: int,
    output_path: Path,
) -> None:
    """Plot train/validation MSE with a compact config text box."""
    epochs = [row["epoch"] for row in history]
    train_losses = [max(row["train_loss"], 1e-12) for row in history]
    val_losses = [max(row["val_loss"], 1e-12) for row in history]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, train_losses, label="train", linewidth=1.8)
    ax.plot(epochs, val_losses, label="val", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"exp0513 loss curve | task={config.task_name}")
    ax.grid(alpha=0.3)
    ax.legend()

    text = "\n".join([
        f"task={config.task_name}",
        f"seed={config.seed}",
        f"lr_bp={config.lr_bp}",
        f"eta_int={config.eta_int}",
        f"gamma={config.gamma}",
        f"lambda={config.lambda_value}",
        f"epochs={config.epochs}",
        f"batch={config.batch_size}",
        f"m={q_dim}",
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
    assignment_metadata: dict,
    edge_count: int,
    summary: dict[str, object],
    source_checkpoint: dict[str, object] | None,
) -> dict[str, object]:
    """Final checkpoint payload for resume support."""
    return {
        "format_version": "exp0513_v2",
        "config": config.to_resolved_dict(),
        "epochs": config.epochs,
        "lambda": config.lambda_value,
        "assignment_metadata": assignment_metadata,
        "edge_count": edge_count,
        "q_dim": solve_v1_q_dim(),
        "architecture": model.arch_dict(),
        "model_state_dict": model.state_dict(),
        "summary": summary,
        "source_checkpoint": source_checkpoint,
    }


def _make_summary(
    config: ExperimentConfig,
    train_history: list[float],
    val_history: list[float],
    q_weight_change_norm: float,
    q_dim: int,
    edge_count: int,
    resume_info: dict[str, object] | None,
) -> dict[str, object]:
    """Compact JSON summary for the current run."""
    return {
        "run_name": config.run_name,
        "task_name": config.task_name,
        "epochs": len(train_history),
        "lambda": config.lambda_value,
        "final_train_loss": float(train_history[-1]),
        "final_val_loss": float(val_history[-1]),
        "best_val_loss": float(min(val_history)),
        "q_head_weight_change_norm": float(q_weight_change_norm),
        "resume_from": config.resume_from,
        "m": q_dim,
        "N": edge_count,
        "resume_info": resume_info,
    }


def _load_resume_checkpoint(resume_from: str) -> dict[str, object]:
    """Load an old or new exp0513 checkpoint for continuation."""
    payload = torch.load(resume_from, map_location="cpu")
    if "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint at {resume_from} does not contain model_state_dict.")
    return payload


def run_experiment(
    config: ExperimentConfig | None = None,
    config_path: Path | None = None,
    run_dir: Path | None = None,
    live_status_path: Path | None = None,
    stop_requested: Callable[[], bool] | None = None,
    status_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run one exp0513 training job using the current lambda configuration."""
    config = config or ExperimentConfig()
    torch.manual_seed(config.seed)

    q_dim = solve_v1_q_dim()
    model = SelfModulatedMLP(
        q_dim=q_dim,
    )
    flat_weights, index_map = flatten_controllable_weights(model)
    edge_count = flat_weights.numel()
    _, B_tilde, assignment_metadata = build_static_assignment(edge_count, q_dim)

    base_dir = Path(__file__).resolve().parents[1] / "runs"
    run_dir = run_dir or make_run_dir(base_dir, config.run_name)
    live_status_path = live_status_path or (run_dir / "live_viewer.json")
    if live_status_path.parent != run_dir:
        live_status_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        copy_config_to_run_dir(config_path, run_dir)
    else:
        dump_config_to_yaml(config, run_dir / "config.yaml")
    write_resolved_config(
        config,
        extra={
            "resolved_q_dim": q_dim,
            "architecture": model.arch_dict(),
            "N": edge_count,
        },
        output_path=run_dir / "resolved_config.json",
    )

    resume_info: dict[str, object] | None = None
    if config.resume_from:
        resume_payload = _load_resume_checkpoint(config.resume_from)
        model.load_state_dict(resume_payload["model_state_dict"])
        resume_info = {
            "path": str(Path(config.resume_from).resolve()),
            "source_lambda": resume_payload.get("lambda"),
            "source_epochs": resume_payload.get("epochs"),
            "source_summary": resume_payload.get("summary"),
            "source_phase": resume_payload.get("phase"),
        }

    dataset = build_dataset(
        task_name=config.task_name,
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

    q_head_before = model.q_head.weight.detach().clone()
    stopped_early = False

    write_live_status(
        live_status_path,
        _build_live_status(
            config=config,
            run_dir=run_dir,
            q_dim=q_dim,
            edge_count=edge_count,
            status="running",
            epoch=0,
            train_loss=None,
            val_loss=None,
            best_val_loss=None,
        ),
        status_callback=status_callback,
    )

    try:
        for epoch in range(1, config.epochs + 1):
            train_loss, metrics = _epoch_train(model, loader, B_tilde, index_map, config)
            val_loss, _, _ = _evaluate(model, dataset.x_val, dataset.y_val)
            metrics.update({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            history.append(metrics)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            write_live_status(
                live_status_path,
                _build_live_status(
                    config=config,
                    run_dir=run_dir,
                    q_dim=q_dim,
                    edge_count=edge_count,
                    status="running",
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=min(val_losses),
                ),
                status_callback=status_callback,
            )

            if stop_requested is not None and stop_requested():
                stopped_early = True
                break
    except Exception as exc:
        write_live_status(
            live_status_path,
            _build_live_status(
                config=config,
                run_dir=run_dir,
                q_dim=q_dim,
                edge_count=edge_count,
                status="failed",
                epoch=len(train_losses),
                train_loss=train_losses[-1] if train_losses else None,
                val_loss=val_losses[-1] if val_losses else None,
                best_val_loss=min(val_losses) if val_losses else None,
                error=str(exc),
            ),
            status_callback=status_callback,
        )
        raise

    q_weight_change = float((model.q_head.weight.detach() - q_head_before).norm().item())
    summary = _make_summary(
        config=config,
        train_history=train_losses,
        val_history=val_losses,
        q_weight_change_norm=q_weight_change,
        q_dim=q_dim,
        edge_count=edge_count,
        resume_info=resume_info,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(
        _checkpoint_payload(
            model=model,
            config=config,
            assignment_metadata=assignment_metadata,
            edge_count=edge_count,
            summary=summary,
            source_checkpoint=resume_info,
        ),
        run_dir / "model.pt",
    )

    plot_loss_curve(history, config, q_dim, edge_count, run_dir / "loss_curve.png")
    if config.enable_diagnostics:
        plot_q_diagnostics(history, run_dir / "q_diagnostics.png")
        plot_update_diagnostics(history, run_dir / "update_diagnostics.png")
        save_history_json(history, run_dir / "diagnostics_history.json")

    final_status = "stopped" if stopped_early else "completed"
    write_live_status(
        live_status_path,
        _build_live_status(
            config=config,
            run_dir=run_dir,
            q_dim=q_dim,
            edge_count=edge_count,
            status=final_status,
            epoch=len(train_losses),
            train_loss=train_losses[-1],
            val_loss=val_losses[-1],
            best_val_loss=min(val_losses),
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
