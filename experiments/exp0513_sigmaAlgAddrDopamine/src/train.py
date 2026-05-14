"""Two-phase training loop for exp0513 V1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path

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
    from .data import build_sin_mix_dataset
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
        mix_phase_b_updates,
        phase_a_controllable_updates,
    )
except ImportError:  # pragma: no cover - script mode
    from assignment import build_static_assignment, flatten_controllable_weights
    from data import build_sin_mix_dataset
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
        mix_phase_b_updates,
        phase_a_controllable_updates,
    )


@dataclass
class ExperimentConfig:
    """Fixed V1 experiment configuration."""

    seed: int = 42
    n_in: int = 1
    trunk_dims: tuple[int, int] = (16, 16)
    y_dim: int = 1
    q_dim: int = 9
    num_train: int = 500
    num_val: int = 200
    num_plot: int = 500
    x_min: float = -2.0 * math.pi
    x_max: float = 2.0 * math.pi
    batch_size: int = 64
    lr_bp: float = 1e-2
    eta_int: float = 1e-4
    gamma: float = 1.0
    phase_a_epochs: int = 1000
    phase_b_epochs: int = 1000
    lambda_phase_b: float = 0.5
    run_name: str = "mixedsin_phase_switch"


def make_run_dir(base_dir: Path, run_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
    phase: str,
) -> tuple[float, dict[str, float]]:
    """Run one epoch of either Phase A or Phase B."""
    model.train()
    batch_losses: list[float] = []
    batch_metric_rows: list[dict[str, float]] = []
    q_head_start = index_map[-1].start
    q_head_end = index_map[-1].end
    bp_mask = torch.zeros(index_map[-1].end, dtype=torch.bool)
    bp_mask[:q_head_start] = True

    for bx, by in loader:
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        y_pred, q_batch = model(bx)
        loss = torch.mean((y_pred - by) ** 2)
        loss.backward()

        bp_flat, _ = extract_flat_bp_update(model, index_map, config.lr_bp)
        bias_updates = extract_bias_bp_updates(model, config.lr_bp)

        if phase == "phase_a":
            int_flat = torch.zeros_like(bp_flat)
            total_flat, controllable_updates = phase_a_controllable_updates(bp_flat, index_map)
            q_mean = q_batch.detach().mean(dim=0)
            s = torch.zeros_like(bp_flat)
            s[:q_head_start] = 0.0
            s[q_head_start:q_head_end] = 0.0
            total_mask = bp_mask
        elif phase == "phase_b":
            int_flat, s, q_mean = compute_internal_update(q_batch, B_tilde, config.eta_int, config.gamma)
            total_flat, controllable_updates, total_mask = mix_phase_b_updates(
                bp_flat=bp_flat,
                int_flat=int_flat,
                index_map=index_map,
                lambda_value=config.lambda_phase_b,
            )
        else:
            raise ValueError(f"Unsupported phase: {phase}")

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


def plot_fit_curves(history: list[dict[str, float]], phase_a_epochs: int, output_path: Path) -> None:
    """Plot train/validation MSE for both phases."""
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train", linewidth=1.8)
    ax.plot(epochs, [row["val_loss"] for row in history], label="val", linewidth=1.8)
    ax.axvline(phase_a_epochs, color="black", linestyle="--", linewidth=1, label="Phase switch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("exp0513 fit curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_prediction(
    x_plot: torch.Tensor,
    y_true: torch.Tensor,
    y_phase_a: torch.Tensor,
    y_phase_b: torch.Tensor,
    output_path: Path,
) -> None:
    """Overlay target with Phase A and Phase B predictions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_np = x_plot.squeeze(-1).cpu().numpy()
    ax.plot(x_np, y_true.squeeze(-1).cpu().numpy(), label="target", linewidth=2.0)
    ax.plot(x_np, y_phase_a.squeeze(-1).cpu().numpy(), label="phase_a", linewidth=1.6)
    ax.plot(x_np, y_phase_b.squeeze(-1).cpu().numpy(), label="phase_b", linewidth=1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("exp0513 sin_mix fit")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _checkpoint_payload(
    model: SelfModulatedMLP,
    config: ExperimentConfig,
    assignment_metadata: dict,
    phase: str,
    epoch: int,
) -> dict[str, object]:
    """Standard checkpoint payload for both phases."""
    return {
        "phase": phase,
        "epoch": epoch,
        "config": asdict(config),
        "assignment_metadata": assignment_metadata,
        "model_state_dict": model.state_dict(),
    }


def _phase_summary(
    phase: str,
    train_history: list[float],
    val_history: list[float],
    q_weight_change_norm: float,
) -> dict[str, object]:
    """Compact JSON summary for one phase."""
    return {
        "phase": phase,
        "epochs": len(train_history),
        "final_train_loss": float(train_history[-1]),
        "final_val_loss": float(val_history[-1]),
        "best_val_loss": float(min(val_history)),
        "q_head_weight_change_norm": float(q_weight_change_norm),
    }


def run_experiment(
    config: ExperimentConfig | None = None,
    run_dir: Path | None = None,
) -> dict[str, object]:
    """Run the full V1 two-phase mixed-sin experiment."""
    config = config or ExperimentConfig()
    torch.manual_seed(config.seed)

    q_dim = solve_v1_q_dim()
    if config.q_dim != q_dim:
        raise ValueError(f"V1 plan expects q_dim={q_dim}, got {config.q_dim}.")

    model = SelfModulatedMLP(
        input_dim=config.n_in,
        trunk_dims=config.trunk_dims,
        y_dim=config.y_dim,
        q_dim=config.q_dim,
    )
    flat_weights, index_map = flatten_controllable_weights(model)
    B, B_tilde, assignment_metadata = build_static_assignment(flat_weights.numel(), config.q_dim)

    dataset = build_sin_mix_dataset(
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

    base_dir = Path(__file__).resolve().parents[1] / "runs"
    run_dir = run_dir or make_run_dir(base_dir, config.run_name)

    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps({
        **asdict(config),
        "resolved_q_dim": q_dim,
        "architecture": model.arch_dict(),
    }, indent=2))
    (run_dir / "assignment_summary.json").write_text(json.dumps(assignment_metadata, indent=2))

    history: list[dict[str, float]] = []
    phase_a_train_losses: list[float] = []
    phase_a_val_losses: list[float] = []
    phase_b_train_losses: list[float] = []
    phase_b_val_losses: list[float] = []

    q_head_before_a = model.q_head.weight.detach().clone()
    y_phase_a_plot = None

    for epoch in range(1, config.phase_a_epochs + 1):
        train_loss, metrics = _epoch_train(model, loader, B_tilde, index_map, config, "phase_a")
        val_loss, _, _ = _evaluate(model, dataset.x_val, dataset.y_val)
        metrics.update({
            "epoch": epoch,
            "phase": "phase_a",
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        history.append(metrics)
        phase_a_train_losses.append(train_loss)
        phase_a_val_losses.append(val_loss)

    _, y_phase_a_plot, _ = _evaluate(model, dataset.x_plot, dataset.y_plot)
    q_weight_change_a = float((model.q_head.weight.detach() - q_head_before_a).norm().item())
    phase_a_summary = _phase_summary("phase_a", phase_a_train_losses, phase_a_val_losses, q_weight_change_a)
    (run_dir / "phase_a_summary.json").write_text(json.dumps(phase_a_summary, indent=2))
    torch.save(
        _checkpoint_payload(model, config, assignment_metadata, "phase_a", config.phase_a_epochs),
        run_dir / "checkpoint_phase_a.pt",
    )

    q_head_before_b = model.q_head.weight.detach().clone()
    for epoch in range(config.phase_a_epochs + 1, config.phase_a_epochs + config.phase_b_epochs + 1):
        train_loss, metrics = _epoch_train(model, loader, B_tilde, index_map, config, "phase_b")
        val_loss, _, _ = _evaluate(model, dataset.x_val, dataset.y_val)
        metrics.update({
            "epoch": epoch,
            "phase": "phase_b",
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        history.append(metrics)
        phase_b_train_losses.append(train_loss)
        phase_b_val_losses.append(val_loss)

    _, y_phase_b_plot, _ = _evaluate(model, dataset.x_plot, dataset.y_plot)
    q_weight_change_b = float((model.q_head.weight.detach() - q_head_before_b).norm().item())
    phase_b_summary = _phase_summary("phase_b", phase_b_train_losses, phase_b_val_losses, q_weight_change_b)
    (run_dir / "phase_b_summary.json").write_text(json.dumps(phase_b_summary, indent=2))
    torch.save(
        _checkpoint_payload(model, config, assignment_metadata, "phase_b", config.phase_b_epochs),
        run_dir / "checkpoint_phase_b.pt",
    )

    plot_fit_curves(history, config.phase_a_epochs, run_dir / "fit_curves.png")
    plot_prediction(dataset.x_plot, dataset.y_plot, y_phase_a_plot, y_phase_b_plot, run_dir / "prediction_plot.png")
    plot_q_diagnostics(history, config.phase_a_epochs, run_dir / "q_diagnostics.png")
    plot_update_diagnostics(history, config.phase_a_epochs, run_dir / "update_diagnostics.png")
    save_history_json(history, run_dir / "diagnostics_history.json")

    return {
        "run_dir": str(run_dir),
        "phase_a_summary": phase_a_summary,
        "phase_b_summary": phase_b_summary,
        "assignment_metadata": assignment_metadata,
    }
