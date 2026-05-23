"""Training loop for exp0522 constant-pulse external clock experiments."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .config import ExperimentConfig, copy_config_to_run_dir, write_resolved_config
    from .model import ExternalClockMLP
    from .plots import plot_rollout_diagnostics, plot_training_curves
    from .task import build_rollout_targets
except ImportError:  # pragma: no cover - script mode
    from config import ExperimentConfig, copy_config_to_run_dir, write_resolved_config
    from model import ExternalClockMLP
    from plots import plot_rollout_diagnostics, plot_training_curves
    from task import build_rollout_targets


def make_run_dir(base_dir: Path, run_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(output_path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, indent=2))


def _to_serializable_history(history: list[dict[str, float]]) -> list[dict[str, float]]:
    return [
        {key: float(value) for key, value in row.items()}
        for row in history
    ]


def _evaluate_rollout(
    model: ExternalClockMLP,
    *,
    num_steps: int,
    cycle_steps: int,
    pulse_value: float,
    disable_language: bool = False,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        phase, target = build_rollout_targets(num_steps, cycle_steps, device)
        prediction, messages, hidden = model.rollout(
            num_steps=num_steps,
            pulse_value=pulse_value,
            disable_language=disable_language,
            return_hidden=True,
        )
        mse = torch.mean((prediction - target) ** 2).item()
        message_norm = torch.linalg.norm(messages, dim=1) if messages.numel() else torch.zeros(num_steps, device=device)

    return {
        "mse": float(mse),
        "phase": phase.cpu(),
        "target": target.squeeze(1).cpu(),
        "prediction": prediction.squeeze(1).cpu(),
        "messages": messages.cpu(),
        "message_norm": message_norm.cpu(),
        "hidden": hidden.cpu() if hidden is not None else None,
    }


def _scheduled_train_steps(epoch: int, config: ExperimentConfig) -> int:
    """Return the rollout length used for the current training epoch."""
    if config.rollout_schedule == "fixed":
        return int(config.fixed_train_steps)

    # Default: short-to-long curriculum to stabilize oscillator learning.
    if config.train_steps <= config.cycle_steps:
        return config.train_steps

    stage_lengths = [
        config.cycle_steps,
        min(config.train_steps, 2 * config.cycle_steps),
        min(config.train_steps, 3 * config.cycle_steps),
        config.train_steps,
    ]
    stage_lengths = list(dict.fromkeys(stage_lengths))
    progress = epoch / max(config.epochs, 1)
    stage_idx = min(int(progress * len(stage_lengths)), len(stage_lengths) - 1)
    return int(stage_lengths[stage_idx])


def _train_single_model(
    *,
    model_name: str,
    model: ExternalClockMLP,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    target_cache: dict[int, torch.Tensor] = {}
    cache_steps = {
        _scheduled_train_steps(1, config),
        config.eval_steps,
        config.long_steps,
    }
    if config.rollout_schedule == "curriculum":
        cache_steps.update(
            {
                config.cycle_steps,
                min(config.train_steps, 2 * config.cycle_steps),
                min(config.train_steps, 3 * config.cycle_steps),
                config.train_steps,
            }
        )
    else:
        cache_steps.add(config.fixed_train_steps)

    for steps in cache_steps:
        _phase, target = build_rollout_targets(steps, config.cycle_steps, device)
        target_cache[int(steps)] = target

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        effective_steps = _scheduled_train_steps(epoch, config)
        train_target = target_cache[effective_steps]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _messages, _hidden = model.rollout(
            num_steps=effective_steps,
            pulse_value=config.pulse_value,
            disable_language=False,
            return_hidden=False,
        )
        train_loss = torch.mean((prediction - train_target) ** 2)
        train_loss.backward()
        if config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        val_result = _evaluate_rollout(
            model,
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            disable_language=False,
        )
        row = {
            "epoch": float(epoch),
            "train_steps": float(effective_steps),
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_result["mse"]),
        }
        history.append(row)

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            print(
                f"[{model_name}] epoch {epoch:4d}/{config.epochs} "
                f"steps={effective_steps:3d} "
                f"train={row['train_loss']:.6f} val={row['val_loss']:.6f}"
            )

    return {
        "model": model,
        "history": history,
    }


def _save_checkpoint(model: ExternalClockMLP, output_path: Path, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )


def _save_rollout_csv(
    *,
    output_path: Path,
    normal_eval: dict[str, Any],
    baseline_eval: dict[str, Any],
    mute_deaf_eval: dict[str, Any],
    language_dim: int,
) -> None:
    header = [
        "step",
        "phase",
        "target",
        "full_prediction",
        "baseline_prediction",
        "mute_deaf_prediction",
        "message_norm",
    ]
    header.extend([f"message_{idx}" for idx in range(language_dim)])

    rows = [header]
    num_steps = len(normal_eval["target"])
    for step in range(num_steps):
        row = [
            step,
            float(normal_eval["phase"][step].item()),
            float(normal_eval["target"][step].item()),
            float(normal_eval["prediction"][step].item()),
            float(baseline_eval["prediction"][step].item()),
            float(mute_deaf_eval["prediction"][step].item()),
            float(normal_eval["message_norm"][step].item()),
        ]
        if language_dim > 0:
            row.extend(
                float(normal_eval["messages"][step, idx].item())
                for idx in range(language_dim)
            )
        rows.append(row)
    output_path.write_text("\n".join(",".join(str(value) for value in row) for row in rows))


def _build_summary(
    *,
    config: ExperimentConfig,
    full_history: list[dict[str, float]],
    baseline_history: list[dict[str, float]],
    full_eval: dict[str, Any],
    full_long: dict[str, Any],
    full_mute_eval: dict[str, Any],
    full_mute_long: dict[str, Any],
    baseline_eval: dict[str, Any],
    baseline_long: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": {
            "run_name": config.run_name,
            "epochs": config.epochs,
            "rollout_schedule": config.rollout_schedule,
            "fixed_train_steps": config.fixed_train_steps,
            "trunk_dims": list(config.trunk_dims),
            "language_dim": config.language_dim,
            "language_readout_coverage": config.language_readout_coverage,
            "cycle_steps": config.cycle_steps,
            "train_steps": config.train_steps,
            "eval_steps": config.eval_steps,
            "long_steps": config.long_steps,
            "pulse_value": config.pulse_value,
            "seed": config.seed,
        },
        "full_language": {
            "final_train_mse": float(full_history[-1]["train_loss"]),
            "final_val_mse": float(full_history[-1]["val_loss"]),
            "eval_mse": float(full_eval["mse"]),
            "long_mse": float(full_long["mse"]),
            "mute_deaf_eval_mse": float(full_mute_eval["mse"]),
            "mute_deaf_long_mse": float(full_mute_long["mse"]),
        },
        "no_language_baseline": {
            "final_train_mse": float(baseline_history[-1]["train_loss"]),
            "final_val_mse": float(baseline_history[-1]["val_loss"]),
            "eval_mse": float(baseline_eval["mse"]),
            "long_mse": float(baseline_long["mse"]),
        },
        "comparisons": {
            "eval_gap_vs_baseline": float(baseline_eval["mse"] - full_eval["mse"]),
            "long_gap_vs_baseline": float(baseline_long["mse"] - full_long["mse"]),
            "eval_gap_vs_mute_deaf": float(full_mute_eval["mse"] - full_eval["mse"]),
            "long_gap_vs_mute_deaf": float(full_mute_long["mse"] - full_long["mse"]),
        },
        "success_checks": {
            "better_than_baseline_eval": bool(full_eval["mse"] < baseline_eval["mse"]),
            "better_than_baseline_long": bool(full_long["mse"] < baseline_long["mse"]),
            "better_than_mute_deaf_eval": bool(full_eval["mse"] < full_mute_eval["mse"]),
            "better_than_mute_deaf_long": bool(full_long["mse"] < full_mute_long["mse"]),
        },
    }


def run_experiment(config: ExperimentConfig, config_path: Path) -> dict[str, Any]:
    """Run the exp0522 constant-pulse external clock experiment."""
    _seed_everything(config.seed)

    root = config_path.resolve().parent
    run_root = (root / config.output_root).resolve()
    run_dir = make_run_dir(run_root, config.run_name)
    plots_dir = run_dir / "plots"
    ckpt_dir = run_dir / "checkpoints"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    copy_config_to_run_dir(config_path, run_dir)
    write_resolved_config(config, run_dir / "resolved_config.json")

    device = torch.device("cpu")

    full_model = ExternalClockMLP(
        trunk_dims=config.trunk_dims,
        language_dim=config.language_dim,
        language_readout_coverage=config.language_readout_coverage,
        use_language=True,
        seed=config.seed,
    ).to(device)
    baseline_model = ExternalClockMLP(
        trunk_dims=config.trunk_dims,
        language_dim=config.language_dim,
        language_readout_coverage=config.language_readout_coverage,
        use_language=False,
        seed=config.seed,
    ).to(device)

    full_result = _train_single_model(
        model_name="full_language",
        model=full_model,
        config=config,
        device=device,
    )
    baseline_result = _train_single_model(
        model_name="no_language",
        model=baseline_model,
        config=config,
        device=device,
    )

    full_eval = _evaluate_rollout(
        full_result["model"],
        num_steps=config.eval_steps,
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        disable_language=False,
    )
    full_long = _evaluate_rollout(
        full_result["model"],
        num_steps=config.long_steps,
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        disable_language=False,
    )
    full_mute_eval = _evaluate_rollout(
        full_result["model"],
        num_steps=config.eval_steps,
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        disable_language=True,
    )
    full_mute_long = _evaluate_rollout(
        full_result["model"],
        num_steps=config.long_steps,
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        disable_language=True,
    )
    baseline_eval = _evaluate_rollout(
        baseline_result["model"],
        num_steps=config.eval_steps,
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        disable_language=False,
    )
    baseline_long = _evaluate_rollout(
        baseline_result["model"],
        num_steps=config.long_steps,
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        disable_language=False,
    )

    _write_json(run_dir / "history_full_language.json", _to_serializable_history(full_result["history"]))
    _write_json(run_dir / "history_no_language.json", _to_serializable_history(baseline_result["history"]))

    _save_rollout_csv(
        output_path=run_dir / "eval_rollout.csv",
        normal_eval=full_eval,
        baseline_eval=baseline_eval,
        mute_deaf_eval=full_mute_eval,
        language_dim=config.language_dim,
    )
    _save_rollout_csv(
        output_path=run_dir / "long_rollout.csv",
        normal_eval=full_long,
        baseline_eval=baseline_long,
        mute_deaf_eval=full_mute_long,
        language_dim=config.language_dim,
    )

    plot_training_curves(
        full_history=full_result["history"],
        baseline_history=baseline_result["history"],
        output_path=plots_dir / "training_curves.png",
        config=config,
    )
    plot_rollout_diagnostics(
        short_full=full_eval,
        short_baseline=baseline_eval,
        short_mute_deaf=full_mute_eval,
        long_full=full_long,
        long_baseline=baseline_long,
        long_mute_deaf=full_mute_long,
        output_path=plots_dir / "rollout_diagnostics.png",
        config=config,
    )

    _save_checkpoint(
        full_result["model"],
        ckpt_dir / "full_language.pt",
        metadata={"config": config.to_resolved_dict(), "model_name": "full_language"},
    )
    _save_checkpoint(
        baseline_result["model"],
        ckpt_dir / "no_language.pt",
        metadata={"config": config.to_resolved_dict(), "model_name": "no_language"},
    )

    summary = _build_summary(
        config=config,
        full_history=full_result["history"],
        baseline_history=baseline_result["history"],
        full_eval=full_eval,
        full_long=full_long,
        full_mute_eval=full_mute_eval,
        full_mute_long=full_mute_long,
        baseline_eval=baseline_eval,
        baseline_long=baseline_long,
    )
    _write_json(run_dir / "summary.json", summary)

    print("Final summary:")
    print(json.dumps(summary, indent=2))

    return {
        "run_dir": str(run_dir),
        "summary": summary,
    }
