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
    now = datetime.now()
    date_dir = base_dir / now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_dir = date_dir / f"{timestamp}_{run_name}"
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
    target_kind: str,
    mixed_sin_second_harmonic_amplitude: float,
    start_step: int = 0,
    initial_message: torch.Tensor | None = None,
    disable_language: bool = False,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        phase, target = build_rollout_targets(
            num_steps,
            cycle_steps,
            device,
            start_step=start_step,
            target_kind=target_kind,
            mixed_sin_second_harmonic_amplitude=mixed_sin_second_harmonic_amplitude,
        )
        prediction, messages, hidden, final_message = model.rollout(
            num_steps=num_steps,
            pulse_value=pulse_value,
            initial_message=initial_message,
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
        "final_message": final_message.cpu(),
        "start_step": int(start_step),
    }


def _evaluate_continuous_stream(
    model: ExternalClockMLP,
    *,
    measured_steps: int,
    warmup_steps: int,
    cycle_steps: int,
    pulse_value: float,
    target_kind: str,
    mixed_sin_second_harmonic_amplitude: float,
    disable_language: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    initial_message = None
    if warmup_steps > 0:
        with torch.no_grad():
            _prediction, _messages, _hidden, final_message = model.rollout(
                num_steps=warmup_steps,
                pulse_value=pulse_value,
                initial_message=None,
                disable_language=disable_language,
                return_hidden=False,
            )
            if model.use_language:
                initial_message = final_message.detach()

    return _evaluate_rollout(
        model,
        num_steps=measured_steps,
        cycle_steps=cycle_steps,
        pulse_value=pulse_value,
        target_kind=target_kind,
        mixed_sin_second_harmonic_amplitude=mixed_sin_second_harmonic_amplitude,
        start_step=warmup_steps,
        initial_message=initial_message,
        disable_language=disable_language,
    )


def _build_train_target(
    *,
    num_steps: int,
    cycle_steps: int,
    device: torch.device,
    train_phase_mode: str,
    start_step: int,
    target_kind: str,
    mixed_sin_second_harmonic_amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = start_step if train_phase_mode == "continuous" else 0
    return build_rollout_targets(
        num_steps,
        cycle_steps,
        device,
        start_step=offset,
        target_kind=target_kind,
        mixed_sin_second_harmonic_amplitude=mixed_sin_second_harmonic_amplitude,
    )


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
    target_cache: dict[tuple[int, int], torch.Tensor] = {}
    cache_steps = {
        config.fixed_train_steps,
        config.eval_steps,
        config.long_steps,
        config.continuous_eval_steps,
    }

    if config.train_phase_mode == "reset":
        for steps in cache_steps:
            _phase, target = _build_train_target(
                num_steps=steps,
                cycle_steps=config.cycle_steps,
                device=device,
                train_phase_mode="reset",
                start_step=0,
                target_kind=config.target_kind,
                mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            )
            target_cache[(int(steps), 0)] = target

    history: list[dict[str, float]] = []
    train_time_cursor = 0
    train_message_state: torch.Tensor | None = None
    for epoch in range(1, config.epochs + 1):
        effective_steps = int(config.fixed_train_steps)
        train_window_start = train_time_cursor if config.train_phase_mode == "continuous" else 0
        if config.train_phase_mode == "reset":
            train_target = target_cache[(effective_steps, 0)]
        else:
            _phase, train_target = _build_train_target(
                num_steps=effective_steps,
                cycle_steps=config.cycle_steps,
                device=device,
                train_phase_mode=config.train_phase_mode,
                start_step=train_window_start,
                target_kind=config.target_kind,
                mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _messages, _hidden, final_message = model.rollout(
            num_steps=effective_steps,
            pulse_value=config.pulse_value,
            initial_message=train_message_state,
            disable_language=False,
            return_hidden=False,
        )
        train_loss = torch.mean((prediction - train_target) ** 2)
        train_loss.backward()
        if config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if config.sequence_mode == "continuous_window" and model.use_language:
            train_message_state = final_message.detach()
        elif config.sequence_mode == "continuous_window":
            train_message_state = None
        else:
            train_message_state = None

        if config.train_phase_mode == "continuous":
            train_time_cursor += effective_steps

        val_result = _evaluate_rollout(
            model,
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        row = {
            "epoch": float(epoch),
            "train_steps": float(effective_steps),
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_result["mse"]),
        }
        if config.sequence_mode == "continuous_window":
            row["train_window_start"] = float(train_window_start)
            row["train_window_end"] = float(train_window_start + effective_steps - 1)
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
        "global_step",
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
            int(normal_eval.get("start_step", 0)) + step,
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
    full_reset_eval: dict[str, Any] | None,
    full_reset_long: dict[str, Any] | None,
    full_continuous_eval: dict[str, Any] | None,
    full_mute_reset_eval: dict[str, Any] | None,
    full_mute_reset_long: dict[str, Any] | None,
    full_mute_continuous_eval: dict[str, Any] | None,
    baseline_reset_eval: dict[str, Any] | None,
    baseline_reset_long: dict[str, Any] | None,
    baseline_continuous_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    def _mse(result: dict[str, Any] | None) -> float | None:
        return None if result is None else float(result["mse"])

    def _gap(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> float | None:
        if lhs is None or rhs is None:
            return None
        return float(lhs["mse"] - rhs["mse"])

    def _better(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None) -> bool | None:
        if lhs is None or rhs is None:
            return None
        return bool(lhs["mse"] < rhs["mse"])

    return {
        "config": {
            "run_name": config.run_name,
            "epochs": config.epochs,
            "sequence_mode": config.sequence_mode,
            "fixed_train_steps": config.fixed_train_steps,
            "trunk_dims": list(config.trunk_dims),
            "language_dim": config.language_dim,
            "language_readout_coverage": config.language_readout_coverage,
            "cycle_steps": config.cycle_steps,
            "target_kind": config.target_kind,
            "mixed_sin_second_harmonic_amplitude": config.mixed_sin_second_harmonic_amplitude,
            "eval_steps": config.eval_steps,
            "long_steps": config.long_steps,
            "continuous_eval_steps": config.continuous_eval_steps,
            "pulse_value": config.pulse_value,
            "train_phase_mode": config.train_phase_mode,
            "eval_phase_mode": config.eval_phase_mode,
            "seed": config.seed,
        },
        "full_language": {
            "final_train_mse": float(full_history[-1]["train_loss"]),
            "final_val_mse": float(full_history[-1]["val_loss"]),
            "reset_eval_mse": _mse(full_reset_eval),
            "reset_long_mse": _mse(full_reset_long),
            "continuous_eval_mse": _mse(full_continuous_eval),
            "mute_deaf_reset_eval_mse": _mse(full_mute_reset_eval),
            "mute_deaf_reset_long_mse": _mse(full_mute_reset_long),
            "mute_deaf_continuous_eval_mse": _mse(full_mute_continuous_eval),
        },
        "no_language_baseline": {
            "final_train_mse": float(baseline_history[-1]["train_loss"]),
            "final_val_mse": float(baseline_history[-1]["val_loss"]),
            "reset_eval_mse": _mse(baseline_reset_eval),
            "reset_long_mse": _mse(baseline_reset_long),
            "continuous_eval_mse": _mse(baseline_continuous_eval),
        },
        "comparisons": {
            "reset_eval_gap_vs_baseline": _gap(baseline_reset_eval, full_reset_eval),
            "reset_long_gap_vs_baseline": _gap(baseline_reset_long, full_reset_long),
            "continuous_eval_gap_vs_baseline": _gap(baseline_continuous_eval, full_continuous_eval),
            "reset_eval_gap_vs_mute_deaf": _gap(full_mute_reset_eval, full_reset_eval),
            "reset_long_gap_vs_mute_deaf": _gap(full_mute_reset_long, full_reset_long),
            "continuous_eval_gap_vs_mute_deaf": _gap(full_mute_continuous_eval, full_continuous_eval),
        },
        "success_checks": {
            "better_than_baseline_reset_eval": _better(full_reset_eval, baseline_reset_eval),
            "better_than_baseline_reset_long": _better(full_reset_long, baseline_reset_long),
            "better_than_baseline_continuous_eval": _better(full_continuous_eval, baseline_continuous_eval),
            "better_than_mute_deaf_reset_eval": _better(full_reset_eval, full_mute_reset_eval),
            "better_than_mute_deaf_reset_long": _better(full_reset_long, full_mute_reset_long),
            "better_than_mute_deaf_continuous_eval": _better(full_continuous_eval, full_mute_continuous_eval),
        },
    }


def run_experiment(config: ExperimentConfig, config_path: Path) -> dict[str, Any]:
    """Run the exp0522 constant-pulse external clock experiment."""
    _seed_everything(config.seed)

    root = config_path.resolve().parent
    run_root = (root / config.output_root).resolve()
    run_dir = make_run_dir(run_root, config.run_name)
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
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

    reset_eval_enabled = config.eval_phase_mode in {"reset", "both"}
    continuous_eval_enabled = config.eval_phase_mode in {"continuous", "both"}
    continuous_warmup_steps = config.fixed_train_steps

    full_reset_eval = (
        _evaluate_rollout(
            full_result["model"],
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        if reset_eval_enabled
        else None
    )
    full_reset_long = (
        _evaluate_rollout(
            full_result["model"],
            num_steps=config.long_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        if reset_eval_enabled
        else None
    )
    full_mute_reset_eval = (
        _evaluate_rollout(
            full_result["model"],
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=True,
        )
        if reset_eval_enabled
        else None
    )
    full_mute_reset_long = (
        _evaluate_rollout(
            full_result["model"],
            num_steps=config.long_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=True,
        )
        if reset_eval_enabled
        else None
    )
    baseline_reset_eval = (
        _evaluate_rollout(
            baseline_result["model"],
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        if reset_eval_enabled
        else None
    )
    baseline_reset_long = (
        _evaluate_rollout(
            baseline_result["model"],
            num_steps=config.long_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        if reset_eval_enabled
        else None
    )

    full_continuous_eval = (
        _evaluate_continuous_stream(
            full_result["model"],
            measured_steps=config.continuous_eval_steps,
            warmup_steps=continuous_warmup_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        if continuous_eval_enabled
        else None
    )
    full_mute_continuous_eval = (
        _evaluate_continuous_stream(
            full_result["model"],
            measured_steps=config.continuous_eval_steps,
            warmup_steps=continuous_warmup_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=True,
        )
        if continuous_eval_enabled
        else None
    )
    baseline_continuous_eval = (
        _evaluate_continuous_stream(
            baseline_result["model"],
            measured_steps=config.continuous_eval_steps,
            warmup_steps=continuous_warmup_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_second_harmonic_amplitude=config.mixed_sin_second_harmonic_amplitude,
            disable_language=False,
        )
        if continuous_eval_enabled
        else None
    )

    _write_json(metrics_dir / "history_full_language.json", _to_serializable_history(full_result["history"]))
    _write_json(metrics_dir / "history_no_language.json", _to_serializable_history(baseline_result["history"]))

    if full_reset_eval is not None and baseline_reset_eval is not None and full_mute_reset_eval is not None:
        _save_rollout_csv(
            output_path=metrics_dir / "reset_eval_rollout.csv",
            normal_eval=full_reset_eval,
            baseline_eval=baseline_reset_eval,
            mute_deaf_eval=full_mute_reset_eval,
            language_dim=config.language_dim,
        )
        _save_rollout_csv(
            output_path=metrics_dir / "eval_rollout.csv",
            normal_eval=full_reset_eval,
            baseline_eval=baseline_reset_eval,
            mute_deaf_eval=full_mute_reset_eval,
            language_dim=config.language_dim,
        )
    if full_reset_long is not None and baseline_reset_long is not None and full_mute_reset_long is not None:
        _save_rollout_csv(
            output_path=metrics_dir / "reset_long_rollout.csv",
            normal_eval=full_reset_long,
            baseline_eval=baseline_reset_long,
            mute_deaf_eval=full_mute_reset_long,
            language_dim=config.language_dim,
        )
        _save_rollout_csv(
            output_path=metrics_dir / "long_rollout.csv",
            normal_eval=full_reset_long,
            baseline_eval=baseline_reset_long,
            mute_deaf_eval=full_mute_reset_long,
            language_dim=config.language_dim,
        )
    if (
        full_continuous_eval is not None
        and baseline_continuous_eval is not None
        and full_mute_continuous_eval is not None
    ):
        _save_rollout_csv(
            output_path=metrics_dir / "continuous_eval_rollout.csv",
            normal_eval=full_continuous_eval,
            baseline_eval=baseline_continuous_eval,
            mute_deaf_eval=full_mute_continuous_eval,
            language_dim=config.language_dim,
        )

    plot_training_curves(
        full_history=full_result["history"],
        baseline_history=baseline_result["history"],
        output_path=plots_dir / "training_curves.png",
        config=config,
    )
    plot_rollout_diagnostics(
        short_full=full_reset_eval or full_continuous_eval,
        short_baseline=baseline_reset_eval or baseline_continuous_eval,
        short_mute_deaf=full_mute_reset_eval or full_mute_continuous_eval,
        long_full=full_reset_long or full_continuous_eval,
        long_baseline=baseline_reset_long or baseline_continuous_eval,
        long_mute_deaf=full_mute_reset_long or full_mute_continuous_eval,
        continuous_long_full=full_continuous_eval,
        continuous_long_baseline=baseline_continuous_eval,
        continuous_long_mute_deaf=full_mute_continuous_eval,
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
        full_reset_eval=full_reset_eval,
        full_reset_long=full_reset_long,
        full_continuous_eval=full_continuous_eval,
        full_mute_reset_eval=full_mute_reset_eval,
        full_mute_reset_long=full_mute_reset_long,
        full_mute_continuous_eval=full_mute_continuous_eval,
        baseline_reset_eval=baseline_reset_eval,
        baseline_reset_long=baseline_reset_long,
        baseline_continuous_eval=baseline_continuous_eval,
    )
    _write_json(metrics_dir / "summary.json", summary)

    print("Final summary:")
    print(json.dumps(summary, indent=2))

    return {
        "run_dir": str(run_dir),
        "summary": summary,
    }
