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
    from .plots import plot_rollout_diagnostics, plot_training_curves, plot_training_timeline
    from .task import build_rollout_targets
except ImportError:  # pragma: no cover - script mode
    from config import ExperimentConfig, copy_config_to_run_dir, write_resolved_config
    from model import ExternalClockMLP
    from plots import plot_rollout_diagnostics, plot_training_curves, plot_training_timeline
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
    mixed_sin_components: tuple[tuple[float, float], ...],
    start_step: int = 0,
    initial_message: torch.Tensor | None = None,
    initial_error: torch.Tensor | None = None,
    detach_error_input: bool = True,
    force_zero_error_input: bool = False,
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
            mixed_sin_components=mixed_sin_components,
        )
        prediction, messages, hidden, final_message, final_error = model.rollout(
            num_steps=num_steps,
            pulse_value=pulse_value,
            target_sequence=target,
            initial_message=initial_message,
            initial_error=initial_error,
            detach_error_input=detach_error_input,
            force_zero_error_input=force_zero_error_input,
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
        "final_error": final_error.cpu(),
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
    mixed_sin_components: tuple[tuple[float, float], ...],
    detach_error_input: bool = True,
    force_zero_error_input: bool = False,
    disable_language: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    initial_message = None
    initial_error = None
    if warmup_steps > 0:
        with torch.no_grad():
            _warmup_phase, warmup_target = build_rollout_targets(
                warmup_steps,
                cycle_steps,
                device,
                start_step=0,
                target_kind=target_kind,
                mixed_sin_components=mixed_sin_components,
            )
            _prediction, _messages, _hidden, final_message, final_error = model.rollout(
                num_steps=warmup_steps,
                pulse_value=pulse_value,
                target_sequence=warmup_target,
                initial_message=None,
                initial_error=None,
                detach_error_input=detach_error_input,
                force_zero_error_input=force_zero_error_input,
                disable_language=disable_language,
                return_hidden=False,
            )
            if model.use_language:
                initial_message = final_message.detach()
            if model.use_error_input:
                initial_error = final_error.detach()

    return _evaluate_rollout(
        model,
        num_steps=measured_steps,
        cycle_steps=cycle_steps,
        pulse_value=pulse_value,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
        start_step=warmup_steps,
        initial_message=initial_message,
        initial_error=initial_error,
        detach_error_input=detach_error_input,
        force_zero_error_input=force_zero_error_input,
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
    mixed_sin_components: tuple[tuple[float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = start_step if train_phase_mode == "continuous" else 0
    return build_rollout_targets(
        num_steps,
        cycle_steps,
        device,
        start_step=offset,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
    )


def _analysis_checkpoint_epochs(config: ExperimentConfig) -> tuple[int, ...]:
    if not config.enable_continuous_collapse:
        return ()
    return tuple(sorted({epoch for epoch in config.checkpoint_epochs if 1 <= epoch < config.epochs}))


def _build_training_timeline_panels(config: ExperimentConfig) -> list[dict[str, Any]]:
    total_steps = int(config.epochs) * int(config.fixed_train_steps)
    if total_steps <= 0:
        return []
    window_steps = min(int(config.plot_training_timeline_window_steps), total_steps)
    max_start = max(total_steps - window_steps, 0)
    panel_count = int(config.plot_training_timeline_num_panels)
    if panel_count == 1 or max_start == 0:
        start_candidates = [0]
    else:
        start_candidates = [int(round(value)) for value in np.linspace(0, max_start, panel_count)]
    seen: set[int] = set()
    starts: list[int] = []
    for start in start_candidates:
        if start not in seen:
            starts.append(start)
            seen.add(start)
    return [
        {
            "start_step": int(start),
            "end_step": int(start + window_steps - 1),
            "global_step": [],
            "target": [],
            "prediction": [],
            "update_steps": [],
        }
        for start in starts
    ]


def _record_training_timeline_overlap(
    *,
    panels: list[dict[str, Any]],
    train_window_start: int,
    train_window_end: int,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> None:
    if not panels:
        return
    target_np = target.squeeze(1).detach().cpu().numpy()
    prediction_np = prediction.squeeze(1).detach().cpu().numpy()
    for panel in panels:
        overlap_start = max(train_window_start, int(panel["start_step"]))
        overlap_end = min(train_window_end, int(panel["end_step"]))
        if overlap_start > overlap_end:
            continue
        local_start = overlap_start - train_window_start
        local_end = overlap_end - train_window_start
        steps = list(range(overlap_start, overlap_end + 1))
        panel["global_step"].extend(steps)
        panel["target"].extend(float(value) for value in target_np[local_start : local_end + 1])
        panel["prediction"].extend(float(value) for value in prediction_np[local_start : local_end + 1])
        if panel["start_step"] <= train_window_end <= panel["end_step"]:
            panel["update_steps"].append(int(train_window_end))


def _train_single_model(
    *,
    model_name: str,
    model: ExternalClockMLP,
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    checkpoint_epochs: tuple[int, ...] = (),
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
                mixed_sin_components=config.mixed_sin_components,
            )
            target_cache[(int(steps), 0)] = target

    history: list[dict[str, float]] = []
    timeline_panels = (
        _build_training_timeline_panels(config)
        if config.sequence_mode == "continuous_window" and config.plot_show_training_timeline
        else []
    )
    train_time_cursor = 0
    train_message_state: torch.Tensor | None = None
    train_error_state: torch.Tensor | None = None
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
                mixed_sin_components=config.mixed_sin_components,
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, messages, _hidden, final_message, final_error = model.rollout(
            num_steps=effective_steps,
            pulse_value=config.pulse_value,
            target_sequence=train_target,
            initial_message=train_message_state,
            initial_error=train_error_state,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=False,
            return_hidden=False,
        )
        train_loss = torch.mean((prediction - train_target) ** 2)

        use_aux = (
            config.sequence_mode == "continuous_window"
            and model.use_language
            and config.message_aux_loss_weight > 0.0
        )
        aux_loss_val = 0.0
        if use_aux:
            _, aux_target = _build_train_target(
                num_steps=1,
                cycle_steps=config.cycle_steps,
                device=device,
                train_phase_mode="continuous",
                start_step=train_window_start + effective_steps,
                target_kind=config.target_kind,
                mixed_sin_components=config.mixed_sin_components,
            )
            aux_pred, _, _, _, _ = model.rollout(
                num_steps=1,
                pulse_value=config.pulse_value,
                target_sequence=aux_target,
                initial_message=final_message,
                initial_error=final_error,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                disable_language=False,
                return_hidden=False,
            )
            aux_loss = torch.mean((aux_pred - aux_target) ** 2)
            total_loss = train_loss + config.message_aux_loss_weight * aux_loss
            aux_loss_val = float(aux_loss.item())
        else:
            total_loss = train_loss

        total_loss.backward()
        if config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if config.sequence_mode == "continuous_window":
            _record_training_timeline_overlap(
                panels=timeline_panels,
                train_window_start=int(train_window_start),
                train_window_end=int(train_window_start + effective_steps - 1),
                target=train_target,
                prediction=prediction,
            )

        if config.sequence_mode == "continuous_window" and model.use_language:
            train_message_state = final_message.detach()
        elif config.sequence_mode == "continuous_window":
            train_message_state = None
        else:
            train_message_state = None
        if config.sequence_mode == "continuous_window" and model.use_error_input and config.carry_error_between_windows:
            train_error_state = final_error.detach()
        else:
            train_error_state = None

        if config.train_phase_mode == "continuous":
            train_time_cursor += effective_steps

        val_result = _evaluate_rollout(
            model,
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=False,
        )
        row = {
            "epoch": float(epoch),
            "train_steps": float(effective_steps),
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_result["mse"]),
        }
        if use_aux:
            row["aux_loss"] = aux_loss_val
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

        if checkpoint_dir is not None and epoch in checkpoint_epochs:
            _save_checkpoint(
                model,
                checkpoint_dir / f"{model_name}_epoch_{epoch:04d}.pt",
                metadata={
                    "config": config.to_resolved_dict(),
                    "model_name": model_name,
                    "epoch": int(epoch),
                    "checkpoint_kind": "milestone",
                },
            )

    return {
        "model": model,
        "history": history,
        "training_timeline": {
            "panels": timeline_panels,
        } if timeline_panels else None,
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
    baseline_eval: dict[str, Any] | None,
    mute_deaf_eval: dict[str, Any] | None,
    language_dim: int,
    normal_column_label: str = "full_prediction",
    baseline_column_label: str = "baseline_prediction",
    mute_column_label: str = "mute_deaf_prediction",
) -> None:
    header = [
        "step",
        "global_step",
        "phase",
        "target",
        normal_column_label,
        "message_norm",
    ]
    if baseline_eval is not None:
        header.append(baseline_column_label)
    if mute_deaf_eval is not None:
        header.append(mute_column_label)
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
            float(normal_eval["message_norm"][step].item()),
        ]
        if baseline_eval is not None:
            row.append(float(baseline_eval["prediction"][step].item()))
        if mute_deaf_eval is not None:
            row.append(float(mute_deaf_eval["prediction"][step].item()))
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
    baseline_history: list[dict[str, float]] | None,
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
            "train_baseline": config.train_baseline,
            "eval_mute_deaf": config.eval_mute_deaf,
            "epochs": config.epochs,
            "sequence_mode": config.sequence_mode,
            "fixed_train_steps": config.fixed_train_steps,
            "message_aux_loss_weight": config.message_aux_loss_weight,
            "detach_error_input": config.detach_error_input,
            "carry_error_between_windows": config.carry_error_between_windows,
            "force_zero_error_input": config.force_zero_error_input,
            "trunk_dims": list(config.trunk_dims),
            "activation": config.activation,
            "language_dim": config.language_dim,
            "language_readout_coverage": config.language_readout_coverage,
            "use_error_input": config.use_error_input,
            "cycle_steps": config.cycle_steps,
            "target_kind": config.target_kind,
            "mixed_sin_components": [list(c) for c in config.mixed_sin_components],
            "eval_steps": config.eval_steps,
            "long_steps": config.long_steps,
            "continuous_eval_steps": config.continuous_eval_steps,
            "enable_continuous_collapse": config.enable_continuous_collapse,
            "checkpoint_epochs": list(config.checkpoint_epochs),
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
        "no_language_baseline": None if baseline_history is None else {
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

    reset_eval_enabled = config.eval_phase_mode in {"reset", "both"}
    continuous_eval_enabled = config.eval_phase_mode in {"continuous", "both"}
    continuous_warmup_steps = config.fixed_train_steps

    full_model = ExternalClockMLP(
        trunk_dims=config.trunk_dims,
        activation=config.activation,
        language_dim=config.language_dim,
        language_readout_coverage=config.language_readout_coverage,
        use_error_input=config.use_error_input,
        use_language=True,
        seed=config.seed,
    ).to(device)
    full_checkpoint_epochs = _analysis_checkpoint_epochs(config)
    full_result = _train_single_model(
        model_name="full_language",
        model=full_model,
        config=config,
        device=device,
        checkpoint_dir=ckpt_dir,
        checkpoint_epochs=full_checkpoint_epochs,
    )
    baseline_result: dict[str, Any] | None = None
    if config.train_baseline:
        baseline_model = ExternalClockMLP(
            trunk_dims=config.trunk_dims,
            activation=config.activation,
            language_dim=config.language_dim,
            language_readout_coverage=config.language_readout_coverage,
            use_error_input=False,
            use_language=False,
            seed=config.seed,
        ).to(device)
        baseline_result = _train_single_model(
            model_name="no_language",
            model=baseline_model,
            config=config,
            device=device,
        )

    full_reset_eval = (
        _evaluate_rollout(
            full_result["model"],
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
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
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
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
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=True,
        )
        if reset_eval_enabled and config.eval_mute_deaf
        else None
    )
    full_mute_reset_long = (
        _evaluate_rollout(
            full_result["model"],
            num_steps=config.long_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=True,
        )
        if reset_eval_enabled and config.eval_mute_deaf
        else None
    )
    baseline_reset_eval = (
        _evaluate_rollout(
            baseline_result["model"],
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=False,
            disable_language=False,
        )
        if reset_eval_enabled and baseline_result is not None
        else None
    )
    baseline_reset_long = (
        _evaluate_rollout(
            baseline_result["model"],
            num_steps=config.long_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=False,
            disable_language=False,
        )
        if reset_eval_enabled and baseline_result is not None
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
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
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
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=True,
        )
        if continuous_eval_enabled and config.eval_mute_deaf
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
            mixed_sin_components=config.mixed_sin_components,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=False,
            disable_language=False,
        )
        if continuous_eval_enabled and baseline_result is not None
        else None
    )

    _write_json(metrics_dir / "history_full_language.json", _to_serializable_history(full_result["history"]))
    if full_result["training_timeline"] is not None:
        _write_json(metrics_dir / "training_timeline.json", full_result["training_timeline"])
    if baseline_result is not None:
        _write_json(metrics_dir / "history_no_language.json", _to_serializable_history(baseline_result["history"]))

    if full_reset_eval is not None and (baseline_reset_eval is not None or full_mute_reset_eval is not None):
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
    if full_reset_long is not None and (baseline_reset_long is not None or full_mute_reset_long is not None):
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
    if full_continuous_eval is not None and (baseline_continuous_eval is not None or full_mute_continuous_eval is not None):
        _save_rollout_csv(
            output_path=metrics_dir / "continuous_eval_rollout.csv",
            normal_eval=full_continuous_eval,
            baseline_eval=baseline_continuous_eval,
            mute_deaf_eval=full_mute_continuous_eval,
            language_dim=config.language_dim,
        )

    plot_training_curves(
        full_history=full_result["history"],
        baseline_history=None if baseline_result is None else baseline_result["history"],
        output_path=plots_dir / "training_curves.png",
        config=config,
    )
    if full_result["training_timeline"] is not None:
        plot_training_timeline(
            timeline_payload=full_result["training_timeline"],
            output_path=plots_dir / "training_timeline.png",
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

    full_final_metadata = {
        "config": config.to_resolved_dict(),
        "model_name": "full_language",
        "epoch": int(config.epochs),
        "checkpoint_kind": "final",
    }
    _save_checkpoint(
        full_result["model"],
        ckpt_dir / "full_language_final.pt",
        metadata=full_final_metadata,
    )
    _save_checkpoint(
        full_result["model"],
        ckpt_dir / "full_language.pt",
        metadata=full_final_metadata,
    )
    if baseline_result is not None:
        baseline_final_metadata = {
            "config": config.to_resolved_dict(),
            "model_name": "no_language",
            "epoch": int(config.epochs),
            "checkpoint_kind": "final",
        }
        _save_checkpoint(
            baseline_result["model"],
            ckpt_dir / "no_language_final.pt",
            metadata=baseline_final_metadata,
        )
        _save_checkpoint(
            baseline_result["model"],
            ckpt_dir / "no_language.pt",
            metadata=baseline_final_metadata,
        )

    summary = _build_summary(
        config=config,
        full_history=full_result["history"],
        baseline_history=None if baseline_result is None else baseline_result["history"],
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
