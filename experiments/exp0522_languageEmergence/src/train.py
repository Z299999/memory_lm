"""Training logic for exp0522 constant-pulse external clock experiments."""

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
    from .config import (
        ExperimentConfig,
        copy_config_to_run_dir,
        parse_train_window_schedule,
        train_window_reference_steps,
        write_resolved_config,
    )
    from .eval import _evaluate_rollout
    from .market_data import load_market_series
    from .model import AgentPool, ExternalClockMLP
    from .plots import plot_training_curves, plot_training_timeline
except ImportError:  # pragma: no cover - script mode
    from config import (
        ExperimentConfig,
        copy_config_to_run_dir,
        parse_train_window_schedule,
        train_window_reference_steps,
        write_resolved_config,
    )
    from eval import _evaluate_rollout
    from market_data import load_market_series
    from model import AgentPool, ExternalClockMLP
    from plots import plot_training_curves, plot_training_timeline


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


def _build_train_target(
    *,
    num_steps: int,
    device: torch.device,
    start_step: int,
    config: ExperimentConfig,
    target_split: str = "train",
) -> dict[str, torch.Tensor]:
    try:
        from .task import build_rollout_targets
    except ImportError:
        from task import build_rollout_targets
    offset = start_step if config.train_phase_mode == "continuous" else 0
    return build_rollout_targets(
        num_steps,
        config.cycle_steps,
        device,
        start_step=offset,
        target_kind=config.target_kind,
        mixed_sin_components=config.mixed_sin_components,
        target_split=target_split,
        ticker=config.ticker,
        price_column=config.price_column,
        normalize=config.normalize,
        test_days=config.test_days,
        market_cache_dir=config.market_cache_dir,
    )


def _analysis_checkpoint_epochs(config: ExperimentConfig) -> tuple[int, ...]:
    if not config.enable_continuous_collapse:
        return ()
    return tuple(sorted({epoch for epoch in config.checkpoint_epochs if 1 <= epoch < config.epochs}))


def _sample_train_steps(config: ExperimentConfig) -> int:
    mode, lower, upper = parse_train_window_schedule(config.train_window_schedule)
    if mode == "fixed":
        return lower
    return random.randint(lower, upper)


def _build_training_timeline_panels(config: ExperimentConfig) -> list[dict[str, Any]]:
    if config.target_kind == "yfinance_price":
        series = load_market_series(
            ticker=config.ticker,
            price_column=config.price_column,
            normalize=config.normalize,
            test_days=config.test_days,
            market_cache_dir=config.market_cache_dir,
        )
        total_steps = int(config.market_passes * series.train_length)
    else:
        total_steps = int(round(int(config.epochs) * float(train_window_reference_steps(config.train_window_schedule))))
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


def _save_checkpoint(model: ExternalClockMLP, output_path: Path, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )


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
    target_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    reference_train_steps = train_window_reference_steps(config.train_window_schedule)
    cache_steps = {
        reference_train_steps,
        config.eval_steps,
        config.long_steps,
        config.continuous_eval_steps,
    }

    schedule_mode, _, _ = parse_train_window_schedule(config.train_window_schedule)
    if config.train_phase_mode == "reset" and schedule_mode == "fixed":
        for steps in cache_steps:
            bundle = _build_train_target(
                num_steps=steps,
                device=device,
                start_step=0,
                config=config,
            )
            target_cache[(int(steps), 0)] = bundle

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
        effective_steps = _sample_train_steps(config)
        train_window_start = train_time_cursor if config.train_phase_mode == "continuous" else 0
        if config.train_phase_mode == "reset" and schedule_mode == "fixed":
            train_bundle = target_cache[(effective_steps, 0)]
        else:
            train_bundle = _build_train_target(
                num_steps=effective_steps,
                device=device,
                start_step=train_window_start,
                config=config,
            )
        train_target = train_bundle["train_target"]
        train_target_y = train_bundle["target_y"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, raw_prediction, messages, _hidden, final_message, final_error = model.rollout(
            num_steps=effective_steps,
            pulse_value=config.pulse_value,
            target_sequence=train_target,
            y_target_sequence=train_target_y,
            initial_message=train_message_state,
            initial_error=train_error_state,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=False,
            return_hidden=False,
        )
        train_loss = torch.mean((raw_prediction - train_target) ** 2)

        use_aux = (
            config.sequence_mode == "continuous_window"
            and model.use_language
            and config.message_aux_loss_weight > 0.0
        )
        aux_loss_val = 0.0
        if use_aux:
            aux_bundle = _build_train_target(
                num_steps=1,
                device=device,
                start_step=train_window_start + effective_steps,
                config=config,
            )
            _aux_prediction, aux_raw_prediction, _, _, _, _ = model.rollout(
                num_steps=1,
                pulse_value=config.pulse_value,
                target_sequence=aux_bundle["train_target"],
                y_target_sequence=aux_bundle["target_y"],
                initial_message=final_message,
                initial_error=final_error,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                disable_language=False,
                return_hidden=False,
            )
            aux_loss = torch.mean((aux_raw_prediction - aux_bundle["train_target"]) ** 2)
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
                target=train_target_y,
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
            target_split="train",
            ticker=config.ticker,
            price_column=config.price_column,
            normalize=config.normalize,
            test_days=config.test_days,
            market_cache_dir=config.market_cache_dir,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=False,
        )
        row = {
            "epoch": float(epoch),
            "train_steps": float(effective_steps),
            "train_loss": float(train_loss.item()),
            "heldout_loss": float(val_result["raw_target_mse"]),
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
                f"train={row['train_loss']:.6f} heldout={row['heldout_loss']:.6f}"
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
        "final_train_state": {
            "final_message": train_message_state,
            "final_error": train_error_state,
            "final_step": int(train_time_cursor),
        },
    }


def _train_market_model(
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
    series = load_market_series(
        ticker=config.ticker,
        price_column=config.price_column,
        normalize=config.normalize,
        test_days=config.test_days,
        market_cache_dir=config.market_cache_dir,
    )
    history: list[dict[str, float]] = []
    timeline_panels = (
        _build_training_timeline_panels(config)
        if config.sequence_mode == "continuous_window" and config.plot_show_training_timeline
        else []
    )
    checkpoint_epochs = tuple(sorted(set(checkpoint_epochs)))
    checkpoint_set = set(checkpoint_epochs)
    update_idx = 0
    final_message_state: torch.Tensor | None = None
    final_error_state: torch.Tensor | None = None
    final_step = 0
    val_steps = min(config.eval_steps, series.test_length)

    for pass_idx in range(1, config.market_passes + 1):
        train_cursor = 0
        train_message_state: torch.Tensor | None = None
        train_error_state: torch.Tensor | None = None
        while train_cursor < series.train_length:
            remaining = series.train_length - train_cursor
            effective_steps = min(_sample_train_steps(config), remaining)
            train_bundle = _build_train_target(
                num_steps=effective_steps,
                device=device,
                start_step=train_cursor,
                config=config,
                target_split="train",
            )
            train_target = train_bundle["train_target"]
            train_target_y = train_bundle["target_y"]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, raw_prediction, _messages, _hidden, final_message, final_error = model.rollout(
                num_steps=effective_steps,
                pulse_value=config.pulse_value,
                target_sequence=train_target,
                y_target_sequence=train_target_y,
                initial_message=train_message_state,
                initial_error=train_error_state,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                disable_language=False,
                return_hidden=False,
            )
            train_loss = torch.mean((raw_prediction - train_target) ** 2)

            use_aux = (
                train_cursor + effective_steps < series.train_length
                and model.use_language
                and config.message_aux_loss_weight > 0.0
            )
            aux_loss_val = 0.0
            if use_aux:
                aux_bundle = _build_train_target(
                    num_steps=1,
                    device=device,
                    start_step=train_cursor + effective_steps,
                    config=config,
                    target_split="train",
                )
                _aux_prediction, aux_raw_prediction, _, _, _, _ = model.rollout(
                    num_steps=1,
                    pulse_value=config.pulse_value,
                    target_sequence=aux_bundle["train_target"],
                    y_target_sequence=aux_bundle["target_y"],
                    initial_message=final_message,
                    initial_error=final_error,
                    detach_error_input=config.detach_error_input,
                    force_zero_error_input=config.force_zero_error_input,
                    disable_language=False,
                    return_hidden=False,
                )
                aux_loss = torch.mean((aux_raw_prediction - aux_bundle["train_target"]) ** 2)
                total_loss = train_loss + config.message_aux_loss_weight * aux_loss
                aux_loss_val = float(aux_loss.item())
            else:
                total_loss = train_loss

            total_loss.backward()
            if config.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            global_window_start = (pass_idx - 1) * series.train_length + train_cursor
            _record_training_timeline_overlap(
                panels=timeline_panels,
                train_window_start=int(global_window_start),
                train_window_end=int(global_window_start + effective_steps - 1),
                target=train_target_y,
                prediction=prediction,
            )

            if model.use_language:
                train_message_state = final_message.detach()
            else:
                train_message_state = None
            if model.use_error_input and config.carry_error_between_windows:
                train_error_state = final_error.detach()
            else:
                train_error_state = None

            update_idx += 1
            train_cursor += effective_steps
            final_message_state = train_message_state
            final_error_state = train_error_state
            final_step = series.train_length

            val_result = _evaluate_rollout(
                model,
                num_steps=val_steps,
                cycle_steps=config.cycle_steps,
                pulse_value=config.pulse_value,
                target_kind=config.target_kind,
                mixed_sin_components=config.mixed_sin_components,
                target_split="test",
                ticker=config.ticker,
                price_column=config.price_column,
                normalize=config.normalize,
                test_days=config.test_days,
                market_cache_dir=config.market_cache_dir,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                disable_language=False,
            )
            row = {
                "epoch": float(update_idx),
                "market_pass": float(pass_idx),
                "train_steps": float(effective_steps),
                "train_loss": float(train_loss.item()),
                "heldout_loss": float(val_result["raw_target_mse"]),
                "train_window_start": float(global_window_start),
                "train_window_end": float(global_window_start + effective_steps - 1),
                "market_train_start": float(train_cursor - effective_steps),
                "market_train_end": float(train_cursor - 1),
            }
            if use_aux:
                row["aux_loss"] = aux_loss_val
            history.append(row)

            if update_idx == 1 or update_idx % config.log_every == 0:
                print(
                    f"[{model_name}] update {update_idx:5d} "
                    f"pass={pass_idx:3d}/{config.market_passes} "
                    f"steps={effective_steps:3d} "
                    f"train={row['train_loss']:.6f} heldout={row['heldout_loss']:.6f}"
                )

            if checkpoint_dir is not None and update_idx in checkpoint_set:
                _save_checkpoint(
                    model,
                    checkpoint_dir / f"{model_name}_epoch_{update_idx:04d}.pt",
                    metadata={
                        "config": config.to_resolved_dict(),
                        "model_name": model_name,
                        "epoch": int(update_idx),
                        "checkpoint_kind": "milestone",
                    },
                )

    return {
        "model": model,
        "history": history,
        "training_timeline": {
            "panels": timeline_panels,
        } if timeline_panels else None,
        "final_train_state": {
            "final_message": final_message_state,
            "final_error": final_error_state,
            "final_step": int(final_step),
        },
    }


def train_model(config: ExperimentConfig, config_path: Path) -> Path:
    """Train the full-language model, save checkpoint and training artifacts. Returns run_dir."""
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

    if config.num_agents > 1:
        full_model = AgentPool(
            num_agents=config.num_agents,
            trunk_dims=config.trunk_dims,
            activation=config.activation,
            language_dim=config.language_dim,
            language_readout_coverage=config.language_readout_coverage,
            use_error_input=config.use_error_input,
            use_residual=config.use_residual,
            language_readout_all_layers=config.language_readout_all_layers,
            message_carry_mode=config.message_carry_mode,
            readout_mode=config.readout_mode,
            error_intake_mode=config.error_intake_mode,
            seed=config.seed,
        ).to(device)
    else:
        full_model = ExternalClockMLP(
            trunk_dims=config.trunk_dims,
            activation=config.activation,
            language_dim=config.language_dim,
            language_readout_coverage=config.language_readout_coverage,
            use_error_input=config.use_error_input,
            use_language=True,
            use_residual=config.use_residual,
            language_readout_all_layers=config.language_readout_all_layers,
            message_carry_mode=config.message_carry_mode,
            readout_mode=config.readout_mode,
            seed=config.seed,
        ).to(device)

    full_checkpoint_epochs = _analysis_checkpoint_epochs(config)
    if config.target_kind == "yfinance_price":
        full_result = _train_market_model(
            model_name="full_language",
            model=full_model,
            config=config,
            device=device,
            checkpoint_dir=ckpt_dir,
            checkpoint_epochs=full_checkpoint_epochs,
        )
    else:
        full_result = _train_single_model(
            model_name="full_language",
            model=full_model,
            config=config,
            device=device,
            checkpoint_dir=ckpt_dir,
            checkpoint_epochs=full_checkpoint_epochs,
        )

    _write_json(
        metrics_dir / "history_full_language.json",
        _to_serializable_history(full_result["history"]),
    )
    if full_result["training_timeline"] is not None:
        _write_json(metrics_dir / "training_timeline.json", full_result["training_timeline"])

    final_metadata = {
        "config": config.to_resolved_dict(),
        "model_name": "full_language",
        "epoch": int(full_result["history"][-1]["epoch"]) if full_result["history"] else int(config.epochs),
        "checkpoint_kind": "final",
    }
    _save_checkpoint(full_result["model"], ckpt_dir / "full_language_final.pt", metadata=final_metadata)
    _save_checkpoint(full_result["model"], ckpt_dir / "full_language.pt", metadata=final_metadata)

    if config.sequence_mode == "continuous_window":
        fts = full_result["final_train_state"]
        torch.save(
            {
                "final_message": fts["final_message"],
                "final_error":   fts["final_error"],
                "final_step":    fts["final_step"],
            },
            ckpt_dir / "final_train_state.pt",
        )

    plot_training_curves(
        full_history=full_result["history"],
        baseline_history=None,
        output_path=plots_dir / "training_curves.png",
        config=config,
    )
    if full_result["training_timeline"] is not None:
        plot_training_timeline(
            timeline_payload=full_result["training_timeline"],
            output_path=plots_dir / "training_timeline.png",
            config=config,
        )

    latest_link = run_root / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)

    return run_dir
