"""Training logic for exp0522 constant-pulse external clock experiments."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from .config import (
        ExperimentConfig,
        FreqCurriculumSchedule,
        copy_config_to_run_dir,
        parse_error_degrade,
        parse_freq_curriculum,
        parse_train_window_schedule,
        train_window_reference_steps,
        write_resolved_config,
    )
    from .eval import _evaluate_rollout
    from .model import ExternalClockMLP
    from .plots import plot_training_curves, plot_training_timeline
except ImportError:  # pragma: no cover - script mode
    from config import (
        ExperimentConfig,
        FreqCurriculumSchedule,
        copy_config_to_run_dir,
        parse_error_degrade,
        parse_freq_curriculum,
        parse_train_window_schedule,
        train_window_reference_steps,
        write_resolved_config,
    )
    from eval import _evaluate_rollout
    from model import ExternalClockMLP
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
    cycle_steps: int,
    device: torch.device,
    train_phase_mode: str,
    start_step: int,
    target_kind: str,
    mixed_sin_components: tuple[tuple[float, float], ...],
    prediction_target: str,
    phase_offset: float = 0.0,
) -> dict[str, torch.Tensor]:
    try:
        from .task import build_rollout_targets
    except ImportError:
        from task import build_rollout_targets
    offset = start_step if train_phase_mode == "continuous" else 0
    return build_rollout_targets(
        num_steps,
        cycle_steps,
        device,
        start_step=offset,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
        prediction_target=prediction_target,
        phase_offset=phase_offset,
    )


def _analysis_checkpoint_epochs(_config: ExperimentConfig) -> tuple[int, ...]:
    return ()


def _sample_train_steps(config: ExperimentConfig) -> int:
    schedule = parse_train_window_schedule(config.train_window_schedule)
    if schedule.mode == "fixed":
        return schedule.min_steps
    if schedule.mode == "random_uniform":
        return random.randint(schedule.min_steps, schedule.max_steps)
    raise ValueError("_sample_train_steps is only valid for fixed and random_uniform schedules.")


class _ErrorDegradeGenerator:
    """Online dim-event generator over global training time."""

    def __init__(self, spec: str) -> None:
        self.schedule = parse_error_degrade(spec)
        self._active_gains: list[float] = []
        if self.schedule.mode == "dim" and self.schedule.rate > 0.0:
            avg_len = (self.schedule.min_steps + self.schedule.max_steps) / 2.0
            self._start_prob = min(1.0, self.schedule.rate / avg_len)
        else:
            self._start_prob = 0.0

    def _sample_event_gains(self) -> list[float]:
        length = random.randint(self.schedule.min_steps, self.schedule.max_steps)
        min_gain = self.schedule.pct / 100.0
        ramp = min(self.schedule.ramp_steps, length // 2)
        if ramp <= 0:
            return [min_gain] * length
        ramp_down = np.linspace(1.0, min_gain, ramp + 1, dtype=float)[1:].tolist()
        plateau = [min_gain] * max(0, length - (2 * ramp))
        ramp_up = np.linspace(min_gain, 1.0, ramp + 1, dtype=float)[1:].tolist()
        return ramp_down + plateau + ramp_up

    def _tail_dim_gain(self, local_step: int) -> float:
        if self.schedule.start_step is None or self.schedule.end_step is None or self.schedule.min_pct is None:
            raise ValueError("tail_dim schedule is missing start/end/min_pct.")
        if local_step < self.schedule.start_step:
            return 1.0
        min_gain = self.schedule.min_pct / 100.0
        if local_step >= self.schedule.end_step:
            return min_gain
        progress = (local_step - self.schedule.start_step) / float(self.schedule.end_step - self.schedule.start_step)
        return 1.0 - (1.0 - min_gain) * progress

    def next_gain(self, *, force_zero_error_input: bool, local_step: int) -> float:
        if force_zero_error_input or self.schedule.mode == "none":
            return 1.0
        if self.schedule.mode == "tail_dim":
            return float(self._tail_dim_gain(local_step))
        if self.schedule.rate <= 0.0:
            return 1.0
        if not self._active_gains and random.random() < self._start_prob:
            self._active_gains = self._sample_event_gains()
        if self._active_gains:
            return float(self._active_gains.pop(0))
        return 1.0

    def next_gains(self, num_steps: int, *, force_zero_error_input: bool, device: torch.device) -> torch.Tensor:
        gains = [
            self.next_gain(force_zero_error_input=force_zero_error_input, local_step=idx)
            for idx in range(num_steps)
        ]
        return torch.tensor(gains, dtype=torch.float32, device=device).reshape(num_steps, 1)


class FrequencyScheduler:
    """Adaptive frequency curriculum: starts at low freq, promotes when model tracks well."""

    def __init__(
        self,
        schedule: FreqCurriculumSchedule,
        target_components: tuple[tuple[float, float], ...],
    ) -> None:
        self._target_components = target_components
        self._target_freq = max(freq for freq, _ in target_components)
        if schedule.mode == "none":
            self._current_freq = self._target_freq
            self._done = True
        else:
            self._current_freq = float(schedule.start_freq)
            self._done = self._current_freq >= self._target_freq
        self._schedule = schedule
        self._recent: list[int] = []
        self._epochs_since_change: int = 0

    def active_components(self) -> tuple[tuple[float, float], ...]:
        if self._done:
            return self._target_components
        amp = self._target_components[0][1]
        return ((self._current_freq, amp),)

    def report_window(self, steps: int) -> int:
        """Record window length. Returns +1 if promoted, -1 if retreated, 0 otherwise."""
        if self._done:
            return 0
        self._recent.append(steps)
        if len(self._recent) > self._schedule.patience:
            self._recent.pop(0)
        # Check promotion
        if (
            len(self._recent) == self._schedule.patience
            and all(s >= self._schedule.target_steps for s in self._recent)
        ):
            new_freq = min(self._current_freq * self._schedule.factor, self._target_freq)
            self._current_freq = new_freq
            self._recent.clear()
            self._epochs_since_change = 0
            if self._current_freq >= self._target_freq:
                self._done = True
            return 1
        # Check retreat
        self._epochs_since_change += 1
        if (
            self._schedule.retreat_patience > 0
            and self._epochs_since_change >= self._schedule.retreat_patience
        ):
            self._current_freq = max(
                self._current_freq / self._schedule.factor,
                1e-9,
            )
            self._recent.clear()
            self._epochs_since_change = 0
            return -1
        return 0

    @property
    def current_freq(self) -> float:
        return self._current_freq

    @property
    def is_done(self) -> bool:
        return self._done


def _error_degrade_stats(error_gain_sequence: torch.Tensor) -> tuple[int, float]:
    if error_gain_sequence.numel() == 0:
        return 0, 1.0
    detached = error_gain_sequence.detach()
    return int(torch.sum(detached < 0.999999).item()), float(torch.mean(detached).item())


def _build_training_timeline_panels(config: ExperimentConfig, total_steps: int) -> list[dict[str, Any]]:
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
            "error_gain": [],
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
    error_gain: torch.Tensor | None = None,
) -> None:
    if not panels:
        return
    target_np = target.squeeze(1).detach().cpu().numpy()
    prediction_np = prediction.squeeze(1).detach().cpu().numpy()
    error_gain_np = None
    if error_gain is not None:
        error_gain_np = error_gain.reshape(-1).detach().cpu().numpy()
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
        if error_gain_np is not None:
            panel["error_gain"].extend(float(value) for value in error_gain_np[local_start : local_end + 1])
        if panel["start_step"] <= train_window_end <= panel["end_step"]:
            panel["update_steps"].append(int(train_window_end))


def _materialize_training_timeline(
    *,
    config: ExperimentConfig,
    total_steps: int,
    window_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    panels = _build_training_timeline_panels(config, total_steps)
    if not panels:
        return None
    for record in window_records:
        _record_training_timeline_overlap(
            panels=panels,
            train_window_start=int(record["train_window_start"]),
            train_window_end=int(record["train_window_end"]),
            target=record["target"],
            prediction=record["prediction"],
            error_gain=record.get("error_gain"),
        )
    return {"panels": panels}


def _save_checkpoint(model: ExternalClockMLP, output_path: Path, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )


def _rollout_event_triggered_window(
    *,
    model: ExternalClockMLP,
    pulse_value: float,
    train_bundle: dict[str, torch.Tensor],
    prediction_target: str,
    threshold: float,
    min_steps: int,
    max_steps: int,
    initial_message: torch.Tensor | None,
    initial_error: torch.Tensor | None,
    initial_pipeline_buffers: list[torch.Tensor] | None,
    detach_error_input: bool,
    force_zero_error_input: bool,
    error_degrade_generator: _ErrorDegradeGenerator,
    loss_space: str,
) -> dict[str, Any]:
    predictions: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    messages: list[torch.Tensor] = []
    error_gains: list[float] = []
    cumulative_sse = 0.0

    message_state = initial_message
    error_state = initial_error
    pipeline_buffers_state = initial_pipeline_buffers
    reconstruction_y_prev = train_bundle["init_y_prev"]
    reconstruction_v_prev = train_bundle["init_v_prev"]
    final_message: torch.Tensor | None = None
    final_error: torch.Tensor | None = None

    for step_idx in range(max_steps):
        error_gain = error_degrade_generator.next_gain(
            force_zero_error_input=force_zero_error_input,
            local_step=step_idx,
        )
        prediction_step, raw_prediction_step, messages_step, _hidden, final_message, final_error, final_pipeline_bufs = model.rollout(
            num_steps=1,
            pulse_value=pulse_value,
            target_sequence=train_bundle["train_target"][step_idx : step_idx + 1],
            y_target_sequence=train_bundle["target_y"][step_idx : step_idx + 1],
            initial_message=message_state,
            initial_error=error_state,
            initial_reconstruction_y=reconstruction_y_prev,
            initial_reconstruction_v=reconstruction_v_prev,
            prediction_target=prediction_target,
            detach_error_input=detach_error_input,
            force_zero_error_input=force_zero_error_input,
            error_input_scale=error_gain,
            disable_language=False,
            return_hidden=False,
            initial_pipeline_buffers=pipeline_buffers_state,
        )
        predictions.append(prediction_step)
        raw_predictions.append(raw_prediction_step)
        messages.append(messages_step)
        error_gains.append(error_gain)

        if loss_space == "y":
            target_step = train_bundle["target_y"][step_idx : step_idx + 1]
            cumulative_sse += float(torch.sum((prediction_step - target_step) ** 2).item())
        else:
            target_step = train_bundle["train_target"][step_idx : step_idx + 1]
            cumulative_sse += float(torch.sum((raw_prediction_step - target_step) ** 2).item())
        step_count = step_idx + 1

        message_state = final_message if model.use_language else None
        error_state = final_error if model.use_error_input else None
        pipeline_buffers_state = final_pipeline_bufs
        if prediction_target == "v":
            reconstruction_y_prev = prediction_step
        elif prediction_target == "a":
            reconstruction_v_prev = reconstruction_v_prev + raw_prediction_step
            reconstruction_y_prev = prediction_step

        if step_count >= min_steps and not math.isnan(cumulative_sse) and cumulative_sse >= threshold:
            break

    realized_steps = len(predictions)
    if realized_steps == 0 or final_message is None or final_error is None:
        raise RuntimeError("event_triggered window rollout produced no steps.")

    return {
        "realized_steps": realized_steps,
        "prediction": torch.cat(predictions, dim=0),
        "raw_prediction": torch.cat(raw_predictions, dim=0),
        "messages": torch.cat(messages, dim=0),
        "error_gain_sequence": torch.tensor(error_gains, dtype=torch.float32, device=train_bundle["train_target"].device).reshape(realized_steps, 1),
        "final_message": final_message,
        "final_error": final_error,
        "final_pipeline_buffers": pipeline_buffers_state,
    }


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
    cache_steps = {reference_train_steps, config.eval_steps, config.long_steps, config.continuous_eval_steps}

    schedule = parse_train_window_schedule(config.train_window_schedule)
    freq_scheduler = FrequencyScheduler(
        parse_freq_curriculum(config.freq_curriculum),
        config.mixed_sin_components,
    )
    freq_curriculum_active = config.freq_curriculum is not None
    if config.train_phase_mode == "reset" and schedule.mode == "fixed" and not freq_curriculum_active:
        for steps in cache_steps:
            bundle = _build_train_target(
                num_steps=steps,
                cycle_steps=config.cycle_steps,
                device=device,
                train_phase_mode="reset",
                start_step=0,
                target_kind=config.target_kind,
                mixed_sin_components=config.mixed_sin_components,
                prediction_target=config.prediction_target,
            )
            target_cache[(int(steps), 0)] = bundle

    history: list[dict[str, float]] = []
    timeline_window_records: list[dict[str, Any]] = []
    error_degrade_generator = _ErrorDegradeGenerator(config.error_degrade)
    train_time_cursor = 0
    train_phase_offset: float = 0.0
    _omega = (2.0 * math.pi) / config.cycle_steps
    train_message_state: torch.Tensor | None = None
    train_error_state: torch.Tensor | None = None
    train_pipeline_buffers_state: list | None = None

    if checkpoint_dir is not None:
        _save_checkpoint(
            model,
            checkpoint_dir / f"{model_name}_epoch_0000.pt",
            metadata={
                "config": config.to_resolved_dict(),
                "model_name": model_name,
                "epoch": 0,
                "checkpoint_kind": "initial",
            },
        )

    for epoch in range(1, config.epochs + 1):
        train_window_start = train_time_cursor if config.train_phase_mode == "continuous" else 0
        train_components = freq_scheduler.active_components()
        if schedule.mode == "event_triggered":
            effective_steps = schedule.max_steps
        else:
            effective_steps = _sample_train_steps(config)
        if config.train_phase_mode == "reset" and schedule.mode == "fixed" and not freq_curriculum_active:
            train_bundle = target_cache[(effective_steps, 0)]
        else:
            train_bundle = _build_train_target(
                num_steps=effective_steps,
                cycle_steps=config.cycle_steps,
                device=device,
                train_phase_mode=config.train_phase_mode,
                start_step=train_window_start,
                target_kind=config.target_kind,
                mixed_sin_components=train_components,
                prediction_target=config.prediction_target,
                phase_offset=train_phase_offset,
            )
        train_target = train_bundle["train_target"]
        train_target_y = train_bundle["target_y"]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if schedule.mode == "event_triggered":
            event_result = _rollout_event_triggered_window(
                model=model,
                pulse_value=config.pulse_value,
                train_bundle=train_bundle,
                prediction_target=config.prediction_target,
                threshold=float(schedule.threshold),
                min_steps=schedule.min_steps,
                max_steps=schedule.max_steps,
                initial_message=train_message_state,
                initial_error=train_error_state,
                initial_pipeline_buffers=train_pipeline_buffers_state,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                error_degrade_generator=error_degrade_generator,
                loss_space=config.train_loss_space,
            )
            effective_steps = int(event_result["realized_steps"])
            prediction = event_result["prediction"]
            raw_prediction = event_result["raw_prediction"]
            messages = event_result["messages"]
            error_gain_sequence = event_result["error_gain_sequence"]
            final_message = event_result["final_message"]
            final_error = event_result["final_error"]
            final_pipeline_bufs = event_result["final_pipeline_buffers"]
            train_target = train_target[:effective_steps]
            train_target_y = train_target_y[:effective_steps]
        else:
            error_gain_sequence = error_degrade_generator.next_gains(
                effective_steps,
                force_zero_error_input=config.force_zero_error_input,
                device=device,
            )
            prediction, raw_prediction, messages, _hidden, final_message, final_error, final_pipeline_bufs = model.rollout(
                num_steps=effective_steps,
                pulse_value=config.pulse_value,
                target_sequence=train_target,
                y_target_sequence=train_target_y,
                initial_message=train_message_state,
                initial_error=train_error_state,
                initial_reconstruction_y=train_bundle["init_y_prev"],
                initial_reconstruction_v=train_bundle["init_v_prev"],
                prediction_target=config.prediction_target,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                error_input_scale_sequence=error_gain_sequence,
                disable_language=False,
                return_hidden=False,
                initial_pipeline_buffers=train_pipeline_buffers_state,
            )
        if config.train_loss_space == "y":
            full_prediction, full_target = prediction, train_target_y
        else:
            full_prediction, full_target = raw_prediction, train_target
        tail = config.train_loss_tail_steps
        if tail is not None and tail < effective_steps:
            loss_prediction = full_prediction[-tail:]
            loss_target = full_target[-tail:]
        else:
            loss_prediction = full_prediction
            loss_target = full_target
        train_loss = torch.mean((loss_prediction - loss_target) ** 2)

        use_aux = (
            config.sequence_mode == "continuous_window"
            and model.use_language
            and config.message_aux_loss_weight > 0.0
        )
        aux_loss_val = 0.0
        if use_aux:
            aux_bundle = _build_train_target(
                num_steps=1,
                cycle_steps=config.cycle_steps,
                device=device,
                train_phase_mode="continuous",
                start_step=train_window_start + effective_steps,
                target_kind=config.target_kind,
                mixed_sin_components=train_components,
                prediction_target=config.prediction_target,
                phase_offset=train_phase_offset,
            )
            aux_prediction, aux_raw_prediction, _, _, _, _, _ = model.rollout(
                num_steps=1,
                pulse_value=config.pulse_value,
                target_sequence=aux_bundle["train_target"],
                y_target_sequence=aux_bundle["target_y"],
                initial_message=final_message,
                initial_error=final_error,
                initial_reconstruction_y=aux_bundle["init_y_prev"],
                initial_reconstruction_v=aux_bundle["init_v_prev"],
                prediction_target=config.prediction_target,
                detach_error_input=config.detach_error_input,
                force_zero_error_input=config.force_zero_error_input,
                disable_language=False,
                return_hidden=False,
                initial_pipeline_buffers=final_pipeline_bufs,
            )
            if config.train_loss_space == "y":
                aux_loss = torch.mean((aux_prediction - aux_bundle["target_y"]) ** 2)
            else:
                aux_loss = torch.mean((aux_raw_prediction - aux_bundle["train_target"]) ** 2)
            total_loss = train_loss + config.message_aux_loss_weight * aux_loss
            aux_loss_val = float(aux_loss.item())
        else:
            total_loss = train_loss

        if (
            config.language_readout_trainable
            and config.language_readout_norm_penalty > 0.0
            and model.use_language
        ):
            col_norms = model.language_readout.norm(dim=0)
            total_loss = total_loss + config.language_readout_norm_penalty * ((col_norms - 1.0) ** 2).mean()

        total_loss.backward()
        if config.grad_clip > 0.0:
            total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        else:
            total_grad_norm = torch.stack(
                [p.grad.norm() for p in model.parameters() if p.grad is not None]
            ).norm()
        if total_grad_norm.isfinite():
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)

        if config.sequence_mode == "continuous_window" and config.plot_show_training_timeline:
            timeline_window_records.append(
                {
                    "train_window_start": int(train_window_start),
                    "train_window_end": int(train_window_start + effective_steps - 1),
                    "target": train_target_y.detach().cpu(),
                    "prediction": prediction.detach().cpu(),
                    "error_gain": error_gain_sequence.detach().cpu(),
                }
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
        if config.sequence_mode == "continuous_window" and model.use_pipeline and final_pipeline_bufs is not None:
            train_pipeline_buffers_state = [b.detach() for b in final_pipeline_bufs]
        else:
            train_pipeline_buffers_state = None
        if config.train_phase_mode == "continuous":
            train_time_cursor += effective_steps

        val_result = _evaluate_rollout(
            model,
            num_steps=config.eval_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            prediction_target=config.prediction_target,
            detach_error_input=config.detach_error_input,
            force_zero_error_input=config.force_zero_error_input,
            disable_language=False,
        )
        _freq_before = freq_scheduler.current_freq
        _freq_action = freq_scheduler.report_window(effective_steps)
        if _freq_action != 0:
            train_phase_offset += (_freq_before - freq_scheduler.current_freq) * _omega * train_time_cursor
        if _freq_action == 1:
            print(
                f"[{model_name}] freq_curriculum: promoted → freq={freq_scheduler.current_freq:.3e}"
                + (" (done)" if freq_scheduler.is_done else "")
            )
        elif _freq_action == -1:
            print(
                f"[{model_name}] freq_curriculum: retreated → freq={freq_scheduler.current_freq:.3e}"
            )

        row = {
            "epoch": float(epoch),
            "train_steps": float(effective_steps),
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_result["mse"] if config.train_loss_space == "y" else val_result["raw_target_mse"]),
            "curr_freq": float(freq_scheduler.current_freq),
        }
        error_degrade_steps, error_gain_mean = _error_degrade_stats(error_gain_sequence)
        row["error_degrade_steps"] = float(error_degrade_steps)
        row["error_gain_mean"] = float(error_gain_mean)
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
                f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
                f"freq={freq_scheduler.current_freq:.3e}"
            )

        if (
            config.early_stop_min_steps is not None
            and effective_steps >= config.early_stop_min_steps
            and freq_scheduler.is_done
        ):
            print(
                f"[{model_name}] early stop at epoch {epoch}: "
                f"steps={effective_steps} >= early_stop_min_steps={config.early_stop_min_steps}"
            )
            break

        save_this_epoch = epoch in checkpoint_epochs or (
            config.checkpoint_every is not None and epoch % config.checkpoint_every == 0
        )
        if checkpoint_dir is not None and save_this_epoch:
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
        "training_timeline": _materialize_training_timeline(
            config=config,
            total_steps=int(train_time_cursor),
            window_records=timeline_window_records,
        )
        if config.sequence_mode == "continuous_window" and config.plot_show_training_timeline
        else None,
        "final_train_state": {
            "final_message": train_message_state,
            "final_error": train_error_state,
            "final_step": int(train_time_cursor),
            "final_pipeline_buffers": train_pipeline_buffers_state,
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

    full_model = ExternalClockMLP(
        trunk_dims=config.trunk_dims,
        activation=config.activation,
        language_dim=config.language_dim,
        language_readout_coverage=config.language_readout_coverage,
        use_error_input=config.use_error_input,
        use_language=True,
        use_residual=config.use_residual,
        use_dense=config.use_dense,
        language_readout_all_layers=config.language_readout_all_layers,
        language_readout_trainable=config.language_readout_trainable,
        readout_nonlinearity=config.readout_nonlinearity,
        message_carry_mode=config.message_carry_mode,
        use_pipeline=config.use_pipeline,
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

    _write_json(
        metrics_dir / "history_full_language.json",
        _to_serializable_history(full_result["history"]),
    )
    if full_result["training_timeline"] is not None:
        _write_json(metrics_dir / "training_timeline.json", full_result["training_timeline"])

    final_metadata = {
        "config": config.to_resolved_dict(),
        "model_name": "full_language",
        "epoch": int(config.epochs),
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
                "final_pipeline_buffers": fts.get("final_pipeline_buffers"),
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
