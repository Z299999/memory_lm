"""Evaluation logic for exp0522."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import torch

try:
    from .config import ExperimentConfig, parse_condition, train_window_bounds, train_window_reference_steps
    from .model import ExternalClockMLP
    from .plots import plot_rollout_diagnostics
    from .task import build_rollout_targets
except ImportError:  # pragma: no cover - script mode
    from config import ExperimentConfig, parse_condition, train_window_bounds, train_window_reference_steps
    from model import ExternalClockMLP
    from plots import plot_rollout_diagnostics
    from task import build_rollout_targets


_CONDITION_FLAGS: dict[str, tuple[bool, bool]] = {
    # condition_name: (force_zero_error_input, disable_language)
    "full":        (False, False),
    "sole_eye":    (False, True),
    "sole_speech": (True,  False),
    "neither":     (True,  True),
}

_CONDITIONS_NEEDING_ERROR = frozenset({"sole_speech", "neither", "late_blind", "blink", "stutter"})


def _condition_flags(condition_name: str) -> tuple[bool, bool]:
    try:
        return _CONDITION_FLAGS[condition_name]
    except KeyError:
        raise ValueError(f"Unknown eval condition: {condition_name!r}")


def _write_json(output_path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, indent=2))


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
        target_bundle = build_rollout_targets(
            num_steps,
            cycle_steps,
            device,
            start_step=start_step,
            target_kind=target_kind,
            mixed_sin_components=mixed_sin_components,
        )
        prediction, raw_prediction, messages, hidden, final_message, final_error = model.rollout(
            num_steps=num_steps,
            pulse_value=pulse_value,
            target_sequence=target_bundle["train_target"],
            y_target_sequence=target_bundle["target_y"],
            initial_message=initial_message,
            initial_error=initial_error,
            detach_error_input=detach_error_input,
            force_zero_error_input=force_zero_error_input,
            disable_language=disable_language,
            return_hidden=True,
        )
        mse = torch.mean((prediction - target_bundle["target_y"]) ** 2).item()
        raw_target_mse = torch.mean((raw_prediction - target_bundle["train_target"]) ** 2).item()
        if messages.numel():
            message_norm = torch.linalg.norm(messages, dim=1)  # (T,)
        else:
            message_norm = torch.zeros(num_steps, device=device)

    return {
        "mse": float(mse),
        "raw_target_mse": float(raw_target_mse),
        "phase": target_bundle["phase"].cpu(),
        "target": target_bundle["target_y"].squeeze(1).cpu(),
        "prediction": prediction.squeeze(1).cpu(),
        "raw_prediction": raw_prediction.squeeze(1).cpu(),
        "messages": messages.cpu(),
        "message_norm": message_norm.cpu(),
        "hidden": hidden.cpu() if hidden is not None else None,
        "final_message": final_message.cpu(),
        "final_error": final_error.cpu(),
        "start_step": int(start_step),
    }


def _evaluate_late_transition_rollout(
    model: ExternalClockMLP,
    *,
    num_steps: int,
    transition_step: int,
    post_force_zero_error: bool,
    post_disable_language: bool,
    cycle_steps: int,
    pulse_value: float,
    target_kind: str,
    mixed_sin_components: tuple[tuple[float, float], ...],
    start_step: int = 0,
    initial_message: torch.Tensor | None = None,
    initial_error: torch.Tensor | None = None,
    detach_error_input: bool = True,
) -> dict[str, Any]:
    """Run full model for transition_step steps, then switch to given post-transition flags."""
    effective_transition = min(transition_step, num_steps)
    phase1 = _evaluate_rollout(
        model,
        num_steps=effective_transition,
        cycle_steps=cycle_steps,
        pulse_value=pulse_value,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
        start_step=start_step,
        initial_message=initial_message,
        initial_error=initial_error,
        detach_error_input=detach_error_input,
        force_zero_error_input=False,
        disable_language=False,
    )
    remaining = num_steps - effective_transition
    if remaining <= 0:
        return phase1
    phase2 = _evaluate_rollout(
        model,
        num_steps=remaining,
        cycle_steps=cycle_steps,
        pulse_value=pulse_value,
        target_kind=target_kind,
        mixed_sin_components=mixed_sin_components,
        start_step=start_step + effective_transition,
        initial_message=phase1["final_message"],
        initial_error=phase1["final_error"],
        detach_error_input=detach_error_input,
        force_zero_error_input=post_force_zero_error,
        disable_language=post_disable_language,
    )
    prediction = torch.cat([phase1["prediction"], phase2["prediction"]])
    raw_prediction = torch.cat([phase1["raw_prediction"], phase2["raw_prediction"]])
    target = torch.cat([phase1["target"], phase2["target"]])
    return {
        "mse": float(torch.mean((prediction - target) ** 2).item()),
        "raw_target_mse": None,
        "phase": torch.cat([phase1["phase"], phase2["phase"]]),
        "target": target,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "messages": torch.cat([phase1["messages"], phase2["messages"]]),
        "message_norm": torch.cat([phase1["message_norm"], phase2["message_norm"]]),
        "hidden": None,
        "final_message": phase2["final_message"],
        "final_error": phase2["final_error"],
        "start_step": int(start_step),
    }


def _evaluate_temporary_loss_rollout(
    model: ExternalClockMLP,
    *,
    num_steps: int,
    loss_start: int,
    loss_end: int,
    phase2_force_zero_error: bool,
    phase2_disable_language: bool,
    cycle_steps: int,
    pulse_value: float,
    target_kind: str,
    mixed_sin_components: tuple[tuple[float, float], ...],
    start_step: int = 0,
    initial_message: torch.Tensor | None = None,
    initial_error: torch.Tensor | None = None,
    detach_error_input: bool = True,
) -> dict[str, Any]:
    """Full model → temporarily lose a channel [loss_start, loss_end) → full again."""
    def _rollout(n: int, s: int, msg, err, fze: bool, dl: bool) -> dict[str, Any]:
        return _evaluate_rollout(
            model,
            num_steps=n,
            cycle_steps=cycle_steps,
            pulse_value=pulse_value,
            target_kind=target_kind,
            mixed_sin_components=mixed_sin_components,
            start_step=s,
            initial_message=msg,
            initial_error=err,
            detach_error_input=detach_error_input,
            force_zero_error_input=fze,
            disable_language=dl,
        )

    p1_steps = min(loss_start, num_steps)
    p1 = _rollout(p1_steps, start_step, initial_message, initial_error, False, False)
    if p1_steps >= num_steps:
        return p1

    p2_steps = min(loss_end - loss_start, num_steps - p1_steps)
    p2 = _rollout(
        p2_steps,
        start_step + p1_steps,
        p1["final_message"],
        p1["final_error"],
        phase2_force_zero_error,
        phase2_disable_language,
    )
    if p1_steps + p2_steps >= num_steps:
        parts = [p1, p2]
    else:
        p3_steps = num_steps - p1_steps - p2_steps
        p3 = _rollout(
            p3_steps,
            start_step + p1_steps + p2_steps,
            p2["final_message"],
            p2["final_error"],
            False,
            False,
        )
        parts = [p1, p2, p3]

    prediction = torch.cat([p["prediction"] for p in parts])
    raw_prediction = torch.cat([p["raw_prediction"] for p in parts])
    target = torch.cat([p["target"] for p in parts])
    return {
        "mse": float(torch.mean((prediction - target) ** 2).item()),
        "raw_target_mse": None,
        "phase": torch.cat([p["phase"] for p in parts]),
        "target": target,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "messages": torch.cat([p["messages"] for p in parts]),
        "message_norm": torch.cat([p["message_norm"] for p in parts]),
        "hidden": None,
        "final_message": parts[-1]["final_message"],
        "final_error": parts[-1]["final_error"],
        "start_step": int(start_step),
    }


# Phase-2 flags for temporary-loss conditions: (force_zero_error, disable_language)
_TEMPORARY_LOSS_FLAGS: dict[str, tuple[bool, bool]] = {
    "blink":   (True,  False),  # temporarily lose error channel
    "stutter": (False, True),   # temporarily lose language channel
}

# Post-transition flags for each late-transition condition.
_LATE_TRANSITION_FLAGS: dict[str, tuple[bool, bool]] = {
    # condition: (post_force_zero_error, post_disable_language)
    "late_blind": (True,  False),  # lose error → sole_speech
    "late_mute":  (False, True),   # lose language → sole_eye
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
            warmup_bundle = build_rollout_targets(
                warmup_steps,
                cycle_steps,
                device,
                start_step=0,
                target_kind=target_kind,
                mixed_sin_components=mixed_sin_components,
            )
            _prediction, _raw_prediction, _messages, _hidden, final_message, final_error = model.rollout(
                num_steps=warmup_steps,
                pulse_value=pulse_value,
                target_sequence=warmup_bundle["train_target"],
                y_target_sequence=warmup_bundle["target_y"],
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


def _save_rollout_csv(
    *,
    output_path: Path,
    evals: dict[str, dict],
    language_dim: int,
) -> None:
    ref = evals["full"]
    num_steps = len(ref["target"])
    msg_cols = ref["messages"].shape[1] if ref["messages"].numel() else 0
    norm = ref["message_norm"]

    header = ["step", "global_step", "phase", "target"]
    for condition in evals:
        header.append(f"{condition}_prediction")
    header.append("message_norm")
    header.extend([f"message_{idx}" for idx in range(msg_cols)])

    rows = [header]
    for step in range(num_steps):
        row: list[Any] = [
            step,
            int(ref.get("start_step", 0)) + step,
            float(ref["phase"][step].item()),
            float(ref["target"][step].item()),
        ]
        for condition, rollout in evals.items():
            row.append(float(rollout["prediction"][step].item()))
        row.append(float(norm[step].item()))
        row.extend(float(ref["messages"][step, idx].item()) for idx in range(msg_cols))
        rows.append(row)
    output_path.write_text("\n".join(",".join(str(v) for v in row) for row in rows))


def _build_summary(
    *,
    config: ExperimentConfig,
    full_history: list[dict[str, float]],
    reset_evals: dict[str, dict] | None,
    reset_long_evals: dict[str, dict] | None,
    continuous_evals: dict[str, dict] | None,
) -> dict[str, Any]:
    def _mse(result: dict[str, Any] | None) -> float | None:
        return None if result is None else float(result["mse"])

    def _gap(a: dict | None, b: dict | None) -> float | None:
        if a is None or b is None:
            return None
        return float(a["mse"] - b["mse"])

    train_window_min, train_window_max = train_window_bounds(config.train_window_schedule)

    return {
        "config": {
            "run_name": config.run_name,
            "epochs": config.epochs,
            "sequence_mode": config.sequence_mode,
            "use_language_resolved": config.language_dim > 0,
            "train_window_schedule": config.train_window_schedule,
            "resolved_train_window_min": train_window_min,
            "resolved_train_window_max": train_window_max,
            "resolved_train_window_reference_steps": train_window_reference_steps(config.train_window_schedule),
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
            "eval_conditions": list(config.eval_conditions),
            "enable_continuous_collapse": config.enable_continuous_collapse,
            "checkpoint_epochs": list(config.checkpoint_epochs),
            "pulse_value": config.pulse_value,
            "train_phase_mode": config.train_phase_mode,
            "eval_phase_mode": config.eval_phase_mode,
            "seed": config.seed,
        },
        "training": {
            "final_train_mse": float(full_history[-1]["train_loss"]) if full_history else None,
            "final_val_mse": float(full_history[-1]["val_loss"]) if full_history else None,
        },
        "eval_results": {
            condition: {
                "reset_eval_mse": _mse(reset_evals.get(condition)) if reset_evals else None,
                "reset_long_mse": _mse(reset_long_evals.get(condition)) if reset_long_evals else None,
                "continuous_eval_mse": _mse(continuous_evals.get(condition)) if continuous_evals else None,
            }
            for condition in config.eval_conditions
        },
        "comparisons": {
            f"{condition}_vs_full": {
                "reset_eval_gap": _gap(
                    reset_evals.get(condition) if reset_evals else None,
                    reset_evals.get("full") if reset_evals else None,
                ),
                "reset_long_gap": _gap(
                    reset_long_evals.get(condition) if reset_long_evals else None,
                    reset_long_evals.get("full") if reset_long_evals else None,
                ),
                "continuous_eval_gap": _gap(
                    continuous_evals.get(condition) if continuous_evals else None,
                    continuous_evals.get("full") if continuous_evals else None,
                ),
            }
            for condition in config.eval_conditions
            if condition != "full"
        },
    }


def evaluate_model(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Load checkpoint, run all eval_conditions, save plots and metrics. Returns summary."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "checkpoints" / "full_language_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ExternalClockMLP(
        trunk_dims=config.trunk_dims,
        activation=config.activation,
        language_dim=config.language_dim,
        language_readout_coverage=config.language_readout_coverage,
        use_error_input=config.use_error_input,
        use_language=True,
        use_residual=config.use_residual,
        language_readout_all_layers=config.language_readout_all_layers,
        message_carry_mode=config.message_carry_mode,
        seed=config.seed,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for condition in config.eval_conditions:
        base, _ = parse_condition(condition)
        if base in _CONDITIONS_NEEDING_ERROR and not config.use_error_input:
            equivalent = "full" if base == "sole_speech" else "sole_eye"
            warnings.warn(
                f"eval_condition '{condition}' sets force_zero_error_input=True but "
                f"model.use_error_input=False — this condition is equivalent to '{equivalent}'.",
                UserWarning,
                stacklevel=2,
            )

    reset_eval_enabled = config.eval_phase_mode in {"reset", "both"}
    continuous_eval_enabled = config.eval_phase_mode in {"continuous", "both"}
    continuous_warmup_steps = train_window_reference_steps(config.train_window_schedule)

    common_kwargs: dict[str, Any] = dict(
        cycle_steps=config.cycle_steps,
        pulse_value=config.pulse_value,
        target_kind=config.target_kind,
        mixed_sin_components=config.mixed_sin_components,
        detach_error_input=config.detach_error_input,
    )

    # Load final training state for seamless continuous eval (continuous_window runs only).
    train_state_path = run_dir / "checkpoints" / "final_train_state.pt"
    if train_state_path.exists():
        _ts = torch.load(train_state_path, map_location=device, weights_only=False)
        continuous_init_msg   = _ts["final_message"] if model.use_language else None
        continuous_init_err   = _ts["final_error"]   if model.use_error_input else None
        continuous_start_step = int(_ts["final_step"])
        has_train_state = True
    else:
        has_train_state = False
        continuous_init_msg   = None
        continuous_init_err   = None
        continuous_start_step = 0

    reset_evals: dict[str, dict] | None = None
    reset_long_evals: dict[str, dict] | None = None
    continuous_evals: dict[str, dict] | None = None

    def _run_condition(
        condition: str,
        num_steps: int,
        start_step: int = 0,
        initial_message=None,
        initial_error=None,
    ) -> dict[str, Any]:
        base, params = parse_condition(condition)
        if base in _TEMPORARY_LOSS_FLAGS:
            p2_fze, p2_dl = _TEMPORARY_LOSS_FLAGS[base]
            return _evaluate_temporary_loss_rollout(
                model,
                num_steps=num_steps,
                loss_start=params[0],
                loss_end=params[1],
                phase2_force_zero_error=p2_fze,
                phase2_disable_language=p2_dl,
                    start_step=start_step,
                    initial_message=initial_message,
                    initial_error=initial_error,
                    **common_kwargs,
                )
        if base in _LATE_TRANSITION_FLAGS:
            post_fze, post_dl = _LATE_TRANSITION_FLAGS[base]
            return _evaluate_late_transition_rollout(
                model,
                num_steps=num_steps,
                transition_step=params[0],
                post_force_zero_error=post_fze,
                post_disable_language=post_dl,
                    start_step=start_step,
                    initial_message=initial_message,
                    initial_error=initial_error,
                    **common_kwargs,
                )
        fze, dl = _condition_flags(base)
        return _evaluate_rollout(
            model,
            num_steps=num_steps,
            force_zero_error_input=fze,
            disable_language=dl,
                start_step=start_step,
                initial_message=initial_message,
                initial_error=initial_error,
                **common_kwargs,
            )

    if reset_eval_enabled:
        reset_evals = {c: _run_condition(c, config.eval_steps) for c in config.eval_conditions}
        reset_long_evals = {c: _run_condition(c, config.long_steps) for c in config.eval_conditions}

    if continuous_eval_enabled:
        continuous_evals = {}
        for condition in config.eval_conditions:
            _base, _ = parse_condition(condition)
            if has_train_state:
                # Seamlessly continue from where training ended.
                continuous_evals[condition] = _run_condition(
                    condition, config.continuous_eval_steps,
                    start_step=continuous_start_step,
                    initial_message=continuous_init_msg,
                    initial_error=continuous_init_err,
                )
            elif _base in _LATE_TRANSITION_FLAGS or _base in _TEMPORARY_LOSS_FLAGS:
                warmup_result = _evaluate_rollout(
                    model,
                    num_steps=continuous_warmup_steps,
                    force_zero_error_input=False,
                    disable_language=False,
                    **common_kwargs,
                ) if continuous_warmup_steps > 0 else None
                init_msg = warmup_result["final_message"] if warmup_result and model.use_language else None
                init_err = warmup_result["final_error"] if warmup_result and model.use_error_input else None
                continuous_evals[condition] = _run_condition(
                    condition, config.continuous_eval_steps,
                    start_step=continuous_warmup_steps,
                    initial_message=init_msg,
                    initial_error=init_err,
                )
            else:
                fze, dl = _condition_flags(_base)
                continuous_evals[condition] = _evaluate_continuous_stream(
                    model,
                    measured_steps=config.continuous_eval_steps,
                    warmup_steps=continuous_warmup_steps,
                    force_zero_error_input=fze,
                    disable_language=dl,
                    **common_kwargs,
                )

    if reset_evals is not None:
        _save_rollout_csv(
            output_path=metrics_dir / "reset_eval_rollout.csv",
            evals=reset_evals,
            language_dim=config.language_dim,
        )
    if reset_long_evals is not None:
        _save_rollout_csv(
            output_path=metrics_dir / "reset_long_rollout.csv",
            evals=reset_long_evals,
            language_dim=config.language_dim,
        )
    if continuous_evals is not None:
        _save_rollout_csv(
            output_path=metrics_dir / "continuous_eval_rollout.csv",
            evals=continuous_evals,
            language_dim=config.language_dim,
        )

    plot_rollout_diagnostics(
        reset_evals=reset_evals,
        reset_long_evals=reset_long_evals,
        continuous_evals=continuous_evals,
        output_path=plots_dir / "eval_rollout_diagnostics.png",
        config=config,
    )

    full_history_path = metrics_dir / "history_full_language.json"
    full_history: list[dict[str, float]] = (
        json.loads(full_history_path.read_text()) if full_history_path.exists() else []
    )
    summary = _build_summary(
        config=config,
        full_history=full_history,
        reset_evals=reset_evals,
        reset_long_evals=reset_long_evals,
        continuous_evals=continuous_evals,
    )
    _write_json(metrics_dir / "summary.json", summary)
    print("Final summary:")
    print(json.dumps(summary, indent=2))
    return summary
