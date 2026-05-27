"""Evaluation helpers for exp0526."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from .config import ExperimentConfig, parse_condition
    from .env import ControllerEnv
    from .model import SelfTalkController
except ImportError:  # pragma: no cover
    from config import ExperimentConfig, parse_condition
    from env import ControllerEnv
    from model import SelfTalkController


def _condition_flags(condition: str, local_step: int) -> tuple[bool, bool]:
    base, params = parse_condition(condition)
    if base == "full":
        return False, False
    if base == "sole_eye":
        return False, True
    if base == "sole_speech":
        return True, False
    if base == "neither":
        return True, True
    if base == "blink":
        start, end = params
        return start <= local_step < end, False
    if base == "stutter":
        start, end = params
        return False, start <= local_step < end
    raise ValueError(f"Unknown condition: {condition!r}")


def _observation(config: ExperimentConfig, eta: torch.Tensor, *, device: torch.device, disable_observation: bool) -> torch.Tensor:
    x_obs = torch.zeros_like(eta) if disable_observation else eta
    pulse = torch.full((1, 1), config.env.pulse_value, device=device)
    return torch.cat([x_obs, pulse], dim=1)


def _bounded_control(raw_u: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    return config.env.u_max * torch.tanh(raw_u)


def rollout_condition(
    *,
    model: SelfTalkController,
    env: ControllerEnv,
    state: torch.Tensor,
    message: torch.Tensor | None,
    start_step: int,
    num_steps: int,
    condition: str,
    config: ExperimentConfig,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    current_state = state.detach().clone()
    current_message = message.detach().clone() if message is not None else None

    rows: dict[str, list[Any]] = {
        "x": [],
        "x_sq": [],
        "abs_x": [],
        "raw_u": [],
        "u": [],
        "message_norm": [],
        "global_step": [],
    }
    messages: list[torch.Tensor] = []

    with torch.no_grad():
        for local in range(num_steps):
            global_step = start_step + local
            eta, derived = env.eta(current_state, global_step)
            disable_observation, disable_language = _condition_flags(condition, local)
            observation = _observation(config, eta, device=device, disable_observation=disable_observation)
            raw_u, next_message, _hidden = model.forward_step(
                observation,
                message_prev=current_message,
                disable_language=disable_language,
            )
            u = _bounded_control(raw_u, config)
            diagnostics = env.diagnostics(current_state, derived)

            rows["x"].append(diagnostics["x"])
            rows["x_sq"].append(diagnostics["x_sq"])
            rows["abs_x"].append(diagnostics["abs_x"])
            rows["raw_u"].append(float(raw_u.item()))
            rows["u"].append(float(u.item()))
            rows["message_norm"].append(float(torch.linalg.norm(next_message).item()) if next_message.numel() else 0.0)
            rows["global_step"].append(int(global_step))
            messages.append(next_message.squeeze(0).detach().cpu())

            current_state, _ = env.step(current_state, u, global_step)
            current_message = next_message.detach() if model.use_language else None

    messages_tensor = torch.stack(messages) if messages else torch.zeros(num_steps, model.language_dim)
    x_sq = torch.tensor(rows["x_sq"], dtype=torch.float32)
    return {
        **rows,
        "messages": messages_tensor,
        "mean_x_sq": float(torch.mean(x_sq).item()) if x_sq.numel() else 0.0,
        "final_x_sq": float(x_sq[-1].item()) if x_sq.numel() else 0.0,
        "mean_u": float(torch.tensor(rows["u"]).mean().item()) if rows["u"] else 0.0,
        "final_state": current_state.detach().cpu(),
        "final_message": current_message.detach().cpu() if current_message is not None else torch.zeros(1, 0),
        "start_step": int(start_step),
        "condition": condition,
    }


def evaluate_conditions(
    *,
    model: SelfTalkController,
    env: ControllerEnv,
    state: torch.Tensor,
    message: torch.Tensor | None,
    start_step: int,
    config: ExperimentConfig,
) -> dict[str, dict[str, Any]]:
    return {
        condition: rollout_condition(
            model=model,
            env=env,
            state=state,
            message=message,
            start_step=start_step,
            num_steps=config.eval.future_steps,
            condition=condition,
            config=config,
        )
        for condition in config.eval.eval_conditions
    }


def write_summary(output_path: Path, evals: dict[str, dict[str, Any]], history: list[dict[str, float]], config: ExperimentConfig) -> None:
    payload = {
        "history_final": history[-1] if history else {},
        "language_dim": int(config.model.language_dim),
        "use_language_resolved": bool(config.model.language_dim > 0),
        "conditions": {
            name: {
                "mean_x_sq": result["mean_x_sq"],
                "final_x_sq": result["final_x_sq"],
                "mean_u": result["mean_u"],
            }
            for name, result in evals.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))


def write_rollout_csv(output_path: Path, evals: dict[str, dict[str, Any]]) -> None:
    max_msg = 0
    for result in evals.values():
        max_msg = max(max_msg, int(result["messages"].shape[1]) if result["messages"].numel() else 0)
    header = [
        "condition",
        "step",
        "global_step",
        "x",
        "x_sq",
        "abs_x",
        "raw_u",
        "u",
        "message_norm",
    ]
    header.extend(f"message_{idx}" for idx in range(max_msg))
    lines = [",".join(header)]
    for condition, result in evals.items():
        steps = len(result["x"])
        messages = result["messages"]
        for step in range(steps):
            row: list[Any] = [
                condition,
                step,
                result["global_step"][step],
                result["x"][step],
                result["x_sq"][step],
                result["abs_x"][step],
                result["raw_u"][step],
                result["u"][step],
                result["message_norm"][step],
            ]
            for idx in range(max_msg):
                value = float(messages[step, idx].item()) if idx < messages.shape[1] else ""
                row.append(value)
            lines.append(",".join(str(value) for value in row))
    output_path.write_text("\n".join(lines))
