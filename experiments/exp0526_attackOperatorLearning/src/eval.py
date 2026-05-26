"""Evaluation helpers for exp0526."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from .config import ExperimentConfig, parse_condition
    from .env import AgeStructuredPredatorPreyEnv
    from .model import SelfTalkController
except ImportError:  # pragma: no cover
    from config import ExperimentConfig, parse_condition
    from env import AgeStructuredPredatorPreyEnv
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


def rollout_condition(
    *,
    model: SelfTalkController,
    env: AgeStructuredPredatorPreyEnv,
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
        "eta1": [], "eta2": [], "eta_norm_sq": [], "raw_u": [], "u": [],
        "message_norm": [], "min_population": [], "population1": [], "population2": [],
        "equilibrium_population1": [], "equilibrium_population2": [], "global_step": [],
        "k1_amp": [], "k2_amp": [], "mu1_base": [], "mu2_base": [],
        "mu1_sen_amp": [], "mu2_sen_amp": [], "zeta1": [], "zeta2": [],
    }
    messages: list[torch.Tensor] = []

    with torch.no_grad():
        for local in range(num_steps):
            global_step = start_step + local
            eta, derived = env.eta(current_state, global_step)
            observation = torch.cat(
                [eta, torch.full((1, 1), config.env.pulse_value, device=device)],
                dim=1,
            )
            force_zero_error, disable_language = _condition_flags(condition, local)
            raw_u, next_message, _hidden = model.forward_step(
                observation,
                error_view=eta,
                message_prev=current_message,
                force_zero_error=force_zero_error,
                disable_language=disable_language,
            )
            u = torch.nn.functional.softplus(raw_u)
            diagnostics = env.diagnostics(current_state, derived)

            rows["eta1"].append(float(eta[0, 0].item()))
            rows["eta2"].append(float(eta[0, 1].item()))
            rows["eta_norm_sq"].append(float(torch.sum(eta ** 2).item()))
            rows["raw_u"].append(float(raw_u.item()))
            rows["u"].append(float(u.item()))
            rows["message_norm"].append(float(torch.linalg.norm(next_message).item()) if next_message.numel() else 0.0)
            rows["min_population"].append(diagnostics["min_population"])
            rows["population1"].append(diagnostics["population1"])
            rows["population2"].append(diagnostics["population2"])
            rows["equilibrium_population1"].append(diagnostics["equilibrium_population1"])
            rows["equilibrium_population2"].append(diagnostics["equilibrium_population2"])
            rows["global_step"].append(int(global_step))
            for key in ("k1_amp", "k2_amp", "mu1_base", "mu2_base", "mu1_sen_amp", "mu2_sen_amp", "zeta1", "zeta2"):
                rows[key].append(float(diagnostics[key]))
            messages.append(next_message.squeeze(0).detach().cpu())

            current_state, _ = env.step(current_state, u, global_step)
            current_message = next_message.detach() if model.use_language else None

    messages_tensor = torch.stack(messages) if messages else torch.zeros(num_steps, model.language_dim)
    eta_norm = torch.tensor(rows["eta_norm_sq"], dtype=torch.float32)
    return {
        **rows,
        "messages": messages_tensor,
        "mean_eta_norm_sq": float(torch.mean(eta_norm).item()) if eta_norm.numel() else 0.0,
        "final_eta_norm_sq": float(eta_norm[-1].item()) if eta_norm.numel() else 0.0,
        "mean_u": float(torch.tensor(rows["u"]).mean().item()) if rows["u"] else 0.0,
        "final_state": current_state.detach().cpu(),
        "final_message": current_message.detach().cpu() if current_message is not None else torch.zeros(1, 0),
        "start_step": int(start_step),
        "condition": condition,
    }


def evaluate_conditions(
    *,
    model: SelfTalkController,
    env: AgeStructuredPredatorPreyEnv,
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


def write_summary(output_path: Path, evals: dict[str, dict[str, Any]], history: list[dict[str, float]]) -> None:
    payload = {
        "history_final": history[-1] if history else {},
        "conditions": {
            name: {
                "mean_eta_norm_sq": result["mean_eta_norm_sq"],
                "final_eta_norm_sq": result["final_eta_norm_sq"],
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
        "condition", "step", "global_step", "eta1", "eta2", "eta_norm_sq",
        "raw_u", "u", "message_norm", "min_population",
        "population1", "population2", "equilibrium_population1", "equilibrium_population2",
        "k1_amp", "k2_amp", "mu1_base", "mu2_base", "mu1_sen_amp", "mu2_sen_amp", "zeta1", "zeta2",
    ]
    header.extend(f"message_{idx}" for idx in range(max_msg))
    lines = [",".join(header)]
    for condition, result in evals.items():
        steps = len(result["eta1"])
        messages = result["messages"]
        for step in range(steps):
            row: list[Any] = [
                condition,
                step,
                result["global_step"][step],
                result["eta1"][step],
                result["eta2"][step],
                result["eta_norm_sq"][step],
                result["raw_u"][step],
                result["u"][step],
                result["message_norm"][step],
                result["min_population"][step],
                result["population1"][step],
                result["population2"][step],
                result["equilibrium_population1"][step],
                result["equilibrium_population2"][step],
                result["k1_amp"][step],
                result["k2_amp"][step],
                result["mu1_base"][step],
                result["mu2_base"][step],
                result["mu1_sen_amp"][step],
                result["mu2_sen_amp"][step],
                result["zeta1"][step],
                result["zeta2"][step],
            ]
            for idx in range(max_msg):
                value = float(messages[step, idx].item()) if idx < messages.shape[1] else ""
                row.append(value)
            lines.append(",".join(str(value) for value in row))
    output_path.write_text("\n".join(lines))
