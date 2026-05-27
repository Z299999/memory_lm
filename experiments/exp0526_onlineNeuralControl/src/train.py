"""Training loop for exp0526 online neural control."""

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
    from .config import ExperimentConfig, copy_config_to_run_dir, parse_train_window_schedule, write_resolved_config
    from .env import build_env
    from .eval import evaluate_conditions, rollout_condition, write_rollout_csv, write_summary
    from .model import SelfTalkController
    from .plots import plot_rollout_diagnostics, plot_training_curves, plot_training_timeline
except ImportError:  # pragma: no cover
    from config import ExperimentConfig, copy_config_to_run_dir, parse_train_window_schedule, write_resolved_config
    from env import build_env
    from eval import evaluate_conditions, rollout_condition, write_rollout_csv, write_summary
    from model import SelfTalkController
    from plots import plot_rollout_diagnostics, plot_training_curves, plot_training_timeline


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_run_dir(base_dir: Path, run_name: str) -> Path:
    now = datetime.now()
    date_dir = base_dir / now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_dir = date_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _sample_window_steps(config: ExperimentConfig) -> int:
    mode, lower, upper = parse_train_window_schedule(config.train.train_window_schedule)
    if mode == "fixed":
        return lower
    return random.randint(lower, upper)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _build_model(config: ExperimentConfig, *, device: torch.device, observation_dim: int) -> SelfTalkController:
    return SelfTalkController(
        trunk_dims=config.model.trunk_dims,
        activation=config.model.activation,
        language_dim=config.model.language_dim,
        language_readout_coverage=config.model.language_readout_coverage,
        use_residual=config.model.use_residual,
        language_readout_all_layers=config.model.language_readout_all_layers,
        message_carry_mode=config.model.message_carry_mode,
        seed=config.run.seed,
        observation_dim=observation_dim,
    ).to(device)


def _observation(config: ExperimentConfig, eta: torch.Tensor, *, device: torch.device, disable_observation: bool = False) -> torch.Tensor:
    x_obs = torch.zeros_like(eta) if disable_observation else eta
    pulse = torch.full((1, 1), config.env.pulse_value, device=device)
    return torch.cat([x_obs, pulse], dim=1)


def _bounded_control(raw_u: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    return config.env.u_max * torch.tanh(raw_u)


def train_model(config: ExperimentConfig, config_path: Path) -> Path:
    _seed_everything(config.run.seed)
    root = config_path.resolve().parent
    run_root = (root / config.run.output_root).resolve()
    run_dir = _make_run_dir(run_root, config.run.run_name)
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    copy_config_to_run_dir(config_path, run_dir)
    write_resolved_config(config, run_dir / "resolved_config.json")

    device = torch.device("cpu")
    env = build_env(config, device=device)
    model = _build_model(config, device=device, observation_dim=env.state_dim + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    state = env.initial_state().to(device)
    message: torch.Tensor | None = None
    global_step = 0
    history: list[dict[str, float]] = []
    timeline_records: list[dict[str, Any]] = []
    last_val = 0.0

    for epoch in range(1, config.train.epochs + 1):
        window_steps = _sample_window_steps(config)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        state_losses: list[torch.Tensor] = []
        control_losses: list[torch.Tensor] = []
        record: dict[str, Any] = {
            "start_step": int(global_step),
            "end_step": int(global_step + window_steps - 1),
            "steps": [],
            "u": [],
        }
        for state_idx in range(env.state_dim):
            record[f"state_{state_idx}"] = []
        record["state_norm_sq"] = []
        max_abs_state = 0.0
        final_state_sq = 0.0

        for _local in range(window_steps):
            eta, _derived = env.eta(state, global_step)
            observation = _observation(config, eta, device=device)
            raw_u, next_message, _hidden = model.forward_step(
                observation,
                message_prev=message,
                disable_language=False,
            )
            u = _bounded_control(raw_u, config)
            next_state, _ = env.step(state, u, global_step)
            next_eta, next_derived = env.eta(next_state, global_step + 1)
            state_sq = torch.sum(next_eta ** 2)

            state_losses.append(state_sq)
            control_losses.append(torch.mean(u ** 2))
            diagnostics = env.diagnostics(next_state, next_derived)
            max_abs_state = max(max_abs_state, float(diagnostics.get("abs_state_max", 0.0)))
            final_state_sq = float(state_sq.detach().item())
            record["steps"].append(int(global_step))
            for state_idx in range(env.state_dim):
                record[f"state_{state_idx}"].append(float(diagnostics[f"state_{state_idx}"]))
            record["state_norm_sq"].append(float(diagnostics["state_norm_sq"]))
            record["u"].append(float(u.detach().item()))

            state = next_state
            message = next_message if model.use_language else None
            global_step += 1

        state_loss = torch.stack(state_losses).mean()
        control_loss = torch.stack(control_losses).mean()
        total_loss = state_loss + config.train.control_loss_weight * control_loss
        total_loss.backward()
        if config.train.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        state = state.detach()
        if message is not None:
            message = message.detach()
        if config.plot.plot_show_training_timeline:
            timeline_records.append(record)

        if epoch == 1 or epoch % config.run.log_every == 0 or epoch == config.train.epochs:
            val = rollout_condition(
                model=model,
                env=env,
                state=state,
                message=message,
                start_step=global_step,
                num_steps=config.eval.eval_steps,
                condition="full",
                config=config,
            )
            last_val = float(val["mean_state_norm_sq"])

        row = {
            "epoch": float(epoch),
            "global_step": float(global_step),
            "train_steps": float(window_steps),
            "total_loss": float(total_loss.detach().item()),
            "state_loss": float(state_loss.detach().item()),
            "control_loss": float(control_loss.detach().item()),
            "val_state_loss": float(last_val),
            "max_abs_state": float(max_abs_state),
            "final_state_sq": float(final_state_sq),
        }
        history.append(row)
        if epoch == 1 or epoch % config.run.log_every == 0 or epoch == config.train.epochs:
            print(
                f"[exp0526] epoch {epoch:4d}/{config.train.epochs} "
                f"steps={window_steps:3d} x2={row['state_loss']:.6g} "
                f"u2={row['control_loss']:.6g} val={row['val_state_loss']:.6g}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {"config": config.to_dict(), "epoch": config.train.epochs, "global_step": global_step},
        },
        ckpt_dir / "full_controller_final.pt",
    )
    torch.save(
        {
            "state": state.detach().cpu(),
            "message": message.detach().cpu() if message is not None else torch.zeros(1, 0),
            "global_step": int(global_step),
        },
        ckpt_dir / "final_train_state.pt",
    )

    evals = evaluate_conditions(
        model=model,
        env=env,
        state=state.detach(),
        message=message.detach() if message is not None else None,
        start_step=global_step,
        config=config,
    )
    _write_json(metrics_dir / "history_full_controller.json", history)
    if timeline_records:
        _write_json(metrics_dir / "training_timeline.json", {"windows": timeline_records})
    write_summary(metrics_dir / "summary.json", evals, history, config)
    write_rollout_csv(metrics_dir / "eval_rollout.csv", evals)

    plot_training_curves(history=history, output_path=plots_dir / "training_curves.png", config=config)
    if timeline_records:
        plot_training_timeline(records=timeline_records, output_path=plots_dir / "training_timeline.png", config=config)
    plot_rollout_diagnostics(
        evals=evals,
        output_path=plots_dir / "eval_rollout_diagnostics.png",
        config=config,
        env=env,
    )

    latest = run_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir, target_is_directory=True)
    return run_dir
