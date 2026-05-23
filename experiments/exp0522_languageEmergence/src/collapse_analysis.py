"""Offline continuous-collapse analysis for exp0522."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .config import ExperimentConfig, config_from_user_dict
    from .model import ExternalClockMLP
    from .train import _evaluate_continuous_stream
except ImportError:  # pragma: no cover - script mode
    from config import ExperimentConfig, config_from_user_dict
    from model import ExternalClockMLP
    from train import _evaluate_continuous_stream


def _write_json(output_path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    output_path.write_text(json.dumps(payload, indent=2))


def _load_resolved_config(run_dir: Path) -> ExperimentConfig:
    payload = json.loads((run_dir / "resolved_config.json").read_text())
    payload.pop("resolved", None)
    return config_from_user_dict(payload)


def _build_full_model(config: ExperimentConfig) -> ExternalClockMLP:
    return ExternalClockMLP(
        trunk_dims=config.trunk_dims,
        activation=config.activation,
        language_dim=config.language_dim,
        language_readout_coverage=config.language_readout_coverage,
        use_language=True,
        seed=config.seed,
    )


def _checkpoint_records(run_dir: Path) -> list[dict[str, Any]]:
    ckpt_dir = run_dir / "checkpoints"
    records: list[dict[str, Any]] = []
    for ckpt_path in sorted(ckpt_dir.glob("full_language_epoch_*.pt")):
        stem = ckpt_path.stem
        epoch = int(stem.rsplit("_", 1)[-1])
        records.append(
            {
                "epoch": epoch,
                "label": f"epoch_{epoch:04d}",
                "path": ckpt_path,
                "is_final": False,
            }
        )
    final_path = ckpt_dir / "full_language_final.pt"
    if final_path.exists():
        final_payload = torch.load(final_path, map_location="cpu")
        final_epoch = int(final_payload.get("metadata", {}).get("epoch", -1))
        records = [record for record in records if record["epoch"] != final_epoch]
        records.append(
            {
                "epoch": final_epoch,
                "label": "final",
                "path": final_path,
                "is_final": True,
            }
        )
    return sorted(records, key=lambda record: (record["epoch"], record["is_final"]))


def _corrcoef_safe(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_std = float(np.std(lhs))
    rhs_std = float(np.std(rhs))
    if lhs_std <= 1e-12 or rhs_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _rollout_metrics(result: dict[str, Any]) -> dict[str, float]:
    prediction = result["prediction"].numpy()
    target = result["target"].numpy()
    messages = result["messages"].numpy()
    if messages.shape[1] == 0:
        temporal_variance = 0.0
        mean_step_delta = 0.0
    else:
        temporal_variance = float(np.var(messages, axis=0).mean())
        deltas = np.diff(messages, axis=0)
        mean_step_delta = float(np.linalg.norm(deltas, axis=1).mean()) if deltas.size else 0.0
    return {
        "pred_std": float(np.std(prediction)),
        "corr_pred_target": _corrcoef_safe(prediction, target),
        "message_temporal_variance": temporal_variance,
        "mean_step_message_delta": mean_step_delta,
    }


def _snapshot_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_step": int(result.get("start_step", 0)),
        "phase": result["phase"].tolist(),
        "target": result["target"].tolist(),
        "prediction": result["prediction"].tolist(),
        "message_norm": result["message_norm"].tolist(),
        "messages": result["messages"].tolist(),
    }


def _plot_collapse_metrics(
    *,
    output_path: Path,
    checkpoint_entries: list[dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    epochs = [entry["epoch"] for entry in checkpoint_entries]
    labels = [
        ("pred_std", "Prediction Std"),
        ("corr_pred_target", "Corr(pred, target)"),
        ("message_temporal_variance", "Message Temporal Variance"),
        ("mean_step_message_delta", "Mean Step Message Delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for axis, (metric_key, title) in zip(axes.flat, labels):
        values = [entry["metrics"][metric_key] for entry in checkpoint_entries]
        axis.plot(epochs, values, marker="o", linewidth=1.6)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=config.plot_grid_alpha)
    fig.suptitle("Continuous Collapse Metrics", fontsize=13)
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)


def _plot_checkpoint_rollouts(
    *,
    output_path: Path,
    checkpoint_entries: list[dict[str, Any]],
    config: ExperimentConfig,
) -> None:
    nrows = len(checkpoint_entries)
    pred_steps = min(config.plot_long_steps, config.continuous_eval_steps)
    message_steps = min(config.plot_message_steps, config.continuous_eval_steps)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(12.5, max(3.0 * nrows, 4.5)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, entry in enumerate(checkpoint_entries):
        result = entry["rollout"]
        pred_axis = axes[row_idx, 0]
        msg_axis = axes[row_idx, 1]

        pred_x = np.arange(pred_steps) + int(result.get("start_step", 0))
        pred_axis.plot(
            pred_x,
            result["target"][:pred_steps].numpy(),
            color=config.plot_target_color,
            linestyle=config.plot_target_linestyle,
            linewidth=config.plot_target_linewidth,
            label="target",
        )
        pred_axis.plot(
            pred_x,
            result["prediction"][:pred_steps].numpy(),
            linewidth=config.plot_series_linewidth,
            label="full",
        )
        pred_axis.set_title(f"{entry['label']} prediction", fontsize=11)
        pred_axis.set_xlabel("Global Step")
        pred_axis.grid(alpha=config.plot_grid_alpha)
        pred_axis.legend(loc="upper right", fontsize=8)

        msg_x = np.arange(message_steps) + int(result.get("start_step", 0))
        messages = result["messages"][:message_steps].numpy()
        if messages.shape[1] > 0:
            for msg_idx in range(messages.shape[1]):
                msg_axis.plot(
                    msg_x,
                    messages[:, msg_idx],
                    linewidth=config.plot_aux_linewidth,
                    label=f"m{msg_idx}",
                )
        msg_axis.set_title(f"{entry['label']} messages", fontsize=11)
        msg_axis.set_xlabel("Global Step")
        msg_axis.grid(alpha=config.plot_grid_alpha)
        if messages.shape[1] > 0:
            msg_axis.legend(loc="upper right", ncols=min(messages.shape[1], 4), fontsize=8)
    fig.suptitle("Continuous Collapse Rollout Snapshots", fontsize=13)
    fig.savefig(output_path, dpi=config.plot_dpi)
    plt.close(fig)


def analyze_continuous_collapse(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config = _load_resolved_config(run_dir)
    checkpoint_entries = _checkpoint_records(run_dir)
    if not checkpoint_entries:
        raise FileNotFoundError(f"No full_language checkpoints found in {run_dir / 'checkpoints'}.")

    analysis_dir = run_dir / "analysis" / "continuous_collapse"
    snapshots_dir = analysis_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    analyzed_entries: list[dict[str, Any]] = []
    for record in checkpoint_entries:
        checkpoint_payload = torch.load(record["path"], map_location="cpu")
        model = _build_full_model(config)
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        model.eval()
        rollout = _evaluate_continuous_stream(
            model,
            measured_steps=config.continuous_eval_steps,
            warmup_steps=config.fixed_train_steps,
            cycle_steps=config.cycle_steps,
            pulse_value=config.pulse_value,
            target_kind=config.target_kind,
            mixed_sin_components=config.mixed_sin_components,
            disable_language=False,
        )
        metrics = _rollout_metrics(rollout)
        snapshot_filename = f"{record['label']}.json"
        _write_json(snapshots_dir / snapshot_filename, _snapshot_payload(rollout))
        analyzed_entries.append(
            {
                "epoch": int(record["epoch"]),
                "label": str(record["label"]),
                "checkpoint_file": record["path"].name,
                "snapshot_file": f"snapshots/{snapshot_filename}",
                "metrics": metrics,
                "rollout": rollout,
            }
        )

    _plot_collapse_metrics(
        output_path=analysis_dir / "collapse_metrics.png",
        checkpoint_entries=analyzed_entries,
        config=config,
    )
    _plot_checkpoint_rollouts(
        output_path=analysis_dir / "checkpoint_rollouts.png",
        checkpoint_entries=analyzed_entries,
        config=config,
    )

    metrics_payload = {
        "analysis_kind": "continuous_collapse",
        "analysis_config": {
            "continuous_eval_steps": config.continuous_eval_steps,
            "warmup_steps": config.fixed_train_steps,
            "checkpoint_epochs": list(config.checkpoint_epochs),
        },
        "checkpoints": [
            {
                "epoch": entry["epoch"],
                "label": entry["label"],
                "checkpoint_file": entry["checkpoint_file"],
                "snapshot_file": entry["snapshot_file"],
                **entry["metrics"],
            }
            for entry in analyzed_entries
        ],
    }
    _write_json(analysis_dir / "metrics.json", metrics_payload)

    return {
        "analysis_dir": str(analysis_dir),
        "num_checkpoints": len(analyzed_entries),
        "metrics_path": str(analysis_dir / "metrics.json"),
    }
