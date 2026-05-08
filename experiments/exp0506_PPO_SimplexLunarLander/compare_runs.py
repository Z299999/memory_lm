#!/usr/bin/env python3
"""Generate a single MLP-vs-SMN comparison figure from explicit run directories."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

_here = Path(__file__).parent
_runtime_cache_dir = _here / ".runtime_cache"
_mpl_cache_dir = _runtime_cache_dir / "mpl"
_xdg_cache_dir = _runtime_cache_dir / "xdg"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache_dir))

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-train-run", required=True, help="exp0501 train run directory")
    parser.add_argument("--mlp-eval-run", required=True, help="exp0501 eval run directory")
    parser.add_argument("--smn-train-run", required=True, help="exp0506 train run directory")
    parser.add_argument("--smn-eval-run", required=True, help="exp0506 eval run directory")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to exp0506/runs/comparisons/<timestamp>",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_pair(train_run: Path, eval_run: Path) -> tuple[dict, dict]:
    train_summary = load_json(train_run / "train_summary.json")
    eval_summary = load_json(eval_run / "eval_summary.json")
    return train_summary, eval_summary


def smoothed_rewards(rewards: list[float], window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rewards_arr = np.asarray(rewards, dtype=float)
    if rewards_arr.size == 0:
        return np.array([]), np.array([])
    use_window = min(window, rewards_arr.size)
    kernel = np.ones(use_window, dtype=float) / use_window
    smooth = np.convolve(rewards_arr, kernel, mode="valid")
    x = np.arange(smooth.size)
    return x, smooth


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(float(value), 60.0)
    if minutes >= 1.0:
        return f"{minutes:.1f}m ({seconds:.0f}s rem)"
    return f"{seconds:.1f}s"


def mlp_label(train_summary: dict) -> str:
    actor = train_summary["actor_structure"]
    critic = train_summary["critic_structure"]
    return f"MLP actor={actor} critic={critic}"


def smn_label(train_summary: dict) -> str:
    return (
        f"SMN n={train_summary['smn_n']}, m={train_summary['smn_m']}, "
        f"act={train_summary.get('smn_activation', 'relu')}"
    )


def panel_lines(train_summary: dict, eval_summary: dict) -> list[str]:
    if train_summary["model_type"] == "MLP":
        architecture = mlp_label(train_summary)
    else:
        architecture = smn_label(train_summary)
    return [
        architecture,
        f"map_scale: train={train_summary['map_scale']} eval={eval_summary['map_scale']}",
        f"params: actor={train_summary['actor_param_count']} critic={train_summary['critic_param_count']} total={train_summary['total_param_count']}",
        f"train: Avg(100ep)={train_summary['final_avg_100ep']:.2f} best_ep={train_summary['best_episode_reward']:.2f}",
        f"train time: {format_seconds(train_summary['total_training_time_sec'])}",
        f"avg update: {train_summary['avg_update_time_sec']:.2f}s",
        f"eval success: {eval_summary['success_count']}/{eval_summary['num_eval_episodes']} ({eval_summary['success_rate'] * 100:.1f}%)",
        f"eval reward: {eval_summary['mean_reward']:.2f} ± {eval_summary['std_reward']:.2f}",
        f"eval time: {format_seconds(eval_summary['total_eval_time_sec'])}",
    ]


def draw_panel(ax, title: str, lines: list[str], facecolor: str) -> None:
    ax.set_title(title)
    ax.set_facecolor(facecolor)
    ax.axis("off")
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    args = parse_args()
    mlp_train_run = Path(args.mlp_train_run).resolve()
    mlp_eval_run = Path(args.mlp_eval_run).resolve()
    smn_train_run = Path(args.smn_train_run).resolve()
    smn_eval_run = Path(args.smn_eval_run).resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (_here / "runs" / "comparisons" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mlp_train, mlp_eval = load_run_pair(mlp_train_run, mlp_eval_run)
    smn_train, smn_eval = load_run_pair(smn_train_run, smn_eval_run)

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_curve = fig.add_subplot(grid[0, :])
    ax_mlp = fig.add_subplot(grid[1, 0])
    ax_smn = fig.add_subplot(grid[1, 1])

    mlp_x, mlp_y = smoothed_rewards(mlp_train["ep_rewards"])
    smn_x, smn_y = smoothed_rewards(smn_train["ep_rewards"])
    ax_curve.plot(mlp_x, mlp_y, linewidth=2.0, color="tab:blue", label="MLP baseline")
    ax_curve.plot(smn_x, smn_y, linewidth=2.0, color="tab:orange", label="SMN backbone")
    ax_curve.axhline(200, color="tab:red", linestyle="--", linewidth=1.0, label="solved threshold")
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Reward (50-episode moving average)")
    ax_curve.set_title("MLP vs SMN Training Curve Overlay")
    ax_curve.legend(fontsize=9)
    ax_curve.grid(alpha=0.2, linewidth=0.6)

    draw_panel(
        ax_mlp,
        f"MLP Baseline\nactor={mlp_train['actor_structure']} critic={mlp_train['critic_structure']}",
        panel_lines(mlp_train, mlp_eval),
        "#f4f8ff",
    )
    draw_panel(
        ax_smn,
        f"SMN Variant\nn={smn_train['smn_n']}, m={smn_train['smn_m']}, act={smn_train.get('smn_activation', 'relu')}",
        panel_lines(smn_train, smn_eval),
        "#fff7ef",
    )

    fig.suptitle(
        "exp0501 vs exp0506 Comparison\n"
        f"MLP train={mlp_train_run.name} eval={mlp_eval_run.name} | "
        f"SMN train={smn_train_run.name} eval={smn_eval_run.name}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    comparison_png = output_dir / "comparison.png"
    fig.savefig(comparison_png, dpi=160)
    plt.close(fig)

    comparison_summary = {
        "comparison_type": "MLP_vs_SMN",
        "mlp_train_run": str(mlp_train_run),
        "mlp_eval_run": str(mlp_eval_run),
        "smn_train_run": str(smn_train_run),
        "smn_eval_run": str(smn_eval_run),
        "outputs": {
            "comparison_png": str(comparison_png),
        },
        "mlp": {
            "train_summary": mlp_train,
            "eval_summary": mlp_eval,
        },
        "smn": {
            "train_summary": smn_train,
            "eval_summary": smn_eval,
        },
    }
    comparison_json = output_dir / "comparison_summary.json"
    write_json(comparison_json, comparison_summary)

    print(f"comparison figure saved: {comparison_png}")
    print(f"comparison summary saved: {comparison_json}")


if __name__ == "__main__":
    main()
