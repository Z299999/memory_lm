#!/usr/bin/env python3
"""exp0506 — PPO + SMN 评估脚本。

读取 eval_config.yaml，加载 checkpoint，在指定环境中贪婪跑 N 集，输出：
  - runs/<ts>_eval/eval.png   每集奖励柱状图 + 奖励分布直方图
  - runs/<ts>_eval/demo.gif   前 3 集录像
  - 终端：success_rate, mean ± std

用法：
    python3 ppoEval.py
"""

import json
import os
import sys
import time
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

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import yaml
from torch.distributions import Normal

_repo_root = _here.parent.parent
_simplex_src = _repo_root / "TOOLS" / "SimplexNet" / "src"
if str(_simplex_src) not in os.sys.path:
    os.sys.path.insert(0, str(_simplex_src))

from simplexnet import SMN

# =============================================================================
# 1. 配置（必须在 lunar_ballistic import 之前设好 MAP_SCALE）
# =============================================================================

eval_cfg = yaml.safe_load(open(_here / "configEval.yaml"))

os.environ["MAP_SCALE"] = str(eval_cfg.get("map_scale", 1.0))


def log(msg: str) -> None:
    print(msg, flush=True)

# 物理参数来源：train config 或 eval config
if eval_cfg.get("use_train_physics", False):
    _train_cfg = yaml.safe_load(open(_here / "configTrain.yaml"))
    physics_cfg = _train_cfg
    log("物理参数来源：configTrain.yaml")
else:
    physics_cfg = eval_cfg
    log("物理参数来源：configEval.yaml")

# =============================================================================
# 2. 加载 checkpoint
# =============================================================================

ckpt_path = _here / eval_cfg.get("checkpoint", "models/latest.pt")
if not ckpt_path.exists():
    sys.exit(f"[Error] Checkpoint not found: {ckpt_path}")

ckpt = torch.load(ckpt_path, weights_only=False)
smn_n = ckpt.get("smn_n", eval_cfg.get("smn_n", 4))
smn_m = ckpt.get("smn_m", eval_cfg.get("smn_m", 9))
smn_activation = ckpt.get("smn_activation", eval_cfg.get("smn_activation", "relu"))

# =============================================================================
# 3. 环境工厂
# =============================================================================


def make_env(render_mode=None):
    if eval_cfg.get("use_ballistic", False):
        from lunar_ballistic import BallisticLunarLander
        return BallisticLunarLander(
            render_mode       = render_mode,
            continuous        = eval_cfg.get("continuous",        True),
            init_speed            = physics_cfg.get("init_speed",            5.0),
            init_flight_angle_deg = physics_cfg.get("init_flight_angle_deg", 45.0),
            random_side           = physics_cfg.get("random_side",           True),
            init_body_angle_deg   = physics_cfg.get("init_body_angle_deg",   13.5),
            init_angular_velocity = physics_cfg.get("init_angular_velocity", 0.0),
            init_altitude_m       = physics_cfg.get("init_altitude_m",       None),
            init_x_m              = physics_cfg.get("init_x_m",              0.0),
        )
    kwargs = {"render_mode": render_mode} if render_mode else {}
    kwargs["continuous"] = eval_cfg.get("continuous", True)
    return gym.make(eval_cfg["env_name"], **kwargs)


_probe  = make_env()
obs_dim = _probe.observation_space.shape[0]
act_dim = _probe.action_space.shape[0]
_probe.close()

log(f"评估环境：{eval_cfg['env_name']}  map_scale={eval_cfg.get('map_scale',1.0)}")
log(f"Checkpoint：{ckpt_path}  (trained to update {ckpt.get('update','?')})")

run_dir = _here / f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval"
run_dir.mkdir(parents=True, exist_ok=True)
log(f"输出目录：{run_dir}\n")


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

# =============================================================================
# 4. 网络（与 ppoTrain.py 相同定义）
# =============================================================================


class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, act_dim, smn_n, smn_m, smn_activation):
        super().__init__()
        self.net = SMN(
            n=smn_n,
            m=smn_m,
            n_in=obs_dim,
            n_out=act_dim,
            activation=smn_activation,
            output_activation="identity",
            scale_output=True,
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def greedy(self, s):
        with torch.no_grad():
            if s.dim() == 1:
                s = s.unsqueeze(0)
            return torch.tanh(self.net(s)).clamp(-1, 1).squeeze(0).numpy()


actor = ContinuousActor(obs_dim, act_dim, smn_n, smn_m, smn_activation)
actor.load_state_dict(ckpt["actor"])
actor.eval()
actor_param_count = sum(p.numel() for p in actor.parameters())
log(f"SMN actor config: n={smn_n}, m={smn_m}, activation={smn_activation}")

# =============================================================================
# 5. 贪婪评估
# =============================================================================

n_episodes = eval_cfg.get("num_eval_episodes", 50)
max_steps  = eval_cfg.get("max_steps", 1000)

rewards    = []
outcomes   = []   # "success" / "crash" / "timeout"
ep_lengths = []

env = make_env()
eval_start = time.perf_counter()
for ep in range(n_episodes):
    s, _ = env.reset()
    total, length, outcome = 0.0, 0, "timeout"
    for _ in range(max_steps):
        a_env = actor.greedy(torch.FloatTensor(s))
        s, r, terminated, truncated, _ = env.step(a_env)
        total += r
        length += 1
        if terminated:
            outcome = "success" if r > 0 else "crash"
            break
        if truncated:
            break
    rewards.append(total)
    outcomes.append(outcome)
    ep_lengths.append(length)
    if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
        elapsed_sec = time.perf_counter() - eval_start
        avg_ep_sec = elapsed_sec / (ep + 1)
        eta_sec = avg_ep_sec * (n_episodes - ep - 1)
        log(f"  Eval episode {ep + 1:3d}/{n_episodes} | "
            f"reward: {total:7.2f} | outcome: {outcome:7s} | "
            f"eta: {eta_sec/60.0:4.1f}m")

env.close()
total_eval_time = time.perf_counter() - eval_start

n_success = outcomes.count("success")
n_crash   = outcomes.count("crash")
n_timeout = outcomes.count("timeout")
log(f"评估结果（{n_episodes} 集，贪婪）：")
log(f"  成功：{n_success}/{n_episodes} ({n_success/n_episodes*100:.1f}%)")
log(f"  坠毁：{n_crash}/{n_episodes} ({n_crash/n_episodes*100:.1f}%)")
log(f"  超时：{n_timeout}/{n_episodes} ({n_timeout/n_episodes*100:.1f}%)")
log(f"  平均奖励：{np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
log(f"  最高奖励：{max(rewards):.2f}  最低：{min(rewards):.2f}")
log(f"  平均集长：{np.mean(ep_lengths):.1f} 步")
log(f"  评估用时：{total_eval_time:.2f}s")

# =============================================================================
# 6. 评估图（2 列：每集奖励柱状图 | 奖励分布直方图）
# =============================================================================

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

OUTCOME_COLORS = {"success": "tab:green", "crash": "tab:red", "timeout": "tab:orange"}

# 上：每集奖励折线图，三色散点
ep_idx = np.arange(n_episodes)
rewards_arr = np.array(rewards)
ax1.plot(ep_idx, rewards_arr, color="tab:blue", linewidth=1.2, alpha=0.5)
for outcome, color in OUTCOME_COLORS.items():
    mask = np.array([o == outcome for o in outcomes])
    if mask.any():
        ax1.scatter(ep_idx[mask], rewards_arr[mask],
                    color=color, s=20, zorder=3, label=outcome)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.axhline(np.mean(rewards), color="gray", linewidth=0.8, linestyle=":",
            label=f"mean={np.mean(rewards):.1f}")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title(f"Per-Episode Reward  map_scale={eval_cfg.get('map_scale',1.0)}\n"
              f"success={n_success}  crash={n_crash}  timeout={n_timeout}")
ax1.legend(fontsize=8)

# 下：奖励分布直方图（三色堆叠）
for outcome, color in OUTCOME_COLORS.items():
    r_sub = [r for r, o in zip(rewards, outcomes) if o == outcome]
    if r_sub:
        ax2.hist(r_sub, bins=15, color=color, alpha=0.6,
                 label=f"{outcome} ({len(r_sub)})", edgecolor="white")
ax2.axvline(np.mean(rewards), color="black", linestyle="--",
            linewidth=1.5, label=f"mean={np.mean(rewards):.1f}")
ax2.set_xlabel("Total Reward")
ax2.set_ylabel("Count")
ax2.set_title(f"Reward Distribution  ({n_episodes} episodes)\n"
              f"init_speed={physics_cfg.get('init_speed',5.0)}  "
              f"flight_angle={physics_cfg.get('init_flight_angle_deg',45)}°")
ax2.legend(fontsize=8)

plt.tight_layout()
plot_path = run_dir / "eval.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
log(f"\n评估图已保存：{plot_path}")

eval_summary = {
    "experiment": "exp0506_PPO_SimplexLunarLander",
    "model_type": "SMN",
    "run_type": "eval",
    "run_dir": str(run_dir),
    "config_path": str(_here / "configEval.yaml"),
    "checkpoint_path": str(ckpt_path),
    "env_name": eval_cfg["env_name"],
    "map_scale": eval_cfg.get("map_scale", 1.0),
    "num_eval_episodes": n_episodes,
    "max_steps": max_steps,
    "obs_dim": obs_dim,
    "act_dim": act_dim,
    "smn_n": smn_n,
    "smn_m": smn_m,
    "smn_activation": smn_activation,
    "actor_param_count": actor_param_count,
    "total_eval_time_sec": total_eval_time,
    "success_count": n_success,
    "crash_count": n_crash,
    "timeout_count": n_timeout,
    "success_rate": n_success / n_episodes if n_episodes else 0.0,
    "mean_reward": float(np.mean(rewards)),
    "std_reward": float(np.std(rewards)),
    "best_reward": float(max(rewards)),
    "worst_reward": float(min(rewards)),
    "avg_episode_length": float(np.mean(ep_lengths)),
    "rewards": rewards,
    "outcomes": outcomes,
    "episode_lengths": ep_lengths,
    "eval_plot": str(plot_path),
}

# =============================================================================
# 7. GIF（前 3 集）+ live demo
# =============================================================================

if eval_cfg.get("render", True):
    import imageio

    vid_n   = min(3, n_episodes)
    vid_env = make_env("rgb_array")
    vid_path = run_dir / "demo.mp4"
    writer  = imageio.get_writer(str(vid_path), fps=30)
    log(f"\n开始录制视频：{vid_n} 集 → {vid_path}")
    for ep in range(vid_n):
        s, _ = vid_env.reset()
        total = 0
        for _ in range(max_steps):
            writer.append_data(vid_env.render())   # 流式写入，不攒内存
            a_env = actor.greedy(torch.FloatTensor(s))
            s, r, terminated, truncated, _ = vid_env.step(a_env)
            total += r
            if terminated or truncated:
                break
        log(f"  Video episode {ep + 1}: reward = {total:.1f}")
    writer.close()
    vid_env.close()
    log(f"  视频已保存：{vid_path}")
    eval_summary["video_path"] = str(vid_path)

    if eval_cfg.get("render_live", False):
        log(f"\n演示模式（{vid_n} 集）... 按 Ctrl+C 可提前退出")
        live_env = make_env("human")
        for ep in range(vid_n):
            s, _ = live_env.reset()
            total = 0
            for _ in range(max_steps):
                a_env = actor.greedy(torch.FloatTensor(s))
                s, r, terminated, truncated, _ = live_env.step(a_env)
                total += r
                if terminated or truncated:
                    break
            log(f"  Live episode {ep + 1}: reward = {total:.1f}")
        live_env.close()

summary_path = run_dir / "eval_summary.json"
write_json(summary_path, eval_summary)
log(f"评估摘要已保存：{summary_path}")
