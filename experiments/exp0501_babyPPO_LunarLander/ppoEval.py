#!/usr/bin/env python3
"""exp0501 — babyPPO 评估脚本。

读取 eval_config.yaml，加载 checkpoint，在指定环境中贪婪跑 N 集，输出：
  - runs/<ts>_eval/eval.png   每集奖励柱状图 + 奖励分布直方图
  - runs/<ts>_eval/demo.gif   前 3 集录像
  - 终端：success_rate, mean ± std

用法：
    python3 ppoEval.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import yaml
from torch.distributions import Normal

# =============================================================================
# 1. 配置（必须在 lunar_ballistic import 之前设好 MAP_SCALE）
# =============================================================================

_here    = Path(__file__).parent
eval_cfg = yaml.safe_load(open(_here / "configEval.yaml"))

os.environ["MAP_SCALE"] = str(eval_cfg.get("map_scale", 1.0))

# 物理参数来源：train config 或 eval config
if eval_cfg.get("use_train_physics", False):
    _train_cfg = yaml.safe_load(open(_here / "configTrain.yaml"))
    physics_cfg = _train_cfg
    print("物理参数来源：configTrain.yaml")
else:
    physics_cfg = eval_cfg
    print("物理参数来源：configEval.yaml")

# =============================================================================
# 2. 加载 checkpoint
# =============================================================================

ckpt_path = _here / eval_cfg.get("checkpoint", "models/latest.pt")
if not ckpt_path.exists():
    sys.exit(f"[Error] Checkpoint not found: {ckpt_path}")

ckpt = torch.load(ckpt_path, weights_only=False)
hidden_layers = ckpt.get("hidden_layers", eval_cfg.get("hidden_layers", [64, 64]))

# =============================================================================
# 3. 环境工厂
# =============================================================================


def make_env(render_mode=None):
    if eval_cfg.get("use_ballistic", False):
        from lunar_ballistic import BallisticLunarLander
        return BallisticLunarLander(
            render_mode       = render_mode,
            continuous        = eval_cfg.get("continuous",        True),
            entry_speed       = physics_cfg.get("entry_speed",       5.0),
            entry_angle_deg   = physics_cfg.get("entry_angle_deg",   45.0),
            random_side       = physics_cfg.get("random_side",       True),
            angle_tilt_factor = physics_cfg.get("angle_tilt_factor", 0.3),
            init_angvel       = physics_cfg.get("init_angvel",       0.0),
            init_height_m     = physics_cfg.get("init_height_m",     None),
            init_x_offset_m   = physics_cfg.get("init_x_offset_m",  0.0),
        )
    kwargs = {"render_mode": render_mode} if render_mode else {}
    kwargs["continuous"] = eval_cfg.get("continuous", True)
    return gym.make(eval_cfg["env_name"], **kwargs)


_probe  = make_env()
obs_dim = _probe.observation_space.shape[0]
act_dim = _probe.action_space.shape[0]
_probe.close()

print(f"评估环境：{eval_cfg['env_name']}  map_scale={eval_cfg.get('map_scale',1.0)}")
print(f"Checkpoint：{ckpt_path}  (trained to update {ckpt.get('update','?')})")

run_dir = _here / f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"输出目录：{run_dir}\n")

# =============================================================================
# 4. 网络（与 ppoTrain.py 相同定义）
# =============================================================================


def _mlp(in_dim, out_dim, hidden):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net     = _mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def greedy(self, s):
        with torch.no_grad():
            return torch.tanh(self.net(s)).clamp(-1, 1).numpy()


actor = ContinuousActor(obs_dim, act_dim, hidden_layers)
actor.load_state_dict(ckpt["actor"])
actor.eval()

# =============================================================================
# 5. 贪婪评估
# =============================================================================

n_episodes = eval_cfg.get("num_eval_episodes", 50)
max_steps  = eval_cfg.get("max_steps", 1000)

rewards    = []
outcomes   = []   # "success" / "crash" / "timeout"
ep_lengths = []

env = make_env()
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

env.close()

n_success = outcomes.count("success")
n_crash   = outcomes.count("crash")
n_timeout = outcomes.count("timeout")
print(f"评估结果（{n_episodes} 集，贪婪）：")
print(f"  成功：{n_success}/{n_episodes} ({n_success/n_episodes*100:.1f}%)")
print(f"  坠毁：{n_crash}/{n_episodes} ({n_crash/n_episodes*100:.1f}%)")
print(f"  超时：{n_timeout}/{n_episodes} ({n_timeout/n_episodes*100:.1f}%)")
print(f"  平均奖励：{np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"  最高奖励：{max(rewards):.2f}  最低：{min(rewards):.2f}")
print(f"  平均集长：{np.mean(ep_lengths):.1f} 步")

# =============================================================================
# 6. 评估图（2 列：每集奖励柱状图 | 奖励分布直方图）
# =============================================================================

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
              f"entry_speed={eval_cfg.get('entry_speed',5.0)}  "
              f"angle={eval_cfg.get('entry_angle_deg',45)}°")
ax2.legend(fontsize=8)

plt.tight_layout()
plot_path = run_dir / "eval.png"
plt.savefig(plot_path, dpi=150)
print(f"\n评估图已保存：{plot_path}")

# =============================================================================
# 7. GIF（前 3 集）+ live demo
# =============================================================================

if eval_cfg.get("render", True):
    vid_n   = min(3, n_episodes)
    vid_env = make_env("rgb_array")
    vid_path = run_dir / "demo.mp4"
    writer  = imageio.get_writer(str(vid_path), fps=30)
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
        print(f"  Video episode {ep + 1}: reward = {total:.1f}")
    writer.close()
    vid_env.close()
    print(f"  视频已保存：{vid_path}")

    if eval_cfg.get("render_live", False):
        print(f"\n演示模式（{vid_n} 集）... 按 Ctrl+C 可提前退出")
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
            print(f"  Live episode {ep + 1}: reward = {total:.1f}")
        live_env.close()
