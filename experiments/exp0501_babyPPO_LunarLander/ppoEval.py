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
            render_mode     = render_mode,
            continuous      = eval_cfg.get("continuous", True),
            entry_speed     = eval_cfg.get("entry_speed",     5.0),
            entry_angle_deg = eval_cfg.get("entry_angle_deg", 45.0),
            random_side     = eval_cfg.get("random_side",     True),
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

rewards     = []
successes   = []
ep_lengths  = []

env = make_env()
for ep in range(n_episodes):
    s, _ = env.reset()
    total, length, success = 0.0, 0, False
    for _ in range(max_steps):
        a_env = actor.greedy(torch.FloatTensor(s))
        s, r, terminated, truncated, _ = env.step(a_env)
        total += r
        length += 1
        if terminated:
            success = r > 0   # terminal +100 = 成功，-100 = 坠毁/越界
            break
        if truncated:
            break
    rewards.append(total)
    successes.append(success)
    ep_lengths.append(length)

env.close()

success_rate = np.mean(successes) * 100
print(f"评估结果（{n_episodes} 集，贪婪）：")
print(f"  成功率：{success_rate:.1f}%")
print(f"  平均奖励：{np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"  最高奖励：{max(rewards):.2f}  最低：{min(rewards):.2f}")
print(f"  平均集长：{np.mean(ep_lengths):.1f} 步")

# =============================================================================
# 6. 评估图（2 列：每集奖励柱状图 | 奖励分布直方图）
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# 左：每集奖励柱状图，绿=成功，红=失败
colors = ["tab:green" if s else "tab:red" for s in successes]
ax1.bar(range(n_episodes), rewards, color=colors, alpha=0.8, width=0.8)
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.set_title(f"Per-Episode Reward  (map_scale={eval_cfg.get('map_scale',1.0)})\n"
              f"green=success, red=failure  |  success_rate={success_rate:.1f}%")

# 右：奖励分布直方图
ax2.hist(rewards, bins=20, color="tab:blue", alpha=0.7, edgecolor="white")
ax2.axvline(np.mean(rewards), color="red", linestyle="--",
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
    gif_n = min(3, n_episodes)
    gif_env = make_env("rgb_array")
    frames  = []
    for ep in range(gif_n):
        s, _ = gif_env.reset()
        total = 0
        for _ in range(max_steps):
            frames.append(gif_env.render())
            a_env = actor.greedy(torch.FloatTensor(s))
            s, r, terminated, truncated, _ = gif_env.step(a_env)
            total += r
            if terminated or truncated:
                break
        print(f"  GIF episode {ep + 1}: reward = {total:.1f}")
    gif_env.close()
    gif_path = run_dir / "demo.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"  GIF 已保存：{gif_path}")

    print(f"\n演示模式（{gif_n} 集）... 按 Ctrl+C 可提前退出")
    live_env = make_env("human")
    for ep in range(gif_n):
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
