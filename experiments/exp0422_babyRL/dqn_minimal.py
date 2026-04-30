#!/usr/bin/env python3
"""BabyRL: 最简 DQN 实现 — 一个脚本理解强化学习。

用法:
    python3 dqn_minimal.py

配置:
    修改 config.yaml 调整超参数
"""

import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from cartpole_custom import CustomCartPole
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


# =============================================================================
# 1. 加载配置
# =============================================================================

_here = Path(__file__).parent
config = yaml.safe_load(open(_here / "config.yaml"))
num_seeds = config.get("num_seeds", 3)

# =============================================================================
# 2. 创建环境（只用来读维度）
# =============================================================================


def make_env(render_mode=None):
    if config.get("use_custom_env", False):
        return CustomCartPole(
            render_mode=render_mode,
            pos_weight    = config.get("pos_weight",    0.1),
            vel_weight    = config.get("vel_weight",    0.0),
            angle_weight  = config.get("angle_weight",  0.0),
            angvel_weight = config.get("angvel_weight", 0.0),
        )
    kwargs = {"render_mode": render_mode} if render_mode else {}
    return gym.make(config["env_name"], **kwargs)


env = make_env()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

print(f"环境：{config['env_name']}")
print(f"观测维度：{obs_dim}, 动作数：{act_dim}")

# 输出目录
run_dir = _here / f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"输出目录：{run_dir}\n")

# =============================================================================
# 3. 定义 Q 网络
# =============================================================================


class QNetwork(nn.Module):
    """简单的 MLP Q 网络."""

    def __init__(self, obs_dim, act_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4. 单次训练（一个 seed）
# =============================================================================


def run_one_seed(seed: int):
    """跑一次完整训练，返回 (rewards_history, loss_history, q_net)."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env_local = make_env()

    q_net = QNetwork(obs_dim, act_dim, config["hidden_layers"])
    target_net = QNetwork(obs_dim, act_dim, config["hidden_layers"])
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=config["lr"])

    buffer = []

    def store(s, a, r, s_, done):
        buffer.append((s, a, r, s_, done))
        if len(buffer) > config["buffer_size"]:
            buffer.pop(0)

    def sample_batch(batch_size):
        return random.sample(buffer, batch_size)

    epsilon = config["epsilon_start"]
    gamma = config["gamma"]
    rewards_history = []
    loss_history = []

    # Resume from checkpoint if requested
    models_dir = _here / "models"
    ckpt_path  = models_dir / "latest.pt"
    start_episode = 0
    if config.get("resume", False) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        q_net.load_state_dict(ckpt["q_net"])
        target_net.load_state_dict(ckpt["target_net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epsilon         = ckpt["epsilon"]
        start_episode   = ckpt["episode"] + 1
        rewards_history = ckpt["rewards_history"]
        loss_history    = ckpt["loss_history"]
        print(f"  [seed {seed}] Resumed from episode {start_episode}")

    end_episode = start_episode + config["num_episodes"]
    for episode in range(start_episode, end_episode):
        s, _ = env_local.reset()
        total_reward = 0

        for _ in range(config["max_steps"]):
            if random.random() < epsilon:
                a = env_local.action_space.sample()
            else:
                with torch.no_grad():
                    a = q_net(torch.FloatTensor(s)).argmax().item()

            s_, r, terminated, truncated, _ = env_local.step(a)
            done = terminated or truncated
            store(s, a, r, s_, done)
            s = s_
            total_reward += r

            if len(buffer) >= config["batch_size"]:
                batch = sample_batch(config["batch_size"])
                ss, aa, rr, ss_, dones = zip(*batch)

                ss_t  = torch.FloatTensor(np.array(ss))
                aa_t  = torch.LongTensor(aa)
                rr_t  = torch.FloatTensor(rr)
                ss__t = torch.FloatTensor(np.array(ss_))

                q_sa = q_net(ss_t).gather(1, aa_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(ss__t).max(dim=1)[0]
                target = rr_t + gamma * q_next * (1 - torch.FloatTensor(dones))

                loss = nn.MSELoss()(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

            if done:
                break

        if episode % config["target_update_freq"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon * config["epsilon_decay"], config["epsilon_end"])
        rewards_history.append(total_reward)

        if episode % config["print_every"] == 0:
            avg_100 = sum(rewards_history[-100:]) / min(100, len(rewards_history))
            print(f"  [seed {seed}] Episode {episode:4d} | Reward: {total_reward:5.1f} | "
                  f"Avg(100): {avg_100:6.2f} | Epsilon: {epsilon:.3f}")

    env_local.close()

    # Save checkpoint
    models_dir.mkdir(exist_ok=True)
    torch.save({
        "q_net":           q_net.state_dict(),
        "target_net":      target_net.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "epsilon":         epsilon,
        "episode":         end_episode - 1,
        "rewards_history": rewards_history,
        "loss_history":    loss_history,
    }, ckpt_path)
    print(f"  [seed {seed}] Checkpoint saved → {ckpt_path}")

    return rewards_history, loss_history, q_net


# =============================================================================
# 5. 多 seed 训练
# =============================================================================

all_rewards = []
all_losses  = []

q_net_last = None
for seed in range(num_seeds):
    print(f"\n── Seed {seed} ──────────────────────────────────")
    rewards, losses, q_net_last = run_one_seed(seed)
    all_rewards.append(rewards)
    all_losses.append(losses)

# =============================================================================
# 6. 汇总打印
# =============================================================================

print("\n" + "=" * 50)
print("训练完成!")
rewards_arr = np.array(all_rewards)  # (num_seeds, num_episodes)
final_avg = rewards_arr[:, -100:].mean()
best_avg  = rewards_arr.max(axis=1).mean()
print(f"平均最终奖励 (后100集, across seeds): {final_avg:.2f}")
print(f"平均最佳单集奖励 (across seeds):      {best_avg:.2f}")

# =============================================================================
# 7. 测试（贪婪模式，用最后一个 seed 的训练好的 q_net）
# =============================================================================

print("\n测试（贪婪模式，10 集）...")
test_rewards = []
test_env = make_env()
for _ in range(10):
    s, _ = test_env.reset()
    total = 0
    for _ in range(config["max_steps"]):
        with torch.no_grad():
            a = q_net_last(torch.FloatTensor(s)).argmax().item()
        s, r, terminated, truncated, _ = test_env.step(a)
        total += r
        if terminated or truncated:
            break
    test_rewards.append(total)
test_env.close()
print(f"测试奖励 (10集贪婪): {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")

# =============================================================================
# 8. 画图
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ── 左图：Reward 曲线 ──
episodes = np.arange(len(all_rewards[0]))
r_mean = rewards_arr.mean(axis=0)
r_std  = rewards_arr.std(axis=0)

for i, r in enumerate(all_rewards):
    ax1.plot(episodes, r, alpha=0.2, linewidth=0.8)

ax1.fill_between(episodes, r_mean - r_std, r_mean + r_std, alpha=0.3, label="mean ± std")
ax1.plot(episodes, r_mean, linewidth=1.5, label="mean")
ax1.axhline(475, color="red", linestyle="--", linewidth=1, label="solved (475)")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_title("Reward per Episode")
ax1.legend(fontsize=8)

# ── 右图：TD Loss 曲线 ──
min_len = min(len(l) for l in all_losses)
losses_arr = np.array([l[:min_len] for l in all_losses])
steps = np.arange(min_len)
l_mean = losses_arr.mean(axis=0)
l_std  = losses_arr.std(axis=0)

for l in all_losses:
    ax2.plot(np.arange(len(l)), l, alpha=0.2, linewidth=0.8)

ax2.fill_between(steps, l_mean - l_std, l_mean + l_std, alpha=0.3, label="mean ± std")
ax2.plot(steps, l_mean, linewidth=1.5, label="mean")
ax2.set_xlabel("Training step")
ax2.set_ylabel("TD Loss")
ax2.set_title("TD Loss")
ax2.legend(fontsize=8)

plt.tight_layout()
plot_path = run_dir / "training.png"
plt.savefig(plot_path, dpi=150)
print(f"\n图表已保存：{plot_path}")

# =============================================================================
# 9. 演示（render_mode="human" + 保存 GIF）
# =============================================================================


def run_demo(q_net, n_episodes: int = 3) -> None:
    import imageio

    # 录制 rgb_array 帧用于保存 GIF
    gif_env = make_env("rgb_array")
    frames = []
    for ep in range(n_episodes):
        s, _ = gif_env.reset()
        total = 0
        for _ in range(config["max_steps"]):
            frames.append(gif_env.render())
            with torch.no_grad():
                a = q_net(torch.FloatTensor(s)).argmax().item()
            s, r, terminated, truncated, _ = gif_env.step(a)
            total += r
            if terminated or truncated:
                break
        print(f"  Demo episode {ep + 1}: reward = {total:.0f}")
    gif_env.close()

    gif_path = run_dir / "demo.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"  GIF 已保存：{gif_path}")

    # 弹出实时窗口
    print(f"\n演示模式（{n_episodes} 集）... 按 Ctrl+C 可提前退出")
    live_env = make_env("human")
    for ep in range(n_episodes):
        s, _ = live_env.reset()
        total = 0
        for _ in range(config["max_steps"]):
            with torch.no_grad():
                a = q_net(torch.FloatTensor(s)).argmax().item()
            s, r, terminated, truncated, _ = live_env.step(a)
            total += r
            if terminated or truncated:
                break
        print(f"  Live episode {ep + 1}: reward = {total:.0f}")
    live_env.close()


if config.get("render", True):
    run_demo(q_net_last)
