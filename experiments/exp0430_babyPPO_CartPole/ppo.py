#!/usr/bin/env python3
"""exp0501 — PPO on CustomCartPole.

同样的倒立摆任务，同样的 shaped reward，但用 PPO 代替 DQN。

PPO 与 DQN 的核心区别：
  - 策略显式：Actor 输出动作概率，而非 Q 值
  - On-policy：rollout 用完即丢，无 replay buffer
  - 探索靠熵奖励 + 随机采样，而非 epsilon-greedy
  - 两张网络：Actor（决策）+ Critic（评价），而非 Q + target Q

用法：
    python3 ppo.py
"""

import random
from datetime import datetime
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Categorical, Normal

from cartpole_custom import CustomCartPole

# =============================================================================
# 1. 配置
# =============================================================================

_here = Path(__file__).parent
config = yaml.safe_load(open(_here / "config.yaml"))
num_seeds  = config.get("num_seeds", 1)
CONTINUOUS = config.get("continuous_action", False)

# =============================================================================
# 2. 环境工厂
# =============================================================================


def make_env(render_mode=None):
    if config.get("use_custom_env", False):
        return CustomCartPole(
            render_mode=render_mode,
            pos_weight    = config.get("pos_weight",    0.0),
            vel_weight    = config.get("vel_weight",    0.0),
            angle_weight  = config.get("angle_weight",  0.0),
            angvel_weight = config.get("angvel_weight", 0.0),
            acc_weight    = config.get("acc_weight",    0.0),
            angacc_weight = config.get("angacc_weight", 0.0),
            continuous    = CONTINUOUS,
        )
    kwargs = {"render_mode": render_mode} if render_mode else {}
    return gym.make(config["env_name"], **kwargs)


_probe = make_env()
obs_dim = _probe.observation_space.shape[0]
act_dim = _probe.action_space.shape[0] if CONTINUOUS else _probe.action_space.n
_probe.close()

mode_str = "continuous F∈[-10,10]" if CONTINUOUS else f"discrete {act_dim} actions"
print(f"环境：{config['env_name']}  obs={obs_dim}  action={mode_str}")

run_dir = _here / f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"输出目录：{run_dir}\n")

# =============================================================================
# 3. 网络定义
# =============================================================================


def _mlp(in_dim, out_dim, hidden):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """输出动作概率分布。"""
    def __init__(self):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, config["hidden_layers"])

    def forward(self, s):
        """返回 (action, log_prob, entropy)。"""
        dist = Categorical(logits=self.net(s))
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def log_prob_entropy(self, s, a):
        """给定 (s, a) 计算 log π(a|s) 和熵，用于 PPO 更新。"""
        dist = Categorical(logits=self.net(s))
        return dist.log_prob(a), dist.entropy()


class ContinuousActor(nn.Module):
    """Gaussian policy: outputs μ(s), learned log σ (state-independent)."""
    def __init__(self):
        super().__init__()
        self.net     = _mlp(obs_dim, act_dim, config["hidden_layers"])
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _dist(self, s):
        mu  = torch.tanh(self.net(s)) * 10.0           # squash to [-10, 10]
        std = self.log_std.exp().clamp(0.01, 5.0)
        return Normal(mu, std)

    def forward(self, s):
        dist   = self._dist(s)
        action = dist.sample().clamp(-10.0, 10.0)
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def log_prob_entropy(self, s, a):
        dist = self._dist(s)
        return dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


class Critic(nn.Module):
    """估计状态价值 V(s)。"""
    def __init__(self):
        super().__init__()
        self.net = _mlp(obs_dim, 1, config["hidden_layers"])

    def forward(self, s):
        return self.net(s).squeeze(-1)


# =============================================================================
# 4. GAE 计算
# =============================================================================


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Generalized Advantage Estimation.

    advantage[t] 衡量"动作 a_t 比平均水平好多少"：
        delta_t = r_t + γ·V(s_{t+1}) - V(s_t)   ← 单步 TD error
        A_t = delta_t + γλ·A_{t+1}               ← 递推平滑
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# =============================================================================
# 5. 单次训练（一个 seed）
# =============================================================================


def run_one_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = make_env()
    actor  = ContinuousActor() if CONTINUOUS else Actor()
    critic = Critic()
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config["lr"]
    )

    gamma      = config["gamma"]
    gae_lambda = config["gae_lambda"]
    clip_eps   = config["clip_eps"]
    value_coef = config["value_coef"]
    entropy_coef = config["entropy_coef"]
    n_steps    = config["n_steps"]
    n_epochs   = config["n_epochs"]
    batch_size = config["batch_size"]
    num_updates = config["num_updates"]

    all_ep_rewards = []   # 每个完整 episode 的总 reward
    all_actor_losses  = []
    all_critic_losses = []

    # ── Resume ───────────────────────────────────────────────────────────────
    models_dir   = _here / "models"
    ckpt_path    = models_dir / "latest.pt"
    start_update = 0
    if config.get("resume", False) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update      = ckpt["update"] + 1
        all_ep_rewards    = ckpt["ep_rewards"]
        all_actor_losses  = ckpt["actor_losses"]
        all_critic_losses = ckpt["critic_losses"]
        print(f"  [seed {seed}] Resumed from update {start_update}")

    s, _ = env.reset()

    end_update = start_update + num_updates
    for update in range(start_update, end_update):
        # ── Step 1: 收集 rollout ──────────────────────────────────────────────
        buf_s, buf_a, buf_logp, buf_r, buf_done, buf_v = [], [], [], [], [], []
        ep_reward = 0.0
        ep_rewards_this_update = []

        for _ in range(n_steps):
            s_t = torch.FloatTensor(s)
            with torch.no_grad():
                a, logp, _ = actor(s_t)
                v = critic(s_t).item()

            a_env = a.numpy() if CONTINUOUS else a.item()
            s2, r, terminated, truncated, _ = env.step(a_env)
            done = terminated or truncated

            buf_s.append(s)
            a_store = a.numpy() if CONTINUOUS else a.item()
            buf_a.append(a_store)
            buf_logp.append(logp.item())
            buf_r.append(r)
            buf_done.append(float(done))
            buf_v.append(v)

            ep_reward += r
            s = s2

            if done:
                ep_rewards_this_update.append(ep_reward)
                ep_reward = 0.0
                s, _ = env.reset()

        # 最后一步的 bootstrap value
        with torch.no_grad():
            last_v = critic(torch.FloatTensor(s)).item()

        # ── Step 2: GAE ───────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            np.array(buf_r, dtype=np.float32),
            np.array(buf_v, dtype=np.float32),
            np.array(buf_done, dtype=np.float32),
            last_v, gamma, gae_lambda
        )
        # 归一化 advantage，稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转成 tensor
        t_s    = torch.FloatTensor(np.array(buf_s))
        t_a    = torch.FloatTensor(np.array(buf_a)) if CONTINUOUS \
                 else torch.LongTensor(buf_a)
        t_logp = torch.FloatTensor(buf_logp)
        t_adv  = torch.FloatTensor(advantages)
        t_ret  = torch.FloatTensor(returns)

        # ── Step 3: PPO 更新（多 epoch + minibatch）───────────────────────────
        idx = np.arange(n_steps)
        ep_actor_losses, ep_critic_losses = [], []

        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n_steps, batch_size):
                mb = idx[start: start + batch_size]

                new_logp, entropy = actor.log_prob_entropy(t_s[mb], t_a[mb])
                new_v = critic(t_s[mb])

                # Clipped surrogate objective
                ratio = torch.exp(new_logp - t_logp[mb])
                adv_mb = t_adv[mb]
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_v, t_ret[mb])
                entropy_loss = -entropy_coef * entropy.mean()

                loss = actor_loss + value_coef * critic_loss + entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5
                )
                optimizer.step()

                ep_actor_losses.append(actor_loss.item())
                ep_critic_losses.append(critic_loss.item())

        all_ep_rewards.extend(ep_rewards_this_update)
        all_actor_losses.append(np.mean(ep_actor_losses))
        all_critic_losses.append(np.mean(ep_critic_losses))

        if (update + 1) % config["print_every"] == 0:
            recent = all_ep_rewards[-20:] if all_ep_rewards else [0]
            avg = np.mean(recent)
            print(f"  [seed {seed}] Update {update+1:4d}/{num_updates} | "
                  f"Avg reward (last 20 ep): {avg:7.2f} | "
                  f"Actor loss: {all_actor_losses[-1]:.4f} | "
                  f"Critic loss: {all_critic_losses[-1]:.4f}")

    env.close()

    # ── Save checkpoint ───────────────────────────────────────────────────────
    models_dir.mkdir(exist_ok=True)
    torch.save({
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "update":        end_update - 1,
        "ep_rewards":    all_ep_rewards,
        "actor_losses":  all_actor_losses,
        "critic_losses": all_critic_losses,
        "continuous":    CONTINUOUS,
    }, ckpt_path)
    print(f"  [seed {seed}] Checkpoint saved → {ckpt_path}")

    return all_ep_rewards, all_actor_losses, all_critic_losses, actor


# =============================================================================
# 6. 多 seed 训练
# =============================================================================

all_rewards_seeds = []
all_actor_seeds   = []
all_critic_seeds  = []
actor_last = None

for seed in range(num_seeds):
    print(f"\n── Seed {seed} ──────────────────────────────────")
    ep_rewards, actor_losses, critic_losses, actor_last = run_one_seed(seed)
    all_rewards_seeds.append(ep_rewards)
    all_actor_seeds.append(actor_losses)
    all_critic_seeds.append(critic_losses)

# =============================================================================
# 7. 汇总打印
# =============================================================================

print("\n" + "=" * 50)
print("训练完成!")
all_final = [np.mean(r[-100:]) if len(r) >= 100 else np.mean(r)
             for r in all_rewards_seeds]
print(f"平均最终奖励 (后100集, across seeds): {np.mean(all_final):.2f}")
print(f"平均最佳单集奖励 (across seeds):      {np.mean([max(r) for r in all_rewards_seeds]):.2f}")

# 测试（贪婪：取 argmax logits）
print("\n测试（贪婪模式，10 集）...")
test_env = make_env()
test_rewards = []
for _ in range(10):
    s, _ = test_env.reset()
    total = 0
    for _ in range(config["max_steps"]):
        with torch.no_grad():
            s_t = torch.FloatTensor(s)
            if CONTINUOUS:
                a = (torch.tanh(actor_last.net(s_t)) * 10.0).numpy()
            else:
                a = actor_last.net(s_t).argmax().item()
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

# 左图：episode reward（滑动平均）
for rewards in all_rewards_seeds:
    ep_idx = np.arange(len(rewards))
    window = min(50, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    ax1.plot(ep_idx[:len(smoothed)], smoothed, alpha=0.8, linewidth=1.2)
ax1.axhline(475, color="red", linestyle="--", linewidth=1, label="solved (475)")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward (smoothed)")
ax1.set_title("Reward per Episode")
ax1.legend(fontsize=8)

# 右图：actor loss + critic loss（双 y 轴）
updates = np.arange(1, len(all_actor_seeds[0]) + 1)
ax2.plot(updates, all_actor_seeds[0], color="tab:blue", linewidth=1.2, label="Actor loss")
ax2b = ax2.twinx()
ax2b.plot(updates, all_critic_seeds[0], color="tab:orange", linewidth=1.2, label="Critic loss")
ax2.set_xlabel("Update")
ax2.set_ylabel("Actor loss", color="tab:blue")
ax2b.set_ylabel("Critic loss", color="tab:orange")
ax2.set_title("PPO Losses")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
plot_path = run_dir / "training.png"
plt.savefig(plot_path, dpi=150)
print(f"\n图表已保存：{plot_path}")

# =============================================================================
# 9. Demo + GIF
# =============================================================================


def run_demo(actor, n_episodes=3):
    import imageio

    gif_env = make_env("rgb_array")
    frames = []
    for ep in range(n_episodes):
        s, _ = gif_env.reset()
        total = 0
        for _ in range(config["max_steps"]):
            frames.append(gif_env.render())
            with torch.no_grad():
                s_t = torch.FloatTensor(s)
                if CONTINUOUS:
                    a = (torch.tanh(actor_last.net(s_t)) * 10.0).numpy()
                else:
                    a = actor_last.net(s_t).argmax().item()
            s, r, terminated, truncated, _ = gif_env.step(a)
            total += r
            if terminated or truncated:
                break
        print(f"  Demo episode {ep + 1}: reward = {total:.0f}")
    gif_env.close()
    gif_path = run_dir / "demo.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"  GIF 已保存：{gif_path}")

    print(f"\n演示模式（{n_episodes} 集）... 按 Ctrl+C 可提前退出")
    live_env = make_env("human")
    for ep in range(n_episodes):
        s, _ = live_env.reset()
        total = 0
        for _ in range(config["max_steps"]):
            with torch.no_grad():
                s_t = torch.FloatTensor(s)
                if CONTINUOUS:
                    a = (torch.tanh(actor_last.net(s_t)) * 10.0).numpy()
                else:
                    a = actor_last.net(s_t).argmax().item()
            s, r, terminated, truncated, _ = live_env.step(a)
            total += r
            if terminated or truncated:
                break
        print(f"  Live episode {ep + 1}: reward = {total:.0f}")
    live_env.close()


if config.get("render", True):
    run_demo(actor_last)
