#!/usr/bin/env python3
"""exp0501 — babyPPO 训练脚本。

读取 train_config.yaml，训练后保存：
  - models/latest.pt       checkpoint
  - runs/<ts>/training.png reward + loss + entropy 三列曲线

用法：
    python3 ppoTrain.py
"""

import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Normal

# =============================================================================
# 1. 配置（必须在 lunar_ballistic import 之前设好 MAP_SCALE）
# =============================================================================

_here  = Path(__file__).parent
config = yaml.safe_load(open(_here / "train_config.yaml"))

os.environ["MAP_SCALE"] = str(config.get("map_scale", 1.0))

num_seeds = config.get("num_seeds", 1)

# =============================================================================
# 2. 环境工厂
# =============================================================================


def make_env(render_mode=None):
    if config.get("use_ballistic", False):
        from lunar_ballistic import BallisticLunarLander
        return BallisticLunarLander(
            render_mode     = render_mode,
            continuous      = config.get("continuous", True),
            entry_speed     = config.get("entry_speed",     5.0),
            entry_angle_deg = config.get("entry_angle_deg", 45.0),
            random_side     = config.get("random_side",     True),
        )
    kwargs = {"render_mode": render_mode} if render_mode else {}
    kwargs["continuous"] = config.get("continuous", True)
    return gym.make(config["env_name"], **kwargs)


_probe  = make_env()
obs_dim = _probe.observation_space.shape[0]
act_dim = _probe.action_space.shape[0]
_probe.close()

print(f"环境：{config['env_name']}  obs={obs_dim}  act={act_dim}  map_scale={config.get('map_scale',1.0)}")
print(f"Solved 标准：连续100集均值 ≥ {config['solved_threshold']}")

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


class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net     = _mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def _dist(self, s):
        mean = torch.tanh(self.net(s))
        std  = self.log_std.clamp(-3, 1).exp()
        return Normal(mean, std)

    def forward(self, s):
        dist = self._dist(s)
        a    = dist.sample()
        return a, dist.log_prob(a).sum(-1), dist.entropy().sum(-1)

    def log_prob_entropy(self, s, a):
        dist = self._dist(s)
        return dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden)

    def forward(self, s):
        return self.net(s).squeeze(-1)


# =============================================================================
# 4. GAE
# =============================================================================


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        mask  = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae   = delta + gamma * lam * mask * gae
        advantages[t] = gae
    return advantages, advantages + values


# =============================================================================
# 5. 单次训练
# =============================================================================


def run_one_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    hidden = config["hidden_layers"]
    env    = make_env()
    actor  = ContinuousActor(obs_dim, act_dim, hidden)
    critic = Critic(obs_dim, hidden)
    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config["lr"]
    )

    gamma        = config["gamma"]
    gae_lambda   = config["gae_lambda"]
    clip_eps     = config["clip_eps"]
    value_coef   = config["value_coef"]
    entropy_coef = config["entropy_coef"]
    n_steps      = config["n_steps"]
    n_epochs     = config["n_epochs"]
    batch_size   = config["batch_size"]
    num_updates  = config["num_updates"]

    all_ep_rewards    = []
    all_actor_losses  = []
    all_critic_losses = []
    all_entropies     = []

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
        all_ep_rewards    = ckpt.get("ep_rewards",    [])
        all_actor_losses  = ckpt.get("actor_losses",  [])
        all_critic_losses = ckpt.get("critic_losses", [])
        all_entropies     = ckpt.get("entropies",     [])
        print(f"  [seed {seed}] Resumed from update {start_update}")

    s, _ = env.reset()
    end_update = start_update + num_updates

    for update in range(start_update, end_update):
        # ── Rollout ───────────────────────────────────────────────────────────
        buf_s, buf_a, buf_logp, buf_r, buf_done, buf_v = [], [], [], [], [], []
        ep_reward = 0.0
        ep_rewards_this_update = []

        for _ in range(n_steps):
            s_t = torch.FloatTensor(s)
            with torch.no_grad():
                a, logp, _ = actor(s_t)
                v = critic(s_t).item()

            a_env = a.clamp(-1, 1).numpy()
            s2, r, terminated, truncated, _ = env.step(a_env)
            done = terminated or truncated

            buf_s.append(s);    buf_a.append(a.numpy())
            buf_logp.append(logp.item())
            buf_r.append(r);    buf_done.append(float(done)); buf_v.append(v)

            ep_reward += r
            s = s2
            if done:
                ep_rewards_this_update.append(ep_reward)
                ep_reward = 0.0
                s, _ = env.reset()

        with torch.no_grad():
            last_v = critic(torch.FloatTensor(s)).item()

        # ── GAE ───────────────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            np.array(buf_r,    dtype=np.float32),
            np.array(buf_v,    dtype=np.float32),
            np.array(buf_done, dtype=np.float32),
            last_v, gamma, gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_s    = torch.FloatTensor(np.array(buf_s))
        t_a    = torch.FloatTensor(np.array(buf_a))
        t_logp = torch.FloatTensor(buf_logp)
        t_adv  = torch.FloatTensor(advantages)
        t_ret  = torch.FloatTensor(returns)

        # ── PPO 更新 ──────────────────────────────────────────────────────────
        idx = np.arange(n_steps)
        ep_actor_losses, ep_critic_losses, ep_entropies = [], [], []

        for _ in range(n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n_steps, batch_size):
                mb = idx[start: start + batch_size]

                new_logp, entropy = actor.log_prob_entropy(t_s[mb], t_a[mb])
                new_v = critic(t_s[mb])

                ratio  = torch.exp(new_logp - t_logp[mb])
                adv_mb = t_adv[mb]
                surr1  = ratio * adv_mb
                surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                actor_loss   = -torch.min(surr1, surr2).mean()
                critic_loss  = nn.MSELoss()(new_v, t_ret[mb])
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
                ep_entropies.append(entropy.mean().item())

        all_ep_rewards.extend(ep_rewards_this_update)
        all_actor_losses.append(np.mean(ep_actor_losses))
        all_critic_losses.append(np.mean(ep_critic_losses))
        all_entropies.append(np.mean(ep_entropies))

        if (update + 1) % config["print_every"] == 0:
            recent = all_ep_rewards[-100:] if all_ep_rewards else [0]
            avg    = np.mean(recent)
            solved = "✅ SOLVED" if avg >= config["solved_threshold"] else ""
            print(f"  [seed {seed}] Update {update+1:4d}/{end_update} | "
                  f"Avg(100ep): {avg:7.2f} | "
                  f"Actor: {all_actor_losses[-1]:.4f} | "
                  f"Critic: {all_critic_losses[-1]:.4f} | "
                  f"Entropy: {all_entropies[-1]:.4f}  {solved}")

    env.close()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    models_dir.mkdir(exist_ok=True)
    torch.save({
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "update":        end_update - 1,
        "ep_rewards":    all_ep_rewards,
        "actor_losses":  all_actor_losses,
        "critic_losses": all_critic_losses,
        "entropies":     all_entropies,
        "hidden_layers": hidden,
    }, ckpt_path)
    print(f"  [seed {seed}] Checkpoint saved → {ckpt_path}")

    return all_ep_rewards, all_actor_losses, all_critic_losses, all_entropies, actor


# =============================================================================
# 6. 多 seed 训练
# =============================================================================

all_rewards_seeds  = []
all_actor_seeds    = []
all_critic_seeds   = []
all_entropy_seeds  = []
actor_last = None

for seed in range(num_seeds):
    print(f"\n── Seed {seed} ──────────────────────────────────")
    ep_rewards, actor_losses, critic_losses, entropies, actor_last = run_one_seed(seed)
    all_rewards_seeds.append(ep_rewards)
    all_actor_seeds.append(actor_losses)
    all_critic_seeds.append(critic_losses)
    all_entropy_seeds.append(entropies)

# =============================================================================
# 7. 汇总打印
# =============================================================================

print("\n" + "=" * 50)
print("训练完成!")
all_final = [np.mean(r[-100:]) if len(r) >= 100 else np.mean(r)
             for r in all_rewards_seeds]
print(f"平均最终奖励 (后100集): {np.mean(all_final):.2f}")
print(f"最佳单集奖励:           {np.mean([max(r) for r in all_rewards_seeds]):.2f}")

# =============================================================================
# 8. 训练图（3 列：reward | loss | entropy）
# =============================================================================

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

# 左：episode reward 滑动平均
for rewards in all_rewards_seeds:
    ep_idx   = np.arange(len(rewards))
    window   = min(50, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax1.plot(ep_idx[:len(smoothed)], smoothed, alpha=0.8, linewidth=1.2)
ax1.axhline(config["solved_threshold"], color="red", linestyle="--",
            linewidth=1, label=f"solved ({config['solved_threshold']})")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward (smoothed)")
ax1.set_title(f"Reward — map_scale={config.get('map_scale',1.0)}")
ax1.legend(fontsize=8)

# 中：actor + critic loss 双 y 轴
updates = np.arange(1, len(all_actor_seeds[0]) + 1)
ax2.plot(updates, all_actor_seeds[0],  color="tab:blue",   linewidth=1.2, label="Actor loss")
ax2b = ax2.twinx()
ax2b.plot(updates, all_critic_seeds[0], color="tab:orange", linewidth=1.2, label="Critic loss")
ax2.set_xlabel("Update")
ax2.set_ylabel("Actor loss",  color="tab:blue")
ax2b.set_ylabel("Critic loss", color="tab:orange")
ax2.set_title("PPO Losses")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# 右：entropy
ax3.plot(updates, all_entropy_seeds[0], color="tab:green", linewidth=1.2)
ax3.set_xlabel("Update")
ax3.set_ylabel("Entropy")
ax3.set_title("Policy Entropy")

plt.tight_layout()
plot_path = run_dir / "training.png"
plt.savefig(plot_path, dpi=150)
print(f"\n训练图已保存：{plot_path}")
