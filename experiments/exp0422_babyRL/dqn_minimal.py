#!/usr/bin/env python3
"""BabyRL: 最简 DQN 实现 — 一个脚本理解强化学习。

用法:
    python3 dqn_minimal.py

配置:
    修改 config.yaml 调整超参数
"""

import yaml
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# 1. 加载配置
# =============================================================================

config = yaml.safe_load(open("config.yaml"))

# =============================================================================
# 2. 创建环境
# =============================================================================

env = gym.make(config["env_name"])
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

print(f"环境：{config['env_name']}")
print(f"观测维度：{obs_dim}, 动作数：{act_dim}")

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


q_net = QNetwork(obs_dim, act_dim, config["hidden_layers"])
target_net = QNetwork(obs_dim, act_dim, config["hidden_layers"])
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=config["lr"])

# 打印网络结构
print(f"Q 网络：{q_net.net}")
n_params = sum(p.numel() for p in q_net.parameters())
print(f"参数量：{n_params}")

# =============================================================================
# 4. 经验回放（简单列表）
# =============================================================================

buffer = []


def store(s, a, r, s_, done):
    """存储一条经验."""
    buffer.append((s, a, r, s_, done))
    if len(buffer) > config["buffer_size"]:
        buffer.pop(0)


def sample(batch_size):
    """随机采样一个 batch."""
    return random.sample(buffer, batch_size)

# =============================================================================
# 5. 训练循环
# =============================================================================


epsilon = config["epsilon_start"]
gamma = config["gamma"]
rewards_history = []

print(f"\n开始训练 {config['num_episodes']} 集...")
print("-" * 50)

for episode in range(config["num_episodes"]):
    s, _ = env.reset()
    total_reward = 0

    for step in range(config["max_steps"]):
        # Epsilon-greedy 选择动作
        if random.random() < epsilon:
            a = env.action_space.sample()  # 探索
        else:
            with torch.no_grad():
                q = q_net(torch.FloatTensor(s))
                a = q.argmax().item()  # 利用

        # 执行动作
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # 存储经验
        store(s, a, r, s_, done)

        s = s_
        total_reward += r

        # 训练 Q 网络
        if len(buffer) >= config["batch_size"]:
            batch = sample(config["batch_size"])
            ss, aa, rr, ss_, dones = zip(*batch)

            ss_t = torch.FloatTensor(np.array(ss))
            aa_t = torch.LongTensor(aa)
            rr_t = torch.FloatTensor(rr)
            ss__t = torch.FloatTensor(np.array(ss_))

            # Q(s, a) - 当前网络预测
            q_sa = q_net(ss_t).gather(1, aa_t.unsqueeze(1)).squeeze(1)

            # Target: r + γ * max Q(s', a') - 用 target network 计算
            with torch.no_grad():
                q_next = target_net(ss__t).max(dim=1)[0]
            target = rr_t + gamma * q_next * (1 - torch.FloatTensor(dones))

            # MSE Loss + 反向传播
            loss = nn.MSELoss()(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # 每 N 集更新 target network
    if episode % config["target_update_freq"] == 0:
        target_net.load_state_dict(q_net.state_dict())

    # 衰减 epsilon
    epsilon = max(epsilon * config["epsilon_decay"], config["epsilon_end"])

    # 记录历史
    rewards_history.append(total_reward)

    # 打印进度
    if episode % config["print_every"] == 0:
        avg_100 = sum(rewards_history[-100:]) / min(100, len(rewards_history))
        print(f"Episode {episode:4d} | Reward: {total_reward:5.1f} | "
              f"Avg(100): {avg_100:6.2f} | Epsilon: {epsilon:.3f}")

# =============================================================================
# 6. 训练完成
# =============================================================================

print("-" * 50)
print("训练完成!")
print(f"最终奖励：{rewards_history[-1]:.1f}")
print(f"最佳奖励：{max(rewards_history):.1f}")
print(f"平均奖励 (后 100 集): {sum(rewards_history[-100:])/min(100, len(rewards_history)):.2f}")

# =============================================================================
# 7. 测试（贪婪模式）
# =============================================================================

print("\n测试（贪婪模式，10 集）...")
test_rewards = []

for _ in range(10):
    s, _ = env.reset()
    total = 0
    for _ in range(config["max_steps"]):
        with torch.no_grad():
            q = q_net(torch.FloatTensor(s))
            a = q.argmax().item()
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        s = s_
        total += r
        if done:
            break
    test_rewards.append(total)

print(f"测试奖励：{sum(test_rewards)/len(test_rewards):.2f} +/- {torch.std(torch.FloatTensor(test_rewards)):.2f}")
