# exp0430 — babyPPO CartPole

## 实验目标

在自定义 CartPole 环境上实现 PPO，学习 PPO 算法机制，并与 exp0422 DQN 对比。

## 算法

**PPO（Proximal Policy Optimization）**，两张独立网络：

| 网络 | 结构 | 输出 |
|------|------|------|
| Actor | 4 → 64 → 64 → 1 | μ(s)，力的均值 |
| Critic | 4 → 64 → 64 → 1 | V(s)，状态价值 |

- σ（探索幅度）：全局可学习参数，与状态无关（state-independent）
- 优势估计：GAE（Generalized Advantage Estimation），λ=0.95
- 更新：clipped surrogate loss，clip_eps=0.2，每次 rollout 训练 10 epoch

## 动作空间

由 `config.yaml` 的 `continuous_action` 控制：

| 模式 | 动作 | Actor 输出 |
|------|------|-----------|
| `false` | 21 离散整数 F∈{-10,...,10}N | Categorical(logits[21]) |
| `true` | 连续 F∈[-10,10]N | Normal(μ, σ) |

## 奖励函数

在 CartPole 原始奖励（每步 +1）基础上加 6 项惩罚：

```
r = 1 - w₁·x² - w₂·ẋ² - w₃·θ² - w₄·θ̇² - w₅·ẍ² - w₆·θ̈²
```

物理引擎由 `cartpole_custom.py` 从头实现（Lagrangian 动力学，Euler 积分）。

## 结果

| 模式 | 收敛 episode | 测试奖励 |
|------|------------|---------|
| 离散 PPO (21 actions) | ~1000 ep | 496.59 ± 0.49 |
| 连续 PPO | ~600 ep | 495.95 ± 3.49 |

对比 exp0422 DQN（~3500 ep 收敛）：**PPO 收敛更快**，但稳定性相近。

## 局限性

1. **σ 与状态无关**：所有状态共用同一探索幅度，无法在"杆快倒"时收紧、在"很稳"时放宽。state-dependent σ（网络直接输出 [μ, log σ]）是自然的下一步升级。

2. **CartPole 太简单**：DQN 和 PPO 都能轻松 solved，两者的本质差异（样本效率、连续动作、稳定性）在这个任务上无法体现。

3. **训练不稳定**：reward 曲线有 catastrophic forgetting，学好了又崩是常态，未能持续稳定在 475 以上。

4. **连续动作偏人为**：CartPole 物理本身用固定力推动，连续化虽然合理，但任务不需要精细的力控制，连续动作优势无法体现。

5. **单 seed**：所有结果基于单次训练，缺乏统计验证。换一个 seed 可能差别很大。

## 下一步

→ **exp0501 LunarLander**：更复杂的环境（连续奖励、多维动作、着陆精度要求），才能真正体现 PPO 相对 DQN 的优势，以及 state-dependent σ 的价值。
