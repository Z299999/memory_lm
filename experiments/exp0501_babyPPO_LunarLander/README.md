# exp0501 — babyPPO LunarLander

## 实验目标

在 `LunarLander-v3`（离散，4个动作）上运行 PPO，验证 PPO 在比 CartPole 更复杂的任务上的优势。

## 为什么选 LunarLander

CartPole 太简单，DQN 和 PPO 都能轻松 solved，算法差异无法体现。LunarLander 具备：

- **稀疏+连续奖励**：着陆精度、速度、燃料消耗多项奖励叠加，奖励信号更复杂
- **更长的时序依赖**：需要多步协调（下降→减速→对准→着陆）
- **4 个离散动作**：不推、左引擎、主引擎、右引擎
- **标准 benchmark**：solved 标准为连续 100 集均值 ≥ 200

## 动作空间

```
0: 不推（无推力）
1: 左侧引擎
2: 主引擎（向下推力）
3: 右侧引擎
```

## 计划

- 算法：离散 PPO（复用 exp0430 的 Actor/Critic 结构，去掉 cartpole_custom.py）
- 使用 gymnasium 原生 `LunarLander-v3`，不自定义物理
- 奖励函数：使用 gymnasium 原生奖励（不做 shaping），观察 PPO 能否自己学会
- 对比项：是否需要奖励 shaping？state-dependent σ 是否有帮助？

## 状态空间（8维）

```
[x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
```

## 代码状态

⏳ 待实现
