# exp0422_babyRL — 最简 DQN 实现

一个 Python 文件 + 一个 YAML 配置，理解 DQN 强化学习。

## 文件结构

```
exp0422_babyRL/
├── dqn_minimal.py    # 主代码（~180 行）
├── config.yaml       # 配置参数
└── idea.md           # 本文档
```

## 快速开始

```bash
cd experiments/exp0422_babyRL
python3 dqn_minimal.py
```

## 配置说明

编辑 `config.yaml` 调整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `env_name` | CartPole-v1 | Gymnasium 环境 |
| `max_steps` | 500 | 每集最大步数 |
| `hidden_layers` | [] | Q 网络隐藏层，[] = 线性，[16] = 单层 |
| `gamma` | 0.99 | 折扣因子 |
| `lr` | 0.01 | 学习率 |
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_end` | 0.01 | 最小探索率 |
| `epsilon_decay` | 0.995 | 每集衰减 |
| `buffer_size` | 10000 | 经验回放容量 |
| `batch_size` | 32 | 批次大小 |
| `target_update_freq` | 100 | 每 N 集更新 target |
| `num_episodes` | 500 | 训练集数 |
| `print_every` | 50 | 打印间隔 |

## DQN 核心思想

```
1. 用神经网络学习 Q 函数：Q(s, a) ≈ 预期累积奖励
2. 经验回放：存储 (s, a, r, s')，打破样本相关性
3. Target Network：固定目标值，防止训练发散
4. Epsilon-Greedy：平衡探索与利用
```

## 代码结构

```python
1. 加载配置
2. 创建环境
3. 定义 Q 网络
4. 经验回放（列表）
5. 训练循环:
   - epsilon-greedy 选动作
   - 执行动作，观察 (r, s')
   - 存储经验
   - 采样 batch，计算 TD loss
   - 反向传播
   - 更新 target network
```

## 技巧

### 简化网络

```yaml
# 单层线性（最简单）
hidden_layers: []

# 加一个隐藏层
hidden_layers: [16]

# 两个隐藏层
hidden_layers: [32, 16]
```

### 调整探索率

```yaml
# 更多探索（适合复杂环境）
epsilon_start: 1.0
epsilon_end: 0.1
epsilon_decay: 0.999

# 更少探索（适合简单环境）
epsilon_start: 0.5
epsilon_end: 0.01
epsilon_decay: 0.99
```

### 调整学习率

```yaml
# 更稳定（但更慢）
lr: 0.001

# 更快（但可能不稳定）
lr: 0.01
```

## 预期结果

CartPole-v1 用默认配置（线性网络）：
- 500 集后：平均奖励 ~20-50
- 最佳情况：~100+

加一个隐藏层 `[16]`：
- 500 集后：平均奖励 ~50-100
- 最佳情况：~200+（接近解决）

## 下一步

- 尝试不同隐藏层结构
- 调整超参数看效果
- 换其他环境（Acrobot-v1, MountainCar-v0）
