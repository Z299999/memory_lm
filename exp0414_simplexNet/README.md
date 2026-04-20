# exp0414_simplexNet — Simplex Memory Networks

基于单纯形格点的几何前馈神经网络架构。

## 快速开始

### 1. 运行对比实验

```bash
cd exp0414_simplexNet
python3 run.py
```

输出：`runs/<日期>/<实验名>/comparison.png` — SMN vs MLP 四图对比。

### 2. 在自己的代码中使用

```python
import sys
sys.path.insert(0, 'src')

from smn_fitter import SMNModule, SMNFitter
import torch

# --- 方式 A: SMNModule（原始 nn.Module）---
# 如果你想完全控制训练过程
module = SMNModule(n=2, m=3, n_in=1, n_out=1, activation='relu')
x = torch.randn(32, 1)
y = module(x)  # 前向传播

# --- 方式 B: SMNFitter（高级 API）---
# 如果你想快速实验，使用内置训练
smn = SMNFitter(n=2, m=3, n_in=1, n_out=1)
smn.fit(epochs=300)  # 使用内置 sin_mix 数据训练
smn.plot(output_path="result.png")  # 两图：scatter + loss 曲线
```

---

## 目录

1. [SMNModule 完整 API 说明](#smnmodule-完整 api-说明)
2. [SMNFitter 高级 API](#smnfitter-高级训练包装器)
3. [强化学习示例](#强化学习示例)
4. [配置说明](#配置 paramsyaml)
5. [数学背景](#数学背景简述)

---

## SMNModule 完整 API 说明

`SMNModule` 是一个纯粹的 PyTorch `nn.Module`，没有任何训练逻辑依赖。适合：
- **强化学习**（自定义 loss、策略梯度）
- **嵌入其他项目**（作为子模块）
- **完全自定义训练循环**

### 初始化参数

```python
from smn_fitter import SMNModule

module = SMNModule(
    n=2,                    # 单纯形维度：2=三角形，3=四面体，...
    m=3,                    # 分辨率：每条边上的格点数
    n_in=1,                 # 输入维度
    n_out=1,                # 输出维度
    activation='relu',      # 激活函数：'relu', 'leaky_relu', 'gelu', 'tanh'
    x_bounds=None,          # 每通道输入范围：[(min, max), ...]
)
```

**参数详解：**

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `n` | int | 2 | 否 | 单纯形维度。`n=2` 是三角形，`n=3` 是四面体 |
| `m` | int | 3 | 否 | 分辨率。每条边有 `m` 个格点 |
| `n_in` | int | 1 | 否 | 输入维度（状态空间维度） |
| `n_out` | int | 1 | 否 | 输出维度（动作数或 Q 值数） |
| `activation` | str | 'relu' | 否 | 隐藏层激活函数 |
| `x_bounds` | list | None | 否 | 每通道输入范围，如 `[(-1,1), (-2,2)]`。None 时默认为 `[(-1,1)]*n_in` |

**注意：** `x_bounds` 用于自动将输入归一化到 `[-1, 1]`。如果你的环境观测值范围是 `[-4.8, 4.8]`，则应设置 `x_bounds=[(-4.8, 4.8)]`。

---

### 属性（Properties）

| 属性 | 返回类型 | 说明 |
|------|----------|------|
| `module.arch_str` | str | 人类可读的架构描述 |
| `module.param_count` | int | 可训练参数总数 |
| `module.graph` | SimplexMemoryGraph | 底层图结构（可访问 `edge_count`, `core_node_count` 等） |

**示例：**
```python
print(module.arch_str)
# 输出："SMN(n=2, m=3, 1→1), nodes=6, edges=11, params=18"

print(module.param_count)  # 输出：18
```

---

### 方法（Methods）

#### `forward(x: Tensor) -> Tensor`

**用途：** 前向传播。通常通过 `module(x)` 调用。

**参数：**
- `x`: 输入张量，形状 `(batch, n_in)`。当 `n_in=1` 时，也接受 `(batch,)`

**返回：**
- 输出张量，形状 `(batch, n_out)`

**示例：**
```python
# SISO (1 输入 1 输出)
x = torch.randn(32, 1)      # 32 个样本，1 维输入
y = module(x)               # 输出：(32, 1)

# MIMO (2 输入 2 输出)
module2 = SMNModule(n_in=2, n_out=2)
x = torch.randn(64, 2)      # 64 个样本，2 维输入
y = module2(x)              # 输出：(64, 2)

# 单样本推理
x = torch.randn(1, 4)       # 1 个样本，4 维输入（如 CartPole 观测）
y = module(x)               # 输出：(1, 2) → 2 个动作的 Q 值
```

**内部流程：**
1. 根据 `x_bounds` 将输入归一化到 `[-1, 1]`
2. 沿单纯形 DAG 逐层传播信号
3. 输出层用 `tanh` 激活，输出范围 `(-1, 1)`

---

#### `state_dict() -> dict`

**用途：** 获取模型权重。用于保存/加载检查点。

**示例：**
```python
# 保存
torch.save(module.state_dict(), 'smn_checkpoint.pth')

# 加载
module.load_state_dict(torch.load('smn_checkpoint.pth'))
```

---

#### `parameters() -> Iterator[Parameter]`

**用途：** 获取可训练参数。传给优化器。

**示例：**
```python
optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
```

---

### 继承自 `nn.Module` 的常用方法

`SMNModule` 继承自 `torch.nn.Module`，因此所有 `nn.Module` 的方法都可用：

| 方法 | 说明 |
|------|------|
| `module.train()` | 设为训练模式（启用 dropout 等，但 SMN 无 dropout） |
| `module.eval()` | 设为评估模式 |
| `module.to(device)` | 移动到 GPU/CPU |
| `module.zero_grad()` | 清零梯度 |
| `module.cuda()` | 移动到 GPU |

---

## SMNFitter 高级训练包装器

如果你不想写训练循环，`SMNFitter` 提供一站式解决方案：

```python
from smn_fitter import SMNFitter

smn = SMNFitter(n=2, m=3, n_in=1, n_out=1)
smn.fit(x_train, y_train, epochs=300)  # 训练
smn.predict(x_test)                     # 预测
smn.plot(output_path="result.png")      # 画图
```

详细 API 见后文。

---

## 强化学习示例

`SMNModule` 可以直接作为强化学习中的 **Q 网络** 或 **策略网络**。

### 示例 1: DQN (Deep Q-Network)

**场景：** CartPole 环境，状态 → Q 值（每个动作一个 Q 值）

**损失函数如何传入：** DQN 使用 MSE loss 比较 预测 Q 值 和 目标 Q 值。损失函数在训练循环中调用，**不是**传给 `SMNModule`，而是你自己在 `backward()` 时调用。

```python
from smn_fitter import SMNModule
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 创建 Q 网络
# CartPole: 4 维观测，2 个动作
q_net = SMNModule(
    n=2, m=4,
    n_in=4, n_out=2,
    x_bounds=[(-4.8, 4.8), (-5, 5), (-0.42, 0.42), (-5, 5)],  # CartPole 观测范围
    activation='relu'
)

# 2. 创建优化器
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()  # ← 损失函数在这里定义

# 3. 训练循环
for step in range(1000):
    # 获取一批经验 (state, action, reward, next_state, done)
    state = torch.randn(64, 4)        # 示例：64 个状态
    action = torch.randint(0, 2, (64,))
    reward = torch.randn(64,)
    next_state = torch.randn(64, 4)
    done = torch.zeros(64,)

    # 4. 前向传播：计算当前 Q 值
    current_q = q_net(state)                          # (64, 2)
    current_q_for_action = current_q.gather(1, action.unsqueeze(1)).squeeze(1)  # (64,)

    # 5. 计算目标 Q 值 (Bellman 方程)
    with torch.no_grad():
        next_q = q_net(next_state).max(dim=1)[0]     # (64,)
    target_q = reward + (1 - done) * 0.99 * next_q   # (64,)

    # 6. 计算损失 ← 这里调用损失函数
    loss = criterion(current_q_for_action, target_q)

    # 7. 反向传播 ← 这里传入梯度
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step={step}, loss={loss.item():.4f}")
```

**关键点：**
- `SMNModule` 只负责 `state → Q 值` 的前向传播
- 损失函数 (`MSELoss`) 在训练循环中调用，不是传给 module
- `loss.backward()` 自动计算梯度并更新 `q_net.parameters()`

---

### 示例 2: 策略梯度 (REINFORCE)

**场景：** CartPole，状态 → 动作概率分布

**损失函数如何传入：** 策略梯度的 loss 是 `-log_prob(action) * return`。你手动计算这个 loss，然后调用 `backward()`。

```python
from smn_fitter import SMNModule
import torch
import torch.optim as optim
import torch.nn.functional as F

# 1. 创建策略网络
# 输出：2 个动作的 logits（未归一化的对数概率）
policy_net = SMNModule(
    n=2, m=4,
    n_in=4, n_out=2,
    x_bounds=[(-4.8, 4.8), (-5, 5), (-0.42, 0.42), (-5, 5)],
    activation='relu'
)

optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
gamma = 0.99

# 2. 一集的交互
def run_episode():
    state = env.reset()
    log_probs = []
    rewards = []

    for t in range(500):
        # 前向传播：state → logits → 概率 → 采样动作
        logits = policy_net(torch.FloatTensor(state).unsqueeze(0))  # (1, 2)
        probs = F.softmax(logits, dim=1)                            # (1, 2)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()                                      # (1,)
        
        log_probs.append(dist.log_prob(action))                     # 标量
        rewards.append(reward)

        state, reward, done, _ = env.step(action.item())
        if done:
            break

    return log_probs, rewards

# 3. 训练
for episode in range(1000):
    log_probs, rewards = run_episode()

    # 计算 returns (折现累积奖励)
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns)

    # 策略梯度 loss: -sum(log_prob * return)
    # ← 损失函数在这里手动定义，不是传给 module
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)
    
    loss = torch.cat(policy_loss).sum()  # 标量

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"episode={episode}, total_reward={sum(rewards)}")
```

**关键点：**
- `policy_net` 输出 logits，你用 `softmax` 转概率，再采样动作
- loss 是你手动计算的（`-log_prob * return`），不是传给 module
- `loss.backward()` 更新 `policy_net.parameters()`

---

### 示例 3: Actor-Critic (A2C)

**场景：** 同时学习策略（Actor）和价值函数（Critic）

```python
from smn_fitter import SMNModule
import torch
import torch.nn as nn
import torch.optim as optim

# Actor 和 Critic 共享一个 SMN 骨干
actor = SMNModule(n=2, m=4, n_in=4, n_out=2)  # 输出 2 个动作的 logits
critic = SMNModule(n=2, m=4, n_in=4, n_out=1) # 输出 1 个状态价值

optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()),
    lr=7e-4
)

# 训练一步
def train_step(states, actions, rewards, next_states, dones):
    # Actor: state → action logits
    logits = actor(states)               # (batch, 2)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)   # (batch,)

    # Critic: state → value
    values = critic(states).squeeze(1)   # (batch,)
    next_values = critic(next_states).squeeze(1).detach()

    # Advantage (TD 误差)
    td_target = rewards + 0.99 * next_values * (1 - dones)
    advantage = td_target - values

    # Actor loss: -log_prob * advantage
    actor_loss = -(log_probs * advantage.detach()).mean()

    # Critic loss: MSE
    critic_loss = nn.MSELoss()(values, td_target)

    # 总 loss
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item()
```

---

### 为什么损失函数不是传给 `SMNModule`？

**设计原理：**
- `SMNModule` 是纯粹的 `nn.Module`，只做 `输入 → 输出` 的映射
- 损失函数是**训练循环**的一部分，不是网络的一部分
- 这样设计让你可以用任何 loss（MSE、交叉熵、策略梯度、contrastive loss 等）

**对比：**

| 方式 | 优点 | 缺点 |
|------|------|------|
| loss 传给 module | 语法简洁 | 只能支持预设的 loss，不灵活 |
| loss 在循环中调用 | 完全灵活 | 代码稍多 |

---

## 配置 (params.yaml)

用于 `python3 run.py` 的配置文件：

```yaml
# Simplex Memory Network Configuration

# Model type: "smn" or "mlp"
model_type: smn

# Experiment name (used in run folder name)
run_name: exp0414_smn

# SMN architecture parameters
n: 4            # Simplex dimension (2 = triangle, 3 = tetrahedron)
m: 4            # Resolution (each edge has m lattice points)
n_in: 2         # Input dimension
n_out: 2        # Output dimension

# MLP baseline architecture (used when model_type: mlp)
mlp_layers: [8, 8, 8]

# Activation function: "relu", "leaky_relu", "gelu", "tanh"
node_activation: relu

# Task registry (task → n_in, n_out):
#   1i1o: sin, sin_mix, poly_wave, piecewise
#   1i2o: sin_cos
#   2i1o: sin_sum, sin_product, quadratic
#   2i2o: trig_2d
task_name: trig_2d
custom_function: ""  # For custom tasks (eval-based, use with care)

# Training parameters
lr: 0.001
batch_size: 64
epochs: 5000

# Moving window training (set window_width > 0 to enable)
window_width: 0.0    # 0.0 = disabled (full domain training)
window_hold: 20      # epochs per window position

# Training set size
num_train: 500

# Data range (for 1D tasks)
x_min: -6.283185307179586  # -2π
x_max: 6.283185307179586   #  2π

# Whether to train an MLP baseline and show 4-panel comparison plot
compare_mlp: true
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | str | "smn" | 模型类型："smn" 或 "mlp" |
| `n` | int | 2 | 单纯形维度（2=三角形，3=四面体） |
| `m` | int | 3 | 分辨率（每条边上的格点数） |
| `n_in` | int | 1 | 输入维度 |
| `n_out` | int | 1 | 输出维度 |
| `node_activation` | str | "relu" | 激活函数："relu", "leaky_relu", "gelu", "tanh" |
| `task_name` | str | "piecewise" | 目标任务函数名 |
| `lr` | float | 1e-3 | 学习率 |
| `batch_size` | int | 64 | 批次大小 |
| `epochs` | int | 300 | 训练轮数 |
| `num_train` | int | 10000 | 训练集大小 |
| `x_min` / `x_max` | float | ±2π | 1D 任务的数据范围 |
| `x_bounds` | list | None | MIMO 任务的每通道范围，如 `[[-1,1], [-2,2]]` |
| `compare_mlp` | bool | true | **是否训练 MLP 基线并显示 4-panel 对比图**。如果 false，只训练 SMN 并显示 2-panel 图（scatter + loss 曲线） |
| `window_width` | float | 0.0 | 移动窗口宽度（0.0=禁用，0.5=50% 域宽） |
| `window_hold` | int | 1 | 每个窗口位置的训练轮数 |

---

## 数学背景简述

（此处省略，与之前相同）

---

## 完整 RL 示例代码

`examples/smn_for_rl.py` 包含完整的可运行示例：
- DQN 训练 CartPole
- REINFORCE 训练 CartPole

运行：
```bash
cd exp0414_simplexNet
pip install gymnasium[classic-control]
PYTHONPATH=src python3 examples/smn_for_rl.py
```
