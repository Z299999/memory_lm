# exp0420_RLforSMN — Simplex Memory Networks for Reinforcement Learning

基于单纯形记忆网络（SMN）的强化学习实验框架。

## 目标

探索 SMN 在强化学习任务中的应用，从简单的 SISO 轨迹跟踪开始，逐步扩展到更复杂的任务。

**核心思想：**
- RL = 深度学习 + 时间维度
- 用先验知识预测未来，根据误差反向传播更新参数
- 类比记忆/语言系统预测未来

---

## 快速开始

### 1. 验证 SMNModule

```bash
cd exp0420_RLforSMN
python3 verify_smn.py
```

输出：
```
Testing SMNModule basic forward pass...
Created: SMNModule(SMN(n=2, m=3, 1→1), nodes=6, edges=15, params=22)
All SMNModule tests PASSED!
```

### 2. 安装依赖

```bash
pip install gymnasium torch numpy matplotlib
```

---

## 项目结构

```
exp0420_RLforSMN/
├── README.md              # 本文件
├── plan.md                # 详细实验计划
├── verify_smn.py          # SMNModule 验证脚本
├── src/
│   ├── __init__.py
│   ├── smn_module.py      # SMN 核心模块（纯 nn.Module）
│   ├── agents/            # RL Agent 定义
│   │   ├── __init__.py
│   │   ├── dqn_agent.py   # DQN agent（待实现）
│   │   └── reinforce_agent.py  # REINFORCE agent（待实现）
│   ├── envs/              # 自定义环境
│   │   ├── __init__.py
│   │   └── siso_tracker.py  # SISO 轨迹跟踪环境（待实现）
│   └── utils/             # 工具函数
│       └── __init__.py
├── experiments/           # 实验脚本（待实现）
│   ├── run_dqn_siso.py
│   └── run_reinforce_siso.py
└── results/               # 实验结果（.gitignore）
```

---

## SMNModule API

### 初始化

```python
from src.smn_module import SMNModule

module = SMNModule(
    n=2,                    # 单纯形维度：2=三角形，3=四面体
    m=4,                    # 分辨率：每条边上的格点数
    n_in=1,                 # 输入维度（状态空间维度）
    n_out=7,                # 输出维度（动作数或 Q 值数）
    activation='relu',      # 激活函数：'relu', 'leaky_relu', 'gelu', 'tanh'
    x_bounds=[(-10, 10)],   # 每通道输入范围
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n` | int | 2 | 单纯形维度。`n=2` 是三角形，`n=3` 是四面体 |
| `m` | int | 3 | 分辨率。每条边有 `m` 个格点 |
| `n_in` | int | 1 | 输入维度（状态空间维度） |
| `n_out` | int | 1 | 输出维度（动作数或 Q 值数） |
| `activation` | str | 'relu' | 隐藏层激活函数 |
| `x_bounds` | list | None | 每通道输入范围，如 `[(-10, 10)]`。None 时默认为 `[(-1,1)]*n_in` |

### 属性

| 属性 | 返回类型 | 说明 |
|------|----------|------|
| `module.arch_str` | str | 人类可读的架构描述 |
| `module.param_count` | int | 可训练参数总数 |

### 方法

#### `forward(x: Tensor) -> Tensor`

前向传播。通常通过 `module(x)` 调用。

```python
# SISO (1 输入 1 输出)
x = torch.randn(32, 1)      # 32 个样本，1 维输入
y = module(x)               # 输出：(32, 1)

# DQN (1 输入 7 输出 - 7 个离散动作的 Q 值)
module_dqn = SMNModule(n_in=1, n_out=7)
x = torch.randn(64, 1)
q_values = module_dqn(x)    # 输出：(64, 7) → 7 个动作的 Q 值

# REINFORCE (1 输入 2 输出 - 动作分布参数)
module_policy = SMNModule(n_in=1, n_out=2)
x = torch.randn(64, 1)
params = module_policy(x)   # 输出：(64, 2) → [mean, log_std]
```

---

## 推荐配置

### DQN（离散动作空间）

```python
q_network = SMNModule(
    n=2, m=4,               # 架构：三角形，4 个格点/边
    n_in=1,                 # 1 维状态：跟踪误差
    n_out=7,                # 7 个离散动作
    activation='relu',
    x_bounds=[(-10, 10)],   # 误差范围
)
```

### REINFORCE（连续动作空间）

```python
policy_net = SMNModule(
    n=2, m=4,
    n_in=1,
    n_out=2,                # [action_mean, action_log_std]
    activation='relu',
    x_bounds=[(-10, 10)],
)

# 采样动作
params = policy_net(state)
mean, log_std = params[0, 0], params[0, 1]
std = torch.exp(log_std)
dist = torch.distributions.Normal(mean, std)
action = dist.sample()
```

---

## 实验计划

详细计划见 [`plan.md`](plan.md)

### 阶段概览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 环境搭建与验证 | ✅ 完成 |
| 2 | SISO 轨迹跟踪环境 | 🔄 进行中 |
| 3 | DQN Agent 实现 | ⏳ 待开始 |
| 4 | REINFORCE Agent 实现 | ⏳ 待开始 |
| 5 | 实验与调参 | ⏳ 待开始 |
| 6 | 文档与总结 | ⏳ 待开始 |

---

## 背景

本实验基于 0414_simplexNet 项目中的 SMNModule，将其应用于强化学习任务。

**参考：**
- 0414_simplexNet: `exp0414_simplexNet/src/smn_fitter.py` — SMNModule 完整实现
- 0414_simplexNet: `exp0414_simplexNet/examples/smn_for_rl.py` — DQN/REINFORCE 示例

---

## 下一步

1. 实现 `src/envs/siso_tracker.py` — SISO 轨迹跟踪环境
2. 实现 `src/agents/dqn_agent.py` — DQN agent
3. 运行第一个 RL 实验！
