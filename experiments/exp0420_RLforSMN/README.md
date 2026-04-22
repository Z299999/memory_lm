# exp0420_RLforSMN — Simplex Memory Networks for Reinforcement Learning

基于单纯形记忆网络（SMN）的强化学习实验框架。

## 快速开始

```bash
# 运行 DQN 实验
cd exp0420_RLforSMN
python3 run.py

# 验证 SMNModule
python3 verify_smn.py
```

## 项目结构

```
exp0420_RLforSMN/
├── run.py                 # 主实验脚本
├── verify_smn.py          # SMNModule 验证
├── src/
│   ├── smn_module.py      # SMN 核心模块
│   ├── dqn_agent.py       # DQN agent
│   ├── siso_tracker.py    # SISO 轨迹跟踪环境
│   └── plot_utils.py      # 画图工具
└── results/               # 实验结果
```

## SMNModule 使用

```python
from src import SMNModule

# DQN: 状态 → Q 值（7 个离散动作）
q_net = SMNModule(n=2, m=4, n_in=2, n_out=7, x_bounds=[(-10, 10), (-5, 5)])

# REINFORCE: 状态 → 动作分布参数 [mean, log_std]
policy = SMNModule(n=2, m=4, n_in=1, n_out=2, x_bounds=[(-10, 10)])
```

## 实验结果

### DQN SISO 轨迹跟踪

**任务：** 跟踪 `sin(0.05*t)` 轨迹  
**配置：** 状态=[error, velocity], 7 个离散动作

**结果：**
- Best 100-ep avg reward: **~-33**
- Final 100-ep avg reward: **~-33**

## 进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1-3 | 环境 + DQN + 实验 | ✅ |
| 4 | REINFORCE Agent | ⏳ |
