# exp0420_RLforSMN — Simplex Memory Networks for Reinforcement Learning

基于单纯形记忆网络（SMN）的强化学习实验框架。

## 快速开始

```bash
# 1. 验证 SMNModule
cd exp0420_RLforSMN
python3 verify_smn.py

# 2. 运行 DQN 实验
PYTHONPATH=. python3 experiments/run_dqn_siso.py

# 3. 查看结果
ls results/*/
```

## 项目结构

```
exp0420_RLforSMN/
├── src/
│   ├── smn_module.py      # SMN 核心模块
│   ├── envs/
│   │   └── siso_tracker.py  # SISO 轨迹跟踪环境
│   ├── agents/
│   │   └── dqn_agent.py   # DQN agent
│   └── utils/
│       └── plot_utils.py  # 画图工具
├── experiments/
│   └── run_dqn_siso.py    # DQN 训练脚本
└── results/               # 实验结果
```

## SMNModule 使用

```python
from src.smn_module import SMNModule

# DQN: 状态 → Q 值（7 个离散动作）
q_net = SMNModule(n=2, m=4, n_in=2, n_out=7, x_bounds=[(-10, 10), (-5, 5)])

# REINFORCE: 状态 → 动作分布参数 [mean, log_std]
policy = SMNModule(n=2, m=4, n_in=1, n_out=2, x_bounds=[(-10, 10)])
```

| 参数 | 说明 |
|------|------|
| `n`, `m` | SMN 架构（单纯形维度，分辨率） |
| `n_in` | 输入维度（状态空间） |
| `n_out` | 输出维度（动作数或 Q 值数） |
| `x_bounds` | 每通道输入范围 |

## 实验结果

### DQN SISO 轨迹跟踪

**任务：** 跟踪 `sin(0.05*t)` 轨迹

**配置：** 状态=[error, velocity], 7 个离散动作，300 episodes

**结果：**
- Best 100-ep avg reward: **-73.00**
- Final 100-ep avg reward: **-122.70**

**输出图表：**
- `training.png` — 训练曲线
- `tracking.png` — 轨迹跟踪效果
- `error_dist.png` — 误差分布

## 进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 环境搭建与验证 | ✅ |
| 2 | SISO 环境 + DQN Agent | ✅ |
| 3 | 运行 RL 实验 | ✅ |
| 4 | REINFORCE Agent | ⏳ |

## 参考

- 0414_simplexNet: `exp0414_simplexNet/src/smn_fitter.py`
- 0414_simplexNet: `exp0414_simplexNet/examples/smn_for_rl.py`
