# TOOLS/SimplexNet 项目规划

## 项目定位

**TOOLS/SimplexNet** 是一个封装完善的 Simplex Memory Network (SMN) 工具包，基于 exp0414_simplexNet 的代码实现和 writing/w0001-simplex-theory 的理论基础。

**目标用户：**
- 想要快速使用 SMN 进行实验的研究者
- 需要将 SMN 集成到强化学习系统中的开发者
- 想要评估 SMN vs MLP 的算法工程师

**核心价值：**
- 开箱即用：最小依赖，清晰接口
- 可恢复训练：完整的 checkpoint 生命周期管理
- 向后兼容：保持与 exp0414 的互操作性

---

## 目录结构

```
TOOLS/SimplexNet/
├── src/
│   └── simplexnet/             # 包源码
│       ├── __init__.py
│       ├── core/               # 核心模块
│       │   ├── __init__.py
│       │   ├── SimplexMemoryGraph.py   # 单纯形图结构（DAG）
│       │   ├── SMNmodule.py            # PyTorch 神经网络模块
│       │   └── SMN_RL.py               # RL 高层封装
│       ├── rl/                 # 强化学习模块
│       │   ├── __init__.py
│       │   └── algorithms/
│       │       ├── __init__.py
│       │       └── dqn.py              # DQN 算法
│       └── tools/              # 工具模块
│           ├── __init__.py
│           ├── checkpoint.py           # 数据持久化
│           ├── logger.py               # 训练日志
│           └── plot.py                 # 可视化
├── examples/                   # 示例脚本
│   └── train_rl.py
├── runs/                       # 运行结果（.gitignore）
│   └── simplexnet/
│       ├── checkpoints/
│       ├── logs/
│       └── plots/
├── README.md                   # 使用说明
└── PLAN.md                     # 项目规划
```

---

## 与 exp0414 的对比

| 方面 | exp0414_simplexNet (现状) | TOOLS/SimplexNet (目标) |
|------|---------------------------|-------------------------|
| 定位 | 研究原型 + 实验验证 | 生产级工具包 |
| 代码组织 | 扁平结构 (6 个文件) | 模块化分层 (core/training/problems/utils) |
| Checkpoint | 仅内存 best_state | 完整的磁盘 checkpoint 管理 |
| 日志 | 无 | 追加式训练日志 |
| 配置 | 单一 params.yaml | 分离 model.yaml / train.yaml |
| 确认机制 | 无 | 配置兼容性检查 + 用户确认 |
| 向后兼容 | N/A | 保持与 exp0414 API 兼容 |

---

## 核心模块设计

### 1. SMNmodule.py — 网络模型类

**职责：** PyTorch nn.Module 实现，纯网络前向传播

```python
class SMNmodule(nn.Module):
    """Simplex Memory Network as a PyTorch module.

    Args:
        n: Simplex dimension (n=2 → triangle, n=3 → tetrahedron)
        m: Resolution (lattice points per edge)
        n_in: Input dimension
        n_out: Output dimension
        activation: Hidden node activation
        x_bounds: Per-channel input normalization bounds

    Example::

        # SISO function approximation
        module = SMNmodule(n=2, m=4, n_in=1, n_out=1)

        # DQN Q-network (CartPole)
        q_net = SMNmodule(n=2, m=4, n_in=4, n_out=2)

        # REINFORCE policy network
        policy = SMNmodule(n=2, m=4, n_in=4, n_out=2)  # outputs [mean, log_std]
    """

    def __init__(self, n=2, m=3, n_in=1, n_out=1,
                 activation='relu', x_bounds=None):
        super().__init__()
        # 初始化单纯形图、网络参数...

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the DAG."""

    @property
    def arch_str(self) -> str:
        """Human-readable architecture string."""

    @property
    def param_count(self) -> int:
        """Total trainable parameters."""
```

**依赖：** 内部使用 `rl/graph.py` 的 `SimplexMemoryGraph`（不对外暴露）

---

### 2. SMN_RL.py — 强化学习封装类

**职责：** RL 算法、MDP 定义、训练/测试、可视化、持久化、GUI

```python
class SMN_RL:
    """Reinforcement Learning with SMN.

    Attributes:
        module (SMNmodule): The underlying SMN network
        env (gym.Env): Gymnasium environment
        algorithm (str): 'DQN', 'PPO', 'REINFORCE'
        gamma (float): Discount factor
        observation_space: From env
        action_space: From env

    Methods:
        train(episodes, render): Train the agent
        test(episodes): Evaluate the agent
        plot(save_path): Visualize training curves
        save_checkpoint(path): Save training state
        load_checkpoint(path): Resume from checkpoint
        launch_gui(): Open PySide interactive window
    """

    def __init__(self, module: SMNmodule, env: gym.Env,
                 algorithm='DQN', gamma=0.99, lr=1e-3):
        self.module = module
        self.env = env
        self.algorithm = algorithm
        self.gamma = gamma
        self.lr = lr
        # MDP definition
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # Training state
        self.episode_rewards = []
        self.episode_losses = []
        # Persistent modules
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'

    def train(self, episodes=1000, render=False, 
              checkpoint_interval=100) -> dict:
        """Training loop.

        Returns:
            dict with 'rewards', 'losses', 'final_stats'
        """

    def test(self, episodes=100, render=False) -> dict:
        """Evaluation.

        Returns:
            dict with 'mean_reward', 'std_reward', 'rewards'
        """

    def plot(self, save_path=None, show=True):
        """Plot training curves (reward, loss)."""

    def save_checkpoint(self, path: str, metadata: dict = None):
        """Save full training state."""

    def load_checkpoint(self, path: str) -> dict:
        """Resume from checkpoint.

        Returns:
            metadata dict
        """

    def launch_gui(self):
        """Launch PySide interactive window."""
```

**内部依赖：**
- `rl/algorithms/dqn.py` — DQN 实现
- `rl/algorithms/ppo.py` — PPO 实现
- `tools/checkpoint.py` — CheckpointManager
- `tools/logger.py` — TrainingLogger
- `tools/plot.py` — plot_training_curves
- `tools/gui.py` — PySide 主窗口

---

## 子模块设计

### 1. rl/ — 强化学习算法模块

```
rl/
├── __init__.py          # 导出算法类
├── algorithms/
│   ├── dqn.py           # DQN 算法（Q-network + replay buffer + target network）
│   ├── ppo.py           # PPO 算法（actor-critic + clip loss）
│   └── reinforce.py     # REINFORCE 算法（policy gradient）
├── mdp.py               # MDP 定义基类（状态空间、动作空间、转移、代价）
└── env_wrapper.py       # Gymnasium 环境封装
```

**DQN 算法接口示例：**
```python
from rl.algorithms import DQN

dqn = DQN(
    q_network=smn_module,
    obs_dim=4, act_dim=2,
    gamma=0.99, lr=1e-3,
    buffer_size=10000,
    batch_size=64
)

dqn.select_action(state, epsilon=0.1)
dqn.store_transition(state, action, reward, next_state, done)
loss = dqn.train_step()
```

---

### 2. tools/ — 工具模块

```
tools/
├── __init__.py
├── checkpoint.py        # CheckpointManager — 保存/加载/兼容性检查
├── logger.py            # TrainingLogger — JSON Lines 追加日志
├── plot.py              # 可视化函数
│   ├── plot_training_curves()
│   ├── plot_trajectory()
│   └── plot_action_distribution()
└── gui.py               # PySide 交互窗口
    ├── TrainingWindow   # 训练控制面板
    ├── MonitorWindow    # 实时曲线显示
    └── LogPanel         # Terminal log 显示
```

**CheckpointManager 接口：**
```python
from tools import CheckpointManager

ckpt_mgr = CheckpointManager('./checkpoints')

# 保存
ckpt_mgr.save(
    module=smn,
    optimizer=optimizer,
    episode=100,
    rewards=rewards,
    metadata={'env': 'CartPole-v1'}
)

# 加载
checkpoint = ckpt_mgr.load_latest()
if checkpoint:
    smn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
```

**TrainingLogger 接口：**
```python
from tools import TrainingLogger

logger = TrainingLogger('./logs')
logger.log('train_start', config={...})
logger.log('epoch', episode=1, reward=100, loss=0.5)
logger.log('checkpoint_saved', path='...')
```

**GUI 窗口功能：**
- 参数调节滑块（n, m, lr, gamma, epsilon）
- 训练控制按钮（Start, Pause, Resume, Stop）
- 实时曲线显示（reward, loss, epsilon）
- Terminal log 面板（滚动显示）
- Checkpoint 管理（Load, Save, Delete）

---

## 实现优先级

### Phase 1 (高优先级) — 最小可用版本

- [x] `SMNmodule.py` — 网络模型（复制并简化自 exp0414/smn_fitter.py）
- [x] `SMN_RL.py` — RL 封装类
- [x] `rl/algorithms/dqn.py` — DQN 算法
- [x] `tools/checkpoint.py` — CheckpointManager（核心功能）
- [x] `tools/logger.py` — TrainingLogger
- [x] `tools/plot.py` — 训练曲线可视化
- [x] `__init__.py` — 导出公共 API
- [x] `README.md` — 基本使用说明
- [x] `examples/train_rl.py` — RL 训练示例
- [x] src/ layout 重构 — 开源准备

### Phase 2 (中优先级) — 增强功能

- [ ] `rl/algorithms/ppo.py` — PPO 算法
- [ ] `rl/algorithms/reinforce.py` — REINFORCE 算法
- [ ] `rl/mdp.py` — MDP 定义基类
- [ ] `tools/gui.py` — PySide 交互窗口
- [ ] `tools/checkpoint.py` — 完整兼容性检查

### Phase 3 (低优先级) — 完善生态

- [ ] `tests/` — 单元测试
- [ ] `examples/` — 更多示例（CartPole, MountainCar, 自定义环境）
- [ ] 文档网站

---

## 下一步行动

**计划已就绪，等待用户指令开始 Phase 1 实现。**

实现顺序：
1. `SMNmodule.py` — 网络模型（基于 exp0414/smn_fitter.py 的 SMNModule）
2. `rl/` 子文件夹创建
3. `rl/algorithms/dqn.py` — DQN 算法（基于 exp0420/dqn_agent.py）
4. `tools/` 子文件夹创建
5. `tools/checkpoint.py` — CheckpointManager
6. `tools/logger.py` — TrainingLogger
7. `tools/plot.py` — 训练曲线可视化
8. `SMN_RL.py` — RL 封装类（整合以上模块）
9. `__init__.py` + `README.md`

---

## 工程化建议对照

| report.tex 建议 | TOOLS/SimplexNet 实现 | 状态 |
|-----------------|-----------------------|------|
| **模块拆分：problem / model / train / test** | rl/ (model+alg) + tools/ (utilities) + SMN_RL (unified) | ✅ |
| **Checkpoint 保守原则（5 条）** | `CheckpointManager` 完整实现 | ✅ |
| **配置分离（3 类 yaml）** | Phase 1: 单一 params.yaml; Phase 2: 三分离 | ⏳ |
| **GUI 职责边界** | tools/gui.py 为独立交互层，不侵入核心逻辑 | ✅ |

---

## 参考文档

- exp0414 工程说明文稿：`exp0414_simplexNet/report_tex/report.tex`
- SMN 理论文档：`writing/w0001-simplex-theory/report.tex`
- exp0414 代码：`exp0414_simplexNet/src/`
- exp0420 RL 代码：`exp0420_RLforSMN/src/`