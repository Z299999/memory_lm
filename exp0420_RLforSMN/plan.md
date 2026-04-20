# exp0420_RLforSMN — SISO 轨迹跟踪 RL 实验计划

## 背景与目标

**背景：**
- 0414_simplexNet 已完成 SMNModule 封装，可作为纯 nn.Module 用于强化学习
- 已有 CartPole 示例（4 维状态→2 个动作 Q 值），但我们需要从更简单的 SISO 任务开始
- RL = 深度学习 + 时间维度，用先验知识预测未来，根据误差反向传播更新参数

**目标：**
在 0420 文件夹中搭建 SISO 轨迹跟踪 RL 实验框架，逐步增加任务难度。

---

## 任务清单（TODO List）

### 阶段 1：环境搭建与验证（Day 1）

- [ ] **1.1 创建项目结构**
  ```
  exp0420_RLforSMN/
  ├── README.md           # 实验说明文档
  ├── params.yaml         # 配置文件（可选，或用 Python config）
  ├── src/
  │   ├── __init__.py
  │   ├── envs/           # 自定义环境
  │   │   ├── __init__.py
  │   │   └── siso_tracker.py  # SISO 轨迹跟踪环境
  │   ├── agents/         # RL Agent 定义
  │   │   ├── __init__.py
  │   │   ├── dqn_agent.py
  │   │   └── reinforce_agent.py
  │   └── utils/          # 工具函数
  │       ├── __init__.py
  │       └── plot_utils.py
  ├── experiments/        # 实验脚本
  │   ├── run_dqn_siso.py
  │   └── run_reinforce_siso.py
  └── results/            # 实验结果（.gitignore）
  ```

- [ ] **1.2 复制 0414 的 SMNModule 依赖**
  - 复制 `src/smn_fitter.py` 中的 SMNModule 类到 0420/src
  - 复制 `src/graph.py`（SMNModule 依赖）
  - 验证 import 路径正确

- [ ] **1.3 安装依赖**
  ```bash
  pip install gymnasium torch numpy matplotlib
  ```

- [ ] **1.4 验证 SMNModule 可用**
  ```python
  from src.smn_fitter import SMNModule
  module = SMNModule(n=2, m=3, n_in=1, n_out=1, x_bounds=[(-1, 1)])
  x = torch.randn(32, 1)
  y = module(x)
  assert y.shape == (32, 1)
  print("SMNModule OK!")
  ```

---

### 阶段 2：SISO 轨迹跟踪环境（Day 2-3）

- [ ] **2.1 设计 SISO 轨迹跟踪环境**
  
  **环境规格：**
  - **观测空间（state）**：1 维 — 当前跟踪误差 `e(t) = target(t) - position(t)`
  - **动作空间（action）**：1 维 — 控制力/加速度 `u(t)`
  - **奖励函数（reward）**：`r(t) = -e(t)² - 0.1*u(t)²`（惩罚误差 + 控制 effort）
  - **动力学**：简化的质点运动 `pos(t+1) = pos(t) + vel(t)*dt`, `vel(t+1) = vel(t) + u(t)*dt`
  - **目标轨迹**：`target(t) = sin(ω*t)` 或 `target(t) = 0.5*sin(t) + 0.3*sin(2*t)`

- [ ] **2.2 实现 `src/envs/siso_tracker.py`**
  ```python
  import gymnasium as gym
  from gymnasium import spaces
  import numpy as np

  class SISOTrajectoryTracker(gym.Env):
      """SISO 轨迹跟踪环境
      
      状态：跟踪误差 e(t) = target(t) - pos(t)
      动作：控制力 u(t)
      奖励：-e² - 0.1*u²
      """
      def __init__(self, target_freq=0.1, episode_length=200):
          super().__init__()
          self.observation_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
          self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
          self.target_freq = target_freq
          self.episode_length = episode_length
          # 内部状态：position, velocity, time_step
          
      def reset(self, seed=None):
          # 重置内部状态，返回初始观测（误差）
          
      def step(self, action):
          # 更新动力学，计算新误差，返回 (obs, reward, terminated, truncated, info)
  ```

- [ ] **2.3 测试环境**
  ```bash
  python3 -c "
  from src.envs.siso_tracker import SISOTrajectoryTracker
  env = SISOTrajectoryTracker()
  obs, _ = env.reset()
  for _ in range(100):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, _ = env.step(action)
      print(f'obs={obs}, reward={reward}')
      if terminated or truncated:
          break
  print('Environment OK!')
  "
  ```

---

### 阶段 3：DQN Agent 实现（Day 4-5）

- [ ] **3.1 实现 `src/agents/dqn_agent.py`**
  
  参考 0414 的 `examples/smn_for_rl.py` DQNAgent，调整为 SISO：
  ```python
  from src.smn_fitter import SMNModule
  
  class DQNAgent:
      def __init__(self, state_dim=1, action_dim=1):
          # SMN 作为 Q 网络：state → Q 值
          # 对于连续动作空间，需要离散化或使用 DDPG/TD3
          # 简化版：将动作离散化为 N 个档位
          self.q_network = SMNModule(
              n=2, m=4,
              n_in=state_dim,      # 1 维：跟踪误差
              n_out=action_dim,    # N 个离散动作的 Q 值
              activation='relu',
              x_bounds=[(-10, 10)], # 误差范围
          )
          # ... 其余同 0414 示例
  ```

- [ ] **3.2 动作离散化策略**
  - 将连续动作 `u ∈ [-1, 1]` 离散化为 5-7 个档位：`[-1.0, -0.5, -0.3, 0, 0.3, 0.5, 1.0]`
  - Q 网络输出：每个档位的 Q 值
  - 选择动作：`argmax(Q)` 或 ε-greedy

- [ ] **3.3 实现训练循环 `experiments/run_dqn_siso.py`**
  ```python
  from src.envs.siso_tracker import SISOTrajectoryTracker
  from src.agents.dqn_agent import DQNAgent
  
  env = SISOTrajectoryTracker()
  agent = DQNAgent(state_dim=1, action_dim=7)  # 7 个离散动作
  
  for episode in range(500):
      state, _ = env.reset()
      total_reward = 0
      for t in range(200):
          action = agent.select_action(state)
          next_state, reward, terminated, truncated, _ = env.step(action)
          agent.store_transition(state, action, reward, next_state, done)
          agent.train_step()
          state = next_state
          total_reward += reward
          if done: break
      # 记录 episode_reward
  ```

- [ ] **3.4 添加结果可视化**
  - 绘制 `episode_reward` 曲线
  - 绘制典型 episode 的 `target(t)`, `position(t)`, `error(t)`, `action(t)`

---

### 阶段 4：REINFORCE Agent 实现（Day 6-7）

- [ ] **4.1 实现 `src/agents/reinforce_agent.py`**
  
  参考 0414 的 PolicyAgent，调整为 SISO 连续动作：
  ```python
  class REINFORCEAgent:
      def __init__(self, state_dim=1):
          # 策略网络：state → (action_mean, action_log_std)
          # 输出 2 维：动作分布的参数
          self.policy_net = SMNModule(
              n=2, m=4,
              n_in=state_dim,
              n_out=2,         # [mean, log_std]
              activation='relu',
              x_bounds=[(-10, 10)],
          )
          
      def select_action(self, state):
          params = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
          mean, log_std = params[0, 0], params[0, 1]
          std = torch.exp(log_std)
          dist = torch.distributions.Normal(mean, std)
          action = dist.sample()
          return action.clip(-1, 1), dist.log_prob(action)
  ```

- [ ] **4.2 实现训练循环 `experiments/run_reinforce_siso.py`**

- [ ] **4.3 对比 DQN vs REINFORCE**
  - DQN：离散动作，off-policy，样本效率高
  - REINFORCE：连续动作，on-policy，可能需要更多样本

---

### 阶段 5：实验与调参（Day 8-10）

- [ ] **5.1 基准实验**
  - 运行 DQN 和 REINFORCE 各 5 次（不同随机种子）
  - 记录收敛速度、最终性能

- [ ] **5.2 超参数搜索**
  - SMN 架构：`n ∈ {2, 3}`, `m ∈ {3, 4, 5}`
  - 学习率：`lr ∈ {1e-4, 3e-4, 1e-3}`
  - 激活函数：`{relu, gelu}`

- [ ] **5.3 任务难度递进**
  - Level 1: `target(t) = sin(0.1*t)`（低频正弦）
  - Level 2: `target(t) = 0.5*sin(t) + 0.3*sin(2*t)`（多频混合）
  - Level 3: 加入外部扰动 `d(t) ~ N(0, 0.1)`

- [ ] **5.4 结果分析**
  - 哪种架构最好？
  - 学习曲线是否稳定？
  - 与 0414 的回归任务相比，RL 任务需要不同的超参数吗？

---

### 阶段 6：文档与总结（Day 11）

- [ ] **6.1 编写 README.md**
  - 环境说明
  - 快速开始指南
  - 实验结果

- [ ] **6.2 整理代码**
  - 添加 docstring
  - 统一代码风格

- [ ] **6.3 总结报告**
  - 什么工作良好
  - 遇到的问题
  - 下一步方向（MIMO 扩展？）

---

## 技术细节

### SMNModule 配置建议

| 任务类型 | n | m | n_in | n_out | x_bounds | activation |
|----------|---|---|------|-------|----------|------------|
| SISO DQN | 2 | 4 | 1 | 7 | [(-10, 10)] | relu |
| SISO REINFORCE | 2 | 4 | 1 | 2 | [(-10, 10)] | relu |

### 关键设计决策

1. **为什么从 SISO 开始？**
   - 验证 RL 流程可行性
   - 调试简单（1 维状态可视化方便）
   - 为 MIMO 积累经验

2. **为什么用轨迹跟踪？**
   - 直观：可以画图看跟踪效果
   - 有明确的性能指标（tracking error）
   - 可以逐步增加难度

3. **DQN vs REINFORCE 选择？**
   - 两者都实现，对比效果
   - DQN 更适合离散控制
   - REINFORCE 更适合连续控制

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SMN 在 RL 中训练不稳定 | 高 | 用小学习率，梯度裁剪，batch normalization |
| 收敛太慢 | 中 | 增加 replay buffer size，调整 target network 更新频率 |
| 跟踪性能差 | 中 | 调整奖励函数权重，增加 episode length |

---

## 成功标准

- [ ] SISO 环境能稳定运行
- [ ] DQN agent 能学会跟踪正弦轨迹（平均误差 < 0.5）
- [ ] REINFORCE agent 能达到类似性能
- [ ] 代码可复用，易于扩展到 MIMO

---

## 参考

- 0414_simplexNet: `exp0414_simplexNet/src/smn_fitter.py` — SMNModule 实现
- 0414_simplexNet: `exp0414_simplexNet/examples/smn_for_rl.py` — DQN/REINFORCE 示例
- Gymnasium 文档：https://gymnasium.farama.org/
