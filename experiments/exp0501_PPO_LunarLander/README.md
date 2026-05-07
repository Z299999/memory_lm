# exp0501 — PPO LunarLander

## Overview

这个实验把 PPO 用在一个 **自定义弹道入射版** `LunarLander-v3` 上，而不是直接使用 Gymnasium 默认重置方式。

核心目标不是“让 LunarLander 跑起来”，而是研究：

- PPO 在更复杂初始条件下能否稳定学会着陆
- 课程学习（curriculum）是否能把策略从简单着陆逐步带到目标弹道任务
- 连续动作 PPO 在带有姿态、速度、来向扰动的控制任务里表现如何

当前项目已经完成从 `Phase 1` 到 `Phase 5` 的整套课程推进，并在最终目标条件下达到稳定成功。


## Current Task

当前默认训练与评估配置都固定在最终目标条件：

```yaml
init_speed: 5.0
init_flight_angle_deg: 45.0
random_side: true
init_body_angle_deg: 13.5
init_angular_velocity: 0.0
init_altitude_m: 10.0
init_x_m: 0.0
```

也就是说，agent 需要处理：

- 较强的初始速度
- 更平的入射角
- 左右随机来向
- 初始机身倾斜


## Environment

训练环境由 [lunar_ballistic.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/lunar_ballistic.py) 提供，在 Gymnasium `LunarLander-v3` 的基础上加入：

- 自定义初始速度、入射角、来向
- 自定义初始姿态
- 自定义初始高度和水平位置
- 面向弹道入射任务的奖励 shaping

状态仍然是 LunarLander 的标准 8 维观测：

```text
[x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
```


## Policy and Value Network

PPO 实现在 [ppoTrain.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/ppoTrain.py)。

- Actor:
  - 输入：8 维状态
  - 网络：MLP，默认 `8 -> 64 -> 64 -> 2`
  - 输出：2 维连续动作分布的均值 `mu(s)`
- Critic:
  - 输入：8 维状态
  - 网络：MLP，默认 `8 -> 64 -> 64 -> 1`
  - 输出：状态价值 `V(s)`

当前策略头采用：

- **state-dependent mean**
- **state-independent std**

也就是：

- `mu(s)` 由神经网络输出
- `log_std` 是单独可学习参数，不随状态变化

这是一种常见的简洁版连续动作 PPO 实现。


## Action Space

本实验使用 **连续动作 LunarLander**，动作维度是 2：

- 第 1 维：主发动机控制
- 第 2 维：侧向发动机控制

训练时 Actor 从高斯分布中采样动作，再裁剪到 `[-1, 1]` 后送入环境。


## Training Result

课程学习完整记录见 [TRAINING_PLAN.md](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/TRAINING_PLAN.md)。

当前最重要的结论是：

- `Phase 1`：打通
- `Phase 2`：打通
- `Phase 3`：打通
- `Phase 4`：打通
- `Phase 5`：打通

最终 `Phase 5` 的代表性结果：

- 训练后 100 集平均奖励：`152.98`
- 贪婪评估：`50/50 success`
- 平均评估奖励：`160.18 ± 3.78`


## Project Strengths

这个项目目前最有价值的地方，我认为有这几项：

- **任务定义清楚**：不是泛泛地跑 PPO，而是明确研究“弹道入射 + 课程训练 + 连续控制”。
- **环境改造有意义**：`lunar_ballistic.py` 把默认 LunarLander 变成了更像真正控制问题的任务。
- **训练路径被验证过**：不是只跑通一次，而是从 `Phase 1` 到 `Phase 5` 逐步推进并全部打通。
- **评估流程完整**：不仅有训练，还包含独立评估、成功/坠毁/超时统计、图表和视频输出。
- **实验可复现性较好**：训练配置、评估配置、课程计划、checkpoint 路径都在本地明确保存。
- **工程体验比早期版本成熟**：训练和评估脚本已经补上进度日志、ETA 和缓存处理，不再像之前那样容易看起来“卡死”。


## Worth Improving

这个项目已经不再是“baby”，但仍然有几个非常值得继续提高的方向：

- **策略方差仍然太简单**：当前 `std` 是全局共享的可学习参数，不是 `state-dependent std`。更完整的策略头可以直接输出 `[mu(s), log_std(s)]`。
- **奖励 shaping 仍然偏手工**：当前 shaping 已经够用，但速度和姿态的持续 shaping 仍然比较弱，未来可以更系统地设计。
- **统计验证还不够**：目前主结果主要是单条课程推进线。下一步应该做多 seed 对比，而不是只看一次成功跑通。
- **泛化测试还不够强**：还可以继续测：
  - `init_altitude_m: null`
  - 更高 `init_speed`
  - 更大 `init_body_angle_deg`
  - 不同 `map_scale`
- **结构还可以进一步模块化**：`ppoTrain.py` 和 `ppoEval.py` 目前已经好用，但仍然偏实验脚本风格，后续可以抽出共享模块。
- **README 里的结果图还没直接展示**：如果后面想把它做成更正式的项目页，可以把代表性训练曲线和评估图嵌进 README。


## Files

- [ppoTrain.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/ppoTrain.py)
  - PPO 训练入口
- [ppoEval.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/ppoEval.py)
  - 贪婪评估、评估图和演示视频
- [lunar_ballistic.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/lunar_ballistic.py)
  - 自定义弹道入射环境
- [configTrain.yaml](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/configTrain.yaml)
  - 训练配置
- [configEval.yaml](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/configEval.yaml)
  - 评估配置
- [TRAINING_PLAN.md](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/TRAINING_PLAN.md)
  - 课程训练设计与执行记录


## Next Step

如果继续往前推，我最推荐的顺序是：

1. 把 `state-dependent std` 做成下一版 PPO。
2. 做多 seed 统计，确认当前结果不是偶然。
3. 在更强泛化条件下测最终 `Phase 5` 策略。
