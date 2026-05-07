# exp0506 — PPO SimplexLunarLander

## Overview

这个实验的目标是复刻 [exp0501_PPO_LunarLander](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0501_PPO_LunarLander/README.md) 的完整实验设定，但把其中的两张 MLP 网络替换成 `TOOLS/SimplexNet` 的 `SMN` 网络。

核心原则非常简单：

- PPO 算法不变
- 环境不变
- 课程训练不变
- 评估流程不变
- 只替换 Actor 和 Critic 的 backbone

因此，`exp0506` 的定位不是“新的 PPO 变种”，而是一个 **backbone ablation / controlled comparison**：

- `exp0501`：MLP PPO baseline
- `exp0506`：SMN PPO variant


## Research Question

在 **相近参数量** 的前提下，Simplex Memory Network 作为 Actor/Critic backbone，是否比 MLP：

- 学得更快
- 更省样本
- 在相同课程训练下更容易推进到目标任务

这个实验不试图证明 “SMN 一定更强”，而是要回答一个更具体也更可信的问题：

**在尽量控制其他变量不变的情况下，SMN 是否展现出更好的学习效率。**


## What Will Stay the Same

为了让比较尽可能干净，`exp0506` 应该尽量继承 `exp0501` 的以下内容：

- 同一 ballistic LunarLander 环境
- 同一 reward shaping
- 同一课程训练阶段设计
- 同一训练/评估脚本结构
- 同一 PPO 更新流程
- 同一 checkpoint / plotting / video 输出逻辑
- 同一训练超参数，除非实验结果证明必须调整


## What Will Change

唯一核心变化是：

- `exp0501` 使用 MLP Actor + MLP Critic
- `exp0506` 使用 SMN Actor + SMN Critic

### Planned Actor

- 输入：环境状态 `obs_dim = 8`
- 输出：动作均值 `mu(s)`，维度 `act_dim = 2`
- backbone：`SMN(n=?, m=?, n_in=8, n_out=2, ...)`

当前计划是保留 `exp0501` 的高斯策略头做法：

- `mu(s)` 由 SMN 输出
- `log_std` 仍然是独立可学习参数

这样比较的重点就只放在 backbone 上，而不是把 `state-dependent std` 这种额外变量混进来。

### Planned Critic

- 输入：环境状态 `obs_dim = 8`
- 输出：状态价值 `V(s)`，维度 `1`
- backbone：`SMN(n=?, m=?, n_in=8, n_out=1, ...)`


## Fair Comparison Rule

这个实验最重要的约束不是“都叫 PPO”，而是 **参数量要尽量接近**。

因此后续实验会重点记录：

- MLP Actor 参数量
- SMN Actor 参数量
- MLP Critic 参数量
- SMN Critic 参数量
- 总参数量

理想状态不是“结构长得像”，而是：

- 参数预算接近
- 训练流程一致
- 课程推进一致


## Success Metrics

“学得更快”需要提前定义，避免后面解释不清。

本实验准备使用两类指标：

### Primary

- reward vs environment steps
- success rate vs environment steps

这反映样本效率。

### Secondary

- reward vs wall-clock time
- success rate vs wall-clock time

这反映真实训练成本。

因为 SMN 可能：

- 用更少样本学会
- 但每一步更慢

也可能相反。


## Planned First Version

第一版不追求一次性扫很多 `n, m`，而是先做一个最干净的最小比较：

1. 复刻 `exp0501` 的 PPO 实验框架
2. 先接入一组主打的 SMN 参数
3. 保留一个 MLP baseline
4. 如果需要，再补一个参数量更接近 SMN 的 MLP baseline

也就是说，第一版重点不是“搜最优 SMN”，而是先回答：

**把 backbone 从 MLP 换成 SMN 之后，实验会发生什么。**


## Open Design Decisions

这些问题在正式写代码前需要继续定清楚：

1. `SMN` 的第一主配置使用哪组 `n, m`
2. 是否额外加入一个参数量更贴近 SMN 的 MLP baseline
3. 第一轮比较是直接在 `Phase 5` 做，还是先在 `Phase 3/4` 预热
4. 是否在第一版就做多 seed，还是先单 seed 打通工程


## Expected Folder Structure

后续大概率会扩展成：

```text
exp0506_PPO_SimplexLunarLander/
├── README.md
├── configTrain.yaml
├── configEval.yaml
├── ppoTrain.py
├── ppoEval.py
├── TRAINING_PLAN.md
└── models/ runs/
```

其中大部分文件会参考 `exp0501`，但网络定义部分会替换为 SMN 版本。


## Current Status

当前状态：

- 目录已创建
- 实验目标已定
- README 已写好
- 代码尚未开始实现

下一步建议：

1. 先计算 `exp0501` 当前 MLP actor/critic 参数量
2. 选出 1 到 3 组合适的 `SMN(n, m)` 候选
3. 再开始实现 `exp0506` 的最小可运行版本
