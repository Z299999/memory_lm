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


## How SMN Is Imported

`exp0506` 当前没有依赖额外安装好的 `simplexnet` 包，而是直接从仓库内的源码导入。

做法是：

- 以当前脚本目录为起点定位 repo 根目录
- 拼出 `TOOLS/SimplexNet/src`
- 把这个路径动态加入 `sys.path`
- 再执行：

```python
from simplexnet import SMN
```

这意味着 `exp0506` 始终使用仓库当前版本的 `SimplexNet` 源码，而不是某个独立安装版本。

### Planned Actor

- 输入：环境状态 `obs_dim = 8`
- 输出：动作均值 `mu(s)`，维度 `act_dim = 2`
- backbone：第一轮主配置暂定为 `SMN(n=4, m=9, n_in=8, n_out=2, ...)`

当前计划是保留 `exp0501` 的高斯策略头做法：

- `mu(s)` 由 SMN 输出
- `log_std` 仍然是独立可学习参数

这样比较的重点就只放在 backbone 上，而不是把 `state-dependent std` 这种额外变量混进来。

### Planned Critic

- 输入：环境状态 `obs_dim = 8`
- 输出：状态价值 `V(s)`，维度 `1`
- backbone：第一轮主配置暂定为 `SMN(n=4, m=9, n_in=8, n_out=1, ...)`


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

当前已知的第一轮对照参数量如下：

- `exp0501` 当前 MLP baseline
  - Actor: `4868`（含 `log_std`）
  - Critic: `4801`
  - Pair total: `9669`
- `SMN(n=4, m=9)` 候选
  - Actor: `4459`（再加 PPO 的 `log_std`）
  - Critic: `4291`
  - Pair total: `8750`
- `SMN(n=5, m=7)` 备选
  - Actor: `4834`
  - Critic: `4621`
  - Pair total: `9455`

因此第一轮实验策略是：

- 主配置：`SMN(n=4, m=9)`，略小于 MLP，测试参数效率
- 备配置：`SMN(n=5, m=7)`，更接近 MLP，测试严格公平对照


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


## Default Comparison Setting

当前默认正式对照口径已经切到：

- `exp0501` train / eval 默认 `map_scale=1.0`
- `exp0506` train / eval 默认 `map_scale=1.0`

这里的含义仅限于 config 文件中的默认设置变更，不涉及环境实现或地图逻辑的额外修改。


## Comparison Artifacts

`exp0506` 现在承担统一对照汇报的职责。第一版 comparison 工作流是：

- 显式传入一组 `exp0501` 的 train / eval run 目录
- 显式传入一组 `exp0506` 的 train / eval run 目录
- 生成：
  - `comparison.png`
  - `comparison_summary.json`

comparison figure 采用 `Overlay + Panels`：

- 上半部分叠加 `MLP` 和 `SMN` 的训练曲线
- 下半部分左右各一个 summary panel

图和摘要中会明确写出：

- `MLP` 结构
- `SMN` 的 `n`、`m`
- actor / critic / total 参数量
- train / eval 的总时长
- 平均 update 时长
- success rate
- 平均奖励
- 当前 `map_scale`


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

1. 是否额外加入一个参数量更贴近 `SMN(n=4, m=9)` 的 MLP baseline
2. 第一轮比较是直接在 `Phase 5` 做，还是先在 `Phase 3/4` 预热
3. 是否在第一版就做多 seed，还是先单 seed 打通工程


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
- README 与 PLAN 已写好
- `exp0501` 的 PPO 实验骨架已复制到 `exp0506`
- `TOOLS/SimplexNet/src` 已通过 repo 内相对定位方式接入训练与评估脚本
- MLP Actor / Critic 已替换为 SMN Actor / Critic
- `SMN(n=4, m=9)` 已作为默认主配置接入训练与评估脚本
- `log_std` 仍然保持为独立可学习参数，便于和 `exp0501` 做干净对照
- `Phase 1` 下已完成一次端到端工程验证：
  - 训练可正常启动、更新、保存 checkpoint
  - 评估脚本可正常读取 SMN checkpoint 并完成图表 / 视频输出
  - 一轮 `Phase 1` 验证结果约为 `150.38` 的后 100 集训练均值，贪婪评估成功率 `34/50`
- `0501` 与 `0506` 的 train / eval 脚本现在都会在各自 run 目录下额外落盘 JSON 摘要，方便 comparison 脚本稳定读取结果
- comparison 脚本已纳入 `exp0506`，作为后续 `MLP vs SMN` 汇报的统一入口
- comparison 脚本已经过一轮真实 run-dir 验证，能够从 `0501` 与 `0506` 的 train / eval 摘要生成统一的 `comparison.png` 与 `comparison_summary.json`

下一步建议：

1. 在固定随机种子或多 seed 条件下继续复跑，避免单次结果波动误导结论
2. 先继续观察主配置 `SMN(n=4, m=9)`，再决定是否补 `SMN(n=5, m=7)` 做更接近参数量的公平对照
3. 如有必要，再把 comparison 脚本扩展到多 seed 汇总图
3. 然后开始正式的 `MLP vs SMN` 对照训练，并把训练时长纳入指标
