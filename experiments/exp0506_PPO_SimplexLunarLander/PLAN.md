# exp0506 Plan

## Goal

在尽量保持 `exp0501_PPO_LunarLander` 不变的前提下，创建一个使用 `TOOLS/SimplexNet` 作为 Actor/Critic backbone 的 PPO 实验，并与 MLP baseline 做相近参数量下的学习效率比较。


## Current Defaults

当前已经先定下来的第一轮默认设计：

- MLP baseline:
  - Actor total: `4868`（含 `log_std`）
  - Critic total: `4801`
  - Pair total: `9669`
- SMN main config:
  - `n=4, m=9`
  - Actor total: `4459`（再加 PPO 的 `log_std`）
  - Critic total: `4291`
  - Pair total: `8750`
- SMN backup config:
  - `n=5, m=7`
  - Actor total: `4834`
  - Critic total: `4621`
  - Pair total: `9455`

当前默认实验意图：

- 主配置 `SMN(n=4, m=9)`：略小于 MLP，测试参数效率
- 备配置 `SMN(n=5, m=7)`：更接近 MLP，测试严格公平对照


## Import Note

当前 `exp0506` 通过仓库内源码路径接入 `SimplexNet`：

- 从脚本所在目录回溯到 repo 根目录
- 定位 `TOOLS/SimplexNet/src`
- 动态加入 `sys.path`
- 然后使用：

```python
from simplexnet import SMN
```

因此当前实验依赖的是仓库里的本地 `SimplexNet` 源码，而不是外部安装版本。


## Completed So Far

- [x] 明确 `exp0501` 当前 MLP Actor 和 Critic 的参数量，并记录为 baseline
- [x] 计算若干组 `SMN(n, m)` 在 `n_in=8, n_out=2` 和 `n_in=8, n_out=1` 下的参数量
- [x] 选出第一轮主配置 `SMN(n=4, m=9)` 和备配置 `SMN(n=5, m=7)`
- [x] 复制 `exp0501` 的实验骨架到 `exp0506`
- [x] 在 `exp0506` 中接入 `TOOLS/SimplexNet` 的 `SMN` 导入路径
- [x] 实现 `SMNActor`，保持当前 PPO 的高斯策略头形式不变
- [x] 实现 `SMNCritic`，输出标量状态价值 `V(s)`
- [x] 保持 `log_std` 为独立可学习参数，避免第一版同时引入 `state-dependent std`
- [x] 确认 `SMNActor` 输出在动作空间上与 `exp0501` 的 `tanh` 行为等价
- [x] 确认 `SMNCritic` 使用 `identity` 输出，不引入额外输出激活
- [x] 复用 `exp0501` 的 `configTrain.yaml` 结构，并增加 `SMN` 所需的 `n`、`m` 等配置项
- [x] 复用 `exp0501` 的 `configEval.yaml` 结构，并保证评估条件可以和训练条件完全对齐
- [x] 复用 `exp0501` 的 ballistic LunarLander 环境，而不是重新实现一套环境
- [x] 复用 `exp0501` 的 reward shaping、checkpoint、plotting、video 输出逻辑
- [x] 在 `exp0506` 中先完成单 seed、单配置的最小可运行版本
- [x] 验证 `exp0506` 是否能在 `Phase 1` 下正常训练，不出现 shape mismatch、NaN 或分布参数异常
- [x] 明确“学得更快”的主指标：reward / success rate vs environment steps
- [x] 明确“学得更快”的副指标：reward / success rate vs wall-clock time


## Todo

- [ ] 决定第一轮对照是否只保留 `0501` 当前 MLP baseline，还是再补一个更接近 SMN 参数量的 MLP baseline
- [ ] 决定第一轮正式对比是直接从 `Phase 5` 开始，还是先在 `Phase 3/4` 预热
- [ ] 为 `exp0506` 写一份自己的 `TRAINING_PLAN.md`，记录阶段推进和对照结果
- [ ] 在训练日志中记录每轮实验的参数量、训练步数、最终成功率和平均奖励
- [ ] 在训练日志中明确记录总训练时长、平均 update 时长、以及达到关键阈值所需时间
- [ ] 生成至少一组 `MLP vs SMN` 的 reward curve 对比图
- [ ] 生成至少一组 `MLP vs SMN` 的 success rate 或 eval result 对比图
- [ ] 检查 `SMN` 是否在相近参数量下表现出更好的样本效率
- [ ] 检查 `SMN` 是否在 wall-clock 时间上仍然具有竞争力
- [ ] 如果第一轮结果不稳定，决定是调整 `n,m` 还是先回到更简单阶段继续验证
- [ ] 更新 `exp0506` README，把最终实验设计变成实际实现说明


## Notes

- 第一版重点是 **控制变量比较 backbone**，不是重新设计 PPO。
- 第一版重点是 **先跑通并建立可比较基线**，不是一次性搜索最优 `SMN`。
- 除非实验明确需要，否则尽量不要改 `exp0501` 已验证过的训练流程。
