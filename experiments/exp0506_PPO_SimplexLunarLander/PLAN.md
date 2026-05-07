# exp0506 Plan

## Goal

在尽量保持 `exp0501_PPO_LunarLander` 不变的前提下，创建一个使用 `TOOLS/SimplexNet` 作为 Actor/Critic backbone 的 PPO 实验，并与 MLP baseline 做相近参数量下的学习效率比较。


## Todo

- [ ] 明确 `exp0501` 当前 MLP Actor 和 Critic 的参数量，并记录为 baseline
- [ ] 计算若干组 `SMN(n, m)` 在 `n_in=8, n_out=2` 和 `n_in=8, n_out=1` 下的参数量
- [ ] 选出第一轮主打的 `SMN` 配置，以及必要时的备选配置
- [ ] 决定第一轮对照是否只保留 `0501` 当前 MLP baseline，还是再补一个更接近 SMN 参数量的 MLP baseline
- [ ] 复制 `exp0501` 的实验骨架到 `exp0506`，但暂时不要改 PPO 训练逻辑
- [ ] 在 `exp0506` 中接入 `TOOLS/SimplexNet` 的 `SMN` 导入路径
- [ ] 实现 `SMNActor`，保持当前 PPO 的高斯策略头形式不变
- [ ] 实现 `SMNCritic`，输出标量状态价值 `V(s)`
- [ ] 保持 `log_std` 为独立可学习参数，避免第一版同时引入 `state-dependent std`
- [ ] 确认 `SMNActor` 输出在动作空间上与 `exp0501` 的 `tanh` 行为等价
- [ ] 确认 `SMNCritic` 使用 `identity` 输出，不引入额外输出激活
- [ ] 复用 `exp0501` 的 `configTrain.yaml` 结构，并增加 `SMN` 所需的 `n`、`m` 等配置项
- [ ] 复用 `exp0501` 的 `configEval.yaml` 结构，并保证评估条件可以和训练条件完全对齐
- [ ] 复用 `exp0501` 的 ballistic LunarLander 环境，而不是重新实现一套环境
- [ ] 复用 `exp0501` 的 reward shaping、checkpoint、plotting、video 输出逻辑
- [ ] 在 `exp0506` 中先完成单 seed、单配置的最小可运行版本
- [ ] 验证 `exp0506` 是否能在 `Phase 1` 下正常训练，不出现 shape mismatch、NaN 或分布参数异常
- [ ] 决定第一轮正式对比是直接从 `Phase 5` 开始，还是先在 `Phase 3/4` 预热
- [ ] 为 `exp0506` 写一份自己的 `TRAINING_PLAN.md`，记录阶段推进和对照结果
- [ ] 明确“学得更快”的主指标：reward / success rate vs environment steps
- [ ] 明确“学得更快”的副指标：reward / success rate vs wall-clock time
- [ ] 在训练日志中记录每轮实验的参数量、训练步数、最终成功率和平均奖励
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
