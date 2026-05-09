# exp0506 Plan

## Goal

在尽量保持 `exp0501_PPO_LunarLander` 不变的前提下，创建一个使用 `TOOLS/SimplexNet` 作为 Actor/Critic backbone 的 PPO 实验，并与 MLP baseline 做相近参数量下的学习效率比较。


## Current Defaults

当前已经先定下来的第一轮默认设计：

- `exp0501` 与 `exp0506` 的正式对照默认 `map_scale=1.0`

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
- [x] 把 `exp0501` 与 `exp0506` 的默认正式对照口径统一收成 `map_scale=1.0`（仅修改 config 默认值）
- [x] 让 train / eval 在 run 目录下产出稳定的 JSON 摘要，供 comparison 脚本复用
- [x] 在 `exp0506` 中新增显式 run-dir 输入的 comparison 脚本设计
- [x] 在 `exp0506` 中实现显式 run-dir 输入的 comparison 脚本
- [x] 用 `map_scale=1.0` 的默认配置重新跑一组 `0501` train / eval run
- [x] 用 `map_scale=1.0` 的默认配置重新跑一组 `0506` train / eval run
- [x] 用 comparison 脚本生成第一张 `MLP vs SMN` 对照图和 JSON 摘要
- [x] 修复 `exp0501` 与 `exp0506` 的 train / eval 环境 seeding，使环境 reset、评估和视频录制进入可复现实验轨道
- [x] 用修复后的可复现 seeding 重新跑一组 `map_scale=1.0` 的 `0501` train / eval run
- [x] 用修复后的可复现 seeding 重新跑一组 `map_scale=1.0` 的 `0506` train / eval run
- [x] 用修复后的 run 重新生成一张更可信的 `MLP vs SMN` comparison 图和 JSON 摘要


## Todo

- [x] 为 `exp0506` 写一份自己的 `TRAINING_PLAN.md`，记录阶段推进和对照结果
- [ ] 检查 `SMN` 是否在相近参数量下表现出更好的样本效率
- [ ] 检查 `SMN` 是否在 wall-clock 时间上仍然具有竞争力
- [ ] 如果第一轮结果不稳定，决定是继续用 `n=4,m=9`，还是切到 `n=5,m=7` 做更接近参数量的公平对照
- [ ] 决定下一轮是先做多 seed 正式比较，还是先切到 `SMN(n=5,m=7)` 做更接近参数量的公平对照


## Notes

- 第一版重点是 **控制变量比较 backbone**，不是重新设计 PPO。
- 第一版重点是 **先跑通并建立可比较基线**，不是一次性搜索最优 `SMN`。
- 除非实验明确需要，否则尽量不要改 `exp0501` 已验证过的训练流程。
