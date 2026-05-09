# exp0506 Training Plan

## Goal

在 `map_scale=1.0`、相同 `Phase 1` 初始条件、相同 PPO 超参数下，对比：

- `exp0501_PPO_LunarLander` 的 `MLP baseline`
- `exp0506_PPO_SimplexLunarLander` 的 `SMN(n=4, m=9)`

核心问题不是“谁最后分更高”，而是：

- `SMN` 是否更省样本
- `SMN` 是否更省真实训练时间
- 这些结论在可复现实验控制下是否成立


## Comparison Setup

- 地图：`map_scale=1.0`
- 任务：`Phase 1`
  - `init_speed=2.0`
  - `init_flight_angle_deg=80.0`
  - `random_side=false`
  - `init_body_angle_deg=0.0`
  - `init_altitude_m=8.0`
- PPO 超参数：沿用 `0501`
- MLP baseline：
  - actor total `4868`
  - critic total `4801`
  - pair total `9669`
- SMN main config：
  - `n=4, m=9`
  - actor total `4459`
  - critic total `4291`
  - pair total `8750`


## Important Note

在 `2026-05-08` 之前，我们发现训练脚本虽然设置了：

- `torch.manual_seed`
- `random.seed`
- `np.random.seed`

但环境 `env.reset()` 并没有显式传 seed，因此单次对照结果并不真正可复现。  
这会让单次 comparison 出现误导性的“MLP 崩溃而 SMN 正常”现象。

当前状态：

- `train` 与 `eval` 的环境 seeding 已修复
- environment reset、评估与视频录制都采用显式 seed 策略
- 后续比较应优先使用修复后的 run


## Run Log

### Run A — First comparison after map_scale=1.0 switch, before env seeding fix

Comparison artifact:

- [comparison_summary.json](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0506_PPO_SimplexLunarLander/runs/comparisons/20260507_214356/comparison_summary.json)

MLP:

- train run: `20260507_213019_train`
- eval run: `20260507_213211_eval`
- final train `Avg(100ep) = 24.75`
- eval success rate: `0/50`
- eval mean reward: `22.20`
- total training time: `106.93s`

SMN:

- train run: `20260507_213217_train`
- eval run: `20260507_214349_eval`
- final train `Avg(100ep) = 192.46`
- eval success rate: `49/50`
- eval mean reward: `201.56`
- total training time: `687.49s`

Interpretation:

- 这轮结果极端到不可信
- 主要暴露出“环境未显式 seed”的实验控制问题
- 不应直接据此得出 “SMN 明显优于 MLP” 的 backbone 结论


### Run B — Reproducible comparison after env seeding fix

Comparison artifact:

- [comparison_summary.json](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0506_PPO_SimplexLunarLander/runs/comparisons/20260508_133444/comparison_summary.json)

MLP:

- train run: `20260508_131946_train`
- eval run: `20260508_132144_eval`
- final train `Avg(100ep) = 119.93`
- eval success rate: `48/50`
- eval mean reward: `180.79`
- total training time: `112.46s`

SMN:

- train run: `20260508_132152_train`
- eval run: `20260508_133435_eval`
- final train `Avg(100ep) = 197.01`
- eval success rate: `48/50`
- eval mean reward: `177.69`
- total training time: `757.15s`

Interpretation:

- 这轮结果更可信
- `MLP` 并没有“始终失败”，环境 seeding 修复后恢复正常
- `SMN` 训练后段平台更高，但最终 eval 成功率没有明显压过 `MLP`
- `SMN` 的 wall-clock training cost 仍然显著更高


## Current Takeaway

目前更稳的结论是：

- 单次未控随机性的结果不能直接用来判断 backbone 优劣
- 在可复现实验控制下，`MLP` 和 `SMN(n=4,m=9)` 都能在 `Phase 1` 学会任务
- 当前证据更像是：
  - `SMN` 可能有不错的训练后段表现
  - 但 wall-clock 成本更高
  - 最终 eval 指标并未明显甩开 `MLP`


## Next Steps

1. 做多 seed 正式比较

- 至少 `3 seeds`
- 重新生成 comparison 图和汇总表

2. 再决定是否切换到 `SMN(n=5,m=7)`

- 如果要强调“更接近参数量”的公平性，这会是下一组主候选

3. 检查两个主问题

- `SMN` 是否更省样本
- `SMN` 是否值得接受更高 wall-clock cost
