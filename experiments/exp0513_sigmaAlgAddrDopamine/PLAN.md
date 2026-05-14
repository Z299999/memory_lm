# exp0513 V1.5 Plan

## Goal

把 `exp0513` 收紧成一个真正可运行、可续训、可切换任务的实验目录：

- 用户只需要改 `config.yaml`，再运行 `python3 run.py`
- 任务支持 `sin`、`sin_mix`、`poly_wave`、`piecewise`
- 训练接口统一成：
  - `epochs`
  - `lambda`
  - `resume_from`
- 默认输出压缩成：
  - `model.pt`
  - `loss_curve.png`
  - `summary.json`
  - copied `config.yaml`
  - `resolved_config.json`
- `q/update` 诊断图仅在 `enable_diagnostics: true` 时额外生成

## Locked Decisions

- [x] `B` 是静态 assignment matrix，初始化后不再变化
- [x] `q_head` 不接受梯度更新，也不使用 auxiliary loss
- [x] 整网激活函数统一使用 `tanh`
- [x] controllable set `N` 取全网所有 weight，不含 bias
- [x] 用户主入口收紧为根目录 `run.py + config.yaml`
- [x] 训练不再向用户暴露旧的 phase 语义
- [x] 续训允许自由调整 `lambda`，不做单调性限制

## Model And Mechanism

### Architecture

- 输入维度：`n_in = 1`
- 共享 trunk：`1 -> 16 -> 16`
- `y_head`: `16 -> 1`
- `q_head`: `16 -> m`
- trunk 和 `y_head` 使用 bias
- `q_head` 使用 `bias=False`
- `y = tanh(y_head(h))`
- `q = tanh(q_head(h))`

### Controllable Set

`N` 只包含所有 weight，不包含任何 bias。固定展开顺序为：

1. trunk 第 1 层 weight
2. trunk 第 2 层 weight
3. `y_head.weight`
4. `q_head.weight`

### `m` And Static Assignment

- `m` 按最小可区分公式自动求解
- 对默认架构固定得到：
  - `m = 9`
- `B ∈ {0,1}^{N×m}`，固定不变
- 为每条 edge 分配唯一非零 binary address
- 选码顺序固定为：
  - 先按 Hamming weight 与 `m/2` 的接近程度排序
  - 再按 lexicographic 排序
- 再构造：
  - `k_i = Σ_j B_{ji}`
  - `\widetilde B_{ji} = B_{ji}/sqrt(k_i)`

## User Interface

### Main Entry

- 默认入口：
  - `python3 run.py`
- 配置入口：
  - `config.yaml`
- `run.py` 负责：
  - 读取配置
  - 构造模型和静态 `B`
  - 可选加载 `resume_from`
  - 训练指定 `epochs`
  - 保存最终 artifact

### Supported Tasks

第一版与 `exp0414` 对齐，支持这些 1D 目标函数：

- `sin`
- `sin_mix`
- `poly_wave`
- `piecewise`

### Key Config Fields

- `run_name`
- `task_name`
- `seed`
- `epochs`
- `lambda`
- `resume_from`
- `batch_size`
- `lr_bp`
- `eta_int`
- `gamma`
- `num_train`
- `num_val`
- `num_plot`
- `x_min`
- `x_max`
- `enable_diagnostics`

## Training Semantics

### Single-Run Rule

用户侧不再区分 `Phase A / Phase B`，当前 run 的语义统一为：

- 用当前 `lambda`
- 连续训练 `epochs` 轮

这意味着：

- `lambda = 0.0`
  - 等价于纯 BP 训练
  - `q_head.weight` 不更新
- `lambda > 0`
  - trunk 和 `y_head` 使用 mixed update
  - `q_head.weight` 仅由 internal update 驱动

### Resume Rule

- `resume_from` 指向旧 run 的 `model.pt`
- 新 run 会加载：
  - 模型参数
  - `m`
  - `N`
  - assignment metadata
  - 旧配置摘要
- 新 run 的真实训练参数以当前 `config.yaml` 为准
- 因此用户可以：
  - 从 `lambda = 0.0` 的预训练 checkpoint 继续
  - 把 `lambda` 调大或调小
  - 换 `epochs`
  - 保留相同机制定义继续观察

## Outputs

默认每次 run 只保存：

- `model.pt`
- `loss_curve.png`
- `summary.json`
- copied `config.yaml`
- `resolved_config.json`

若 `enable_diagnostics: true`，额外保存：

- `q_diagnostics.png`
- `update_diagnostics.png`
- `diagnostics_history.json`

### Plot Requirements

`loss_curve.png` 需要：

- 在同一张图上画出 `train loss` 与 `val loss`
- 在图中嵌入参数文本框，至少标注：
  - `task_name`
  - `seed`
  - `lr_bp`
  - `eta_int`
  - `gamma`
  - `lambda`
  - `epochs`
  - `batch_size`
  - `m`
  - `N`
  - `resume_from` 简短来源标记

## Primary Observations

- `q` 是否从杂乱输出变成稳定分工
- `s = \widetilde B q` 是否对不同 edge 形成稳定差异
- `Δw_int` 与 `Δw_bp` 的相对量级和结构是否稳定

## Secondary Observations

- 不同 `task_name` 下的拟合误差变化
- 在全 `tanh` 饱和趋势下，mixed update 是否仍能工作
- 从旧 checkpoint 继续训练时，`lambda` 改变后行为是否连续可解释

## Acceptance Criteria

- [x] `python3 run.py` 可直接从 `config.yaml` 启动
- [x] `task_name` 可切换 `sin`、`sin_mix`、`poly_wave`、`piecewise`
- [x] `resume_from` 支持从旧 `model.pt` 继续训练
- [x] `lambda` 在续训时允许增大或减小
- [x] 默认输出收紧为单 checkpoint、单主图、单 summary
- [x] `enable_diagnostics=false` 时不产出诊断图
- [x] `enable_diagnostics=true` 时可产出诊断图
- [x] theory hand-example 校验入口保持兼容
- [x] 完成默认入口、任务切换、续训、诊断开关四类烟测

## First Fallback Order

如果 mixed update 明显失稳，按以下顺序回退：

1. 降低 `eta_int`
2. 保持当前 `lambda` 不变，先缩小内部更新尺度
3. 不先改任务、不先改架构
