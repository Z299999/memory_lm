# exp0513 V1 Plan

## Goal

把 `sigma-algebra edge addressing` 的自调制机制收成一个可直接运行的第一版实验：

- 先复现 theory 里的 hand-computable example
- 再在 1D `sin_mix` 拟合任务上跑两阶段训练
- 重点看 `q / s / 更新模式` 是否出现稳定结构
- 拟合误差只作为次要观察

## Locked Decisions

- [x] `B` 是静态 assignment matrix，初始化后不再变化
- [x] `q_head` 不接受梯度更新，也不使用 auxiliary loss
- [x] 整网激活函数统一使用 `tanh`
- [x] 第一版任务采用 `exp0414` 的 1D `sin_mix`
- [x] 训练 protocol 采用 `Phase A 纯 BP -> Phase B 固定 lambda=0.5`
- [x] 第一版只做单 seed，不做多 seed 统计

## Task Definition

### Hand Example

- 先用代码复现 [theory/theory.tex](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/theory/theory.tex) 中的小 MLP 手算例子
- 必须验证：
  - `Δw_bp`
  - `Δw_int`
  - mixed update

### Main Toy Task

- 任务：1D `sin_mix`
- 定义：
  - `y = 0.5*sin(x) + 0.3*sin(2x) + 0.2*sin(3x)`
  - `x ∈ [-2π, 2π]`
- 训练方式：标准监督训练，不做在线学习
- 默认数据规模：
  - `num_train = 500`
  - `num_val = 200`
  - `num_plot = 500`

## Model And Parameter Partition

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

参数分组固定为：

- `theta_trunk_w`：属于 `N`
- `theta_y_w`：属于 `N`
- `theta_q_w`：属于 `N`
- `theta_bias`：不属于 `N`

## `m` And Static Assignment

### `m` Choice

`m` 按最小可区分公式自动定，并考虑 `q_head.weight` 自身也在 `N` 里：

- `N = 16*1 + 16*16 + 16*1 + 16*m = 288 + 16m`
- 取最小整数 `m` 使得 `m = ceil(log2(N+1))`
- 对 V1 默认架构，解为：
  - `m = 9`

### `B` Construction

- `B ∈ {0,1}^{N×m}`，固定不变
- 使用长度为 `m` 的所有非零 binary codes
- 为每条 edge 分配一个唯一地址
- 选择顺序固定为：
  - 先按 Hamming weight 与 `m/2` 的接近程度排序
  - 再按 lexicographic 排序
- 由此优先选接近平衡的地址
- 构造
  - `k_i = Σ_j B_{ji}`
  - `\widetilde B_{ji} = B_{ji}/sqrt(k_i)`

## Training Protocol

### Shared Defaults

- `lr_bp = 1e-2`
- `eta_int = 1e-4`
- `gamma = 1.0`
- `batch_size = 64`
- 初始化：Xavier uniform with gain for `tanh`
- 优化方式：手动 plain SGD，不使用 Adam

### Phase A

- 训练目标：纯 BP 预训练
- `epochs = 1000`
- 更新规则：
  - `theta_trunk_w`, `theta_y_w`, `theta_bias`：纯 BP
  - `theta_q_w`：不更新

### Phase B

- 从 Phase A checkpoint 继续训练
- `epochs = 1000`
- 固定 `lambda = 0.5`
- 更新规则：
  - `theta_trunk_w`, `theta_y_w`：
    - `(1-lambda)Δw_bp + lambdaΔw_int`
  - `theta_q_w`：
    - 视为 `Δw_bp = 0`
    - 实际更新为 `lambdaΔw_int`
  - `theta_bias`：
    - 继续纯 BP

## Primary Observations

- `q` 是否从杂乱输出变成稳定分工
- `s = \widetilde B q` 是否对不同 edge 形成稳定差异
- `Δw_int` 与 `Δw_bp` 的相对量级和结构是否稳定

## Secondary Observations

- val MSE 在 Phase B 是否维持、改善或退化
- 在全 `tanh` 饱和趋势下，mixed update 是否仍能工作

## Outputs

每次正式 run 至少保存：

- `config.json`
- `assignment_summary.json`
- `phase_a_summary.json`
- `phase_b_summary.json`
- `checkpoint_phase_a.pt`
- `checkpoint_phase_b.pt`
- `fit_curves.png`
- `prediction_plot.png`
- `q_diagnostics.png`
- `update_diagnostics.png`

## Acceptance Criteria

- [ ] `m` fixed-point solver 对默认架构返回 `9`
- [ ] `B` 所有 row 非零、唯一，列和合理平衡
- [ ] flatten / unflatten 映射往返一致
- [ ] hand example 数值复现成功
- [ ] Phase A 能稳定跑完，且 `theta_q_w` 保持不变
- [ ] Phase B 能从 Phase A checkpoint 继续训练
- [ ] `q / s / 更新模式` 的诊断图可以正常生成

## First Fallback Order

如果 Phase B 明显失稳，按以下顺序回退：

1. 降低 `eta_int`
2. 保持 `lambda = 0.5` 不变
3. 不先改任务、不先改架构
