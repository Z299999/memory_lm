# exp0513 Current Plan

## Current Direction

`exp0513` 当前已经切换到新的主线：

- 删除专门的 `q_head`
- 从 hidden layers 中静态随机选择 `m` 个 dopamine nodes
- 这些 hidden neurons 直接用自己的 `tanh` 激活调控整网 weight
- 每条 edge 的调控覆盖采用静态随机冗余 assignment
- 当前阶段暂时忽略 theory 中的 addressability / unique address

## Locked Decisions

- [x] 整网激活函数继续统一使用 `tanh`
- [x] dopamine nodes 从所有 hidden neurons 的总池中全局随机抽取
- [x] dopamine signal 直接使用 hidden activation，不再额外加新 head
- [x] 调控对象覆盖整张网络的所有 weight，不包含 bias
- [x] `coverage_c` 表示全局平均每条 edge 被覆盖约多少次
- [x] 静态 dopamine-edge assignment 必须写入 run 产物和 checkpoint
- [x] resume 时原样继承 dopamine nodes 与 assignment，不重新采样
- [x] dashboard 中 `epochs` 表示“继续训练多少轮”，不是目标总 epoch
- [x] dashboard 右侧状态区显示全局 epoch 区间，例如 `1001-2000`

## Current Mechanics

### Network

- 默认架构：`1 -> 16 -> 16 -> 1`
- hidden pool 大小：`32`
- controllable set：所有 weight
- non-controllable set：所有 bias

### Dopamine Coverage

- 设总 edge 数为 `N`
- 设目标全局平均覆盖数为 `c`
- 目标总关联数为 `K = cN`
- 每条 edge 至少先分配 1 个 dopamine node
- 剩余关联位随机补齐
- 同一条 edge 与同一个 dopamine node 之间最多只有一条关联
- 不强求逐边恰好 `c` 个覆盖，只要求全局平均约为 `c`

### `m` Recommendation

- 默认推荐：
  - `m = 10c`
- 直觉是让平均每个 dopamine node 控制的 edge 数不超过 `N/10`
- 当前默认架构下 `H = 32`
- 因此自动推荐模式下，通常只允许：
  - `c <= 3`

若用户手动填写 `dopamine_m_override`：

- 必须是正整数
- 不能超过 hidden pool 大小
- 若不合法则直接报错

## Training Semantics

### Single Run

- 用户主入口：
  - `run.py + config.yaml`
- 当前 run 用当前 `lambda`
- 连续训练 `epochs` 轮

### Resume

- `resume_from` 指向旧 run 的 `model.pt`
- 本次填写的 `epochs` 表示新增训练轮数
- 新 run 会继承：
  - 模型参数
  - dopamine node 选择结果
  - static dopamine-edge assignment
  - `global_epoch_completed`

因此 resume 的语义是：

- checkpoint 从哪里停下
- 新 run 就从那里继续往后数

## Dashboard Status

当前 dashboard 已支持：

- 编辑配置并启动单个 active run
- `Run / Stop / Refresh / Reset form`
- 点击 dopamine hidden node 高亮它控制的 edges
- 点击 edge 查看 controlling dopamine nodes
- 显示：
  - local epoch
  - global epoch
  - global epoch range
  - train/val/best loss
  - `coverage_c`
  - `dopamine_m`

## Documentation Note

- `theory/` 里的 addressable `q_head` 版本目前视为历史草稿
- 当前代码主线已经不再要求 unique address
- 如果后面需要重新把 `c_r`、heavy-tail coverage 或 addressability 写回理论，需要重新整理 theory 文稿，而不是直接把旧稿当成当前实现说明

## Next Step Candidates

1. 把 dopamine assignment 的统计量进一步放进 dashboard
- 例如 rank 后的 `c_r`
- 每层覆盖分布

2. 研究 `coverage_c` 与 `dopamine_m` 的关系
- 不同 `c`
- 不同手动 override 的 `m`
- 对训练稳定性和 loss 的影响

3. 如果后面重新碰 theory
- 先把 hidden-dopamine 版本的符号和旧 `q_head` 版本明确分开
