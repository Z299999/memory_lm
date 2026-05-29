# exp0522 Language Emergence Plan

## Goal

这个实验的目标不是训练一个显式语言模块，而是观察：

- agent 在只有任务 loss 的情况下
- 是否会自发利用一个固定的 communication channel
- 把对未来有用的信息留给未来的自己，或者未来的其他 agent

这里我们强调的是 language emergence，而不是 supervised language training。

## Current Framing

我们目前把语言理解为一种通过环境留下的痕迹：

- agent 有正常的 task input / task output
- 另外还有 `k` 个 language output nodes 和 `k` 个 language input nodes
- language output 不参与训练，不直接接受反向传播更新
- language signal 会在下一时刻作为环境信号重新进入输入端

因此，这里的关键不是“语言头学会说话”，而是：

> hidden representation 会不会为了完成未来任务，学会把有用信息投影到一个固定语言通道中。

## Locked Decisions For V0

第一版 `v0` 先锁定如下决定：

- 先做单 agent，自言自语
- agent 主体是 MLP，不先上 RNN / Transformer
- 语言头不训练
- 训练目标中不加入任何 language loss
- 只用任务 loss 来塑造 hidden representation
- 暂时不做多 agent
- 暂时不接 dopamine / RL

## Why V0 Should Not Start From Function Fitting

普通函数拟合通常没有充分的“记忆压力”。

如果当前输入已经包含完成任务所需的全部信息，那么语言通道即使存在，也可能只是无功能噪声。这样即使出现某些 pattern，也很难论证那是 emergent language。

所以 `v0` 更适合从 partial observability 出发，而不是从静态函数拟合出发。

## V0 Task Proposal

### Task Type

第一版建议使用：

- 单 agent
- 部分可观测
- 延迟记忆任务

### Minimal Episode Structure

一个 episode 可以写成：

1. `t = 0`
agent 看到一个 cue。

2. `t = 1, ..., T - 1`
原始 cue 消失。
agent 只能看到普通环境输入和上一时刻留下的 language signal。

3. `t = T`
agent 需要输出一个正确答案，例如分类结果，或者对初始 cue 的重构结果。

### Why This Task

这个任务的好处是：

- 如果没有语言通道，MLP 很难仅靠当前输入完成延迟任务
- 如果语言通道有用，性能会明显提升
- 任务是否依赖语言，可以通过 ablation 很清楚地验证

## Agent Interface Proposal

设：

- task input 维度为 `d_x`
- task output 维度为 `d_y`
- language channel 维度为 `k`

则 agent 在每个时刻的接口可以写成：

- 输入：`[x_t, m_{t-1}]`
- 输出：
  - `y_t`：任务相关输出
  - `m_t`：language output

其中：

- `x_t` 是普通环境输入
- `m_{t-1}` 是上一时刻留下来的 language signal
- `y_t` 参与任务 loss
- `m_t` 不参与独立监督

## Language Channel Design

### Main Principle

语言头不训练，但语言通道不能退化成完全重复的拷贝。

如果所有 language output heads 都用完全一样的固定权重，例如全部是 `1 / k`，那么多个 channel 很可能塌缩成几乎同一个信号，实际容量会很低。

### Recommended Fixed Readout

第一版优先考虑以下三种固定读出方式：

1. fixed sparse signed readout
- 每个 language head 只连接少量 hidden neurons
- 若某个 head 连接了 `s` 个 hidden neurons，则每条固定边权重取
  - `+1 / sqrt(s)` 或
  - `-1 / sqrt(s)`
- 不同 head 连接的 hidden 子集不同
- 不同 head 的正负号模式也不同

2. fixed block readout
- 每个 language node 读取 hidden state 的一个不同子块
- 子块内使用固定常数权重
- 如果需要更强表达力，可以在块内使用 signed pattern，而不是全部同号

3. fixed Gaussian random readout
- 每个 language node 只连接少量 hidden neurons
- 或者直接对全部 hidden 做一个冻结的随机投影
- 权重固定，不参与训练，符号自然有正有负

当前最推荐的是 `fixed sparse signed readout`，因为：

- 不容易塌缩成多个几乎相同的 channel
- 每个 language head 都有不同的感受野
- 正负混合比纯平均更有表达力
- 仍然足够简单，容易实现和分析

如果希望第一版更规整、更容易解释，则可以退一步用 `fixed block readout`，因为：

- 更容易解释
- 更容易可视化
- 更容易控制每个 language head 的容量

### Scale Choice

若一个 language head 实际连接了 `s` 个 hidden neurons，则其固定权重的尺度更适合设为：

- `1 / sqrt(s)`

而不是简单设为 `1 / k`。

这样可以让不同 `k` 下的 message 振幅更稳定。

### Recommended Default For V0

如果 hidden width 记为 `h`，language width 记为 `k`，则推荐先取：

- `k = 4`
- 每个 language head 的连接数 `s = h / k`
- 若 `h` 不能被 `k` 整除，则让前几个 head 多分到一个 hidden neuron

一个具体例子：

- 若 `h = 32`
- `k = 4`
- 则每个 language head 连接 `s = 8` 个 hidden neurons
- 该 head 的固定权重可以是一个长度为 `8` 的 signed pattern，再整体乘以 `1 / sqrt(8)`

例如某个 head 的固定权重可以写成：

- `[+1, -1, +1, +1, -1, -1, +1, -1] / sqrt(8)`

### What We Should Avoid

第一版应避免以下退化设计：

- 所有 language heads 都使用完全相同的权重向量
- 所有权重都取同号常数，例如全部 `1 / k`
- 所有 heads 都读取完全相同的 hidden 子集

这些设计会让多个语言通道高度相关，导致有效容量远小于名义上的 `k`。

### Optional Receptor-Side Symmetry

除了输出侧 fixed readout，还可以考虑让输入侧的 language receptor 也采用固定 mixing：

- 把 `m_t` 先经过一个固定矩阵
- 再送入下一时刻输入端

这样可以避免输入侧只是对 message 做机械复制。

不过 `v0` 可以先从更简单的版本开始：

- 直接把 `m_{t-1}` 拼接到普通输入 `x_t` 后面

等输出侧机制验证清楚后，再考虑 receptor-side fixed mixing。

## Training Semantics

第一版训练语义：

- 只有任务 loss
- loss 只作用在 task outputs `y_t`
- language outputs `m_t` 没有 label
- language readout 参数固定
- MLP 主体参数可以通过任务 loss 更新

这样即使语言头不训练，主体网络依然可能学会：

- 在 hidden state 中保留未来有用信息
- 并把这些信息投影到固定 language channel 中

## Baselines And Ablations

第一版建议至少做以下对照：

### Baseline A: no-language

- 完全移除 language input / output channel

用途：

- 检验任务本身是否真的需要外部记忆通道

### Baseline B: language channel present but zeroed

- 保留接口
- 但每一步强行令 `m_t = 0`

用途：

- 检验性能提升是否确实来自 communication channel，而不是参数量或接口形式变化

### Ablation C: shuffle message at test time

- 测试时把 `m_t` 打乱

用途：

- 检验模型是否真正依赖 message 里的结构化信息

### Ablation D: vary channel width

- 比较 `k = 1, 2, 4, 8`

用途：

- 观察 communication capacity 与任务性能的关系

### Ablation E: add noise to message

- 在测试时对 `m_t` 加小噪声

用途：

- 检验 emergent code 的鲁棒性

## What Counts As Success In V0

如果出现以下现象，就可以认为 `v0` 是成功的：

- 有 language channel 时，任务性能明显优于 no-language baseline
- 将 message 清零或打乱后，性能显著下降
- 同一个 cue 往往会诱发相对稳定的 message pattern
- 更大的 `k` 往往提供更高的有效容量

第一版先不要求复杂的 linguistic structure，只要求：

- message 是功能性的
- message 是可重复利用的
- message 对未来行为有因果影响

## Suggested V0 Metrics

第一版可以重点记录：

- train / val task loss
- final task accuracy or reconstruction error
- no-language vs language performance gap
- message ablation drop
- message norm statistics
- same-cue message consistency

## Future Stages

### V1: environment-mediated buffer

不是简单把 `m_t` 直接喂回下一时刻输入，而是：

- 把 signal 存在环境 buffer 中
- 让读取过程更接近“through environment”

### V2: multi-agent sender-receiver

- 一个 agent 观察 cue
- 另一个 agent 执行动作或回答问题
- 两者通过环境中的 signal 交互

### V3: survival / ecological world

- 2D 环境
- 觅食、生存、移动控制
- communication signal 留在空间位置上

### V4: combine with other lines

未来可以考虑和以下主线结合：

- `exp0513` dopamine / local modulation
- `exp0422-0501` RL

但这些都不属于 `v0` 的范围。

## Open Questions

目前还没有完全锁定的问题：

1. `v0` 的 cue 是做分类更好，还是做向量重构更好？
2. `m_t` 是直接拼接到输入端，还是先过一个固定 receptor mixing？
3. 每个 episode 的长度 `T` 取多少最合适？
4. fixed block readout 的 hidden partition 应该如何设置？
5. 是否需要在训练中加入 message energy penalty，避免 message 无界增大？

## Immediate Next Step

下一步应该做的不是直接写大工程，而是先把下面四件事完全定清楚：

1. `v0` 具体任务定义
2. agent 输入输出张量接口
3. fixed language readout 的具体形式
4. baseline 和 ablation 的最小实验表
