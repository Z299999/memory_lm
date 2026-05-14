# exp0513 Plan

## Goal

把 `sigma-algebra edge addressing` 这套自调制网络机制，先从理论和最小原型上跑通，再决定要不要扩展成完整实验。

更具体地说，这个实验要回答：

- `O(log N)` 个 modulator heads 能否稳定地给 \(N\) 条 controllable edges 提供可区分地址？
- 这种地址化 internal modulation 能否和普通 backprop 干净地混合？
- 在固定 \(B\)、不引入单独 \(q\)-head 训练目标的前提下，`q` 调控是否会随着训练逐步涌现？

## Locked Decisions

- [x] `B` 是静态 assignment matrix
- [x] 一个输出神经元能调控的突触集合在训练开始前固定
- [x] 不给 `q`-heads 设计单独的 auxiliary loss
- [x] 训练前期以 backprop 为主，后期逐步提高 self-modulation 占比
- [x] 第一版实验的核心观察目标是：`q` 调控机制是否会自行涌现

## Stage 1 — Theory Lock

- [x] 写出 addressability 的基本命题
- [x] 区分 addressability 和 independent controllability
- [x] 引入 assignment matrix \(B\) 与 energy normalization
- [x] 写出 mixed update 公式
- [x] 给出一个 hand-computable MLP 例子
- [x] 补上 reachable signal space 维度上界
- [x] 明确当前第一版实验不为 `q`-heads 额外设计独立训练目标
- [x] 编译并检查 `theory/theory.tex`
- [ ] 决定这份理论稿的正式标题和术语是否保留 `dopamine` 这个词

## Stage 2 — Minimal Prototype

- [ ] 在 `src/` 中实现 assignment matrix 采样器
- [ ] 支持 coverage / distinguishability 检查
- [ ] 支持简单 balance 指标
- [ ] 支持静态 \(B\) 的持久化与复用
- [ ] 实现一个最小 `SelfModulatedMLP`
- [ ] 实现 `bp_update + internal_update + mixed_update`
- [ ] 实现一个从 BP 主导到 self-modulation 主导的 `lambda` schedule
- [ ] 用代码复现理论稿中的手算例子

## Stage 3 — First Toy Tasks

- [ ] 选一个最小任务：
  - 线性回归
  - XOR
  - 小型时序 prediction
- [ ] 比较纯 BP 与 mixed-update 的训练行为
- [ ] 记录 `q`、`s=\widetilde B q`、以及边更新分布
- [ ] 观察 `q` 的调控模式是否随着 `lambda` 提升出现稳定结构

## Current Best Next Step

先做两件最小但关键的事：

1. 开始写最小原型，只实现：
   - assignment matrix \(B\)
   - edge-level modulation signal \(s=\widetilde B q\)
   - mixed update
   - `lambda` 从小到大的 schedule
2. 先复现理论稿里的手算例子，再上 toy task
