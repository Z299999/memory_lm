# exp0513 Plan

## Goal

把 `sigma-algebra edge addressing` 这套自调制网络机制，先从理论和最小原型上跑通，再决定要不要扩展成完整实验。

更具体地说，这个实验要回答：

- `O(log N)` 个 modulator heads 能否稳定地给 \(N\) 条 controllable edges 提供可区分地址？
- 这种地址化 internal modulation 能否和普通 backprop 干净地混合？
- 在真正实现时，`q`-heads 应该如何获得学习信号？

## Stage 1 — Theory Lock

- [x] 写出 addressability 的基本命题
- [x] 区分 addressability 和 independent controllability
- [x] 引入 assignment matrix \(B\) 与 energy normalization
- [x] 写出 mixed update 公式
- [x] 给出一个 hand-computable MLP 例子
- [x] 补上 reachable signal space 维度上界
- [x] 明确指出 `q`-head learning rule 仍是开放设计问题
- [x] 编译并检查 `theory/theory.tex`
- [ ] 决定这份理论稿的正式标题和术语是否保留 `dopamine` 这个词

## Stage 2 — Minimal Prototype

- [ ] 在 `src/` 中实现 assignment matrix 采样器
- [ ] 支持 coverage / distinguishability 检查
- [ ] 支持简单 balance 指标
- [ ] 实现一个最小 `SelfModulatedMLP`
- [ ] 实现 `bp_update + internal_update + mixed_update`
- [ ] 用代码复现理论稿中的手算例子

## Stage 3 — First Toy Tasks

- [ ] 选一个最小任务：
  - 线性回归
  - XOR
  - 小型时序 prediction
- [ ] 比较纯 BP 与 mixed-update 的训练行为
- [ ] 记录 `q`、`s=\widetilde B q`、以及边更新分布

## Stage 4 — Learning Rule For q-Heads

- [ ] 比较三种路线：
  - 固定 / 随机 q-heads
  - auxiliary loss
  - multi-step credit assignment
- [ ] 明确哪一种最适合作为 `0513` 的第一版正式实验

## Current Best Next Step

先做两件最小但关键的事：

1. 编译理论稿，确保数学稿现在是可交付的
2. 开始写最小原型，只实现：
   - assignment matrix \(B\)
   - edge-level modulation signal \(s=\widetilde B q\)
   - mixed update
