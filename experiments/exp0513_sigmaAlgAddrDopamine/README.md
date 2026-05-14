# exp0513 — Sigma-Algebra Edge Addressing for Self-Modulated Networks

这个实验要做的事情，是把“网络自己输出内部调制信号，再去调自己的边权”这件事先用一个尽量干净的数学框架说清楚，再做最小原型验证。

当前核心想法是：

- 把一部分可训练边定义为 controllable edge set \(E\)
- 用 \(m\) 个 modulator heads 输出内部信号 \(q\)
- 用一个静态的二值 assignment matrix \(B\) 把这 \(m\) 个 head 映射到 \(N=|E|\) 条边
- 通过 sigma-algebra / binary address 的视角说明：
  - \(m\) 个二值生成元最多可区分 \(2^m\) 条边
  - 因而 \(m=O(\log N)\) 足以提供 addressability
  - 但这不等于对 \(N\) 条边进行一步独立控制
- 训练策略上，前期由普通 backprop 主导，随后逐步增大 self-modulation 的占比，观察 \(q\)-调控是否会在这个过程中自己涌现出稳定机制

## Current Status

当前 `exp0513` 还处在理论定型阶段：

- 已有一版理论稿：
  - [theory/theory.tex](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/theory/theory.tex)
- 已经明确区分：
  - addressability
  - independent controllability
- 已经写出：
  - \(B\) 的覆盖 / 可区分 / 平衡 / 分散约束
  - \(1/\sqrt{k_i}\) 能量归一化
  - backprop 与 internal modulation 的混合更新
  - 一个可手算的小 MLP 例子
- 已经确定两条实验决策：
  - \(B\) 是静态的，一旦初始化后不再改动
  - 不给 \(q\)-heads 单独设计监督目标，而是先用 BP 训练、再逐步过渡到自我调控

还没有完成的部分：

- 还没有最小可运行代码原型
- 还没有 toy task 实验来验证这种内部调制是否有实际收益
- 还没有验证 \(q\)-调控是否真的会随着训练过程逐步涌现

## Folder Layout

```text
exp0513_sigmaAlgAddrDopamine/
├── README.md
├── PLAN.md
├── theory/
│   └── theory.tex
├── src/
│   └── __init__.py
└── runs/
```

## What This Experiment Is Really About

这个实验当前最重要的不是“证明它一定比普通 BP 强”，而是先回答三个更基本的问题：

1. 数学上，`O(log N)` 个 modulator heads 到底能对多少边提供可区分地址？
2. 工程上，怎样构造一个既满足 coverage / distinguishability，又不过分失衡的 \(B\)？
3. 动力学上，在固定 \(B\)、没有单独 \(q\)-head 训练目标的情况下，\(q\)-调控是否会随着训练从 BP 主导阶段逐步涌现？

## Recommended Next Step

最推荐的推进顺序是：

1. 先把最小 `toy MLP` 原型写出来
2. 把静态 \(B\)、`lambda schedule` 和 mixed update 都落成代码
3. 先验证手算例子和代码结果一致
4. 然后再上 toy task，观察 \(q\)-调控是否会逐步涌现
