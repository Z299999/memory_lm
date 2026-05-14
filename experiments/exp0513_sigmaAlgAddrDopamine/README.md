# exp0513 — Sigma-Algebra Edge Addressing for Self-Modulated Networks

这个实验要做的事情，是把“网络自己输出内部调制信号，再去调自己的边权”这件事先用一个尽量干净的数学框架说清楚，再做最小原型验证。

当前核心想法是：

- 把一部分可训练边定义为 controllable edge set \(E\)
- 用 \(m\) 个 modulator heads 输出内部信号 \(q\)
- 用一个二值 assignment matrix \(B\) 把这 \(m\) 个 head 映射到 \(N=|E|\) 条边
- 通过 sigma-algebra / binary address 的视角说明：
  - \(m\) 个二值生成元最多可区分 \(2^m\) 条边
  - 因而 \(m=O(\log N)\) 足以提供 addressability
  - 但这不等于对 \(N\) 条边进行一步独立控制

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

还没有完成的部分：

- 还没有最小可运行代码原型
- 还没有把 \(q\)-head 自身如何学习这件事定下来
- 还没有 toy task 实验来验证这种内部调制是否有实际收益

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
3. 学习上，\(q\)-heads 的参数到底应该：
   - 固定
   - 用 auxiliary loss 学
   - 还是通过多步 credit assignment / meta-learning 学

## Recommended Next Step

最推荐的推进顺序是：

1. 先把理论稿中的定义、边界条件和 claims 收紧
2. 再做一个最小 `toy MLP` 原型
3. 先验证手算例子和代码结果一致
4. 然后再上更像实验的 toy task
