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

当前 `exp0513` 已经从理论草稿推进到可运行原型阶段：

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

当前已经完成的工程入口：

- 已有单次训练入口与可续训 checkpoint 机制
- 已支持 1D task registry：`sin`, `sin_mix`, `poly_wave`, `piecewise`
- 已支持用 `config.yaml` 选择 `lambda`、任务与 `resume_from`

## Folder Layout

```text
exp0513_sigmaAlgAddrDopamine/
├── checks/
├── run.py
├── server.py
├── viz.py
├── config.yaml
├── README.md
├── PLAN.md
├── theory/
├── src/
├── viz/
└── runs/
```

## Quick Start

```bash
cd experiments/exp0513_sigmaAlgAddrDopamine
python3 run.py
```

常规使用方式：

- 修改 [config.yaml](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/config.yaml)
- 运行 [run.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/run.py)

支持的 1D 任务：

- `sin`
- `sin_mix`
- `poly_wave`
- `piecewise`

如果想接着之前的模型继续训练：

- 在 `config.yaml` 中设置 `resume_from: "/abs/path/to/model.pt"`
- 同时自由修改 `lambda`

如果想验证理论里那个可手算的小例子：

- 运行 [checks/verify_theory_hand_example.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/checks/verify_theory_hand_example.py:1)

如果想打开本地 dashboard：

```bash
cd experiments/exp0513_sigmaAlgAddrDopamine
python3 viz.py
```

然后访问：

- `http://127.0.0.1:8000/viz/`

也可以自己指定端口，或者只启动 server 不自动打开浏览器：

```bash
python3 viz.py --port 8765
python3 viz.py --no-open
python3 server.py --port 8765
```

这个页面现在是一个本地 dashboard：

- 左侧可以修改所有训练参数并发起单个 active run
- 中间显示 `1 -> 16 -> 16 -> (y=1, q=9)` 的默认 forward network
- 点击 `q_i` 高亮它控制的 forward edges
- 点击 forward edge 查看它被哪些 `q_i` 控
- 右侧显示当前 run 的状态、epoch、train/val loss 与最终结果

## What This Experiment Is Really About

这个实验当前最重要的不是“证明它一定比普通 BP 强”，而是先回答三个更基本的问题：

1. 数学上，`O(log N)` 个 modulator heads 到底能对多少边提供可区分地址？
2. 工程上，怎样构造一个既满足 coverage / distinguishability，又不过分失衡的 \(B\)？
3. 动力学上，在固定 \(B\)、没有单独 \(q\)-head 训练目标的情况下，\(q\)-调控是否会随着训练从 BP 主导阶段逐步涌现？

## Recommended Next Step

最推荐的推进顺序是：

1. 用 `lambda: 0.0` 跑一个纯 BP checkpoint
2. 通过 `resume_from` 继续训练，并尝试调大或调小 `lambda`
3. 对比不同 `task_name` 下的 loss 曲线与可选诊断图
4. 观察 \(q\)-调控在不同任务上的稳定性与涌现模式
