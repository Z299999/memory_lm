# exp0513 — Hidden Dopamine Nodes for Self-Modulated Networks

这个实验现在已经从早期的 `q_head + addressability` 原型，重构成了一个更直接的版本：

- 删除专门的 `q_head`
- 从 hidden layers 里静态随机挑选 `m` 个 dopamine nodes
- 这些 hidden neurons 直接用自己当前的 `tanh` 激活来调控整张网络的 weight
- 每条 edge 的 dopamine 覆盖采用静态随机冗余分配

当前阶段**不追求 addressability / unique address**。  
theory 目录里的早期地址化表述暂时保留为历史草稿，但当前代码主线优先研究：

- hidden dopamine nodes 能不能稳定调控整网 weight
- 全局平均覆盖数 `c` 和 dopamine node 个数 `m` 应该怎么选
- 在 `resume` 和不同 `lambda` 下，这套机制能不能平稳继续训练

## Current Status

当前 `exp0513` 已经具备：

- 单入口训练：
  - [run.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/run.py)
  - [config.yaml](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/config.yaml)
- 本地 dashboard：
  - [viz.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/viz.py)
  - [src/dashboard_server.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/src/dashboard_server.py)
- hidden dopamine 版本的训练、checkpoint、resume
- live dashboard 状态轮询
- live dashboard train/val loss 曲线
- 纯前馈网络结构图 + dopamine node 控边高亮
- general MIMO 架构与固定命名 MIMO 任务族

## Folder Layout

```text
exp0513_sigmaAlgAddrDopamine/
├── checks/
├── run.py
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

支持的任务：

- `sin`
- `sin_mix`
- `poly_wave`
- `piecewise`
- `coupled_trig_2d2d`
- `cross_poly_2d2d`
- `piecewise_2d2d`
- `mixed_field_3d3d`
- `coupled_trig_4d4d`

当前配置已经支持显式设置：

- `input_dim`
- `output_dim`
- `trunk_dims`

其中：

- 现有 1D 任务要求 `input_dim=1, output_dim=1`
- 新增 MIMO 任务会各自声明合法维度；若配置不匹配会直接报错

当前内置支持到 `4D -> 4D`。

保留兼容的 1D 任务：

- `sin`
- `sin_mix`
- `poly_wave`
- `piecewise`

## Core Config Fields

- `lambda`
- `coverage_c`
  - 全局平均每条 edge 被 dopamine 覆盖多少次
- `dopamine_m_override`
  - 可选；若留空则系统自动推荐 `m = 10c`
- `resume_from`
  - 指向已有 `model.pt`

当前默认网络是 `1 -> 16 -> 16 -> 1`，但现在模型已经泛化到：

- `input_dim -> trunk_dims -> output_dim`

默认 hidden pool 大小仍是 `32`。  
按当前推荐规则：

- 推荐 `m = 10c`
- 因此默认架构下若使用自动推荐，通常只允许 `c <= 3`

## Resume Semantics

现在的 `resume` 语义是：

- `resume_from` 指向旧 run 的 `model.pt`
- 当前填写的 `epochs` 表示**继续训练多少轮**
- 不是“目标总 epoch”

例如：

- 上一次已经训练到全局 `1000`
- 本次 `resume_from` 指过去，并设置 `epochs: 1000`
- dashboard 会显示这次 run 的全局区间大致是 `1001-2000`

resume 时会继承：

- 模型参数
- 已选中的 dopamine nodes
- 静态 dopamine-edge assignment
- 全局 epoch offset

也就是说，resume **不会重新随机抽 dopamine nodes，也不会重新采样覆盖关系**。

## Dashboard

启动本地 dashboard：

```bash
cd experiments/exp0513_sigmaAlgAddrDopamine
python3 viz.py
```

然后访问：

- `http://127.0.0.1:8000/viz/`

也可以自己指定端口：

```bash
python3 viz.py --port 8765
python3 viz.py --no-open
```

这个页面现在是一个本地 dashboard：

- 左侧可以编辑训练参数并发起单个 active run
- 中间按当前配置动态显示 forward network
- 被选中的 hidden dopamine nodes 会被特别标出来
- 点击某个 dopamine node，会高亮它控制的 forward edges
- 点击 forward edge，可以查看它被哪些 dopamine nodes 覆盖
- 右侧显示当前 run 的状态、局部 epoch、全局 epoch、train/val loss 等信息
- 右侧还会实时重绘当前 run 的 train/val loss 曲线

## Notes

- [checks/verify_theory_hand_example.py](/Users/shzhang/Documents/Codes/memory_lm/experiments/exp0513_sigmaAlgAddrDopamine/checks/verify_theory_hand_example.py:1) 仍然保留，但它验证的是早期 addressable 版本的小例子。
- 当前代码主线优先研究 hidden dopamine coverage，不要求和 theory 里的旧 `q_head` 口径完全一致。
