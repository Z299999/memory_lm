# exp0410 — Triangular Memory Network

`exp0410_triangleNet` 是一个独立的小实验目录，用来实现并验证 `Triangular Memory Network (TMN)`。

第一阶段目标很明确：

- 实现可配置的三角 DAG 网络结构
- 先用 baby case `L=3, n_in=1, n_out=1` 跑通
- 用 1D 函数拟合验证“这种结构可以训练”
- 与一个参数量级接近的 `MLP` baseline 做直接对照

当前阶段不做记忆机制分析，不做 relay node，不做 attention block。

原始动机见 [motivation.md](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/motivation.md)。

## 目录结构

```text
exp0410_triangleNet/
├── params.yaml
├── run.py
├── motivation.md
├── readme.md
├── timeline.md       # 科研进展记录（含 dead ends 和 TODOs）
├── requirements.txt
└── scripts/
    ├── config.py
    ├── data.py
    ├── train.py
    ├── plot.py
    └── model/
        ├── __init__.py
        ├── graph.py
        ├── mlp.py
        └── tmn.py
```

---

## 三维坐标系统与术语（TMN-3D）

TMN 节点使用三元组 `(x, y, z)` 定位，对应标准笛卡尔坐标轴：

| 轴 | 方向 | 术语 | 坐标符号 | 取值范围 | 示例 |
|----|------|------|----------|----------|------|
| **X 轴** | 里→外 | 深度 (Depth) | `x` | `1` 到 `depth` | `x=1` 是最里层，`x=2` 是外层 |
| **Y 轴** | 左→右 | 位置 (Position) | `y` | `1` 到 `L-z+1` | `y=1` 是最左，`y=L-z+1` 是最右 |
| **Z 轴** | 下→上 | 层 (Layer) | `z` | `1` 到 `L` | `z=1` 是底层，`z=L` 是顶层 |

**节点记法**：`(x, y, z)` = `(深度，位置，层)`

**节点示例**（L=4, depth=2）：
- `(2, 1, 4)` = 外层，左边，第 4 层（顶层外层）
- `(1, 1, 4)` = 里层，左边，第 4 层（顶层里层）
- `(2, 4, 1)` = 外层，最右，第 1 层（底层外层）
- `(1, 1, 1)` = 里层，最左，第 1 层（底层里层）

**注意**：当前实现使用**正三角形**结构：
- 第 z 层有 `L-z+1` 个位置（y 从 1 到 L-z+1）
- z=1（底层）有 L 个位置
- z=L（顶层）有 1 个位置

### 连接规则

**禁止的连接**：
- **X 方向连接**：同一位置内的神经元之间**不直接相连**（即 `(x, y, z) → (x+1, y, z)` 不存在）

**允许的连接**（三种）：

| 类型 | 方向 | 规则 | 示例（depth=1） |
|------|------|------|------|
| **层内连接** | Y 方向（左→右） | 相邻位置之间全连接 | `(1, 1, 4) → (1, 2, 4)` |
| **下→上连接** | Z+ 方向（垂直向上） | `(x, y, z) → (x, y, z+1)` | `(1, 1, 3) → (1, 1, 4)` |
| **上→下连接** | Z- 方向（斜向右下） | `(x, y, z) → (x, y+1, z-1)` | `(1, 1, 4) → (1, 2, 3)` |

**连出/连入约定**（当 depth > 1 时）：
- **连出**：从源位置的**最大 X**（最外层，`x = depth`）
- **连入**：到目标位置的**最小 X**（最里层，`x = 1`）
- 同一位置内的多个神经元（不同 x）是并行的，都从相同的前驱接收信号

### 参数表示

```yaml
# 三角形层数（Z 轴，下到上）
L: 4

# 深度（X 轴，里到外），每个位置的神经元数量（所有层相同）
depth: 1  # 原始 TMN（每个位置 1 个神经元）
# 或
depth: 2  # 每层扩充到 2 个神经元
```

**边的数量计算**：

| L | depth | 核心节点数 | 总边数 |
|---|-------|------------|--------|
| 4 | 1 | 10 | 26 |
| 4 | 2 | 20 | 64 |
| 4 | d | 10×d | 18d² + 8d |

边数组成（L=4, depth=d）：
- 层内边（Y 方向）：d² × (1+2+3) = 6d²
- 下→上边（Z+）：d² × (1+2+3) = 6d²
- 上→下边（Z-）：d² × (1+2+3) = 6d²
- 输入边：4d
- 输出边：4d
- 总计：18d² + 8d

---

## 当前实现

### 1. `TMNGraph`

负责：

- 生成输入头、核心节点、输出头
- 按文档规则生成三角 DAG 边
- 构造 `preds / succs / level / topological_levels`
- 检查所有边都从更早层指向更晚层

核心层级公式：

```text
t(T_(r,c)) = L - r + 2c
```

输入层固定为 `1`，输出层固定为 `2L + 1`。

### 2. `TMNNetwork`

**标量节点版本**（depth=1）：

每个节点是一个标量，每条边有一个可学习的标量权重：

```text
node(v) = ReLU( Σ w(u→v) × node(u) + bias(v) )
output  = Tanh( Σ w(u→out) × node(u) + bias_out )
```

**深度扩展版本**（depth=2）：

每个位置有多个神经元，相邻位置之间全连接：
- 连出：从源位置的**最大 X**（最外层，`x = depth`）
- 连入：到目标位置的**最小 X**（最里层，`x = 1`）

```text
位置 (z, y) 内部：神经元 (1, y, z), (2, y, z), ..., (d, y, z)
层内连接（Y 方向）：(d, y, z) → (1, y+1, z)  全连接 d² 条边
跨层连接（Z 方向）：(d, y, z) → (1, y', z-1) 全连接 d² 条边
```

**关于跳层（relay node）**：motivation 里提到过，如果框架不支持跳层连接，需要在中间补"死点"（relay node，权重和偏置固定不更新）来让信号逐层传递。PyTorch 的动态计算图天然支持任意 DAG，forward 时我们用一个 `states` 字典按拓扑序遍历节点，每个节点直接从字典里读取任意更早层的父节点状态，不需要补点。**所以当前实现没有 relay node。**

forward 执行逻辑：

- 按拓扑层从前往后遍历所有核心节点
- 每个节点把所有父节点的状态加权求和，加 bias，过激活函数
- 输出节点聚合右对角线上的父节点，加 bias，过 tanh

### 3. `MLPBaseline`

一个简单全连接基线：

- 若干层 `Linear + ReLU`
- 最终输出用 `tanh`

比较口径是“参数量级接近”，不是严格参数完全相等。

### 4. 数据任务

第一阶段支持这些 1D 回归任务：

- `sin`
- `sin_mix`
- `poly_wave`
- `piecewise`

输入 `x` 会先归一化到稳定范围，目标值保持 `sin(x)`，自然落在 `[-1, 1]`。

## 快速开始

```bash
cd exp0410_triangleNet
pip install -r requirements.txt
```

最简单的方式：

```bash
python3 run.py
```

这会自动：

- 训练 TMN
- 训练 MLP
- 只生成一张四宫格对比图

默认输出目录现在是：

```text
exp0410_triangleNet/runs/<run_name or task_name>/comparison_4panel.png
```

不会再额外保存 `TMN` 和 `MLP` 各自的 `loss_curve.png`、`prediction.png`、checkpoint 之类分散文件。

你真正需要改的参数文件是：

- [params.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/params.yaml)

`run.py` 会自动读取它。

训练 TMN：

```bash
python3 scripts/train.py --model-type tmn --run-name baby_tmn
```

训练更复杂的内置函数：

```bash
python3 scripts/train.py --model-type tmn --task-name sin_mix --run-name mix_tmn
python3 scripts/train.py --model-type tmn --task-name piecewise --run-name piecewise_tmn
```

训练自定义函数：

```bash
python3 scripts/train.py --model-type tmn --custom-function "torch.sin(2*x) + 0.2*torch.cos(7*x)" --run-name custom_tmn
```

训练 MLP baseline：

```bash
python3 scripts/train.py --model-type mlp --run-name baby_mlp
```

## 参数在哪里改

最直接的入口是 [params.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/params.yaml)。

如果你想直接点运行，就只改 [params.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/params.yaml)，然后运行 [run.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/run.py)。

你主要会改这些参数：

- 结构参数：
  - `L` — 三角形层数（上到下）
  - `depth` — 每层的深度（外到里），如 `1` 或 `2`
  - `n_in / n_out` — 输入/输出头数量
- MLP baseline 参数：
  - `mlp_layers`
- 函数参数：
  - `task_name`
  - `custom_function`
  - `node_activation` — 支持 `relu`, `leaky_relu`, `gelu`, `tanh`
  - `output_activation`
- 训练参数：
  - `lr`
  - `batch_size`
  - `epochs`
- 数据参数：
  - `x_min / x_max`

像 `seed`、`weight_decay`、`num_train`、`num_val`、`save_plots` 这些不常改的东西，已经收进内部默认值了，不需要你在文件顶部面对它们。

如果你不想改文件，也可以直接在命令行覆盖，例如：

```bash
python3 scripts/train.py --model-type tmn --run-name test1 --d-model 64 --epochs 500 --task-name sin_mix
python3 scripts/train.py --model-type mlp --run-name test2 --mlp-layers "[128,128,128]" --epochs 500
```

`mlp_layers` 的意思很直接：

- `[8]`：1 个隐藏层，每层 8 个节点
- `[8, 8, 8]`：3 个隐藏层，每层 8 个节点
- `[16, 32, 16]`：3 个隐藏层，宽度分别是 16、32、16

拟合函数逻辑在 [data.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/data.py)：

- 内置函数在 `target_function(...)`
- 命令行传入的自定义表达式在 `target_function_from_config(...)`

例如你可以直接命令行传：

```bash
python3 scripts/train.py --custom-function "torch.sin(2*x) + 0.2*torch.cos(7*x)"
```

如果你仍然想手改源码，最简单的改法就是直接改 `target_function(...)`，比如把

```python
return torch.sin(x)
```

改成

```python
return x ** 2
```

或者

```python
return torch.sin(2 * x)
```

`params.yaml` 现在会更直观一些，例如：

```yaml
L: 3
mlp_layers: [8, 8, 8]
task_name: piecewise
```

第一次阅读代码时，建议按这个顺序看：

1. [params.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/params.yaml)：你实际要改的参数
2. [config.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/config.py)：参数是怎么被读进程序的
3. [graph.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/model/graph.py)：三角 DAG 是怎么生成的
4. [tmn.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/model/tmn.py)：TMN 前向传播怎么走
5. [train.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/train.py)：训练循环和产物保存
6. [plot.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/plot.py)：可视化逻辑
7. [timeline.md](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/timeline.md)：科研进展记录

## 默认配置

- `L=4`
- `depth=1` — 原始 TMN，每个位置 1 个神经元
- `n_in=1`
- `n_out=1`
- `node_activation=relu` — 也支持 `leaky_relu`, `gelu`, `tanh`
- `output_activation=tanh`
- 默认任务：`sin_mix`

注意：当前输出层仍然是 `tanh`，所以最适合拟合值域大致落在 `[-1, 1]` 的目标函数。你如果传一个值域很大的自定义函数，训练会受到输出层限制。

## 输出产物

每次训练会在 `runs/<model_type>/<run_name>/` 下保存：

- `config.json`
- `metrics.json`
- `checkpoint.pt`
- `loss_curve.png`
- `prediction.png`

## 验证建议

最小 sanity check：

```bash
python3 scripts/train.py --model-type tmn --run-name smoke_tmn --epochs 20
python3 scripts/train.py --model-type mlp --run-name smoke_mlp --epochs 20
```

如果你想先看结构是否符合 baby case，可以直接在 Python 里检查：

```bash
python3 - <<'PY'
from model.graph import TMNGraph
g = TMNGraph(L=3, n_in=1, n_out=1)
print("nodes:", g.nodes)
print("edges:", len(g.edges))
print("levels:", g.topological_levels)
PY
```

## 后续方向

这个第一版只是为后续升级铺底：

- 更大的三角形 `L`
- 多输入头、多输出头
- 更强的节点 block
- 更贴近“记忆”的任务与分析
