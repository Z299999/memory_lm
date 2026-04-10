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
├── requirements.txt
└── scripts/
    ├── config.py
    ├── data.py
    ├── train.py
    ├── utils.py
    └── model/
        ├── __init__.py
        ├── graph.py
        ├── mlp.py
        └── tmn.py
```

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

第一版节点块固定为：

```text
Linear + Bias + ReLU
```

执行逻辑：

- 输入头先把标量输入投影到 `d_model`
- 每条边都有一个可学习线性变换
- 核心节点把所有父节点变换后的状态求和，再过节点块
- 输出头聚合右边界父节点后映射到标量
- 最终输出只在输出头做一次 `tanh`

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
  - `L`
  - `n_in / n_out`
  - `d_model`
- MLP baseline 参数：
  - `mlp_layers`
- 函数参数：
  - `task_name`
  - `custom_function`
  - `node_activation`
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
```

## 默认配置

- `L=3`
- `n_in=1`
- `n_out=1`
- `d_model=32`
- `node_activation=relu`
- `output_activation=tanh`
- 默认任务：`sin`

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
