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
├── config.py
├── data.py
├── evaluate.py
├── model/
│   ├── __init__.py
│   ├── graph.py
│   ├── mlp.py
│   └── tmn.py
├── motivation.md
├── readme.md
├── requirements.txt
├── train.py
└── utils.py
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

第一阶段只支持：

- `sin(x)` 1D 回归

输入 `x` 会先归一化到稳定范围，目标值保持 `sin(x)`，自然落在 `[-1, 1]`。

## 快速开始

```bash
cd exp0410_triangleNet
pip install -r requirements.txt
```

训练 TMN：

```bash
python3 train.py --model-type tmn --run-name baby_tmn
```

训练 MLP baseline：

```bash
python3 train.py --model-type mlp --run-name baby_mlp
```

评估一个 checkpoint：

```bash
python3 evaluate.py --checkpoint runs/tmn/baby_tmn/checkpoint.pt
```

## 参数在哪里改

最直接的入口是 [config.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/config.py)。

你主要会改这些参数：

- 结构参数：
  - `L`：三角形边长，默认 `3`
  - `d_model`：TMN 隐藏维度
  - `n_in / n_out`：目前代码保留了配置位，但训练脚本第一阶段只实际支持标量输入输出
- MLP baseline 参数：
  - `mlp_hidden_dim`
  - `mlp_num_layers`
- 训练参数：
  - `lr`
  - `batch_size`
  - `epochs`
  - `weight_decay`
- 数据参数：
  - `num_train / num_val`
  - `x_min / x_max`

如果你不想改文件，也可以直接在命令行覆盖，例如：

```bash
python3 train.py --model-type tmn --run-name test1 --d-model 64 --epochs 500
python3 train.py --model-type mlp --run-name test2 --mlp-hidden-dim 128 --epochs 500
```

拟合函数本身不在 `config.py`，而是在 [data.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/data.py) 的 `target_function(...)` 里改。

例如把

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

第一次阅读代码时，建议按这个顺序看：

1. [config.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/config.py)：所有可调参数
2. [model/graph.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/model/graph.py)：三角 DAG 是怎么生成的
3. [model/tmn.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/model/tmn.py)：TMN 前向传播怎么走
4. [train.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/train.py)：训练循环和产物保存
```

## 默认配置

- `L=3`
- `n_in=1`
- `n_out=1`
- `d_model=32`
- `node_activation=relu`
- `output_activation=tanh`
- 默认任务：`sin`

## 输出产物

每次训练会在 `runs/<model_type>/<run_name>/` 下保存：

- `config.json`
- `metrics.json`
- `checkpoint.pt`
- `loss_curve.png`
- `prediction.png`

评估脚本会额外生成：

- `evaluation/prediction_eval.png`
- `evaluation/report.json`

## 验证建议

最小 sanity check：

```bash
python3 train.py --model-type tmn --run-name smoke_tmn --epochs 20
python3 train.py --model-type mlp --run-name smoke_mlp --epochs 20
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
