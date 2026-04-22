# exp0418_brainSimplex

**果蝇大脑连接组的单纯形结构分析**

本项目分析果蝇（Drosophila）中央脑连接组的有向单纯形结构，遵循 Reimann et al. (2017) 的方法论。

## 数据来源

当前实验默认使用的是一份 **Oxford 项目整理并提供的 Drosophila central brain 数据集**，不是 README 早期版本里提到的 5 万节点随机子集流程。

- 直接数据提供方：Oxford Mathematics Part C, C5.4 Networks Mini-Project
- 本项目实际导入的原始文件：
  - `data/raw/pete_fly_central_edges.csv`
  - `data/raw/pete_fly_central_nodes_metadata.csv`
- 本项目标准化后的中间文件：
  - `data/raw/oxford_edge_list.csv`
  - `data/raw/oxford_nodes.csv`
- 原始生物学数据来源：FlyWire / Drosophila central brain connectome（Nature 2024 相关数据）

这四层关系必须分清：

1. `FlyWire` 是底层生物连接组来源。
2. `Oxford C5.4` 项目整理并分发了本实验直接使用的 central brain 数据文件。
3. 本仓库再把 Oxford 文件转换成统一字段名的 `oxford_edge_list.csv` / `oxford_nodes.csv`。
4. 后续预处理结果写入 `data/processed/`，单纯形统计结果写入 `results/`。

当前这份 Oxford central 数据的规模为：

- 节点数：32,272
- 有向边数：849,981
- 边权：当前统一记为 `weight = 1`

## 快速开始

### 1. 安装依赖

```bash
cd exp0418_brainSimplex
pip install networkx pandas numpy scipy matplotlib
```

### 2. 准备数据

```bash
# 当前正式实验：导入 Oxford central brain 数据
python src/import_oxford_data.py

# 然后做预处理，生成 processed/ 下的 edge list、邻接矩阵和统计信息
python src/preprocessing.py
```

### 3. 运行分析

```bash
# 计算 0 到 15 维单纯形计数
python src/simplex_detection.py 15

# 如需零模型对比，可以额外运行
python src/simplex_detection.py 15 --compare 5

# 生成 0-15 维单纯形柱状图
MPLCONFIGDIR=/tmp/mpl-exp0418 python src/plot_simplex_counts.py
```

## 项目概述

### 研究背景

Reimann et al. (2017) 发现大脑神经回路中存在高维**有向单纯形**（directed simplices），这些结构不是随机的，而是生物 organized 的。

### 当前目标

1. **检测** 有向单纯形（k = 1, 2, 3, ...）
2. **对比** Erdős-Rényi 零模型
3. **量化** 单纯形过剩程度

### 当前结果摘要

基于当前 Oxford central brain 数据和现行计数定义，`0-15` 维 directed simplex 计数为：

| 维度 | 数量 |
|------|------|
| 0 | 32,272 |
| 1 | 716,923 |
| 2 | 1,585,453 |
| 3 | 1,062,581 |
| 4 | 320,647 |
| 5 | 52,739 |
| 6 | 4,952 |
| 7 | 216 |
| 8 | 4 |
| 9-15 | 0 |

这说明该图中不仅存在大量三角形和四面体型结构，还存在一个快速衰减但非零的高维尾部，最高到 `8-simplex`。

## 项目结构

```
exp0418_brainSimplex/
├── PLAN.md                  # 研究计划
├── README.md                # 本文件
├── idea.md                  # 原始笔记
├── requirements.txt         # Python 依赖
├── data/
│   ├── raw/                 # 原始数据
│   │   ├── pete_fly_central_edges.csv          # Oxford 提供的原始边文件
│   │   ├── pete_fly_central_nodes_metadata.csv # Oxford 提供的原始节点文件
│   │   ├── oxford_edge_list.csv                # 仓库标准化后的边列表
│   │   ├── oxford_nodes.csv                    # 仓库标准化后的节点元数据
│   └── processed/           # 处理后的数据
├── src/
│   ├── import_oxford_data.py# 导入 Oxford central 数据
│   ├── preprocessing.py     # 图预处理
│   ├── simplex_detection.py # 快速单纯形计数
│   └── plot_simplex_counts.py # 结果画图
├── results/                 # 分析结果
│   ├── simplex_counts.csv
│   └── null_model_comparison.csv
└── plots/                   # 输出图像
    └── simplex_counts_dim0_15.png
```

## 当前推荐流程

如果你的目标是复现实验当前结果，请使用下面这条路径：

```bash
cd exp0418_brainSimplex
python src/import_oxford_data.py
python src/preprocessing.py
python src/simplex_detection.py 15
```

说明：

- 当前默认分析对象是 Oxford 整理的 central brain 数据，不是 5 万节点子集。
- `preprocessing.py` 现在只读取 `data/raw/oxford_edge_list.csv` 和 `data/raw/oxford_nodes.csv`。
- `simplex_detection.py` 现在使用致密重编号 + 后继集合交集的计数实现，不再依赖大量 NetworkX 子图构造。
- 当前 `1-simplex` 计数是 `716,923`，不是 `849,981`。原因是互反边 `u -> v` 与 `v -> u` 不满足 simplex 对“每对节点恰好一个方向”的要求，因此不计入 `1-simplex`。

## 核心概念

| 术语 | 定义 |
|------|------|
| **有向 k-单纯形** | k+1 个节点的全连接有向无环图（DAG） |
| **有向团** | 同有向单纯形 |
| **零模型** | 用于对比的随机图（Erdős-Rényi） |

## 运行示例

```bash
# 导入 Oxford central brain 数据
python src/import_oxford_data.py

# 预处理，生成 processed/ 文件
python src/preprocessing.py

# 检测 0-15 维单纯形
python src/simplex_detection.py 15

# 如需零模型对比
python src/simplex_detection.py 15 --compare 5

# 生成 0-15 维柱状图
MPLCONFIGDIR=/tmp/mpl-exp0418 python src/plot_simplex_counts.py

# 查看结果
cat results/simplex_counts.csv
cat results/null_model_comparison.csv
```

## 结果可视化

`src/plot_simplex_counts.py` 会读取 `results/simplex_counts.csv`，并在 `plots/simplex_counts_dim0_15.png` 输出一张两行图：

- 上图：线性坐标，适合看低维大数量级
- 下图：对数坐标，适合看 `5-15` 维的高维尾部和零值

如果只看线性坐标，`5` 维以上的柱子几乎会被压扁；因此分析高维尾部时应主要参考对数坐标图。

当前仓库里保存的 `results/null_model_comparison.csv` 是基于 `5` 个 Erdős-Rényi realization 生成的示例输出；由于零模型本身带随机性，重跑时数值会有小幅波动。

## 参考文献

Reimann, M. W., et al. (2017). Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function. *Frontiers in Computational Neuroscience*, 11, 48.

Oxford Mathematics Part C, C5.4 Networks Mini-Project. Dataset files used here: `pete_fly_central_edges.csv` and `pete_fly_central_nodes_metadata.csv`.

FlyWire / Drosophila central brain connectome, Nature (2024-related release lineage).
