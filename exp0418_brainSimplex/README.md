# exp0418_brainSimplex

**果蝇大脑连接组的单纯形结构分析**

本项目分析果蝇（Drosophila）大脑连接组的有向单纯形结构，遵循 Reimann et al. (2017) 的方法论。

## 快速开始

### 1. 安装依赖

```bash
cd exp0418_brainSimplex
pip install networkx pandas numpy scipy matplotlib
```

### 2. 准备数据

```bash
# 生成 5 万节点子集（推荐）
python src/download_subset.py --n-nodes 50000

# 或使用 100 节点样本数据（即时测试）
python src/data_acquisition.py
```

### 3. 运行分析

```bash
# 步骤 1: 预处理
python src/preprocessing.py

# 步骤 2: 单纯形检测 + 零模型对比
python src/simplex_detection.py 4 --compare 5
```

## 项目概述

### 研究背景

Reimann et al. (2017) 发现大脑神经回路中存在高维**有向单纯形**（directed simplices），这些结构不是随机的，而是生物 organized 的。

### 当前目标

1. **检测** 有向单纯形（k = 1, 2, 3, ...）
2. **对比** Erdős-Rényi 零模型
3. **量化** 单纯形过剩程度

## 项目结构

```
exp0418_brainSimplex/
├── PLAN.md                  # 研究计划
├── README.md                # 本文件
├── idea.md                  # 原始笔记
├── requirements.txt         # Python 依赖
├── data/
│   ├── raw/                 # 原始数据
│   │   ├── synapses.csv     # 突触连接
│   │   └── neurons.csv      # 神经元元数据
│   └── processed/           # 处理后的数据
├── src/
│   ├── download_subset.py   # 生成子集数据
│   ├── data_acquisition.py  # 数据下载
│   ├── preprocessing.py     # 图预处理
│   └── simplex_detection.py # 单纯形检测
└── results/                 # 分析结果
    ├── simplex_counts.csv
    └── null_model_comparison.csv
```

## 核心概念

| 术语 | 定义 |
|------|------|
| **有向 k-单纯形** | k+1 个节点的全连接有向无环图（DAG） |
| **有向团** | 同有向单纯形 |
| **零模型** | 用于对比的随机图（Erdős-Rényi） |

## 运行示例

```bash
# 生成 5 万节点子集
python src/download_subset.py --n-nodes 50000

# 预处理
python src/preprocessing.py

# 检测单纯形（最高 4 维），与 5 个零模型对比
python src/simplex_detection.py 4 --compare 5

# 查看结果
cat results/simplex_counts.csv
cat results/null_model_comparison.csv
```

## 参考文献

Reimann, M. W., et al. (2017). Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function. *Frontiers in Computational Neuroscience*, 11, 48.
