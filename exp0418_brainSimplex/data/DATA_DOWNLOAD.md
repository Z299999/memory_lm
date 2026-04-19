# FlyWire 数据集获取指南

## FlyWire 数据集说明

FlyWire 果蝇大脑连接组数据集包含：
- **约 130,000 个神经元**
- **约 5000 万个突触连接**
- 细胞类型标注、半谱系标签、形态学特征

## 数据获取方式

### 方式 1: 通过 CODEx API 下载（推荐）

```bash
# 使用 curl 下载突触数据
curl -o data/raw/synapses.csv \
  "https://codex.flywire.ai/api/v1/table/flywire_neuropil_synapses"

# 下载神经元元数据
curl -o data/raw/neurons.csv \
  "https://codex.flywire.ai/api/v1/table/flywire_neuropil_neurons"
```

### 方式 2: 使用 flywire Python 客户端

```bash
# 从 GitHub 安装（不在 PyPI 上）
pip install git+https://github.com/schlegelhead/flywire.git
```

```python
from flywire import fetch_connectome

# 获取连接组
cn = fetch_connectome(version='783')

# 保存数据
cn.neurons.to_csv('data/raw/neurons.csv')
cn.synapses.to_csv('data/raw/synapses.csv')
```

### 方式 3: Google Cloud Public Datasets

FlyWire 数据在 Google Cloud 上公开可用：
https://console.cloud.google.com/marketplace/product/flywire/flywire

### 方式 4: 直接下载（适用于测试）

如果上述方法都不可用，可以使用我们提供的样本数据：

```bash
cd exp0418_brainSimplex
python3 src/data_acquisition.py
```

这将生成一个包含 100 个节点的测试图。

---

## 下载完整数据后的预处理

```bash
cd exp0418_brainSimplex
python3 src/preprocessing.py
```

生成的文件：
- `data/processed/edge_list.csv` - 加权边列表
- `data/processed/adjacency_matrix.npz` - 邻接矩阵（稀疏格式）
- `data/processed/graph_statistics.json` - 图统计信息
- `data/processed/metadata.json` - 元数据

---

## 注意事项

1. 完整数据集约 2-5GB，确保有足够的磁盘空间
2. 预处理可能需要 10-30 分钟（取决于内存）
3. 建议至少 16GB RAM 来处理完整图
