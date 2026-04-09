# Exp0408 - Memory Backpropagation (Baby Case)

## 实验目标

验证基于年龄结构记忆模型的 memory backpropagation 机制。本实验专注于 **baby case（$A_{\max}=0$）**：记忆只有一层，每轮新回复完全覆盖旧记忆。

模型细节见 [model.md](model.md)。

## 系统架构

```
数据集 (exp0406, 2869题 GSM8K/MATH)
    │
    ▼
┌─────────────────────────────────────────────┐
│  HOST LLM (qwen3.5-plus)                    │
│  - 持有题目 + 标准答案                       │
│  - 判断 student 答案对错                     │
│  - 因材施教生成个性化 feedback               │
│  - 决定：继续反馈 or 推进下一题              │
└───────────┬─────────────────────────────────┘
            │ x_t（题目 or feedback）
            ▼
┌─────────────────────────────────────────────┐
│  STUDENT LLM (qwen3.5-plus → Qwen2.5-0.5B) │
│  输入: x_t + m_t^(0)                        │
│  输出: r_t  →  m_{t+1}^(0) = r_t           │
└─────────────────────────────────────────────┘
```

## 实验流程

### Forward Generation

每道题最多进行 **1轮作答 + 3轮反馈**（事不过三）：

```
x_1 = 题目
r_1 = G(x_1, ∅)          → host 判断
x_2 = feedback(r_1)
r_2 = G(x_2, m_1^(0))    → host 判断
...（最多到 r_4）
换题
```

每一轮产生一个训练样本对 $(m_t^{(0)},\, m_{t+1}^{(0)})$。

### Offline Self-Prediction（Backward）

训练目标：用 $m_t^{(0)}$ 预测 $m_{t+1}^{(0)}$，做反向传播更新 student 模型参数。

### Forward/Backward 比例

由参数 `forward_per_back` 控制：

| 值 | 含义 |
|----|------|
| `3` | 每跑3道题做一次 backward 训练 |
| `inf` | 纯 forward，只收集数据，不训练 |

## Host 输出格式

```
## AGENT_INPUT
{发给 student 的文本：题目 or 个性化反馈}

## EXTRACTED_ANSWER
{从 student 上一轮回复中提取的答案}

## IS_CORRECT
{true / false}

## FEEDBACK_RATIONALE
{host 判断 student 哪里错了，因材施教的依据}

## NEXT_ACTION
{feedback | next_problem}
```

## 可视化方案

实验产出三个层次的可视化，保存在 `runs/<run_id>/plots/`。

### 1. 行为层（验证实验假设）

**目标**：回答"训练有没有效果"。

| 图表 | x 轴 | y 轴 | 说明 |
|------|------|------|------|
| 首次作答正确率曲线 | 题目序号 | 正确率（滑动窗口） | 随训练推进应逐渐上升 |
| 平均 feedback 轮数曲线 | 题目序号 | 平均 feedback 次数 | 随训练推进应逐渐下降 |
| 正确率对比柱状图 | backward 训练轮次 | 训练前/后正确率 | 每次 backward 前后各测一批题对比 |

### 2. 训练层（监控训练过程）

| 图表 | x 轴 | y 轴 | 说明 |
|------|------|------|------|
| Backward loss 曲线 | 训练步数 | self-prediction loss | 应单调下降 |
| 样本利用率 | backward 轮次 | 新增样本数 | 监控数据积累速度 |

### 3. 表示层（探索性，可选）

**目标**：观察训练前后模型内部表示的变化。

#### 降维可视化
- **t-SNE / UMAP**：取最后一层隐向量，对 key token（如 "dozen", "twice", "third"）做降维，对比训练前后聚类结构。

#### 协方差分析
- **协方差矩阵热力图**：直接观察 embedding 维度之间的协变关系，训练前后对比。
- **特征值谱（eigenvalue spectrum）**：协方差矩阵的特征值排列曲线。特征值越集中，说明 embedding 在更低维子空间内表示知识；训练后若曲线变化，反映表示结构的改变。

#### 各向同性（Isotropy）

各向异性含义：embedding 向量不均匀分布在球面，而是集中在一个方向锥体里。有四种互补的可视化方式：

| 方式 | 类型 | 说明 |
|------|------|------|
| **Cosine similarity 分布直方图** | 快照图 | 随机抽 N 对 embedding 算 cosine similarity，画分布。各向异性严重时分布集中在接近 1 处；各向同性时分布中心在 0。训练前后叠加对比。 |
| **平均 cosine similarity 曲线** | 时间序列 | 上图的均值标量随训练步数绘制，即 `isotropy.png`。 |
| **PCA 累计解释方差曲线** | 快照图 | 对所有 embedding 做 PCA，画前 k 个主成分的累计解释方差比。各向异性时曲线陡峭（前 1~2 个 PC 覆盖 90%+ 方差）；各向同性时曲线平缓。与特征值谱本质相同，可合并展示。 |
| **单位球面密度图** | 快照图 | 将 embedding 归一化后取前两个 PCA 方向投影到 2D，画密度热力图。各向异性时所有点堆成一个斑点；各向同性时点均匀铺开。 |

#### 线性探针（Probing）
- 在 embedding 上训练线性分类器，预测"这道题答对了吗"。训练前后 probe 准确率的变化定量反映正确/错误信息是否被线性编码进 embedding。

#### Memory 轨迹（本实验特有）
- 把历轮 $m_t^{(0)}$ 做句子级 embedding（均值池化），画出**时间轨迹**。用箭头表示方向，颜色区分答对/答错，观察 feedback 是否驱动 embedding 向正确答案方向移动。

#### 参数漂移
- 训练过程中 embedding 层参数变化量 $\|W_t - W_0\|_F$（Frobenius 范数）曲线。可进一步分析哪些 token 的 embedding 变化最大（预期：数字词、运算词变化最显著）。

### 实现计划

```
scripts/
└── visualize.py    # 读取 runs/<run_id>/ 产物，生成全部图表
runs/<run_id>/
└── plots/
    │
    │   ── 时间序列图（x 轴为时间 t）──────────────────────────
    ├── accuracy_curve.png          # x: 题目序号 t   | y: 首次作答正确率（滑动窗口）
    ├── feedback_rounds.png         # x: 题目序号 t   | y: 该题所需 feedback 轮数
    ├── loss_curve.png              # x: backward 步数 | y: self-prediction loss
    ├── isotropy.png                # x: backward 轮次 | y: 各向同性得分
    ├── memory_trajectory.png       # x/y: 2D 投影空间 | 点按 t 排列，箭头连线
    ├── param_drift.png             # x: backward 步数 | y: ||W_t - W_0||_F
    │
    │   ── 快照图（无时间轴，需训练前后各生成一张对比）──────────
    ├── embedding_tsne_{step}.png        # x/y: t-SNE 坐标，颜色=概念类别
    ├── covariance_heatmap_{step}.png   # x/y: embedding 维度索引
    ├── eigenvalue_spectrum_{step}.png  # x: 特征值排名  | y: 特征值大小（含累计解释方差）
    ├── cosine_sim_hist_{step}.png      # x: cosine similarity | y: 频数，训练前后叠加
    └── sphere_density_{step}.png       # x/y: 前两个 PCA 方向，颜色=密度
```

快照图在 `step=0`（训练前）和每次 backward 后各生成一张，文件名中带 step 编号。

运行方式：
```bash
python3 scripts/visualize.py --run-id r00001
```

## 文件结构

```
exp0408_memory_backprop/
├── run.py                  # 入口，设置参数
├── config.yaml             # forward_per_back, max_feedback, 模型等
├── scripts/
│   ├── host.py             # Host LLM：判断 + 因材施教 feedback
│   ├── forward.py          # Forward generation 主循环
│   ├── backward.py         # Offline self-prediction 训练
│   ├── student_model.py    # HuggingFace 本地模型推理封装
│   ├── visualize.py        # 可视化：行为层 + 训练层 + 表示层
│   └── llm_client.py       # API 客户端
├── prompts/
│   ├── host.md             # Host prompt 模板
│   └── student.md          # Student prompt 模板
├── runs/                   # 每次实验产物
├── readme.md
└── model.md
```

## 数据来源

使用 `exp0406_distil/data/generated/distillation_data.jsonl`，共 2869 条数学推理题（GSM8K + MATH），格式为标准 messages JSONL。
