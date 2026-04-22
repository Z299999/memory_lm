# exp0409 — 复现 Attention Is All You Need (Transformer, 2017)

本实验从零用 PyTorch 复现 Vaswani et al. 2017 的 Transformer 架构，忠实还原论文中每个模块的设计。
不进行大规模训练（无算力），目标是：实现架构 → 跑通小数据验证 → 为后续微调预训练模型打好基础。

原始论文：[`literature/pdf_papers/b00002_attention-is-all-you-need`](../literature/pdf_papers/b00002_attention-is-all-you-need/)

---

## 实验目标

- 逐模块实现论文架构（Scaled Dot-Product Attention、Multi-Head Attention、Encoder/Decoder Stack 等）
- 通过 shape 测试和玩具数据集验证实现正确性
- 训练循环支持论文原版超参数（Adam + warmup schedule、label smoothing）
- 数据管道使用 HuggingFace `datasets`（opus-100 中英子集），避免手动下载

---

## 项目结构

```
exp0409_atten2017_implement/
├── readme.md                   # 本文件
├── requirements.txt            # 依赖：torch, datasets, tokenizers, sacrebleu
├── config.py                   # 所有超参数（d_model, N, h, d_ff 等）
├── model/
│   ├── __init__.py
│   ├── attention.py            # ScaledDotProductAttention, MultiHeadAttention
│   ├── ffn.py                  # PositionwiseFFN（两层线性 + ReLU）
│   ├── positional_encoding.py  # 正弦/余弦位置编码
│   ├── encoder.py              # EncoderLayer × N，EncoderStack
│   ├── decoder.py              # DecoderLayer × N，DecoderStack
│   └── transformer.py          # 完整 Transformer，共享 embedding
├── data/
│   └── dataset.py              # 加载 opus-100 zh-en，BPE tokenizer
├── train.py                    # 训练循环（含 warmup lr、label smoothing）
└── evaluate.py                 # BLEU 评估，beam search 解码
```

---

## 模型架构（论文原版参数）

| 组件 | 参数 |
|------|------|
| Encoder / Decoder 层数 | N = 6 |
| 模型维度 | d_model = 512 |
| 注意力头数 | h = 8，d_k = d_v = 64 |
| FFN 内层维度 | d_ff = 2048 |
| 位置编码 | 正弦/余弦，固定，不可学习 |
| Dropout | 0.1（子层输出 + embedding） |
| 共享权重 | source emb = target emb = pre-softmax linear |

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

**Multi-Head Attention:**

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**Position-wise FFN:**

```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

**Positional Encoding:**

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

---

## 训练配置（论文原版）

| 参数 | 值 |
|------|----|
| 优化器 | Adam，β1=0.9，β2=0.98，ε=1e-9 |
| 学习率调度 | warmup_steps=4000，`lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))` |
| Label Smoothing | ε_ls = 0.1 |
| 数据集（本地验证用） | HuggingFace `Helsinki-NLP/opus-100`，zh-en，取前 10 万句对 |

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 验证模型（shape 测试，无需数据）
python -c "from model.transformer import Transformer; from config import Config; m = Transformer(Config()); print('OK')"

# 小数据集过拟合测试（验证训练循环）
python train.py --toy --steps 500

# 正式训练（opus-100 子集，单卡或 CPU）
python train.py --data zh-en --max_samples 100000

# 评估 BLEU
python evaluate.py --checkpoint runs/latest/checkpoint.pt
```

---

## 配置说明

`config.py` 中的 `Config` 类包含所有超参数，可通过命令行或直接修改覆盖：

```python
@dataclass
class Config:
    # 模型
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 37000      # BPE 词表大小

    # 训练
    warmup_steps: int = 4000
    batch_size: int = 32
    max_steps: int = 100000
    label_smoothing: float = 0.1

    # 数据
    data_lang_pair: str = "zh-en"
    max_samples: int = 100000    # 截取子集大小
```

---

## 验证方法

1. **Shape 测试**：对每个模块输入 dummy tensor，断言输出 shape 正确
2. **前向传播测试**：完整模型在随机 batch 上运行，无报错
3. **玩具数据过拟合**：100 句对训练 500 步，loss 应从 ~8 降至 ~0.5 以下
4. **Attention 可视化**：对样本句子画 attention weight 热图，验证注意力分布合理

---

## 参考资料

- 原始论文：Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NIPS 2017
- 教学实现参考：[The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/)
- 预训练模型（后续微调用）：`Helsinki-NLP/opus-mt-zh-en`
- 数据集：`Helsinki-NLP/opus-100`（HuggingFace）
