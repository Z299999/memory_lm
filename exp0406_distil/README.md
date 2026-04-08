# Exp0406 - LLM Distillation Project

将 Qwen3.5-Plus 蒸馏到 0.5B 小模型的实验项目。

## 项目目标

- **教师模型**: qwen-plus (Qwen3.5-Plus)
- **学生模型**: Qwen2.5-0.5B-Instruct
- **任务**: 数学推理 (GSM8K, MATH)
- **方法**: 响应蒸馏 (Response Distillation)

## 快速开始

### 1. 安装依赖

```bash
cd exp0406_distil
pip install -r requirements.txt
```

### 2. 配置 API Key

创建 `.env` 文件：

```bash
DASHSCOPE_API_KEY="your_api_key_here"
```

### 3. 生成蒸馏数据

```bash
python scripts/generate_data.py
```

这会：
- 从 GSM8K 和 MATH 数据集采样题目
- 调用教师模型生成解题步骤
- 保存为 `data/generated/distillation_data.jsonl`

### 4. 训练学生模型

```bash
python scripts/train_student.py
```

训练后的模型保存在 `models/student/final/`

### 5. 评估

```bash
python scripts/evaluate.py
```

## 项目结构

```
exp0406_distil/
├── scripts/
│   ├── llm_client.py      # LLM API 客户端
│   ├── generate_data.py   # 生成蒸馏数据
│   ├── train_student.py   # 训练学生模型
│   └── evaluate.py        # 评估模型
├── data/
│   ├── raw/               # 原始数据（可选）
│   ├── generated/         # 教师模型生成的数据
│   └── processed/         # 处理后的数据
├── models/
│   └── student/           # 训练好的学生模型
├── results/
│   └── eval/              # 评估结果
├── config.yaml            # 配置文件
├── requirements.txt       # Python 依赖
└── README.md
```

## 配置说明

编辑 `config.yaml` 修改以下参数：

### 教师模型配置
```yaml
teacher:
  model: "qwen-plus"
  temperature: 0.2
  max_tokens: 2048
```

### 学生模型配置
```yaml
student:
  model: "Qwen/Qwen2.5-0.5B-Instruct"
  max_length: 1024
```

### 训练配置
```yaml
train:
  batch_size: 16
  learning_rate: 5e-5
  num_epochs: 5
```

## 蒸馏方法

本项目使用**响应蒸馏 (Response Distillation)**：

1. 教师模型解题 → 生成详细步骤
2. 学生模型学习模仿教师的输出
3. 损失函数：学生输出与教师输出的交叉熵

## 调试模式

生成少量数据测试流程：

```bash
python scripts/generate_data.py --debug
```

## 参考

- [Qwen2.5 Documentation](https://qwen.readthedocs.io/)
- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [GSM8K Dataset](https://huggingface.co/datasets/gsm8k)
