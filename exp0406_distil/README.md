# Exp0406 - Math Data Generation

使用 LLM API 生成数学训练数据的项目。

## 项目目标

- **数据源**: GSM8K、MATH、WebInstructSub、NuminaMath-CoT 等数学数据集
- **生成方式**: 并行调用 API 生成解题步骤
- **输出格式**: JSONL，包含 user/assistant 对话格式

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

### 3. 生成数据

```bash
# 单进程生成
python run.py generate

# 多进程并行生成（推荐）
python run.py generate-parallel --workers 32

# 调试模式：生成 10 条数据测试
python run.py debug
```

生成的数据保存在 `data/generated/distillation_data.jsonl`

## 项目结构

```
exp0406_distil/
├── scripts/
│   ├── llm_client.py         # LLM API 客户端
│   ├── generate_data.py      # 数据生成脚本（单进程）
│   └── generate_parallel.py  # 数据生成脚本（多进程并行）
├── data/
│   ├── generated/            # 生成的数据
│   ├── raw/                  # 原始数据集（可选）
│   └── processed/            # 处理后的数据（可选）
├── config.yaml               # 配置文件
├── requirements.txt          # Python 依赖
├── run.py                    # 运行入口
└── README.md
```

## 配置说明

编辑 `config.yaml` 修改参数：

### 教师模型配置
```yaml
teacher:
  model: "qwen3.5-plus"
  temperature: 0.2
  max_tokens: 2048
```

### 数据集配置
```yaml
data:
  datasets:
    - name: "gsm8k"
      subset: "main"
      split: "train"
      num_samples: 2000
  advanced_datasets:
    - name: "AI-MO/aimo-validation-math-level-5"
      num_samples: 500
```

## 数据格式

生成的数据为 JSONL 格式，每行一个样本：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Mimi picked up 2 dozen seashells..."
    },
    {
      "role": "assistant",
      "content": "Here is the step-by-step solution..."
    }
  ]
}
```

## 命令行用法

```bash
python run.py <command> [options]

可用命令:
  generate          - 生成蒸馏数据（单进程）
  generate-parallel - 并行生成（支持 --workers 参数）
  status            - 查看当前状态
  debug             - 调试模式

示例:
  python run.py generate-parallel --workers 32
  python run.py status
```

## 参考

- [HuggingFace Datasets](https://huggingface.co/datasets)
- [GSM8K Dataset](https://huggingface.co/datasets/gsm8k)
- [DashScope API](https://help.aliyun.com/zh/dashscope/)
