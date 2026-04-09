# Exp0406 Distillation Project - CLAUDE.md

## 项目概述

使用 LLM API 生成数学训练数据的项目。

**核心目标**：
- 数据源：GSM8K, MATH, WebInstructSub, NuminaMath-CoT 等
- 生成方式：并行调用 Qwen3.5-Plus API 生成解题步骤
- 输出格式：JSONL（user/assistant 对话格式）

## 快速命令

```bash
cd exp0406_distil

# 查看状态
python run.py status

# 生成数据（单进程）
python run.py generate

# 生成数据（多进程并行，推荐）
python run.py generate-parallel --workers 32

# 调试模式：生成 10 条数据测试
python run.py debug
```

## 文件职责

| 文件 | 职责 |
|------|------|
| `run.py` | 统一入口，封装 generate/parallel/status 命令 |
| `scripts/generate_data.py` | 单进程数据生成 |
| `scripts/generate_parallel.py` | 多进程并行数据生成（ThreadPoolExecutor） |
| `scripts/llm_client.py` | DashScope API 客户端 |
| `config.yaml` | 配置文件（数据集、API 参数） |
| `data/generated/` | 生成的蒸馏数据 |

## 重要约束

1. **不要修改** `.env` 中的 API Key 格式
2. **数据格式**：JSONL，每行 `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
3. **生成脚本支持断点续跑**：会自动跳过已存在的问题

## ⚠️ 重要警告

**蒸馏数据是最宝贵的资产！**

- `data/generated/distillation_data.jsonl` 包含使用昂贵 token 生成的训练数据
- **绝对不要** 覆盖或删除此文件
- 合并分支时，保留样本数最多的版本
- 每次 commit 前检查样本数量是否减少
- 生成进程可随时停止/重启，脚本会自动跳过已存在的问题

## 数据来源

| 数据集 | 类型 | 目标数量 |
|--------|------|---------|
| GSM8K | 小学到初中应用题 | 2000 |
| AIMO Math Level 5 | 高中竞赛数学 | 500 |
| WebInstructSub | 大学数学综合 | 1000 |
| NuminaMath-CoT | 数学证明题 | 1000 |
| MMLU college_mathematics | 大学数学 | 200 |
| MMLU abstract_algebra | 抽象代数 | 200 |
| **总计** | | **~4900** |
