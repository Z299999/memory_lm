# Exp0406 Distillation Project - CLAUDE.md

## 项目概述

将 Qwen3.5-Plus（教师模型）蒸馏到 Qwen2.5-0.5B（学生模型）的实验项目。

**核心目标**：
- 学生模型参数量：0.5B
- 任务领域：数学推理（GSM8K, MATH）
- 蒸馏方法：响应蒸馏（Response Distillation）

## 快速命令

```bash
cd exp0406_distil

# 测试流程（生成 10 条数据）
python run.py debug

# 完整流程
python run.py all

# 断点续训：跳过已完成的步骤
python run.py all --resume

# 分步执行
python run.py generate   # 生成数据
python run.py train      # 训练模型（自动查找最新 checkpoint 续训）
python run.py eval       # 评估
```

## 文件职责

| 文件 | 职责 |
|------|------|
| `run.py` | 统一入口，封装 generate/train/eval 流程 |
| `scripts/generate_data.py` | 调用教师模型生成蒸馏数据 |
| `scripts/train_student.py` | 用 TRL SFTTrainer 训练学生模型 |
| `scripts/evaluate.py` | 在测试集上评估准确率 |
| `scripts/llm_client.py` | DashScope API 客户端 |
| `config.yaml` | 统一配置（模型名、超参数等） |

## 重要约束

1. **不要修改** `.env` 中的 API Key 格式
2. **不要添加** wandb/tensorboard 日志（config 中已禁用）
3. **保持** 学生模型为 `Qwen/Qwen2.5-0.5B-Instruct`，除非用户明确要求更换
4. **数据格式**：JSONL，每行 `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

## 开发原则

- 脚本保持简洁，优先复用 HuggingFace/TRL 的默认行为
- 配置优先放在 `config.yaml`，不要硬编码
- 新增功能前先检查 `run.py` 是否需要更新命令入口

## 常见问题

**Q: 数据生成失败？**  
A: 检查 `.env` 中 `DASHSCOPE_API_KEY` 是否有效，网络是否通畅。

**Q: 训练显存不足？**  
A: 降低 `config.yaml` 中的 `batch_size` 或启用 `load_in_8bit`。

**Q: 评估准确率低？**  
A: 增加训练 epochs 或蒸馏数据量。

**Q: 训练中断了怎么办？**  
A: 直接运行 `python run.py all --resume` 或 `python run.py train`，会自动从最新 checkpoint 继续训练。

## ⚠️ 重要警告

**蒸馏数据是最宝贵的资产！**

- `data/generated/distillation_data.jsonl` 包含使用昂贵 token 生成的训练数据
- **绝对不要** 覆盖或删除此文件
- 合并分支时，保留样本数最多的版本
- 每次 commit 前检查样本数量是否减少
- 生成进程可随时停止/重启，脚本会自动跳过已存在的问题
