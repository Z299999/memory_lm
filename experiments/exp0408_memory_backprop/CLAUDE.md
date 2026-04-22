# Exp0408 Memory Backpropagation - CLAUDE.md

## 项目概述

验证基于年龄结构记忆模型的 memory backpropagation 机制，专注于 **baby case（$A_{\max}=0$）**：记忆只有一层，每轮回复完全覆盖旧记忆。

**核心角色**：
- **Host LLM**（`qwen3.5-plus` via DashScope API）：持有题目和标准答案，判断 student 作答，生成因材施教的 feedback
- **Student LLM**（`Qwen2.5-0.5B-Instruct` 本地 HuggingFace）：接收题目/feedback + 上一轮记忆，输出新回复作为新记忆
- **数据集**：`exp0406_distil/data/generated/distillation_data.jsonl`，4558 条多来源数学题（GSM8K、AIMO、WebInstructSub、NuminaMath-CoT、MMLU）

## 快速命令

```bash
cd exp0408_memory_backprop

# smoke test（不调用真实 API，验证代码流程）
python3 -c "
import sys, math; sys.path.insert(0, 'scripts')
import yaml; cfg = yaml.safe_load(open('config.yaml'))
from forward import run_forward
run_forward(data_path=__import__('pathlib').Path('../../experiments/exp0406_distil/data/generated/distillation_data.jsonl'),
    runs_dir=__import__('pathlib').Path('runs'), run_id='smoke',
    host_model=cfg['host']['model'], student_model=cfg['student']['model'],
    host_temperature=cfg['host']['temperature'], student_temperature=cfg['student']['temperature'],
    host_max_tokens=cfg['host']['max_tokens'], student_max_new_tokens=cfg['student']['max_new_tokens'],
    max_feedback_rounds=2, forward_per_back=math.inf, num_problems=2, stub=True)
"

# 正式运行（修改 run.py 里的 RUN_ID / NUM_PROBLEMS 后执行）
python3 run.py

# 生成 embedding 可视化（初始模型 step=0）
python3 scripts/visualize.py --model models/student --step 0 --out plots/step_0000

# 训练后对比可视化（step=N）
python3 scripts/visualize.py --model models/student --step N --out plots/step_00NN
```

## 文件职责

| 文件 | 职责 |
|------|------|
| `run.py` | 入口，修改 `RUN_ID` / `NUM_PROBLEMS` / `FORWARD_PER_BACK` 等参数 |
| `config.yaml` | 模型名、温度、token 数、`forward_per_back`、`max_feedback_rounds` |
| `scripts/forward.py` | Forward 主循环：student 作答 → host 判断 → feedback → 下一题 |
| `scripts/host.py` | Host LLM 调用 + 解析（EXTRACTED_ANSWER / IS_CORRECT / AGENT_INPUT 等） |
| `scripts/student_model.py` | HuggingFace 本地模型推理封装（`HFStudentModel` / `StubStudentModel`） |
| `scripts/backward.py` | Offline self-prediction 训练（当前为 stub，待实现） |
| `scripts/visualize.py` | Embedding 可视化：cosine sim、特征值谱、协方差、球面密度、t-SNE |
| `scripts/llm_client.py` | DashScope OpenAI-compat API 客户端 |
| `prompts/host.md` | Host prompt 模板 |
| `prompts/student.md` | Student prompt 模板 |
| `models/README.md` | 模型下载说明（模型本身不入 git） |

## 运行产物结构

```
runs/<run_id>/
├── transcript.md               # 完整对话记录
├── memory/
│   └── 0001-0050/m_0001.md    # 每50个文件一个子目录
├── host/
│   └── 0001-0050/h_0001_fb1.md
└── training/
    └── samples.jsonl           # (m_prev, m_next) 训练样本对
```

## 关键参数说明

| 参数 | 位置 | 说明 |
|------|------|------|
| `RUN_ID` | `run.py` | 命名格式 `r00001`，方便排序 |
| `NUM_PROBLEMS` | `run.py` | `None` = 跑完全部 2869 题 |
| `STUB` | `run.py` | `True` = 离线 fake 模型，用于测试流程 |
| `forward_per_back` | `config.yaml` | `.inf` = 不训练，只收集数据 |
| `max_feedback_rounds` | `config.yaml` | 每题最多反馈轮数，默认 3（事不过三） |

## 重要约束

1. **不要修改** `.env` 中的 API Key
2. **student model** 推理用本地 `models/student/`，训练用同一路径；模型文件不入 git，见 `models/README.md`
3. **runs/ 和 plots/ 不入 git**，有价值的 run 需在 `.gitignore` 里手动 unignore
4. **backward.py 目前是 stub**，实现时需要传入 `student` 对象（已预留接口）
5. **host 文件命名**：`h_{prob_step_id}_fb{round}.md`，同一道题的所有 feedback 共享同一 `prob_step_id`

## 常见问题

**Q: API 调用失败？**  
A: 检查 `.env` 中 `DASHSCOPE_API_KEY` 是否有效，`DASHSCOPE_BASE_URL` 是否正确。

**Q: 模型加载慢或 OOM？**  
A: `student_model.py` 使用 `device_map="auto"`，Mac 上会自动用 MPS。如果内存不足，可改为 `device_map="cpu"`。

**Q: visualize.py 出现 NaN/Inf 错误？**  
A: `_clean()` 函数会自动过滤，如仍报错检查模型权重是否完整（重新下载）。

**Q: 如何新开一次 run？**  
A: 修改 `run.py` 中的 `RUN_ID`（如 `r00002`），直接运行即可，不会覆盖旧 run。

## ⚠️ 注意

- `exp0406_distil/data/generated/distillation_data.jsonl` 是共享数据（4558条），**不要修改或删除**
- backward 训练实现后，注意保存 checkpoint 到 `models/student/` 的子目录，不要覆盖原始预训练权重
