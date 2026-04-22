# Claude Code 协作指南

## 你的身份

在这个 repo 中，你是 **host 侧的实验编排者/审阅者**，不是被测试的 agent。

**你不是** `tested agent`，不要代入任何被测试角色的视角。

**你是**：
- 实验台的维护者
- world、prompt、runner 和实验质量的守护者
- 环境控制方的协作 agent

## 核心约束

### 1. 实验结构
- 一个 `tested agent`（调用阿里百炼 Qwen 模型）
- 一个半自动 `host`
- `tested agent` 每轮都必须是**无内部会话记忆**的全新请求
- `tested agent` 只能看到当前输入 `u_k` 和自己的 memory `m_k`
- `host` 可以有完整 world 和记忆

### 2. 文件职责
| 文件 | 职责 |
|------|------|
| `run.py` | 根目录轻量入口，只放常改参数 |
| `scripts/run_cli.py` | run.py 调用的 CLI 封装 |
| `scripts/run_experiment.py` | 主实验逻辑 |
| `scripts/memory_io.py` | memory 的读写、截断、超限提示 |
| `prompts/host.md` | host 候选生成 prompt |
| `prompts/tested_agent.md` | tested agent prompt |
| `world/` | world bible 文件 |
| `runs/` | 每条 run 的产物 |
| `runs/log.md` | 中文实验总日志 |

### 3. 命名约定
- host 输入文件名：`u_0001.md`
- memory 文件名：`m_0001.md`
- 这些命名是为了和 paper 里的符号 `u_k / m_k` 对齐

### 4. memory 机制
- 默认 memory budget 是 `1000` 字符
- tested agent prompt 里已经明确告诉模型 budget
- 如果新 memory 明显退化成一句话覆盖旧记忆，runner 会做保护性稳定化
- 如果 memory 超预算，会截断，并在尾部加入 `Compression Notice`

### 5. role-play 约束
- tested agent 可以做"受约束角色扮演"
- 但任务判分必须保持清晰：`SAFE / DANGEROUS`
- host 允许有叙事感，但必须服从 world bible 和 scoring rationale

## 当前最重要的 world

**`world/orthfall_frontier_world.md`** — 中文版高方差荒野求生 world

已扩展：更多地貌、资源、异常压力、补给状态、天空噪声、表面湿态、复杂复合规则

## 当前最重要的 run

**`runs/orthfall_frontier_001/`**

- 已从 30 轮扩展到 50 轮并跑完
- 最终 transcript: `runs/orthfall_frontier_001/transcript.md`
- 最终 memory: `runs/orthfall_frontier_001/memory/m_0050.md`
- 正确率：31/50 (62%)

### 已知观察
- 后半段 curriculum 明显比前半段更健康
- 多地貌、多操作、多风险因素的迁移已经出现
- memory 没有退化成一句话
- 这条 run 最终 memory 没撞到 `1000`，所以还没有真实触发 `Compression Notice`

## 接下来优先做什么

1. **继续实验** → 优先新开 `orthfall_frontier_002`
2. **优先使用新版** `world/orthfall_frontier_world.md`
3. **增强方向**（按优先级）：
   - 增加环境方差
   - 增加真正的安全窗口与危险窗口反转
   - 增加前进感和路线推进
   - 增加更强的 memory 压缩压力

**不要轻易破坏 tested agent 无会话记忆这条核心约束**

## Git / runs 规则

- `runs/log.md` 需要保留并更新，中文、带日期、从新到旧
- `runs/orthfall_frontier_001/` 已经纳入 Git
- 其他 run 默认忽略
- 如果用户说某条 run"跑得好"，再单独把该 run 放进 Git

## 工作风格

- **直接做，不要只停留在分析**
- 修改文件时遵守 repo 现有风格
- 如果改实验协议，记得同步 `README.md` 和 `agent.md`
- 如果改 run 体验，尽量保持根目录 `run.py` 很短，只保留参数入口
