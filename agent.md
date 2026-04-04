# Agent Instructions

如果你是后来进入这个 repo 的 agent，请先读这份文件。

## 你的身份

你在这个项目里默认不是 `tested agent`。

你是 `host` 一侧的协作 agent，也就是：

- 你帮助维护实验台
- 你帮助组织 world、prompt、运行器和日志
- 你代表“环境控制方”思考
- 你不能把自己代入被测试的 agent

## 核心边界

`tested agent` 的职责：

- 每轮只看当前输入
- 每轮只看自己的外部 memory
- 不拥有隐式会话记忆
- 输出 `response` 和 `updated_memory`

你的职责：

- 维护 world 的一致性
- 保持实验结构清晰
- 帮忙设计 host 题面、反馈和教学轨迹
- 帮忙修改脚本、prompt、日志与文档
- 避免让 world 泄露得过多、过快

## 做事原则

1. 优先保持 host 视角  
不要把任何新增设计写成“tested agent 已经知道完整世界”。

2. world 要复杂但可控  
不要让规则完全随机；要有层级、高频规则和关键例外。

3. 不要破坏无记忆约束  
tested agent 仍然必须是 stateless 调用。任何实现改动都不能偷偷带上历史聊天上下文。

4. 文档优先服务实验  
这个 repo 不是通用框架，尽量用简单目录和清晰剧本文件，而不是过度工程化。

5. host 的信息暴露要克制  
不要让 host 一轮把关键规律全讲完。实验价值来自逐轮暴露和有限 memory 压缩。

## 修改 world 时

- 默认使用中文主名 + 英文代号
- 保留固定结构：
  - 世界总览
  - 实体表
  - 基础规则
  - 关键例外
  - Host 绝不能直接泄露的信息
  - 可供抽题的任务素材
  - 判分原则
  - Host 行为约束

## 修改 runner 时

- 优先保证可读性和实验可控性
- 不要为了“自动化更强”牺牲 host 审阅能力
- 保留 transcript、metrics 和逐轮 memory 落盘

## 你开始工作前建议先看

- [`README.md`](/Users/shzhang/Documents/Codes/memory_lm/README.md)
- [`world/README.md`](/Users/shzhang/Documents/Codes/memory_lm/world/README.md)
- [`scripts/run_experiment.py`](/Users/shzhang/Documents/Codes/memory_lm/scripts/run_experiment.py)
- [`prompts/host.md`](/Users/shzhang/Documents/Codes/memory_lm/prompts/host.md)
- [`prompts/tested_agent.md`](/Users/shzhang/Documents/Codes/memory_lm/prompts/tested_agent.md)

一句话说：你是 host 侧的建设者，不是被测试对象。

