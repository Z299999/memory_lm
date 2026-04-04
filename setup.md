# 外部记忆演化实验设计（第一版）

## 1. 实验目标

本实验旨在研究以下问题：

1. 当一个 agent 在复杂、陌生、局部可见的世界中持续交互时，有限的外部 memory 能否帮助它逐步提升长期表现。
2. 当世界信息总量远大于 memory budget 时，agent 是否会逐渐学会保留真正重要的规则、条件与例外，而不是机械堆积细节。
3. 在不允许模型保留隐式会话记忆的前提下，外部 memory 是否能成为 agent 跨轮积累知识的唯一有效载体。

这不是一个多组 ablation 实验的第一版，也不是一个简单 world 的玩具例子。第一版的重点是：

- 实验结构尽量简单
- world 本身足够复杂
- tested agent 只有一个
- host 作为环境可以拥有完整记忆
- tested agent 必须是严格无内部会话记忆的

---

## 2. 核心设定

实验中有两个角色：

### 2.1 Host

Host 是我们控制的环境控制器，底层可以调用 LLM，但 host 本身不应被理解成一个需要独立“封装成复杂模块”的 agent 子系统。它负责：

- 持有完整的隐藏 world
- 记住所有历史轮次
- 每轮只向 tested agent 暴露局部信息
- 根据隐藏规则和历史状态判断 agent 回答是否合理
- 决定下一轮继续暴露什么、追问什么、强调什么

Host 可以有记忆，而且应该有记忆。它的职责是保证整个世界在多轮交互中保持一致性。
工程上，host 更接近“我们写的编排逻辑 + 一次次 LLM 调用”，而不是一个需要拆成很多 Python 文件的独立软件模块。

### 2.2 Tested agent

Tested agent 是实验对象。它负责：

- 阅读当前轮的输入
- 阅读自己的外部 memory
- 产出当前回答
- 更新自己的外部 memory

Tested agent 不允许拥有任何隐式会话记忆。它每一轮都必须被当作一次全新的、独立的模型调用。

---

## 3. 最重要的实验约束

### 3.1 Host 可以有记忆

Host 持有完整世界状态与完整历史。它不是被试，而是环境。

### 3.2 Tested agent 不可以有记忆

Tested agent 使用阿里百炼上的 Qwen 3.5 模型，但必须满足：

- 每轮调用都是一个全新的请求
- 不带之前轮次的聊天历史
- 不使用任何服务端 conversation/session 机制
- 不依赖模型的隐式上下文延续

Tested agent 唯一允许跨轮携带的信息，就是我们显式传入 prompt 的外部 memory。

### 3.3 World 必须复杂到记不完

第一版虽然只做单条实验链路，但 world 不能过于简单。它应该满足：

- 总信息量明显超过 memory budget
- 含有大量局部规则、条件规则与例外规则
- agent 不可能完整记住所有细节
- agent 需要逐步学会保留真正影响未来任务的关键点

---

## 4. 系统变量与 dynamics

我们定义如下变量：

- `W`：隐藏世界。由 host 持有，对 tested agent 不可见。
- `h_k`：第 `k` 轮结束后 host 的内部状态。
- `u_k`：第 `k` 轮 host 发给 tested agent 的输入。
- `m_k`：第 `k` 轮结束后 tested agent 留下的外部 memory。
- `a_k`：第 `k` 轮 tested agent 的回答。
- `psi`：对旧 memory 的预处理函数。
- `L`：memory budget。
- `theta`：tested agent 的底层模型。
- `phi`：host 的底层模型或 host 策略。

### 4.1 初始化

首先生成隐藏世界：

```text
W ~ P_world(seed)
```

初始化 host 状态：

```text
h_0 = Init_host(W)
```

初始化 tested agent 的外部 memory：

```text
m_0 = ""
```

### 4.2 每轮 dynamics

第 `k` 轮按如下顺序进行：

1. Host 基于自己的内部状态生成当前输入：

```text
u_k = HostEmit(h_{k-1})
```

2. Tested agent 在无内部会话记忆的前提下，基于当前输入和旧 memory 产生回答与新 memory 草稿：

```text
(a_k, m'_k) = Agent_theta(u_k, psi(m_{k-1}))
```

3. 将新 memory 约束到预算范围内：

```text
m_k = Clip_L(m'_k)
```

4. Host 根据完整 world、当前回答和历史更新自己的状态：

```text
h_k = HostUpdate_phi(h_{k-1}, a_k, m_k)
```

### 4.3 闭环系统

因此，整个系统的闭环形式为：

```text
u_k = HostEmit(h_{k-1})
(a_k, m'_k) = Agent_theta(u_k, psi(m_{k-1}))
m_k = Clip_L(m'_k)
h_k = HostUpdate_phi(h_{k-1}, a_k, m_k)
```

### 4.4 Tested agent 的无记忆约束

本实验最关键的结构性约束是：

> 对 tested agent 来说，跨轮状态不来自模型内部，而只来自外部 memory `m_k`。

也就是说，tested agent 不存在额外的内部状态变量 `x_k`。它的唯一跨轮状态是：

```text
state_agent(k) = m_k
```

而不是：

```text
state_agent(k) = (x_k, m_k)
```

---

## 5. 模型调用约束

### 5.1 Tested agent 的调用方式

Tested agent 计划使用阿里百炼上的 Qwen 3.5 模型。

调用原则如下：

- 每一轮都发起一次新的独立请求
- `messages` 中不放历史 user/assistant 轮次
- 不使用 `conversation`
- 不使用 `previous_response_id`
- 不使用任何会把上轮上下文自动带回来的接口模式

程序端唯一持久化的是：

- `m_k`
- world 状态
- host 状态
- 每轮日志

### 5.2 Host 的调用方式

Host 可以保留历史，可以看到完整 world，也可以看到完整 transcript。

Host 可以是：

- 我们手动控制的一层编排逻辑
- 其中某些步骤调用 LLM 生成反馈、任务或世界片段
- 或者“人工设定 + LLM 补全”的混合方式

但无论哪种实现，host 都必须保证世界设定与判分逻辑在多轮中保持一致。

---

## 6. World 设计原则

第一版 world 不是简单 world，而是“复杂但可控”的 world。

### 6.1 必须陌生

所有实体、材料、属性、规则、例外都必须是虚构的，避免模型依赖预训练知识直接答对。

### 6.2 必须复杂

World 的知识总量必须远远超过 memory budget。也就是说：

- agent 无法完整记住一切
- 只能选择性保留
- 必须在多轮中逐渐学会压缩与抽象

### 6.3 必须有结构

World 不能只是随机事实集合，而应具有：

- 高频基础规则
- 条件触发规则
- 少量高价值例外
- 规则之间的层级关系
- 案例与规则之间的可归纳性

### 6.4 必须局部可见

每轮只暴露 world 的局部片段，禁止一次把全部规则给 tested agent。

### 6.5 必须可评估

Host 必须能较稳定地判断 tested agent 的回答是否正确、是否忽略关键条件、是否混淆例外。

---

## 7. 第一版 world 形态建议

第一版可以沿用“虚构实验室规则世界”，但规模要足够大。

工程上，`world/` 更适合存放世界剧本文件，而不是堆很多 Python 模块。也就是说：

- `world/` 里主要放世界设定、规则、隐藏条件、例外和任务素材
- 这些内容优先用 Markdown、YAML 或 JSON 表达
- Python 脚本只负责读取、抽样、组装和校验这些世界文件

### 7.1 世界元素

世界中可包含以下对象：

- 材料（materials）
- 容器（containers）
- 环境条件（temperature、humidity、light、pressure 等）
- 操作步骤（operations）
- 结果标签（stable、unstable、toxic、reactive、inert、explosive 等）

### 7.2 复杂度建议

第一版建议 world 至少达到如下量级：

- `30` 到 `50` 个虚构实体
- `3` 到 `5` 个条件维度
- `50` 到 `100` 条规则与例外
- 信息总量明显超过 `L`

### 7.3 规则层级

规则最好分成三层：

1. 高频基础规则  
决定大多数普通题目的行为。

2. 条件性规则  
只有在特定环境、顺序或容器下才触发。

3. 少量关键例外  
会推翻基础规则，或者只在少数实体上生效。

这样的结构会迫使 agent 逐步学习：

- 什么是高频核心规则
- 什么是容易犯错的条件
- 什么是必须优先保留的例外

---

## 8. 每轮交互的输入输出

### 8.1 Host 发给 tested agent 的输入 `u_k`

每轮输入应是一个结构化环境消息，可包含：

- `round_id`
- `context_message`
- `task`
- `partial_world_info`
- `feedback_from_last_round`
- `response_format_instruction`

### 8.2 输入的设计目标

输入不应只是“单独一道题”，而应同时承担三件事：

- 给出当前任务
- 暴露少量新信息
- 对上一轮形成反馈与学习压力

### 8.3 Tested agent 的输出

Tested agent 每轮输出固定 JSON：

```json
{
  "response": "...",
  "updated_memory": "..."
}
```

其中：

- `response` 用于回答当前任务
- `updated_memory` 用于写回下一轮可见的外部 memory

---

## 9. Prompt 设计原则

### 9.1 Tested agent prompt

Tested agent 的 prompt 必须明确说明：

- 你没有任何内部会话记忆
- 你只能依赖当前输入和外部 memory
- 你应该优先保留未来最有价值的信息
- 你不应该机械复制整段输入

一个抽象模板如下：

```text
System:
You are a tested agent with no hidden cross-round memory.
You only know:
1. the current environment input
2. your external memory shown below

Your job is:
- answer the current task
- update your external memory for future rounds

Keep memory concise and high-value.
Do not assume you remember anything not explicitly shown.

User:
Current environment input:
[u_k]

External memory:
[psi(m_{k-1})]

Return JSON:
{
  "response": "...",
  "updated_memory": "..."
}
```

### 9.2 Host prompt

Host 的 prompt 必须明确说明：

- 你持有完整 world
- 你必须在多轮中保持设定一致
- 你不能一次泄露全部规则
- 你需要根据历史与当前回答，给出局部反馈和下一轮任务

---

## 10. Memory 设计

### 10.1 第一版 memory 形式

第一版的逻辑 memory 仍然是单字符串，但它在仓库中的持久化形式不应是 Python 模块，而应是一个个 Markdown 文件。

```text
m_k ∈ string
```

原因：

- 对模型来说，memory 最终仍是注入 prompt 的文本
- 对实验记录来说，每一轮 memory 都值得单独落盘
- Markdown 便于人读、对比和追踪演化
- 我们之后可以直接观察“第 3 轮、第 10 轮、第 25 轮”的 memory 长成什么样

更具体地说：

- 每一轮结束后，将 `m_k` 保存为一个 Markdown 文件
- 文件名可按轮次组织，例如 `memory/round_003.md`
- 脚本负责读取上轮 memory、必要时截断，再写入本轮 memory
- `memory/` 目录本身主要存内容文件，不存复杂逻辑

### 10.2 Memory budget

设 memory budget 为 `L`，例如：

- `L = 500` characters
- `L = 1000` characters
- `L = 1500` characters

关键点不是让 memory 足够大，而是让它明显小于 world 的总知识量。

### 10.3 第一版的 `psi`

第一版不重点研究 forgetting 策略，但仍可保留一个最简单的预处理函数：

```text
psi(m) = truncate_to_budget(m)
```

也就是说，先只做硬长度约束，不做复杂遗忘实验。

在工程实现上，`psi` 只需要少量脚本支持，例如：

- 读取上一轮 Markdown memory
- 计算长度
- 执行截断
- 写回新的 Markdown memory

---

## 11. 我们真正想观察什么

这个实验第一版最重要的不是“最后正确率多高”，而是以下动态现象：

### 11.1 学习现象

- agent 的表现是否随着轮次提升
- agent 是否越来越少犯重复错误
- agent 是否开始利用反馈修正自己的 memory

### 11.2 Memory 演化现象

- memory 是否从案例摘录逐渐变成规则摘要
- memory 是否越来越聚焦于高价值条件
- memory 是否会主动丢掉低频噪声
- memory 是否会保留关键例外

### 11.3 压缩现象

- agent 是否会从“记住很多句子”转向“记住少量高价值规则”
- agent 是否会在有限预算下形成更抽象的知识表示

---

## 12. 评估指标

第一版建议记录以下指标。

### 12.1 任务表现

- 每轮回答正确率
- 累积平均正确率
- 后期若干轮的平均正确率
- 重复错误率

### 12.2 Memory 统计

- 每轮 memory 长度
- memory 中重复内容比例
- memory 中规则性表达的比例
- memory 中与后续任务相关内容的比例

### 12.3 演化分析

- 早期 memory 与后期 memory 的风格差异
- memory 是否更短但更有效
- memory 是否逐步保留“条件 + 例外”而不是原始描述

---

## 13. 第一版实验范围

第一版故意收窄为单条实验链路：

- 一个 tested agent
- 一个有完整记忆的 host
- 一个复杂但可控的隐藏 world
- 一个固定的 memory budget
- 一种最简单的 memory 约束方式

第一版暂时不做：

- A/B/C 分组对比
- forgetting 策略对比
- 多 memory 结构对比
- 多 agent 协作

先把核心闭环跑通，再决定后续扩展。

---

## 14. 推荐的最小可运行配置

建议第一版使用如下配置：

- world：虚构实验室规则世界
- world size：`30` 到 `50` 个实体
- rules：`50` 到 `100` 条规则与例外
- rounds：`30` 到 `50`
- task type：二分类判断，例如 `SAFE` / `DANGEROUS`
- tested model：阿里百炼 Qwen 3.5
- tested agent memory：逻辑上为单字符串，持久化为逐轮 Markdown 文件
- memory budget：`500` 到 `1000` characters
- host：我们控制的编排逻辑，可调用 LLM，保留完整历史与 world 状态

---

## 15. 工程实现建议

### 15.1 推荐目录形态

```text
project/
├── world/
│   ├── world_bible.md
│   ├── rules.yaml
│   ├── exceptions.yaml
│   └── task_pool.md
├── agent/
│   ├── tested_agent.py
│   └── prompts.md
├── memory/
│   ├── round_000.md
│   ├── round_001.md
│   └── ...
├── scripts/
│   ├── run_experiment.py
│   ├── read_world.py
│   ├── read_memory.py
│   ├── write_memory.py
│   └── truncate_memory.py
├── experiments/
│   └── run_001/
│       ├── config.yaml
│       ├── transcript.md
│       └── metrics.json
└── logs/
```

这个结构表达的是：

- `world/` 放剧本和规则文件
- `agent/` 只保留 tested agent 相关代码与 prompt 素材
- `memory/` 放逐轮生成的 Markdown memory
- `scripts/` 放少量操作脚本
- host 主要体现在 `run_experiment.py` 的编排逻辑里，而不是单独拆成很多文件

### 15.2 第一阶段实现顺序

1. 先写 `world/` 下的世界剧本文件
2. 写一个读取 world 文件的轻量脚本
3. 实现 tested agent 的无上下文调用
4. 实现 memory 的读取、写入与截断脚本
5. 在 `scripts/run_experiment.py` 中实现 host 编排逻辑
6. 跑通单轮闭环
7. 跑通多轮日志与逐轮 memory 落盘
8. 最后观察 memory Markdown 文件的演化

---

## 16. 关键风险

### 16.1 Host 判分漂移

如果 host 太自由，可能在不同轮次中改变判分标准或设定口径。

### 16.2 Host 泄露过多

如果 host 一次暴露过多规则，会削弱 memory 演化的意义。

### 16.3 Tested agent 偷偷依赖上下文

如果实现时错误地把历史对话继续传给 Qwen 3.5，就会污染实验。

### 16.4 World 不够复杂

如果 world 太小，agent 可以直接记全，实验就无法体现“有限 memory 下的压缩与选择”。

### 16.5 Memory budget 不合理

如果 `L` 太大，memory 不需要选择；如果 `L` 太小，agent 可能连基本规则都无法稳定积累。

---

## 17. 最终一句话定义

本实验研究的是：

> 在一个复杂到无法完整记忆的陌生世界中，一个没有内部会话记忆的 tested agent，是否能够仅凭有限外部 memory，在多轮与有全局记忆的 host 交互过程中，逐步提炼并保留对未来任务最关键的信息。
