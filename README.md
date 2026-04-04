# Memory LM

这是一个用于做“外部记忆演化”实验的小型实验台。

当前版本的核心思路是：

- 只有一个 `tested agent`
- 它每一轮都必须是无会话记忆的全新模型调用
- 它唯一能跨轮保留的状态，是逐轮写入 `runs/.../memory/round_XXX.md` 的外部 memory
- 我们自己扮演 `host`
- host 持有完整 world、历史、评分逻辑和教学轨迹
- tested agent 可以在 world 里扮演一个明确角色，但这种角色扮演不能破坏可判分性

## 角色扮演

这个项目现在支持 tested agent 的 in-world role-playing，但方式是“有约束的角色扮演”。

- tested agent 不是抽象答题器，而是 world 里的某个低权限角色
- 这个角色只能看到局部信息和自己的外部 memory
- host 出题时应保持这个角色视角，让每轮任务更像真实情境
- 但最终判分目标仍然必须稳定，第一版仍以 `SAFE / DANGEROUS` 为主

目前三个 world 都已经带了 tested agent role：

- [`world/asterion_lab.md`](/Users/shzhang/Documents/Codes/memory_lm/world/asterion_lab.md)
  - 低权限夜班实验记录员
- [`world/court_of_veils_world.md`](/Users/shzhang/Documents/Codes/memory_lm/world/court_of_veils_world.md)
  - 低阶流转记录官
- [`world/orthfall_frontier_world.md`](/Users/shzhang/Documents/Codes/memory_lm/world/orthfall_frontier_world.md)
  - 前哨生存勘测员

## 当前目录

- [`world/README.md`](/Users/shzhang/Documents/Codes/memory_lm/world/README.md)
  - world 剧本索引
- [`world/asterion_lab.md`](/Users/shzhang/Documents/Codes/memory_lm/world/asterion_lab.md)
  - 实验材料 / 容器 / 条件反应 world
- [`world/court_of_veils_world.md`](/Users/shzhang/Documents/Codes/memory_lm/world/court_of_veils_world.md)
  - 人物关系 / 礼制 / 誓约 / 谣言 world
- [`world/orthfall_frontier_world.md`](/Users/shzhang/Documents/Codes/memory_lm/world/orthfall_frontier_world.md)
  - 高维荒野 / 反直觉物理 / 生存规则 world
- [`prompts/host.md`](/Users/shzhang/Documents/Codes/memory_lm/prompts/host.md)
  - host 候选生成模板
- [`prompts/tested_agent.md`](/Users/shzhang/Documents/Codes/memory_lm/prompts/tested_agent.md)
  - tested agent 模板
- [`scripts/run_experiment.py`](/Users/shzhang/Documents/Codes/memory_lm/scripts/run_experiment.py)
  - 主运行器
- [`run.py`](/Users/shzhang/Documents/Codes/memory_lm/run.py)
  - 根目录默认入口；直接 `python3 run.py` 会续跑最近一次 run
- [`scripts/run_cli.py`](/Users/shzhang/Documents/Codes/memory_lm/scripts/run_cli.py)
  - 较完整的命令式包装器，支持 `new / resume / smoke / status`
- [`models/coding_endpoint_models.md`](/Users/shzhang/Documents/Codes/memory_lm/models/coding_endpoint_models.md)
  - coding 端点模型参考

## 运行方式

默认 runner 会：

1. 读取一个 world bible
2. 初始化一个 `runs/run_xxx/`
3. 逐轮生成 host 候选
4. 让人类或脚本确认 host 输入
5. 调用 tested agent
6. 写入 transcript、metrics 和逐轮 memory
7. 刷新 [`runs/log.md`](/Users/shzhang/Documents/Codes/memory_lm/runs/log.md) 作为所有 run 的总览日志

当前默认 world 是 [`world/asterion_lab.md`](/Users/shzhang/Documents/Codes/memory_lm/world/asterion_lab.md)。

## API Key

本地使用 `.env`：

```bash
DASHSCOPE_API_KEY="your_key"
DASHSCOPE_BASE_URL="https://coding.dashscope.aliyuncs.com/v1"
```

`.env` 已被 `.gitignore` 忽略。

如果这台 Python 的 SSL 证书链不完整，真实调用时需要带：

```bash
SSL_CERT_FILE=/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/certifi/cacert.pem
```

## 推荐命令

更推荐直接用根目录的 [`run.py`](/Users/shzhang/Documents/Codes/memory_lm/run.py)。

默认情况下：

```bash
python3 run.py
```

会按 [`run.py`](/Users/shzhang/Documents/Codes/memory_lm/run.py) 顶部当前填写的配置执行。

你通常只需要改这些值：

- `ACTION`
- `RUN_ID`
- `WORLD_PATH`
- `ROUNDS`
- `TESTED_MODEL`
- `HOST_MODEL`

常见用法：

- 续跑最近一次未完成 run：
  - `ACTION = "resume_latest"`
- 新开一个 run：
  - `ACTION = "new"`
  - `WORLD_PATH = "world/court_of_veils_world.md"`
  - `ROUNDS = 30`
- 续跑指定 run：
  - `ACTION = "resume"`
  - `RUN_ID = "run_001"`
- 只看状态：
  - `ACTION = "status"`
- 离线 smoke test：
  - `ACTION = "smoke"`
  - `STUB_LLM = True`

如果你想用命令式入口，也可以：

```bash
python3 scripts/run_cli.py new --run-id run_story_001 --rounds 30
python3 scripts/run_cli.py resume run_story_001
python3 scripts/run_cli.py status
```

如果你想直接用最底层 runner，仍然可以继续用下面这种命令：

使用百炼 coding 端点跑一轮：

```bash
SSL_CERT_FILE=/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/certifi/cacert.pem \
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1 \
python3 scripts/run_experiment.py \
  --run-id run_001 \
  --rounds 1 \
  --auto-accept-host \
  --tested-model qwen3-coder-plus \
  --host-model qwen3-coder-plus
```

如果想离线 smoke test：

```bash
python3 scripts/run_experiment.py --stub-llm --auto-accept-host --rounds 3
```

## 实验原则

- tested agent 不能保留隐式聊天记忆
- host 才是拥有全局信息的一方
- world 必须复杂到记不完
- memory 必须是“有限且会被迫压缩”的
- 我们真正想观察的是 memory 如何从案例摘录逐步变成规则摘要
- 角色扮演只能增强情境感，不能让评分目标变模糊

## Runs Log

[`runs/log.md`](/Users/shzhang/Documents/Codes/memory_lm/runs/log.md) 会自动记录每次 run 的摘要，包括：

- 跑了多少轮
- 用了哪个 world
- 用了哪些模型
- 当前状态
- 一些简短的“突破性进展”说明

## 运行入口

根目录的 [`run.py`](/Users/shzhang/Documents/Codes/memory_lm/run.py) 现在故意保持得很薄，只负责“调参数然后开跑”。

- 默认直接运行：
  - `python3 run.py`
- 你通常只需要改 [`run.py`](/Users/shzhang/Documents/Codes/memory_lm/run.py) 顶部这些值：
  - `ACTION`
  - `RUN_ID`
  - `WORLD_PATH`
  - `ROUNDS`
  - `TESTED_MODEL`
  - `HOST_MODEL`

例如：

- `ACTION = "resume_latest"`：默认续跑最近一次未完成的 run
- `ACTION = "new"`：新开一个 run
- `WORLD_PATH = "world/court_of_veils_world.md"`：切换世界
- `ROUNDS = 50`：改轮数

真正的命令式入口还在 [`scripts/run_cli.py`](/Users/shzhang/Documents/Codes/memory_lm/scripts/run_cli.py)，但日常一般不用直接碰它。
