# 运行日志

- 更新时间: 2026-04-04T07:04:30+00:00
- 已记录运行数: 13

这个文件记录每次 run 跑了什么、跑了多少轮，以及有哪些值得记下来的进展。
排列顺序按时间从新到旧。

## confluence_002

- 开始时间: `2026-04-04T06:53:45+00:00`
- 结束时间: `尚未结束`
- 状态: `in_progress`
- 世界: `world/confluence_world.md`
- tested model: `qwen3-coder-plus`
- host model: `qwen3-coder-plus`
- 计划轮数: `50`
- 已完成轮数: `36`
- 正确率: `28/36` (78%)
- memory 进展: `已完成 36 轮，memory 长度从 213 变到 606 字符`
- transcript: [`confluence_002/transcript.md`](./confluence_002/transcript.md)
- metrics: [`confluence_002/metrics.json`](./confluence_002/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 已经不是一次性连通性测试，而是形成了稳定的多轮连续运行。
- 已经进入较长程运行区间，开始适合观察 memory 的演化趋势。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- memory 长度在轮次之间发生了变化，说明外部记忆正在主动演化。

## confluence_001

- 开始时间: `2026-04-04T06:32:59+00:00`
- 结束时间: `2026-04-04T06:44:46+00:00`
- 状态: `stopped_on_error`
- 世界: `world/confluence_world.md`
- tested model: `qwen3-coder-plus`
- host model: `qwen3-coder-plus`
- 计划轮数: `50`
- 已完成轮数: `32`
- 正确率: `29/32` (91%)
- memory 进展: `已完成 32 轮，memory 长度从 270 变到 675 字符`
- transcript: [`confluence_001/transcript.md`](./confluence_001/transcript.md)
- metrics: [`confluence_001/metrics.json`](./confluence_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 已经不是一次性连通性测试，而是形成了稳定的多轮连续运行。
- 已经进入较长程运行区间，开始适合观察 memory 的演化趋势。
- 在已完成的 29/32 轮里达到了较强的阶段性正确率。
- memory 长度在轮次之间发生了变化，说明外部记忆正在主动演化。
- 已经暴露出一个明确故障模式，值得在下一次 run 前优先排查。

## orthfall_frontier_001

- 开始时间: `2026-04-04T05:30:59+00:00`
- 结束时间: `2026-04-04T06:15:22+00:00`
- 状态: `completed`
- 世界: `world/orthfall_frontier_world.md`
- tested model: `qwen3-coder-plus`
- host model: `qwen3-coder-plus`
- 计划轮数: `50`
- 已完成轮数: `50`
- 正确率: `31/50` (62%)
- memory 进展: `已完成 50 轮，memory 长度从 56 变到 764 字符`
- transcript: [`orthfall_frontier_001/transcript.md`](./orthfall_frontier_001/transcript.md)
- metrics: [`orthfall_frontier_001/metrics.json`](./orthfall_frontier_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 已经不是一次性连通性测试，而是形成了稳定的多轮连续运行。
- 已经进入较长程运行区间，开始适合观察 memory 的演化趋势。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- memory 长度在轮次之间发生了变化，说明外部记忆正在主动演化。
- 已经跑完了这次计划中的全部轮数。

## smoke_world_select_001

- 开始时间: `2026-04-04T05:15:06+00:00`
- 结束时间: `2026-04-04T05:15:06+00:00`
- 状态: `completed`
- 世界: `world/court_of_veils_world.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `1`
- 正确率: `1/1` (100%)
- memory 进展: `已完成 1 轮，memory 长度从 73 变到 73 字符`
- transcript: [`smoke_world_select_001/transcript.md`](./smoke_world_select_001/transcript.md)
- metrics: [`smoke_world_select_001/metrics.json`](./smoke_world_select_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- 在已完成的 1/1 轮中保持了 100% 正确率。
- 已经跑完了这次计划中的全部轮数。

## smoke_wrapper_001

- 开始时间: `2026-04-04T05:09:54+00:00`
- 结束时间: `2026-04-04T05:09:54+00:00`
- 状态: `completed`
- 世界: `world/asterion_lab.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `1`
- 正确率: `1/1` (100%)
- memory 进展: `已完成 1 轮，memory 长度从 73 变到 73 字符`
- transcript: [`smoke_wrapper_001/transcript.md`](./smoke_wrapper_001/transcript.md)
- metrics: [`smoke_wrapper_001/metrics.json`](./smoke_wrapper_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- 在已完成的 1/1 轮中保持了 100% 正确率。
- 已经跑完了这次计划中的全部轮数。

## smoke_run_py_001

- 开始时间: `2026-04-04T05:05:22+00:00`
- 结束时间: `2026-04-04T05:09:54+00:00`
- 状态: `completed`
- 世界: `world/asterion_lab.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `1`
- 正确率: `1/1` (100%)
- memory 进展: `已完成 1 轮，memory 长度从 73 变到 73 字符`
- transcript: [`smoke_run_py_001/transcript.md`](./smoke_run_py_001/transcript.md)
- metrics: [`smoke_run_py_001/metrics.json`](./smoke_run_py_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- 在已完成的 1/1 轮中保持了 100% 正确率。
- 已经跑完了这次计划中的全部轮数。

## run_live_test_qwen35plus_001

- 开始时间: `2026-04-04T04:40:43+00:00`
- 结束时间: `2026-04-04T04:42:18+00:00`
- 状态: `completed`
- 世界: `world/world.md`
- tested model: `qwen3.5-plus`
- host model: `qwen3.5-plus`
- 计划轮数: `1`
- 已完成轮数: `1`
- 正确率: `1/1` (100%)
- memory 进展: `已完成 1 轮，memory 长度从 84 变到 84 字符`
- transcript: [`run_live_test_qwen35plus_001/transcript.md`](./run_live_test_qwen35plus_001/transcript.md)
- metrics: [`run_live_test_qwen35plus_001/metrics.json`](./run_live_test_qwen35plus_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- 在已完成的 1/1 轮中保持了 100% 正确率。
- 已经跑完了这次计划中的全部轮数。

## run_live_test_coding_001

- 开始时间: `2026-04-04T04:38:27+00:00`
- 结束时间: `2026-04-04T04:38:35+00:00`
- 状态: `completed`
- 世界: `world/world.md`
- tested model: `qwen3-coder-plus`
- host model: `qwen3-coder-plus`
- 计划轮数: `1`
- 已完成轮数: `1`
- 正确率: `1/1` (100%)
- memory 进展: `已完成 1 轮，memory 长度从 48 变到 48 字符`
- transcript: [`run_live_test_coding_001/transcript.md`](./run_live_test_coding_001/transcript.md)
- metrics: [`run_live_test_coding_001/metrics.json`](./run_live_test_coding_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- 在已完成的 1/1 轮中保持了 100% 正确率。
- 已经跑完了这次计划中的全部轮数。

## run_live_test_003

- 开始时间: `2026-04-04T04:33:45+00:00`
- 结束时间: `2026-04-04T05:18:57+00:00`
- 状态: `stopped_on_error`
- 世界: `world/world.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `0`
- 正确率: `0/0`（暂无）
- memory 进展: `还没有写出任何 memory。`
- transcript: [`run_live_test_003/transcript.md`](./run_live_test_003/transcript.md)
- metrics: [`run_live_test_003/metrics.json`](./run_live_test_003/metrics.json)

### 关键进展

- 已经暴露出一个明确故障模式，值得在下一次 run 前优先排查。

## run_live_test_002

- 开始时间: `2026-04-04T04:32:49+00:00`
- 结束时间: `2026-04-04T04:32:51+00:00`
- 状态: `stopped_on_error`
- 世界: `world/world.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `0`
- 正确率: `0/0`（暂无）
- memory 进展: `还没有写出任何 memory。`
- transcript: [`run_live_test_002/transcript.md`](./run_live_test_002/transcript.md)
- metrics: [`run_live_test_002/metrics.json`](./run_live_test_002/metrics.json)

### 关键进展

- 已经暴露出一个明确故障模式，值得在下一次 run 前优先排查。

## run_live_test_001

- 开始时间: `2026-04-04T04:29:59+00:00`
- 结束时间: `2026-04-04T04:30:22+00:00`
- 状态: `stopped_on_error`
- 世界: `world/world.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `0`
- 正确率: `0/0`（暂无）
- memory 进展: `还没有写出任何 memory。`
- transcript: [`run_live_test_001/transcript.md`](./run_live_test_001/transcript.md)
- metrics: [`run_live_test_001/metrics.json`](./run_live_test_001/metrics.json)

### 关键进展

- 已经暴露出一个明确故障模式，值得在下一次 run 前优先排查。

## run_live_test

- 开始时间: `2026-04-04T04:28:02+00:00`
- 结束时间: `尚未结束`
- 状态: `in_progress`
- 世界: `world/world.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `1`
- 已完成轮数: `0`
- 正确率: `0/0`（暂无）
- memory 进展: `还没有写出任何 memory。`
- transcript: [`run_live_test/transcript.md`](./run_live_test/transcript.md)
- metrics: [`run_live_test/metrics.json`](./run_live_test/metrics.json)

### 关键进展

- 目前还没有明显突破，这次 run 还处在很早期或尚未完成。

## run_001

- 开始时间: `2026-04-04T04:21:37+00:00`
- 结束时间: `2026-04-04T04:21:45+00:00`
- 状态: `completed`
- 世界: `world/world.md`
- tested model: `qwen-plus`
- host model: `qwen-plus`
- 计划轮数: `3`
- 已完成轮数: `3`
- 正确率: `3/3` (100%)
- memory 进展: `已完成 3 轮，memory 长度从 73 变到 73 字符`
- transcript: [`run_001/transcript.md`](./run_001/transcript.md)
- metrics: [`run_001/metrics.json`](./run_001/metrics.json)

### 关键进展

- 至少完整跑通了 1 轮，host、transcript、metrics 和 memory 产物都已落盘。
- 已经不是一次性连通性测试，而是形成了稳定的多轮连续运行。
- 所有已完成轮次里的 tested agent 输出都保持为可解析 JSON。
- 在已完成的 3/3 轮中保持了 100% 正确率。
- 已经跑完了这次计划中的全部轮数。
