# Coding Endpoint Model Notes

适用前提：

- `DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1`
- 当前 repo 里的实验脚本走的是 OpenAI 兼容 `chat/completions`

下面这份表是基于阿里云百炼 Coding Plan / coding 端点的官方说明整理出来的“当前可关注模型清单”。

## 已确认可用

这两个模型已经在当前 repo 里实际测过：

| model id | 状态 | 备注 |
| --- | --- | --- |
| `qwen3-coder-plus` | 已实测可用 | 已完整跑通 1 轮 host + tested agent 实验链路 |
| `qwen3.5-plus` | 已实测可调用 | 已验证最小直连请求可返回；完整实验链路会更慢 |

## 官方文档里可关注的候选模型

| model id | 类型 | 适合方向 | 当前备注 |
| --- | --- | --- | --- |
| `qwen3.5-plus` | 千问通用 | 推理、总结、一般 agent | 当前 repo 已验证可调用 |
| `qwen3-coder-plus` | 千问编码 | coding、结构化输出 | 当前 repo 已验证完整 1 轮 |
| `qwen3-coder-next` | 千问编码 | 更强的 coding / agent | 官方文档列出，可后续测试 |
| `qwen3-max-2026-01-23` | 千问通用 | 更强推理 | 官方文档列出，可后续测试 |
| `kimi-k2.5` | 第三方 | 通用对话 / reasoning | 官方文档列出，可后续测试 |
| `glm-5` | 第三方 | 通用推理 | 官方文档列出，可后续测试 |
| `glm-4.7` | 第三方 | 通用推理 | 官方文档列出，可后续测试 |
| `MiniMax-M2.5` | 第三方 | 通用对话 | 官方文档列出，可后续测试 |

## 如果你想先试“相对更轻”的模型

下面这个顺序不是官方成本排名，而是我根据模型命名与常见定位做的保守推断，适合先试：

1. `qwen3.5-plus`
2. `glm-4.7`
3. `qwen3-coder-plus`

如果你更看重 coding 表现，而不是绝对轻量，优先试：

1. `qwen3-coder-plus`
2. `qwen3-coder-next`

## 推荐切换命令

```bash
SSL_CERT_FILE=/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/certifi/cacert.pem \
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1 \
python3 scripts/run_experiment.py \
  --run-id run_try_qwen35plus \
  --rounds 1 \
  --auto-accept-host \
  --tested-model qwen3.5-plus \
  --host-model qwen3.5-plus
```

```bash
SSL_CERT_FILE=/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/certifi/cacert.pem \
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1 \
python3 scripts/run_experiment.py \
  --run-id run_try_coder_plus \
  --rounds 1 \
  --auto-accept-host \
  --tested-model qwen3-coder-plus \
  --host-model qwen3-coder-plus
```

## 官方来源

- OpenCode 接入百炼推理服务：<https://help.aliyun.com/zh/model-studio/opencode>
- Coding Plan 概述：<https://help.aliyun.com/zh/model-studio/coding-plan>
- Cline / Coding Plan 接入说明：<https://help.aliyun.com/zh/model-studio/cline-coding-plan>
- 图片理解 Skill（含 coding plan 模型兼容列表）：<https://help.aliyun.com/zh/model-studio/add-vision-skill>

