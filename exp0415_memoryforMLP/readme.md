# exp0415 — MLP Online Learning Smoke Test

`exp0415_memoryforMLP` 验证一个核心问题：**MLP 能否通过在线学习（online learning）在权重里编码序列记忆？**

实验设置极简：

- 输入永远是常量 `1`（代表 `<BOS>`）
- 目标序列是 `0, 1, 0, 1, 0, 1, ...`（无限交替）
- MLP 架构：`[1, 4, 4, 1]`（1 输入，两层 hidden 各 4 个节点，1 输出）
- 每个时刻做一次前向传播 + 一次反向传播 + 权重更新（batchsize=1）
- MLP 没有任何显式记忆单元，唯一的"记忆"是权重本身

**目的**：作为对照基线，为后续引入 simplex/triangle 网络做铺垫。

---

## 运行方式

```bash
cd exp0415_memoryforMLP
python3 run.py
```

输出图片自动保存在 `runs/` 目录下。

---

## 目录结构

```text
exp0415_memoryforMLP/
├── params.yaml        # 实验配置（lr, steps, hidden, optimizer, seed）
├── run.py             # 主入口，一键运行
├── requirements.txt
├── readme.md
└── runs/              # 输出图片（自动生成，不提交）
```

---

## 配置参数（params.yaml）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hidden` | `[4, 4]` | MLP 隐藏层宽度，完整架构为 `[1, hidden..., 1]` |
| `lr` | `10.0` | 学习率（见下方结论） |
| `optimizer` | `sgd` | `sgd` 或 `adam` |
| `steps` | `100` | 在线学习的时刻数 |
| `seed` | `42` | 随机种子 |

---

## 实验结论

本实验对比了两种训练模式：

| 模式 | 描述 |
|---|---|
| **standard** | 每步用当前梯度 grad_t 立即更新权重 |
| **delayed_grad** | 每步用上一步的梯度 grad_{t-1} 更新权重；当前梯度留给下一步使用 |

### 量化对比（lr=0.5，SGD，100 步）

| 模式 | 正确率（\|output−target\|<0.5） | avg last-10 loss |
|---|---|---|
| standard | 3/100 | 0.267 |
| **delayed_grad** | **98/100** | **0.090** |

### 为什么 standard 失败

对单层网络 `output = σ(w + b)` 的解析分析揭示了根本原因：

- 时刻 t 目标为 `0`，梯度让 `w` 减小 → 下一步 t+1 目标为 `1`，但输出已更接近 `0`，反而离目标更远
- 梯度更新**永远服务于当前步**，而这恰好**伤害了下一步的预测**

```
lr=  0.5  y0=0.622  y1_pred=0.552  (need y1≈1)  barely ok
lr=  1.0  y0=0.622  y1_pred=0.479  (need y1≈1)  no
lr=  5.0  y0=0.622  y1_pred=0.081  (need y1≈1)  no
lr= 10.0  y0=0.622  y1_pred=0.005  (need y1≈1)  no
```

网络最终收敛到固定输出（`0` 或 `0.5`），而不是振荡点。

### 为什么 delayed_grad 成功

关键洞察：目标序列以 2 为周期交替，所以 `target_{t-1} = target_{t+1}`。

- 时刻 t 用 grad_{t-1}（该梯度是为了预测 target_{t-1} = (t-1)%2 而算的）更新权重
- 更新后，下一步输出 y_{t+1} 被推向 target_{t-1}
- 而 target_{t-1} ≡ target_{t+1}（交替序列的对称性）

因此，每次权重更新天然对齐**下一步的正确答案**，而不是当前步。梯度的"时间错位"恰好补偿了序列的交替结构。

### LR 扫描（delayed_grad 模式）

| LR | standard 正确率 | delayed_grad 正确率 |
|---|---|---|
| 0.1 | 39/100 | 69/100 |
| **0.5** | 3/100 | **98/100** |
| 1.0 | 0/100 | 75/100 |
| 2.0 | 0/100 | 67/100 |
| 5.0 | 17/100 | 66/100 |

---

## 与后续实验的关系

- **delayed_grad** 的成功揭示：通过修改梯度的时间对齐方式，同一个 MLP 权重可以编码交替序列记忆
- 下一步：将 delayed_grad 机制引入 exp0414/exp0410 的 simplex/triangle 网络，验证更复杂序列是否也能通过梯度时间对齐来学习
