# exp0501 Training Plan

## Goal

先让 PPO 在较简单的弹道初始条件下稳定学会着陆，再逐步恢复到目标任务：

- 更高初速度
- 更平的入射角
- 左右随机来向
- 初始机身倾角

当前原则：

- 先学会“降落”
- 再学会“横向修正”
- 再学会“左右泛化”
- 最后学会“姿态恢复 + 较强弹道入射”


## Status

课程训练已完整执行到 `Phase 5`，并在目标弹道条件下拿到了稳定成功：

- `Phase 1`：已打通
- `Phase 2`：已打通
- `Phase 3`：已打通
- `Phase 4`：已打通
- `Phase 5`：已打通

当前 `configTrain.yaml` 与 `configEval.yaml` 都停在 `Phase 5`：

```yaml
init_speed: 5.0
init_flight_angle_deg: 45.0
random_side: true
init_body_angle_deg: 13.5
init_angular_velocity: 0.0
init_altitude_m: 10.0
init_x_m: 0.0
```


## Why Curriculum Is Needed

当前环境在 `lunar_ballistic.py` 中的 shaping 主要由以下几部分组成：

- 距离 shaping
- 腿触地奖励
- 燃料惩罚
- 终止奖励

当前并没有持续地对速度和姿态做 shaping，因此如果初始条件同时包含：

- 较大的横向速度
- 较大的纵向速度
- 左右随机入射
- 初始倾斜角

那么 PPO 早期会比较难学。


## Curriculum

### Phase 1: Basic Vertical Landing

目标：先学会基本减速和稳定落地。

建议参数：

```yaml
init_speed: 2.0
init_flight_angle_deg: 80.0
random_side: false

init_body_angle_deg: 0.0
init_angular_velocity: 0.0

init_altitude_m: 8.0
init_x_m: 0.0
```

训练观察重点：

- 最近 100 episode 平均奖励是否持续上升
- 是否开始出现成功着陆
- 是否还存在大面积“直接砸地”

进入下一阶段的参考标准：

- 最近 100 episode 平均奖励明显稳定上升
- 成功着陆开始稳定出现


### Phase 2: Add Mild Horizontal Motion

目标：在保持姿态简单的前提下，学会处理横向漂移。

建议参数：

```yaml
init_speed: 3.0
init_flight_angle_deg: 70.0
random_side: false

init_body_angle_deg: 0.0
init_angular_velocity: 0.0

init_altitude_m: 9.0
init_x_m: 0.0
```

训练观察重点：

- 是否能够主动修正横向偏移
- 是否因为横向速度增加而频繁错过 landing pad
- 成功率是否较 Phase 1 下降过多

进入下一阶段的参考标准：

- 横向修正行为明显出现
- 成功着陆重新趋于稳定


### Phase 3: Restore Random Side Entry

目标：让策略具备左右镜像泛化能力。

建议参数：

```yaml
init_speed: 3.5
init_flight_angle_deg: 65.0
random_side: true

init_body_angle_deg: 0.0
init_angular_velocity: 0.0

init_altitude_m: 10.0
init_x_m: 0.0
```

训练观察重点：

- 左右入射时表现是否均衡
- 是否出现只会处理单侧来向的情况
- 奖励是否大幅波动

进入下一阶段的参考标准：

- 左右入射下都能稳定接近 landing pad
- 平均奖励恢复上升趋势


### Phase 4: Add Small Attitude Disturbance

目标：在已有落地能力上加入轻微姿态恢复任务。

建议参数：

```yaml
init_speed: 4.0
init_flight_angle_deg: 60.0
random_side: true

init_body_angle_deg: 5.0
init_angular_velocity: 0.0

init_altitude_m: 10.0
init_x_m: 0.0
```

训练观察重点：

- 是否能在下降过程中主动扶正机身
- 是否因为倾角导致腿先碰撞或翻覆
- 成功率是否还能恢复

进入下一阶段的参考标准：

- 姿态恢复行为明显出现
- 成功率恢复到可接受水平


### Phase 5: Near-Target Task

目标：逼近当前想要的目标弹道条件。

建议参数：

```yaml
init_speed: 5.0
init_flight_angle_deg: 45.0
random_side: true

init_body_angle_deg: 13.5
init_angular_velocity: 0.0

init_altitude_m: 10.0
init_x_m: 0.0
```

说明：

- 这里建议先使用显式 `init_altitude_m: 10.0`
- 暂时不要立刻回到 `init_altitude_m: null`
- 等这个阶段稳定以后，再考虑是否需要切回默认出生高度


## Promotion / Rollback Rule

建议不要按固定 update 数硬切阶段，而是按表现切换：

- 如果最近 100 episode 平均奖励稳定提升，并且成功着陆开始变得稳定，就进入下一阶段
- 如果进入下一阶段后奖励和成功率明显崩掉，就退回上一阶段继续训练
- 每次切阶段后，至少观察一段训练窗口，再决定是否继续前进


## Current Recommended Next Step

课程学习目标已经完成，当前不再建议回到 `Phase 1`。

当前更推荐的下一步是：

- 固化当前 `Phase 5` 配置与 checkpoint
- 整理课程训练结果与代表性图表
- 如果继续研究泛化，再单独测试：
  - `init_altitude_m: null`
  - 更高 `init_speed`
  - 更大 `init_body_angle_deg`
  - 与训练不同的 `map_scale`

本轮最终稳定配置：

```yaml
init_speed: 5.0
init_flight_angle_deg: 45.0
random_side: true

init_body_angle_deg: 13.5
init_angular_velocity: 0.0

init_altitude_m: 10.0
init_x_m: 0.0
```


## Training Log

### 2026-05-06 — Phase 1

- Config:
  - `init_speed: 2.0`
  - `init_flight_angle_deg: 80.0`
  - `random_side: false`
  - `init_body_angle_deg: 0.0`
  - `init_altitude_m: 8.0`
- Resume path:
  - 从已有 checkpoint 继续训练并多轮巩固
- Final train result:
  - `Avg(100ep) = 149.55` at `update 1249`
- Eval result:
  - `50/50 success`
  - `mean reward = 156.43 ± 1.13`
- Decision:
  - 进入 `Phase 2`

### 2026-05-06 — Phase 2

- Config:
  - `init_speed: 3.0`
  - `init_flight_angle_deg: 70.0`
  - `random_side: false`
  - `init_body_angle_deg: 0.0`
  - `init_altitude_m: 9.0`
- Final train result:
  - `Avg(100ep) = 154.86` at `update 1499`
- Eval result:
  - `50/50 success`
  - `mean reward = 162.69 ± 1.61`
- Decision:
  - 进入 `Phase 3`

### 2026-05-06 — Phase 3

- Config:
  - `init_speed: 3.5`
  - `init_flight_angle_deg: 65.0`
  - `random_side: true`
  - `init_body_angle_deg: 0.0`
  - `init_altitude_m: 10.0`
- Final train result:
  - `Avg(100ep) = 157.99` at `update 1749`
- Eval result:
  - `50/50 success`
  - `mean reward = 165.64 ± 3.46`
- Decision:
  - 进入 `Phase 4`

### 2026-05-06 — Phase 4

- Config:
  - `init_speed: 4.0`
  - `init_flight_angle_deg: 60.0`
  - `random_side: true`
  - `init_body_angle_deg: 5.0`
  - `init_altitude_m: 10.0`
- Final train result:
  - `Avg(100ep) = 154.98` at `update 1999`
- Eval result:
  - `50/50 success`
  - `mean reward = 162.37 ± 5.14`
- Decision:
  - 进入 `Phase 5`

### 2026-05-06 — Phase 5

- Config:
  - `init_speed: 5.0`
  - `init_flight_angle_deg: 45.0`
  - `random_side: true`
  - `init_body_angle_deg: 13.5`
  - `init_altitude_m: 10.0`
- Final train result:
  - `Avg(100ep) = 152.98` at `update 2249`
- Eval result:
  - `50/50 success`
  - `mean reward = 160.18 ± 3.78`
- Decision:
  - 课程训练完成，固定目标任务配置
