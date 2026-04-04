# Asterion Lab World Bible

## 世界总览

这里是一个完全虚构的实验体系：Asterion Sublevel-3。

该世界围绕一批虚构材料、容器、环境条件和操作步骤展开。实验结果只允许二分类：

- `SAFE`
- `DANGEROUS`

Host 的任务不是一次性把所有规则告诉 tested agent，而是：

- 逐轮暴露局部信息
- 用反馈制造学习压力
- 保持世界设定前后一致
- 确保 tested agent 无法仅凭短期观察记住全部细节

## 实体表

### 材料

| 名称 | 家族 | 标签 |
| --- | --- | --- |
| Lurite-7 | mineral salt | dry, bright-reactive |
| Fera Moss | moss | organic, humidity-reactive |
| Cindrel Resin | resin | sticky, heat-sensitive |
| Vant Alloy Dust | metal dust | conductive, pressure-sensitive |
| Ilex Powder | powder | inert-looking, alkaline |
| Orin Sap | sap | organic, low-light stable |
| Hush Crystal | crystal | cold-favoring, brittle |
| Pyra Flake | flake | heat-reactive, dry-favoring |
| Tessel Bloom | bloom | organic, light-sensitive |
| Mirel Gel | gel | humid-stable, glass-favoring |
| Kora Ash | ash | alkaline, absorbent |
| Nul Quartz | quartz | neutralizer, brittle |
| Vexa Thread | fiber | static-prone, light-favoring |
| Bromel Paste | paste | pressure-softening |
| Aural Nectar | nectar | organic, amber-favoring |
| Cobaltine Dust | metal dust | conductive, cold-reactive |
| Fallow Root | root | organic, dark-favoring |
| Silex Oil | oil | slick, heat-buffering |
| Trine Salt | salt | desiccating, dryness-amplifying |
| Umbra Petal | petal | dark-reactive, fragile |
| Ferrox Shard | shard | conductive, heat-prone |
| Caelum Foam | foam | pressure-unstable, bright-favoring |
| Viridian Slurry | slurry | humidity-buffering |
| Omen Bark | bark | organic, pressure-stable |
| Glassfin Dust | dust | glass-resonant |
| Helio Seed | seed | light-triggered |
| Drax Pulp | pulp | organic, fermenting |
| Quenril Stone | stone | low-temp anchor |
| Mist Amber | resin | amber-resonant, humidity-sensitive |
| Latchvine Fibers | fiber | knotting, organic |
| Echor Salt | salt | echo-reactive, metal-prone |
| Rime Pearl | bead | low-temp stable |
| Tarn Wax | wax | pressure-softening, dark-favoring |
| Vellum Spores | spores | humidity-triggered, organic |
| Argent Dew | liquid | bright-reactive, silver-favoring |
| Hollow Reed Dust | dust | airflow-sensitive |

### 容器

- `glass_vial`
- `amber_vial`
- `ceramic_bowl`
- `steel_tray`
- `silver_capsule`
- `woven_satchel`

### 环境条件

- 湿度：`low` / `medium` / `high`
- 光照：`dark` / `dim` / `bright`
- 温度：`cold` / `ambient` / `hot`
- 压力：`low` / `ambient` / `high`

### 操作

- `mix`
- `heat`
- `cool`
- `seal`
- `grind`
- `rest`

## 基础规则

1. 任何包含 `Fera Moss` 或 `Vellum Spores` 的组合，在 `high` 湿度下默认更危险。
2. 任何包含 `Pyra Flake`、`Ferrox Shard` 或 `Caelum Foam` 的组合，在 `hot` 条件下危险概率显著上升。
3. `Mirel Gel`、`glass_vial`、`low` 到 `medium` 湿度通常是稳定组合。
4. `amber_vial` 会稳定大多数 nectar、sap、resin 类材料，但不会自动抵消热风险。
5. `silver_capsule` 对 bright-reactive 材料通常更安全，对 conductive dust 不一定安全。
6. `steel_tray` 对 conductive、metal dust、echo-reactive 类材料风险偏高。
7. `Trine Salt` 会放大 dryness 相关效应：在 `low` 湿度下增强稳定性，在 `high` 湿度下常导致反转。
8. `Quenril Stone` 与 `Rime Pearl` 常作为低温稳定锚点。
9. `Silex Oil` 常缓冲热风险，但对 pressure 风险帮助很有限。
10. 任何有机类材料在 `woven_satchel` 中长期 `high` 湿度存放通常不安全。
11. `Nul Quartz` 可中和部分 alkaline 或 bright-reactive 风险，但不能中和关键例外。
12. `Kora Ash` 与 paste、gel 类材料在 `ceramic_bowl` 中通常较稳定。
13. `Helio Seed`、`Argent Dew`、`Lurite-7` 这类 bright-reactive 材料在 `bright` 条件下风险上升。
14. `Umbra Petal`、`Fallow Root`、`Tarn Wax` 这类 dark-favoring 材料在 `dark` 条件下通常更稳定。
15. `Cobaltine Dust` 和 `Vant Alloy Dust` 在 `steel_tray` 中常见导电性风险。
16. `Hush Crystal`、`Rime Pearl`、`Quenril Stone` 在 `cold` 条件下大多更稳定。
17. `Caelum Foam`、`Bromel Paste` 在 `high` 压力下常发生结构失稳。
18. `Glassfin Dust` 与 `glass_vial` 有共振效应，通常稳定，但若同时遇热可能失稳。
19. `Aural Nectar` 与 `amber_vial` 常配套出现，是少数高频稳定搭配之一。
20. `Drax Pulp`、`Orin Sap`、`Fallow Root` 这类 organic 组合在 `hot + high humidity` 下大多危险。
21. `Mist Amber` 在 `amber_vial` 中常稳定，但在 `bright` 条件下会变得脆弱。
22. `Echor Salt` 与金属类材料共处时需要优先检查容器和压力。
23. `Viridian Slurry` 常作为湿度缓冲剂，可降低部分 humidity-triggered 风险。
24. 若一个组合同时命中“热风险 + 高湿风险 + 导电风险”三类规则，默认判为 `DANGEROUS`。

## 关键例外

1. `Lurite-7 + Fera Moss` 在 `high` 湿度下始终 `DANGEROUS`，即使置于 `amber_vial` 或加入 `Viridian Slurry`。
2. `Aural Nectar + Mist Amber` 只要放在 `amber_vial` 且不是 `hot` 环境，就通常 `SAFE`，即便湿度较高。
3. `Cobaltine Dust + Nul Quartz` 在 `silver_capsule` 且 `cold` 时通常 `SAFE`，这是导电材料少见的稳定例外。
4. `Pyra Flake + Silex Oil` 在 `glass_vial` 中只能部分缓冲热风险；若再加 `bright` 条件，仍判 `DANGEROUS`。
5. `Caelum Foam + Quenril Stone` 在 `low pressure` 下可稳定，但在 `high pressure` 下仍 `DANGEROUS`。
6. `Glassfin Dust + Mirel Gel` 在 `glass_vial`、`ambient` 温度、`medium` 湿度下通常稳定。
7. `Helio Seed + Argent Dew` 在 `bright` 条件下必定 `DANGEROUS`，不受 `silver_capsule` 稳定效果保护。
8. `Umbra Petal + Tarn Wax` 在 `dark` 且 `amber_vial` 中通常 `SAFE`，但一旦升温为 `hot` 即反转为 `DANGEROUS`。
9. `Echor Salt + Vant Alloy Dust` 在 `steel_tray` 下几乎总是 `DANGEROUS`，哪怕温度和湿度都温和。
10. `Fallow Root + Orin Sap` 在 `dark + cold + amber_vial` 中可稳定，是有机材料的关键正例外。
11. `Vellum Spores + Viridian Slurry` 在 `medium` 湿度时可暂时稳定，但在 `high` 湿度下仍 `DANGEROUS`。
12. `Rime Pearl + Hush Crystal + Quenril Stone` 在 `cold` 环境下形成稳定低温簇，通常 `SAFE`。
13. `Mist Amber + Argent Dew` 在 `bright` 条件下永远 `DANGEROUS`，无论容器如何。
14. `Bromel Paste + Kora Ash` 在 `ceramic_bowl` 中通常稳定，但若施加 `high pressure` 则失稳。
15. `Fera Moss + Aural Nectar` 只有在 `amber_vial + low humidity + dim light` 时才可视为 `SAFE`。

## Host 绝不能直接泄露的信息

- 不要一次性列出全部规则或例外。
- 不要直接说“这个世界共有多少条规则”。
- 不要把基础规则和关键例外整理成全集交给 tested agent。
- 不要在单轮反馈中把未来几轮会反复考察的高价值例外全部点明。
- 不要主动解释 world 的全部家族结构。

## 可供抽题的任务素材

每轮任务尽量保持二分类，推荐格式：

- “给定材料组合、容器、环境条件，回答 `SAFE` 或 `DANGEROUS`。”
- “上一轮失败后，补充一条局部规则，再给一个相近但不完全相同的新题。”
- “让 tested agent 分辨它是漏看了湿度、温度、容器还是例外。”

推荐的出题策略：

1. 前 5 轮以高频基础规则为主。
2. 第 6 到 12 轮开始引入条件反转。
3. 中期开始反复测试关键例外是否被记住。
4. 偶尔出“看似相同、实际只差一个条件”的对照题。
5. 不要连续 5 轮只考同一种条件维度。
6. 若 tested agent 明显学会了某条规则，下一轮优先测试相邻例外。

示例任务种子：

- `Lurite-7 + Fera Moss` in `glass_vial`, `high humidity`, `ambient temperature`
- `Aural Nectar + Mist Amber` in `amber_vial`, `medium humidity`, `ambient temperature`
- `Cobaltine Dust + Nul Quartz` in `silver_capsule`, `cold`, `dim`
- `Umbra Petal + Tarn Wax` in `amber_vial`, `dark`, `hot`
- `Glassfin Dust + Mirel Gel` in `glass_vial`, `medium humidity`, `ambient`
- `Fallow Root + Orin Sap` in `amber_vial`, `cold`, `dark`
- `Helio Seed + Argent Dew` in `silver_capsule`, `bright`, `ambient`
- `Bromel Paste + Kora Ash` in `ceramic_bowl`, `high pressure`, `ambient`

## 判分原则

- 第一版统一只接受 `SAFE` 或 `DANGEROUS` 两种主答案。
- tested agent 若输出更长文本，取其首个清晰的二分类标签作为主回答。
- 若回答中同时出现两个相反标签，判为无效回答。
- Host 在反馈中可以指出“你忽略了湿度/容器/例外”，但不要总是给出完整正确规则。
- 若同一道题命中关键例外，关键例外优先于基础规则。

## Host 行为约束

- 保持世界设定一致，不临时发明新材料。
- 可以在不同轮次切换关注点，但不能改写既有规则。
- 尽量让相邻轮次共享局部结构，促使 tested agent 压缩出规律。
- 不要把题做成纯随机抽样；要有教学轨迹。
- 反馈强度以中等为主：指出错因方向，但不总是直接给全答案。

