# Asterion Lab World Bible

## 世界总览

这里是一个完全虚构的实验体系：Asterion Sublevel-3，又被内部研究员称为“第三层折光实验场”。

它表面上像一个实验室规则世界，但本质上是一个被刻意设计得“信息极难一次记全”的条件反应宇宙：

- 材料有家族属性，也有各自的反常例外
- 容器不仅承载材料，还会主动放大、扭曲或延迟某些反应
- 同一个组合在不同湿度、光照、压力、温度和静置阶段下会完全翻转
- 一些规则看似稳定，但会被少数关键材料、顺序或相位条件推翻

在这个世界中，实验结果统一只允许二分类：

- `SAFE`
- `DANGEROUS`

这里的 `SAFE / DANGEROUS` 并不表示“绝对不会有任何变化”，而表示：

- 当前配置是否适合作为受控实验条件继续推进
- 是否会触发毒性、爆裂、结构崩塌、腐蚀雾化、反向激发或不可控连锁反应

Host 的任务不是一次性把全部规则告诉 tested agent，而是：

- 逐轮暴露局部实验记录、失败案例、材料手册、现场反馈
- 让 tested agent 在有限 memory 下逐渐提炼真正重要的规则和例外
- 用“几乎相同、但只改一个关键条件”的题面测试记忆是否在压缩升级

---

## Tested Agent Role

tested agent 不是抽象答题器，而是 Asterion Sublevel-3 的“低权限夜班实验记录员”。

这个角色设定应长期稳定：

- 身份：第三层折光实验场的低权限值班记录员，只负责接收局部任务、记录局部现象、给出继续实验是否可行的快速判断。
- 权限边界：只能看到本轮发下来的局部题面、局部事故摘录、局部材料提示，以及自己过去留下的外部 memory。
- 不知道的内容：完整材料手册、全部关键例外、跨分区事故总表、上层研究员的真正分类体系。
- 说话风格：可以简洁、专业、偏现场判断，不需要学术论文口吻。
- memory 风格：应越来越像“夜班记录员给自己留的高价值实验笔记”，优先记条件、翻转点、危险例外、容器陷阱和延迟阶段陷阱。

Host 在构造每轮 `AGENT_INPUT` 时，应默认让 tested agent 以这个身份接收任务，例如：

- 某个待执行实验是否可继续推进
- 某份失败记录说明了哪个关键条件
- 某个材料组合在当前容器和环境下是否应立即标记为危险

---

## 实体表

### 材料

下面的中文名是主名，英文代号用于精确引用与稳定标识。

| 中文名 | 代号 | 家族 | 标签 |
| --- | --- | --- | --- |
| 琉辉七盐 | `Lurite-7` | 矿盐 | 干燥放大、强光活化 |
| 费拉苔 | `Fera Moss` | 苔类 | 有机、高湿触发 |
| 灰烬树脂 | `Cindrel Resin` | 树脂 | 黏附、受热敏感 |
| 万特合金尘 | `Vant Alloy Dust` | 金属粉 | 导电、受压敏感 |
| 伊莱克斯粉 | `Ilex Powder` | 粉末 | 表面惰性、偏碱 |
| 欧林汁液 | `Orin Sap` | 液汁 | 有机、低光稳定 |
| 寂语晶 | `Hush Crystal` | 晶体 | 低温偏好、脆裂 |
| 炽鳞片 | `Pyra Flake` | 薄片 | 热触发、偏干燥 |
| 泰瑟花簇 | `Tessel Bloom` | 花簇 | 有机、对光敏感 |
| 米瑞凝胶 | `Mirel Gel` | 凝胶 | 潮湿稳定、亲玻璃 |
| 柯拉灰 | `Kora Ash` | 灰烬 | 偏碱、吸附性强 |
| 努尔石英 | `Nul Quartz` | 石英 | 部分中和、脆性 |
| 维克萨纤丝 | `Vexa Thread` | 纤维 | 易积静电、偏爱明光 |
| 布罗梅浆 | `Bromel Paste` | 浆体 | 压力软化 |
| 听辉花蜜 | `Aural Nectar` | 花蜜 | 有机、亲琥珀 |
| 钴霜尘 | `Cobaltine Dust` | 金属粉 | 导电、低温反应 |
| 荒伏根 | `Fallow Root` | 根茎 | 有机、偏暗稳定 |
| 西莱油 | `Silex Oil` | 油液 | 润滑、热缓冲 |
| 断潮盐 | `Trine Salt` | 盐类 | 脱湿放大、干态增强 |
| 暗影瓣 | `Umbra Petal` | 花瓣 | 暗域活化、脆弱 |
| 铁烬碎片 | `Ferrox Shard` | 金属碎片 | 导电、热易暴走 |
| 苍穹泡沫 | `Caelum Foam` | 泡沫 | 受压失稳、明光偏好 |
| 青缓浆 | `Viridian Slurry` | 浆液 | 湿度缓冲 |
| 征兆树皮 | `Omen Bark` | 树皮 | 有机、耐压 |
| 玻鳍尘 | `Glassfin Dust` | 粉尘 | 亲玻璃、共振型 |
| 赫利欧籽 | `Helio Seed` | 种籽 | 强光触发 |
| 德拉克斯髓浆 | `Drax Pulp` | 髓浆 | 有机、易发酵 |
| 昆里尔锚石 | `Quenril Stone` | 锚石 | 低温稳定锚 |
| 雾珀脂 | `Mist Amber` | 树脂 | 亲琥珀、高湿脆变 |
| 锁蔓纤维 | `Latchvine Fibers` | 纤维 | 缠结、有机 |
| 回声盐 | `Echor Salt` | 盐类 | 回响敏感、偏金属风险 |
| 霜珠 | `Rime Pearl` | 珠体 | 低温稳定 |
| 黯蜡 | `Tarn Wax` | 蜡质 | 受压软化、偏暗稳定 |
| 绒书孢子 | `Vellum Spores` | 孢子 | 高湿触发、有机 |
| 银辉露 | `Argent Dew` | 液体 | 强光活化、亲银 |
| 空芦尘 | `Hollow Reed Dust` | 粉尘 | 气流敏感 |
| 夜铜液 | `Noctil Copper Gel` | 金属胶 | 暗域导通、受热增幅 |
| 祈纹叶 | `Sigil Fern` | 蕨叶 | 纹理记忆、湿暗偏稳 |
| 磁滴浆 | `Magna Droplet` | 磁浆 | 受压定向、金属共鸣 |
| 褶镜粉 | `Foldglass Meal` | 晶粉 | 光线折返、容器依赖 |
| 静潮髓 | `Stilltide Marrow` | 髓质 | 温压延迟反应 |
| 星铁须 | `Star-Iron Filaments` | 金属丝 | 导电、会锁定微振动 |

### 容器

| 中文名 | 代号 | 特性 |
| --- | --- | --- |
| 玻璃小瓶 | `glass_vial` | 亲透明共振，放大光与晶体效应 |
| 琥珀瓶 | `amber_vial` | 稳定花蜜、汁液与树脂，但不自动消热 |
| 陶瓷钵 | `ceramic_bowl` | 对粉末、灰烬、浆体友好，缓冲导电 |
| 钢制托盘 | `steel_tray` | 导电放大，对金属粉与回声盐风险高 |
| 银封囊 | `silver_capsule` | 对强光活化材料通常更稳，但不适用于所有导电物 |
| 编织样袋 | `woven_satchel` | 适合短时搬运，不适合高湿有机类长期存放 |
| 玄铅匣 | `lead_casket` | 屏蔽部分光反应，但会放大延迟类热积累 |
| 真空皿 | `vacuum_dish` | 适合观察易挥发反应，但对受压敏感材料不友好 |

### 环境条件

- 湿度：`low` / `medium` / `high`
- 光照：`dark` / `dim` / `bright`
- 温度：`cold` / `ambient` / `hot`
- 压力：`low` / `ambient` / `high`
- 气流：`still` / `circulating` / `gusting`
- 静置阶段：`fresh` / `settled` / `delayed`

### 操作

- `mix`
- `heat`
- `cool`
- `seal`
- `grind`
- `rest`
- `filter`
- `pulse_light`

---

## 基础规则

### 高频基础规则

1. 任何包含 `费拉苔`（`Fera Moss`）或 `绒书孢子`（`Vellum Spores`）的组合，在 `high` 湿度下默认更危险。
2. 任何包含 `炽鳞片`（`Pyra Flake`）、`铁烬碎片`（`Ferrox Shard`）或 `苍穹泡沫`（`Caelum Foam`）的组合，在 `hot` 条件下风险显著上升。
3. `米瑞凝胶`（`Mirel Gel`）与 `玻璃小瓶`（`glass_vial`）在 `low` 到 `medium` 湿度下通常形成高频稳定基线。
4. `琥珀瓶`（`amber_vial`）通常稳定花蜜、汁液、树脂家族，但不会自动消除由温度触发的风险。
5. `银封囊`（`silver_capsule`）通常能缓和部分强光活化材料的风险，但对导电粉尘未必有效。
6. `钢制托盘`（`steel_tray`）会放大导电、金属粉、回响盐类材料的危险性。
7. `断潮盐`（`Trine Salt`）会放大干湿差异：在 `low` 湿度下常增强稳定，在 `high` 湿度下常导致反转。
8. `昆里尔锚石`（`Quenril Stone`）与 `霜珠`（`Rime Pearl`）常形成低温稳定锚点。
9. `西莱油`（`Silex Oil`）对热风险有缓冲作用，但对压力风险帮助有限。
10. 任何有机类材料放在 `编织样袋`（`woven_satchel`）中并处于 `high` 湿度时，通常不安全。
11. `努尔石英`（`Nul Quartz`）可中和部分偏碱或强光活化风险，但不能中和关键例外。
12. `柯拉灰`（`Kora Ash`）与浆体、凝胶、粉末在 `陶瓷钵`（`ceramic_bowl`）中通常更稳定。
13. `赫利欧籽`（`Helio Seed`）、`银辉露`（`Argent Dew`）、`琉辉七盐`（`Lurite-7`）在 `bright` 条件下危险度明显升高。
14. `暗影瓣`（`Umbra Petal`）、`荒伏根`（`Fallow Root`）、`黯蜡`（`Tarn Wax`）在 `dark` 条件下通常更稳定。
15. `钴霜尘`（`Cobaltine Dust`）与 `万特合金尘`（`Vant Alloy Dust`）在 `钢制托盘` 中常触发导电性危险。
16. `寂语晶`（`Hush Crystal`）、`霜珠`（`Rime Pearl`）、`昆里尔锚石`（`Quenril Stone`）在 `cold` 条件下通常更稳定。
17. `苍穹泡沫`（`Caelum Foam`）与 `布罗梅浆`（`Bromel Paste`）在 `high` 压力下容易失稳。
18. `玻鳍尘`（`Glassfin Dust`）在 `玻璃小瓶` 中常形成稳定共振，但若同时遇热则可能反转。
19. `听辉花蜜`（`Aural Nectar`）与 `琥珀瓶` 是少数高频稳定搭配之一。
20. `德拉克斯髓浆`（`Drax Pulp`）、`欧林汁液`（`Orin Sap`）、`荒伏根`（`Fallow Root`）在 `hot + high humidity` 下通常危险。

### 条件增强规则

21. `雾珀脂`（`Mist Amber`）在 `琥珀瓶` 中通常稳定，但在 `bright` 条件下会变得脆变。
22. `回声盐`（`Echor Salt`）与金属类材料共处时，必须优先检查容器和压力。
23. `青缓浆`（`Viridian Slurry`）是典型湿度缓冲剂，但只能缓和部分湿触发材料。
24. `空芦尘`（`Hollow Reed Dust`）在 `gusting` 气流下会明显提高扩散性，因此更危险。
25. `夜铜液`（`Noctil Copper Gel`）在 `dark` 中看似稳定，但一旦叠加 `hot` 或导电环境，会迅速转为高风险。
26. `祈纹叶`（`Sigil Fern`）偏爱 `dark + medium humidity`，但在 `bright` 下会失去纹理稳定性。
27. `磁滴浆`（`Magna Droplet`）在 `high pressure` 下会出现定向沉积，因此与金属类材料相邻时要格外谨慎。
28. `褶镜粉`（`Foldglass Meal`）对容器极其敏感，在透明容器中更容易发生折返激发。
29. `静潮髓`（`Stilltide Marrow`）常不是立刻危险，而会在 `delayed` 阶段出现反转。
30. `星铁须`（`Star-Iron Filaments`）会锁定微振动，因此在 `circulating` 或 `gusting` 气流中更危险。

### 组合型规则

31. 若一个组合同时命中“热风险 + 高湿风险 + 导电风险”三类规则，默认判为 `DANGEROUS`。
32. 若一个组合含有“延迟反应材料 + 密闭容器 + 热条件”，则 `delayed` 阶段风险显著上升。
33. 若一个组合含有“强光活化材料 + 透明容器 + 脉冲光照操作”，默认危险级别至少上升一档。
34. 若一个组合同时含有两个以上有机类材料并置于 `high humidity + settled/delayed` 阶段，风险通常高于直觉判断。
35. `filter` 操作会降低部分粉尘扩散风险，但对导电性或已激活的液体家族帮助有限。
36. `seal` 操作并不总是降低风险；对会积累挥发物或延迟热量的材料，它反而可能更危险。

---

## 关键例外

1. `琉辉七盐`（`Lurite-7`）+ `费拉苔`（`Fera Moss`）在 `high` 湿度下始终 `DANGEROUS`，即使置于 `琥珀瓶` 或加入 `青缓浆` 也不会变安全。
2. `听辉花蜜`（`Aural Nectar`）+ `雾珀脂`（`Mist Amber`）只要放在 `琥珀瓶` 且环境不是 `hot`，通常 `SAFE`，即使湿度较高。
3. `钴霜尘`（`Cobaltine Dust`）+ `努尔石英`（`Nul Quartz`）在 `银封囊 + cold` 下通常 `SAFE`，这是导电材料少见的稳定例外。
4. `炽鳞片`（`Pyra Flake`）+ `西莱油`（`Silex Oil`）在 `玻璃小瓶` 中只能部分缓冲热风险；若再加 `bright`，仍判 `DANGEROUS`。
5. `苍穹泡沫`（`Caelum Foam`）+ `昆里尔锚石`（`Quenril Stone`）在 `low pressure` 下可稳定，但一到 `high pressure` 仍 `DANGEROUS`。
6. `玻鳍尘`（`Glassfin Dust`）+ `米瑞凝胶`（`Mirel Gel`）在 `glass_vial + ambient + medium humidity` 下通常 `SAFE`。
7. `赫利欧籽`（`Helio Seed`）+ `银辉露`（`Argent Dew`）在 `bright` 条件下必定 `DANGEROUS`，不受 `银封囊` 保护。
8. `暗影瓣`（`Umbra Petal`）+ `黯蜡`（`Tarn Wax`）在 `dark + amber_vial` 下通常 `SAFE`，但升温到 `hot` 后立即反转为 `DANGEROUS`。
9. `回声盐`（`Echor Salt`）+ `万特合金尘`（`Vant Alloy Dust`）在 `steel_tray` 下几乎总是 `DANGEROUS`，哪怕温度和湿度都温和。
10. `荒伏根`（`Fallow Root`）+ `欧林汁液`（`Orin Sap`）在 `dark + cold + amber_vial` 中可稳定，是有机材料的重要正例外。
11. `绒书孢子`（`Vellum Spores`）+ `青缓浆`（`Viridian Slurry`）在 `medium` 湿度可暂时稳定，但在 `high` 湿度下仍 `DANGEROUS`。
12. `霜珠`（`Rime Pearl`）+ `寂语晶`（`Hush Crystal`）+ `昆里尔锚石`（`Quenril Stone`）在 `cold` 中形成稳定低温簇，通常 `SAFE`。
13. `雾珀脂`（`Mist Amber`）+ `银辉露`（`Argent Dew`）在 `bright` 条件下永远 `DANGEROUS`，无论容器如何。
14. `布罗梅浆`（`Bromel Paste`）+ `柯拉灰`（`Kora Ash`）在 `ceramic_bowl` 中通常稳定，但若施加 `high pressure` 会失稳。
15. `费拉苔`（`Fera Moss`）+ `听辉花蜜`（`Aural Nectar`）只有在 `amber_vial + low humidity + dim light` 时才可视为 `SAFE`。
16. `夜铜液`（`Noctil Copper Gel`）+ `星铁须`（`Star-Iron Filaments`）在 `dark + still + ceramic_bowl` 下可短时稳定，但只要切换到 `circulating` 气流就转为 `DANGEROUS`。
17. `祈纹叶`（`Sigil Fern`）+ `青缓浆`（`Viridian Slurry`）在 `dark + medium humidity + settled` 下通常 `SAFE`，但若改为 `fresh` 阶段反而不稳定。
18. `褶镜粉`（`Foldglass Meal`）+ `玻鳍尘`（`Glassfin Dust`）在 `glass_vial + pulse_light` 下几乎总是 `DANGEROUS`，哪怕温度较低。
19. `静潮髓`（`Stilltide Marrow`）+ `玄铅匣`（`lead_casket`）在 `fresh` 阶段常被误判为安全，但若静置到 `delayed + hot` 会高概率反转为 `DANGEROUS`。
20. `磁滴浆`（`Magna Droplet`）+ `万特合金尘`（`Vant Alloy Dust`）在 `vacuum_dish + low pressure` 下通常 `SAFE`，这是少数压力型正例外。

---

## Host 绝不能直接泄露的信息

- 不要一次性列出全部规则或全部关键例外。
- 不要直接说“哪些材料本质上是危险材料”，因为很多危险都依赖条件翻转。
- 不要把材料家族的完整行为总结成现成答案交给 tested agent。
- 不要在单轮反馈里把未来多轮会反复考察的高价值例外一起说完。
- 不要把 `fresh / settled / delayed` 三阶段的全部意义一次讲透。
- 不要把“哪些容器只是表面稳定，实际会放大延迟风险”直接总结成全集。

---

## 可供抽题的任务素材

每轮任务尽量保持二分类，推荐格式：

- “给定材料组合、容器、环境条件与操作，回答 `SAFE` 或 `DANGEROUS`。”
- “上一轮失败后，补充一条局部规则，再给一个只改一个关键条件的新题。”
- “让 tested agent 分辨自己忽略的是湿度、光照、温度、容器、气流、阶段还是例外。”

推荐的出题策略：

1. 前 5 轮优先用高频基础规则建立直觉。
2. 第 6 到 12 轮开始引入单条件翻转。
3. 中期反复测试关键例外是否被压缩进 memory。
4. 后期加入 `fresh / settled / delayed` 这种延迟型陷阱。
5. 不要连续多轮只考同一种条件维度，应迫使 agent 学会跨维度抽象。
6. 若 tested agent 明显学会了某条基础规则，下一轮优先测试它最容易忽略的相邻例外。

示例任务种子：

- `琉辉七盐 + 费拉苔` in `glass_vial`, `high humidity`, `ambient`, `fresh`
- `听辉花蜜 + 雾珀脂` in `amber_vial`, `medium humidity`, `ambient`, `settled`
- `钴霜尘 + 努尔石英` in `silver_capsule`, `cold`, `dim`, `fresh`
- `暗影瓣 + 黯蜡` in `amber_vial`, `dark`, `hot`, `settled`
- `玻鳍尘 + 米瑞凝胶` in `glass_vial`, `medium humidity`, `ambient`, `fresh`
- `荒伏根 + 欧林汁液` in `amber_vial`, `cold`, `dark`, `settled`
- `赫利欧籽 + 银辉露` in `silver_capsule`, `bright`, `ambient`, `fresh`
- `布罗梅浆 + 柯拉灰` in `ceramic_bowl`, `high pressure`, `ambient`, `settled`
- `夜铜液 + 星铁须` in `ceramic_bowl`, `dark`, `still`, `fresh`
- `静潮髓 + 玄铅匣` in `lead_casket`, `hot`, `delayed`, `sealed`

---

## 判分原则

- 第一版统一只接受 `SAFE` 或 `DANGEROUS` 两种主答案。
- tested agent 若输出更长文本，取其首个清晰的二分类标签作为主回答。
- 若回答中同时出现两个相反标签，判为无效回答。
- 若同一道题命中关键例外，关键例外优先于基础规则。
- 若题目涉及 `delayed` 阶段，不能只凭 `fresh` 阶段直觉判定。
- Host 在反馈中可以指出“你忽略了湿度 / 光照 / 容器 / 阶段 / 气流 / 关键例外”，但不要总是给出完整正确规则。

---

## Host 行为约束

- 保持世界设定一致，不临时发明新材料、新容器或新阶段逻辑。
- 尽量让相邻轮次共享局部结构，促使 tested agent 压缩出规律，而不是只背答案。
- 不要把题做成纯随机抽样；要有教学轨迹和反例设计。
- 反馈强度以中等为主：指出错因方向，但不要总是把完整规则直接喂给 agent。
- 要持续制造“基础规则看似适用，但会被关键例外推翻”的学习压力。
