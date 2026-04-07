## AGENT_INPUT
前哨生存勘测员，收到新地形简报：

你现在位于**玻风平滩**(`Glasswind Flats`)，当前环境条件：
- 光照：`dim` (昏暗)
- 风态：`crosswind` (侧风) 
- 温度：`temperate` (温和)
- 折层活性：`stable` (稳定)

你携带的庇护所装备：**石倚棚**(`stone_lean_to`)

任务：判断在此条件下使用石倚棚扎营是否安全。

请回答 `SAFE` 或 `DANGEROUS`。

## CANONICAL_ANSWER
SAFE

## SCORING_RATIONALE
在玻风平滩，dim光照条件相比splitlight/bright要安全得多，不会产生强烈的反射方向错判。虽然crosswind风态对石倚棚有积尘风险，但在dim光照下，地表反射问题显著减轻，整体风险可控。agent此前已学到splitlight+crosswind组合危险，现在应能区分dim光照相对安全的情况。

## NEXT_ROUND_INTENT
测试agent是否能区分玻风平滩中不同光照等级的风险差异。agent已知bright和splitlight都危险，现在验证其能否识别dim条件下的相对安全性，强化"风险主要来自光照反射强度"而非简单地认为"玻风平滩总是危险"的过度概括。
