## AGENT_INPUT
前哨生存勘测员，收到新地形简报：

你在玻风平滩（Glasswind Flats）区域，当前条件：
- 光照：bright（强光）
- 风态：crosswind（侧风）  
- 折层活性：stable（稳定）
- 静止周期：first_rest（第一静止期）

你拥有以下装备：石倚棚（stone_lean_to）、辉藓（glow_lichen）、镜盐（mirror_salt）

任务：评估在此条件下使用石倚棚扎营是否安全。

请回答 SAFE 或 DANGEROUS。

## CANONICAL_ANSWER
DANGEROUS

## SCORING_RATIONALE
虽然石倚棚比开敞篷布更稳定，但在玻风平滩的bright光照条件下仍然危险。根据规则6，石倚棚在downpull风态下会积尘，但更重要的是玻风平滩的bright条件会产生强反射和方向错判风险（规则1）。此外，玻鬣豺会追逐反射（例外13），而石倚棚的结构可能在强光下产生危险反射。即使折层稳定，光照反射风险仍使此地扎营危险。

## NEXT_ROUND_INTENT
继续强化玻风平滩的光照危险模式，但这次引入不同的庇护所类型（石倚棚vs开敞篷布），测试agent是否理解风险主要来自光照反射而非庇护所类型，并观察agent是否会记住bright条件下的通用危险模式。
