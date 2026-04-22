## AGENT_INPUT
前哨生存勘测员，收到新情况更新：

你现在仍在玻风平滩（Glasswind Flats）区域，但光照条件已从之前的分裂光线（splitlight）变为明亮光照（bright）。其他条件保持不变：侧风风态，折层稳定，第一静止周期，方向清晰。

你的任务是评估相同的扎营方案：使用开敞篷布在当前位置过夜。

基于你之前的经验和当前的地貌情报，请判断这个扎营行动是 SAFE 还是 DANGEROUS？

## CANONICAL_ANSWER
DANGEROUS

## SCORING_RATIONALE
在玻风平滩上，bright光照比splitlight更危险，因为地表反射会制造错误方向感并可能吸引玻鬣豺等追光生物。agent之前在splitlight条件下判断SAFE已经错误，现在光照变得更明亮，危险性进一步增加，正确答案应为DANGEROUS。

## NEXT_ROUND_INTENT
强化agent对玻风平滩光照条件变化的敏感性，特别是从splitlight到bright的危险升级模式，让agent学会区分不同光照等级的风险差异。
