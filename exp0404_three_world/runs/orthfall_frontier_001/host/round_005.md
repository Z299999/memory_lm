## AGENT_INPUT

前哨生存勘测员，收到新任务简报：

你在**玻风平滩**(`Glasswind Flats`)区域执行侦察任务。当前环境条件：
- 光照：`splitlight` (分裂光态)
- 风态：`crosswind` (侧风) 
- 热态：`temperate` (温和)
- 折层活性：`stable` (稳定)
- 静止周期：`first_rest` (第一静止周期)
- 方向可靠度：`uncertain` (不确定)

你携带了**折幕帐**(`fold_tent`)，需要在此处扎营过夜。根据你的勘测经验判断，这个扎营方案是否安全？

请回答：`SAFE` 或 `DANGEROUS`

## CANONICAL_ANSWER

DANGEROUS

## SCORING_RATIONALE

在玻风平滩的splitlight条件下扎营是危险的，因为分裂光态会产生多重反射，制造方向错觉和视觉干扰。虽然折幕帐在stable折层中通常可靠，但玻风平滩的splitlight光照会通过帐篷材料产生额外的反射陷阱，增加迷失风险。这是对agent是否真正掌握玻风平滩光照危险模式的测试，而非仅仅记住bright条件的危险。

## NEXT_ROUND_INTENT

继续测试agent对玻风平滩光照条件的敏感性，准备在下轮引入相同的splitlight条件但改为石倚棚的对比，观察agent是否能识别出风险主要来自光照-地形交互而非庇护所类型差异。
