## AGENT_INPUT

前哨生存勘测员，根据当前环境数据做出判断：

你在**玻风平滩**（Glasswind Flats）区域执行扎营评估任务。当前环境条件：
- 光照：`bright`（明亮光照）
- 风态：`crosswind`（侧风）
- 折层活性：`stable`（稳定）
- 静止周期：`first_rest`（第一静止周期）
- 方向可靠度：`clear`（清晰）

你携带了基础装备，需要在此处过夜。请评估：在这种环境下使用**开敞篷布**（open_tarp）扎营是否安全？

回答 `SAFE` 或 `DANGEROUS`。

## CANONICAL_ANSWER

DANGEROUS

## SCORING_RATIONALE

在玻风平滩的bright光照条件下，地表反射会制造错误方向感并可能导致方向错判风险。虽然当前是first_rest且方向clear，但bright光照结合开敞篷布的暴露特性，在此区域构成了方向错判+暴露风险的复合威胁。agent应基于前两轮对玻风平滩光照风险的学习，识别出bright条件下的扎营危险性。

## NEXT_ROUND_INTENT

测试agent是否将玻风平滩的光照风险规则泛化应用，特别是在bright条件下扎营的具体操作判断。准备在下轮引入相同区域但不同扎营方式（如石倚棚）的对比，强化agent对具体风险组合的记忆。
