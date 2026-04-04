## AGENT_INPUT
前哨生存勘测员，收到当前地貌简报：

你在**玻风平滩**(`Glasswind Flats`)区域边缘，当前状态：
- 光照：`splitlight`（分裂光线）
- 风态：`crosswind`（侧风）
- 折层活性：`stable`（稳定）
- 静止周期：`first_rest`（第一静止周期）
- 方向可靠度：`clear`（清晰）

你携带基础装备，需要判断：在此条件下，使用**开敞篷布**(`open_tarp`)在此地扎营过夜是否安全？

请回答`SAFE`或`DANGEROUS`。
## CANONICAL_ANSWER
DANGEROUS
## SCORING_RATIONALE
在玻风平滩的splitlight条件下，地表反射会制造错误方向感，而且splitlight比bright更危险。虽然其他条件看似稳定，但玻风平滩的分裂光线环境对开敞篷布扎营构成方向迷失风险，应判为DANGEROUS。
## NEXT_ROUND_INTENT
测试agent是否理解玻风平滩的光照风险模式，后续可对比bright条件下的相同任务来强化splitlight的危险性。
