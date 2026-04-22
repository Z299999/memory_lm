## AGENT_INPUT

你在玻风平滩（Glasswind Flats）执行前哨勘测任务。当前条件：splitlight光照，crosswind风态，temperate温度，stable折层活性。你携带石倚棚（stone_lean_to）材料。你的任务是在此地建立临时营地过夜，这是第一静止周期。

基于你的勘测经验和当前条件，判断在此地使用石倚棚扎营是否安全：

A) SAFE - 可以在此扎营
B) DANGEROUS - 不应在此扎营

## CANONICAL_ANSWER

DANGEROUS

## SCORING_RATIONALE

在玻风平滩的splitlight条件下，地表反射会制造错误方向感，无论使用何种庇护所类型（包括石倚棚）都会面临方向迷失风险。虽然agent的external memory提到"折幕帐在splitlight/crosswind条件下安全"，但这与round 2-4中学到的"玻风平滩在bright条件下危险"的核心规则相冲突。实际上，splitlight在玻风平滩本身就构成危险条件，因为分裂光线会干扰方向判断，这与庇护所类型无关。

## NEXT_ROUND_INTENT

继续强化玻风平滩的光照危险核心概念，特别是让agent理解splitlight本身就在该区域构成根本性危险，与庇护所类型选择无关。准备引入相同区域不同光照的对比来巩固这一规则。
