---
name: debug smoke test
description: User guidance about using smoke tests for debugging instead of full runs
type: feedback
---

**Rule**: When debugging or testing new features, use a smoke test with minimal epochs (e.g., 50) instead of running the full experiment.

**Why**: User explicitly pointed out that debug runs should be quick smoke tests, not full 10000-epoch runs. Full runs waste time when the goal is just to verify the code works.

**How to apply**: 
- When testing visualization changes, set `epochs: 50` in params.yaml
- Only run full experiments once the code is verified working
- "smoke test" = fastest possible run to verify no crashes + output is correct
