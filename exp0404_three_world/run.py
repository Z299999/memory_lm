from scripts.run_cli import run_defaults

# How to use:
# 1. Edit the values below.
# 2. Run `python3 run.py`.
#
# Most common setups:
# - Continue the latest unfinished run:
#   ACTION = "resume_latest"
# - Start a brand-new run:
#   ACTION = "new"
#   WORLD_PATH = "world/orthfall_frontier_world.md"
#   ROUNDS = 30
#   HOST_LANGUAGE = "zh"
# - Continue one specific run:
#   ACTION = "resume"
#   RUN_ID = "run_001"
# - Check status only:
#   ACTION = "status"
#
# Available worlds:
# - "world/asterion_lab.md"
# - "world/court_of_veils_world.md"
# - "world/orthfall_frontier_world.md"

ACTION = "resume_latest"  # "new" | "resume" | "resume_latest" | "smoke" | "status"
RUN_ID = "confluence_002"  # only used when ACTION is "resume", or when you want to name a new run
WORLD_PATH = "world/confluence_world.md"  # only used for "new" and "smoke"
ROUNDS = 50  # only used for "new" and "smoke"
MEMORY_BUDGET = 5000  # only used for "new"
MEMORY_MODE = "prepend"  # only used for "new": "rewrite" | "prepend"
TESTED_MODEL = "qwen3-coder-plus"  # only used for "new"
HOST_MODEL = "qwen3-coder-plus"  # only used for "new"
HOST_LANGUAGE = "zh"  # "zh" = host writes Chinese only, "en" = host writes English only
AUTO_ACCEPT_HOST = True  # True = do not pause for host review
STUB_LLM = False  # True = offline fake model for smoke testing

raise SystemExit(
    run_defaults(
        action=ACTION,
        run_id=RUN_ID,
        world_path=WORLD_PATH,
        rounds=ROUNDS,
        memory_budget=MEMORY_BUDGET,
        memory_mode=MEMORY_MODE,
        tested_model=TESTED_MODEL,
        host_model=HOST_MODEL,
        host_language=HOST_LANGUAGE,
        auto_accept_host=AUTO_ACCEPT_HOST,
        stub_llm=STUB_LLM,
    )
)
