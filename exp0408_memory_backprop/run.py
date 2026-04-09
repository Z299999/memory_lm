import math
import sys
from pathlib import Path

import yaml

# ── Parameters ────────────────────────────────────────────────────────────────

RUN_ID          = "r00001"
NUM_PROBLEMS    = 10          # None = all 2869
STUB            = False       # True = offline fake LLM for smoke testing

# Override config.yaml values here (leave as None to use config defaults)
FORWARD_PER_BACK  = None      # e.g. 3, or math.inf for no training
MAX_FEEDBACK      = None      # e.g. 3

# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

from forward import run_forward

fpb_raw = FORWARD_PER_BACK if FORWARD_PER_BACK is not None else cfg["experiment"]["forward_per_back"]
forward_per_back = math.inf if (fpb_raw == ".inf" or fpb_raw is math.inf) else float(fpb_raw)

run_forward(
    data_path              = ROOT / cfg["experiment"]["data_path"],
    runs_dir               = ROOT / cfg["experiment"]["runs_dir"],
    run_id                 = RUN_ID,
    host_model             = cfg["host"]["model"],
    host_base_url          = cfg["host"].get("base_url"),
    host_api_key           = cfg["host"].get("api_key"),
    student_model          = cfg["student"]["model"],
    host_temperature       = cfg["host"]["temperature"],
    student_temperature    = cfg["student"]["temperature"],
    host_max_tokens        = cfg["host"]["max_tokens"],
    student_max_new_tokens = cfg["student"]["max_new_tokens"],
    max_feedback_rounds    = MAX_FEEDBACK if MAX_FEEDBACK is not None else cfg["experiment"]["max_feedback_rounds"],
    forward_per_back       = forward_per_back,
    num_problems           = NUM_PROBLEMS,
    stub                   = STUB,
)
