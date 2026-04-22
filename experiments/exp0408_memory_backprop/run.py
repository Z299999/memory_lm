import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
exp = cfg["experiment"]


def _parse_inf(val) -> float:
    return math.inf if (val == ".inf" or val is math.inf) else float(val)


def _existing_run_ids(runs_dir: Path) -> list[int]:
    """Return sorted list of existing r0000x run numbers."""
    nums = []
    for d in runs_dir.iterdir():
        if d.is_dir():
            m = re.fullmatch(r"r(\d+)", d.name)
            if m:
                nums.append(int(m.group(1)))
    return sorted(nums)


def _is_complete(run_dir: Path) -> bool:
    """Check if a run finished cleanly (transcript ends with 'Finished:')."""
    transcript = run_dir / "transcript.md"
    if not transcript.exists():
        return False
    return "Finished:" in transcript.read_text(encoding="utf-8")


def resolve_run_id(runs_dir: Path, mode: str) -> str:
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_run_ids(runs_dir)

    if mode == "new":
        next_num = (max(existing) + 1) if existing else 1
        run_id = f"r{next_num:05d}"
        print(f"[run] mode=new → {run_id}")
        return run_id

    if mode == "resume":
        # Find the latest incomplete run
        for num in reversed(existing):
            run_dir = runs_dir / f"r{num:05d}"
            if not _is_complete(run_dir):
                run_id = f"r{num:05d}"
                print(f"[run] mode=resume → {run_id}")
                return run_id
        # No incomplete run found — start a new one
        next_num = (max(existing) + 1) if existing else 1
        run_id = f"r{next_num:05d}"
        print(f"[run] mode=resume but no incomplete run found → new {run_id}")
        return run_id

    raise ValueError(f"Unknown mode: {mode!r}. Use 'new' or 'resume'.")


runs_dir = ROOT / exp["runs_dir"]
run_id = resolve_run_id(runs_dir, exp.get("mode", "new"))

from forward import run_forward

run_forward(
    data_path              = ROOT / exp["data_path"],
    runs_dir               = runs_dir,
    run_id                 = run_id,
    host_model             = cfg["host"]["model"],
    host_base_url          = cfg["host"].get("base_url"),
    host_api_key           = cfg["host"].get("api_key"),
    student_model          = cfg["student"]["model"],
    host_temperature       = cfg["host"]["temperature"],
    student_temperature    = cfg["student"]["temperature"],
    host_max_tokens        = cfg["host"]["max_tokens"],
    student_max_new_tokens = cfg["student"]["max_new_tokens"],
    max_feedback_rounds    = exp["max_feedback_rounds"],
    forward_per_back       = _parse_inf(exp["forward_per_back"]),
    viz_every              = _parse_inf(exp["viz_every"]),
    num_problems           = exp["num_problems"],
    stub                   = exp["stub"],
)
