"""Offline analysis entrypoint for continuous-collapse diagnosis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


def _add_src_to_path() -> Path:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze continuous-collapse checkpoints for exp0522.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a completed exp0522 run directory.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional checkpoint prefix to analyze, e.g. full_language (legacy names like v0_open_loop still work).",
    )
    return parser.parse_args()


def main() -> None:
    temp_root = Path(tempfile.gettempdir())
    mpl_dir = temp_root / "exp0522_mplconfig"
    xdg_cache_dir = temp_root / "exp0522_xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    _add_src_to_path()
    from collapse_analysis import analyze_continuous_collapse

    args = parse_args()
    result = analyze_continuous_collapse(
        Path(args.run_dir).expanduser(),
        model_name=args.model_name,
    )
    print(result["analysis_dir"])
    print(result["metrics_path"])


if __name__ == "__main__":
    main()
