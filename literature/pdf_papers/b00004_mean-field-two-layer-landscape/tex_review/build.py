#!/usr/bin/env python3
"""Build script for the LaTeX reading review."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = "main"


def run(cmd):
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        sys.exit(result.returncode)


def build():
    pdflatex = ["pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex"]
    run(pdflatex)
    run(["bibtex", MAIN])
    run(pdflatex)
    run(pdflatex)
    print(f"Done -> {HERE / (MAIN + '.pdf')}")


if __name__ == "__main__":
    build()
