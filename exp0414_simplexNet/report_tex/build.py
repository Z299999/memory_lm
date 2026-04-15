#!/usr/bin/env python3
"""Build script for report.tex with SyncTeX support."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
MAIN = "report"

PDFLATEX = [
    "pdflatex",
    "-synctex=1",
    "-interaction=nonstopmode",
    f"{MAIN}.tex",
]


def run(cmd):
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        sys.exit(result.returncode)


def build():
    run(PDFLATEX)
    run(["bibtex", MAIN])
    run(PDFLATEX)
    run(PDFLATEX)
    print(f"\nDone → {HERE / MAIN}.pdf")


if __name__ == "__main__":
    build()
