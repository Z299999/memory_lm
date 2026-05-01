#!/usr/bin/env python3
"""Minimal forward-pass example for SimplexNet."""

import sys

import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simplexnet import SMN


def main() -> None:
    torch.manual_seed(0)
    model = SMN(n=2, m=4, n_in=3, n_out=2)
    x = torch.randn(8, 3)
    y = model(x)
    print(model.arch_str)
    print(y.shape)


if __name__ == "__main__":
    main()
