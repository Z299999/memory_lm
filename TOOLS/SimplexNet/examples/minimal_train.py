#!/usr/bin/env python3
"""Minimal supervised training example for SimplexNet."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simplexnet import SMN


def main() -> None:
    model = SMN(n=3, m=5, n_in=4, n_out=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    x = torch.randn(32, 4)
    target = torch.randn(32, 1)

    pred = model(x)
    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(model.arch_str)
    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
