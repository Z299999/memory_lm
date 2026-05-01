#!/usr/bin/env python3
"""Minimal supervised training example for SimplexNet with plots."""

from pathlib import Path
import sys

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simplexnet import SMN


def _make_output_dir() -> Path:
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + 0.3 * torch.sin(3.0 * x)


def plot_loss_curve(losses: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(losses, color="#2563eb", linewidth=2)
    ax.set_title("SMN Toy Regression Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fit(
    x: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    save_path: Path,
    final_loss: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x.squeeze(1).cpu().numpy(), target.squeeze(1).cpu().numpy(), label="Target", linewidth=2.4)
    ax.plot(x.squeeze(1).cpu().numpy(), pred.squeeze(1).cpu().numpy(), label="SMN prediction", linewidth=2.0)
    ax.set_title(f"SMN Toy Regression Fit\nFinal MSE: {final_loss:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)

    output_dir = _make_output_dir()
    loss_path = output_dir / "minimal_train_loss_curve.png"
    fit_path = output_dir / "minimal_train_fit.png"

    model = SMN(n=3, m=10, n_in=1, n_out=1, activation="relu")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = torch.nn.MSELoss()

    x = torch.linspace(-2.0 * math.pi, 2.0 * math.pi, 256).unsqueeze(1)
    target = target_function(x)
    losses: list[float] = []

    model.train()
    for _epoch in range(1200):
        pred = model(x)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred = model(x)

    final_loss = losses[-1]
    plot_loss_curve(losses, loss_path)
    plot_fit(x, target, pred, fit_path, final_loss)

    print(model.arch_str)
    print(f"final_loss={final_loss:.6f}")
    print(loss_path)
    print(fit_path)


if __name__ == "__main__":
    main()
