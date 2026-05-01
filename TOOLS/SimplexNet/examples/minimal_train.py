#!/usr/bin/env python3
"""Minimal supervised training example for SimplexNet with plots."""

from pathlib import Path
import sys

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simplexnet import SMN


def _make_output_dir() -> Path:
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + 0.3 * torch.sin(3.0 * x)


class BaselineMLP(nn.Module):
    """Small MLP baseline with parameter count close to the SMN demo."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    lr: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    losses: list[float] = []

    model.train()
    for _epoch in range(epochs):
        pred = model(x)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return losses


def plot_loss_curve(
    smn_losses: list[float],
    mlp_losses: list[float],
    smn_summary: str,
    mlp_summary: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(smn_losses, color="#2563eb", linewidth=2.2, label="SMN")
    ax.plot(mlp_losses, color="#ea580c", linewidth=2.0, label="MLP baseline")
    ax.set_title("Toy Regression Loss Curve\nSMN vs similar-parameter MLP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.98,
        smn_summary + "\n" + mlp_summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fit(
    x: torch.Tensor,
    target: torch.Tensor,
    smn_pred: torch.Tensor,
    mlp_pred: torch.Tensor,
    smn_summary: str,
    mlp_summary: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x.squeeze(1).cpu().numpy(), target.squeeze(1).cpu().numpy(), label="Target", linewidth=2.4)
    ax.plot(
        x.squeeze(1).cpu().numpy(),
        smn_pred.squeeze(1).cpu().numpy(),
        label="SMN prediction",
        linewidth=2.2,
    )
    ax.plot(
        x.squeeze(1).cpu().numpy(),
        mlp_pred.squeeze(1).cpu().numpy(),
        label="MLP prediction",
        linewidth=2.0,
        linestyle="--",
    )
    ax.set_title("Toy Regression Fit\nSMN vs similar-parameter MLP")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.02,
        smn_summary + "\n" + mlp_summary,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)

    output_dir = _make_output_dir()
    loss_path = output_dir / "minimal_train_loss_curve.png"
    fit_path = output_dir / "minimal_train_fit.png"

    model = SMN(n=3, m=10, n_in=1, n_out=1, activation="relu")
    baseline = BaselineMLP(width=32)

    x = torch.linspace(-2.0 * math.pi, 2.0 * math.pi, 256).unsqueeze(1)
    target = target_function(x)
    epochs = 1200
    lr = 2e-3

    smn_losses = train_model(model, x, target, epochs=epochs, lr=lr)

    torch.manual_seed(0)
    baseline = BaselineMLP(width=32)
    mlp_losses = train_model(baseline, x, target, epochs=epochs, lr=lr)

    model.eval()
    baseline.eval()
    with torch.no_grad():
        smn_pred = model(x)
        mlp_pred = baseline(x)

    smn_final_loss = smn_losses[-1]
    mlp_final_loss = mlp_losses[-1]
    smn_summary = (
        f"SMN: n=3, m=10, params={param_count(model)}, "
        f"epochs={epochs}, lr={lr:.0e}, final MSE={smn_final_loss:.4f}"
    )
    mlp_summary = (
        f"MLP: 1-32-32-1, params={param_count(baseline)}, "
        f"epochs={epochs}, lr={lr:.0e}, final MSE={mlp_final_loss:.4f}"
    )
    plot_loss_curve(smn_losses, mlp_losses, smn_summary, mlp_summary, loss_path)
    plot_fit(x, target, smn_pred, mlp_pred, smn_summary, mlp_summary, fit_path)

    print(model.arch_str)
    print(f"smn_final_loss={smn_final_loss:.6f}")
    print(f"mlp_final_loss={mlp_final_loss:.6f}")
    print(f"smn_params={param_count(model)}")
    print(f"mlp_params={param_count(baseline)}")
    print(loss_path)
    print(fit_path)


if __name__ == "__main__":
    main()
