"""exp0415 — MLP Online Learning Smoke Test

A shallow MLP receives a constant input (1.0) at every time step.
The targets alternate: 0, 1, 0, 1, ...

At each step we:
  1. Forward-pass x=1 → output y
  2. Record y and the current target
  3. Compute MSE(y, target) and backpropagate
  4. Update weights

The MLP has no recurrent connection and no explicit memory cell.
Its only "memory" is the set of weights, which change via backprop.
We want to see whether online learning lets the weights encode the
alternating pattern so that the network can track (or even predict)
the next target.

Run:  python3 run.py
Config: params.yaml
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_params(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP: [1] → hidden... → [1] with Tanh activations + Sigmoid output."""

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())   # output ∈ (0, 1); targets are 0 and 1
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def arch_str(self) -> str:
        widths = [m.in_features for m in self.net if isinstance(m, nn.Linear)]
        widths.append(1)   # final output dim
        return "[" + ", ".join(map(str, widths)) + "]"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------

def run_online(params: dict) -> dict:
    seed = params.get("seed", 42)
    torch.manual_seed(seed)

    hidden = params["hidden"]
    lr = params["lr"]
    steps = params["steps"]
    opt_name = params.get("optimizer", "sgd").lower()

    model = MLP(hidden)
    loss_fn = nn.MSELoss()

    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    x = torch.tensor([[1.0]])   # constant input — never changes

    rec_outputs: list[float] = []
    rec_targets: list[float] = []
    rec_losses: list[float] = []

    for t in range(steps):
        target_val = float(t % 2)   # 0, 1, 0, 1, ...
        y_hat = torch.tensor([[target_val]])

        # Record output BEFORE weight update so we see what the current
        # weights would predict, not what they predict after seeing the answer.
        with torch.no_grad():
            y_pre = model(x).item()
        rec_outputs.append(y_pre)
        rec_targets.append(target_val)

        # Now compute loss and update
        y = model(x)
        loss = loss_fn(y, y_hat)
        rec_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "outputs": rec_outputs,
        "targets": rec_targets,
        "losses": rec_losses,
        "model": model,
        "arch": model.arch_str(),
        "params": model.param_count(),
        "lr": lr,
        "optimizer": opt_name,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(result: dict, output_path: Path) -> None:
    outputs = result["outputs"]
    targets = result["targets"]
    losses = result["losses"]
    steps = result["steps"]
    arch = result["arch"]
    lr = result["lr"]
    opt = result["optimizer"]
    n_params = result["params"]

    t_axis = list(range(steps))

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: output vs target (full run) ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # spans both columns
    ax1.plot(t_axis, targets, color="gray", linestyle="--", linewidth=1,
             label="target (0,1,0,1,...)", alpha=0.7)
    ax1.plot(t_axis, outputs, color="steelblue", linewidth=1.5,
             label="MLP output (pre-update)")
    ax1.scatter(t_axis, outputs, s=10, color="steelblue", zorder=3)
    ax1.axhline(0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("time step")
    ax1.set_ylabel("value")
    ax1.set_title(
        f"MLP online learning — arch={arch}  lr={lr}  opt={opt}  params={n_params}\n"
        f"Constant input x=1 at every step; targets alternate 0→1→0→1→…"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: MSE loss per step ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_axis, losses, color="tomato", linewidth=1.2, label="MSE loss")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("MSE loss")
    ax2.set_title("Per-step MSE loss")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: signed prediction error ────────────────────────────────────
    errors = [o - t_ for o, t_ in zip(outputs, targets)]
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(t_axis, errors, color=["tomato" if e > 0 else "steelblue" for e in errors],
            width=0.8, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("time step")
    ax3.set_ylabel("output − target")
    ax3.set_title("Signed prediction error (output − target)")
    ax3.grid(True, alpha=0.25)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    params = load_params(here / "params.yaml")

    print("=" * 55)
    print("exp0415 — MLP Online Learning Smoke Test")
    print("=" * 55)
    print(f"  hidden layers : {params['hidden']}")
    print(f"  lr            : {params['lr']}")
    print(f"  optimizer     : {params.get('optimizer', 'sgd')}")
    print(f"  steps         : {params['steps']}")
    print(f"  seed          : {params.get('seed', 42)}")
    print()

    result = run_online(params)

    print(f"  arch          : {result['arch']}")
    print(f"  total params  : {result['params']}")
    print()

    # Quick summary stats
    outputs = result["outputs"]
    targets = result["targets"]
    correct = sum(
        1 for o, t in zip(outputs, targets)
        if abs(o - t) < 0.5   # rounded prediction matches target
    )
    print(f"  'correct' steps (|output-target| < 0.5): {correct}/{result['steps']}")
    final_loss = sum(result["losses"][-10:]) / 10
    print(f"  avg loss over last 10 steps: {final_loss:.4f}")
    print()

    runs_dir = here / "runs"
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = runs_dir / f"online_mlp_{timestamp}.png"

    save_plot(result, output_path)


if __name__ == "__main__":
    main()
