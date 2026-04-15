"""exp0415 — MLP Online Learning Smoke Test

A shallow MLP receives a constant input (1.0) at every time step.
The targets alternate: 0, 1, 0, 1, ...

Two training modes are compared side by side:

  standard      — apply current step's gradient immediately
  delayed_grad  — apply the PREVIOUS step's gradient; save current gradient
                  for the next step

The delayed-gradient insight:
  At time t, grad_{t-1} was computed for target_{t-1} = (t-1)%2.
  Because the sequence alternates with period 2, target_{t-1} = target_{t+1}.
  So applying grad_{t-1} at time t nudges the weights toward producing
  the CORRECT output at t+1 — naturally aligning updates with future targets.

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
        widths.append(1)
        return "[" + ", ".join(map(str, widths)) + "]"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------

def run_online(params: dict, delayed_grad: bool = False) -> dict:
    """
    Run the online learning loop.

    delayed_grad=False  →  standard: apply grad_t at step t
    delayed_grad=True   →  delayed:  apply grad_{t-1} at step t (save grad_t for t+1)
    """
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

    x = torch.tensor([[1.0]])

    rec_outputs: list[float] = []
    rec_targets: list[float] = []
    rec_losses: list[float] = []

    prev_grads: list[torch.Tensor] | None = None

    for t in range(steps):
        target_val = float(t % 2)
        y_hat = torch.tensor([[target_val]])

        # Record output BEFORE any weight update
        with torch.no_grad():
            y_pre = model(x).item()
        rec_outputs.append(y_pre)
        rec_targets.append(target_val)
        rec_losses.append((y_pre - target_val) ** 2)

        # Forward pass for gradient computation
        y = model(x)
        loss = loss_fn(y, y_hat)

        optimizer.zero_grad()
        loss.backward()

        if delayed_grad:
            # Save current gradients before any step
            current_grads = [p.grad.detach().clone() for p in model.parameters()]

            if prev_grads is not None:
                # Overwrite .grad with the PREVIOUS step's gradient, then step
                for p, g in zip(model.parameters(), prev_grads):
                    p.grad = g.clone()
                optimizer.step()

            prev_grads = current_grads
        else:
            # Standard: apply current gradient immediately
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
        "mode": "delayed_grad" if delayed_grad else "standard",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _correct_count(outputs: list[float], targets: list[float]) -> int:
    return sum(1 for o, t in zip(outputs, targets) if abs(o - t) < 0.5)


def _avg_loss(losses: list[float], n: int = 10) -> float:
    return sum(losses[-n:]) / min(n, len(losses))


# ---------------------------------------------------------------------------
# Plotting — two-mode comparison
# ---------------------------------------------------------------------------

def save_comparison_plot(
    std_result: dict,
    dg_result: dict,
    output_path: Path,
) -> None:
    steps = std_result["steps"]
    arch = std_result["arch"]
    lr = std_result["lr"]
    opt = std_result["optimizer"]
    n_params = std_result["params"]
    t_axis = list(range(steps))

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    colors = {"standard": "steelblue", "delayed_grad": "darkorange"}
    labels = {"standard": "standard (grad_t → step t)",
              "delayed_grad": "delayed  (grad_{t-1} → step t)"}

    targets = std_result["targets"]   # same for both runs

    # ── Row 0: output vs target, standard ───────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_axis, targets, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax0.plot(t_axis, std_result["outputs"], color=colors["standard"],
             linewidth=1.4, label=labels["standard"])
    ax0.scatter(t_axis, std_result["outputs"], s=8, color=colors["standard"], zorder=3)
    ax0.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.35)
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_title(f"standard  |  correct={_correct_count(std_result['outputs'], targets)}/{steps}"
                  f"  |  last10_loss={_avg_loss(std_result['losses']):.4f}")
    ax0.set_ylabel("output")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.22)

    # ── Row 0: output vs target, delayed ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t_axis, targets, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.plot(t_axis, dg_result["outputs"], color=colors["delayed_grad"],
             linewidth=1.4, label=labels["delayed_grad"])
    ax1.scatter(t_axis, dg_result["outputs"], s=8, color=colors["delayed_grad"], zorder=3)
    ax1.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.35)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"delayed grad  |  correct={_correct_count(dg_result['outputs'], targets)}/{steps}"
                  f"  |  last10_loss={_avg_loss(dg_result['losses']):.4f}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.22)

    # ── Row 1: MSE loss ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_axis, std_result["losses"], color=colors["standard"], linewidth=1.2)
    ax2.set_title("MSE loss — standard")
    ax2.set_ylabel("MSE loss")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.22)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_axis, dg_result["losses"], color=colors["delayed_grad"], linewidth=1.2)
    ax3.set_title("MSE loss — delayed grad")
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, alpha=0.22)

    # ── Row 2: signed error ──────────────────────────────────────────────────
    def _bar_colors(outputs, targets):
        return ["tomato" if o - t > 0 else "steelblue" for o, t in zip(outputs, targets)]

    ax4 = fig.add_subplot(gs[2, 0])
    errs_std = [o - t for o, t in zip(std_result["outputs"], targets)]
    ax4.bar(t_axis, errs_std, color=_bar_colors(std_result["outputs"], targets),
            width=0.8, alpha=0.7)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_title("signed error — standard")
    ax4.set_xlabel("time step")
    ax4.set_ylabel("output − target")
    ax4.set_ylim(-1.1, 1.1)
    ax4.grid(True, alpha=0.22)

    ax5 = fig.add_subplot(gs[2, 1])
    errs_dg = [o - t for o, t in zip(dg_result["outputs"], targets)]
    ax5.bar(t_axis, errs_dg, color=_bar_colors(dg_result["outputs"], targets),
            width=0.8, alpha=0.7)
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_title("signed error — delayed grad")
    ax5.set_xlabel("time step")
    ax5.set_ylim(-1.1, 1.1)
    ax5.grid(True, alpha=0.22)

    fig.suptitle(
        f"MLP online learning — arch={arch}  lr={lr}  opt={opt}  params={n_params}\n"
        f"Constant input x=1; targets alternate 0→1→0→1→…   "
        f"Left: standard SGD   Right: delayed-gradient SGD",
        fontsize=11,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    params = load_params(here / "params.yaml")

    print("=" * 60)
    print("exp0415 — MLP Online Learning Smoke Test (both modes)")
    print("=" * 60)
    print(f"  hidden layers : {params['hidden']}")
    print(f"  lr            : {params['lr']}")
    print(f"  optimizer     : {params.get('optimizer', 'sgd')}")
    print(f"  steps         : {params['steps']}")
    print(f"  seed          : {params.get('seed', 42)}")
    print()

    std_result = run_online(params, delayed_grad=False)
    dg_result  = run_online(params, delayed_grad=True)

    steps = std_result["steps"]
    targets = std_result["targets"]

    for label, result in [("standard    ", std_result), ("delayed_grad", dg_result)]:
        correct = _correct_count(result["outputs"], targets)
        ll = _avg_loss(result["losses"])
        print(f"  [{label}]  correct={correct}/{steps}  avg_last10_loss={ll:.4f}")
    print()

    runs_dir = here / "runs"
    runs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = runs_dir / f"comparison_{timestamp}.png"

    save_comparison_plot(std_result, dg_result, output_path)


if __name__ == "__main__":
    main()
