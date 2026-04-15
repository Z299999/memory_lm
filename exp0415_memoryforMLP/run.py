"""exp0415 — MLP Online Learning Smoke Test

A shallow MLP receives a constant input (1.0) at every time step.
The targets repeat a configurable pattern cyclically, e.g.:
  [0, 1]        →  0 1 0 1 0 1 ...
  [-1, 0, 1, 0] → -1 0 1 0 -1 0 1 0 ...

Two training modes are compared side by side:

  standard      — apply current step's gradient immediately
  delayed_grad  — apply the PREVIOUS step's gradient; save current gradient
                  for the next step

The delayed-gradient insight:
  At time t, grad_{t-1} was computed for target_{t-1}.
  For a length-P pattern, target_{t-1} = target_{t-1+P} = target_{t+P-1}.
  When P=2: target_{t-1} = target_{t+1}, so each update aligns perfectly
  with the NEXT correct answer.

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
    """Simple MLP: [1] → hidden... → [1] with Tanh activations + Tanh output.

    Tanh output covers (-1, 1), supporting any target pattern in that range.
    """

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())   # output ∈ (-1, 1); covers all pattern values in [-1, 1]
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
    pattern = [float(v) for v in params.get("target_pattern", [0, 1])]
    period = len(pattern)

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
        target_val = pattern[t % period]
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
        "pattern": pattern,
        "mode": "delayed_grad" if delayed_grad else "standard",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _correct_count(outputs: list[float], targets: list[float], pattern: list[float]) -> int:
    """Count steps where output is closer to the correct target than to any other pattern value."""
    if len(pattern) == 1:
        return sum(1 for o, t in zip(outputs, targets) if abs(o - t) < 0.1)
    # threshold = half the minimum distance between adjacent distinct pattern values
    vals = sorted(set(pattern))
    min_gap = min(abs(vals[i+1] - vals[i]) for i in range(len(vals)-1))
    thresh = min_gap / 2
    return sum(1 for o, t in zip(outputs, targets) if abs(o - t) < thresh)


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

    pattern = std_result["pattern"]
    targets = std_result["targets"]   # same for both runs

    # y-axis range: slightly beyond the pattern value range
    y_lo = min(pattern) - 0.15
    y_hi = max(pattern) + 0.15
    y_mid = (min(pattern) + max(pattern)) / 2
    pat_str = str(pattern).replace(" ", "")

    fig = plt.figure(figsize=(14, 15))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.32)

    colors = {"standard": "steelblue", "delayed_grad": "darkorange"}
    labels = {"standard": "standard (grad_t → step t)",
              "delayed_grad": "delayed  (grad_{t-1} → step t)"}

    # ── Row 0: output vs target, standard ───────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_axis, targets, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax0.plot(t_axis, std_result["outputs"], color=colors["standard"],
             linewidth=1.4, label=labels["standard"])
    ax0.scatter(t_axis, std_result["outputs"], s=8, color=colors["standard"], zorder=3)
    ax0.axhline(y_mid, color="black", linestyle=":", linewidth=0.6, alpha=0.35)
    ax0.set_ylim(y_lo, y_hi)
    ax0.set_title(f"standard  |  correct={_correct_count(std_result['outputs'], targets, pattern)}/{steps}"
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
    ax1.axhline(y_mid, color="black", linestyle=":", linewidth=0.6, alpha=0.35)
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_title(f"delayed grad  |  correct={_correct_count(dg_result['outputs'], targets, pattern)}/{steps}"
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
    err_range = max(abs(y_hi - y_lo), 0.5) + 0.05
    ax4.set_ylim(-err_range, err_range)
    ax4.grid(True, alpha=0.22)

    ax5 = fig.add_subplot(gs[2, 1])
    errs_dg = [o - t for o, t in zip(dg_result["outputs"], targets)]
    ax5.bar(t_axis, errs_dg, color=_bar_colors(dg_result["outputs"], targets),
            width=0.8, alpha=0.7)
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_title("signed error — delayed grad")
    ax5.set_xlabel("time step")
    ax5.set_ylim(-err_range, err_range)
    ax5.grid(True, alpha=0.22)

    # ── Row 3: last-period zoom ──────────────────────────────────────────────
    # Show the last 5 complete periods so convergence quality is visible
    # even when total steps is very large.
    period = len(pattern)
    n_show = 5 * period
    n_show = min(n_show, steps)
    # align to a complete period boundary
    n_show = (n_show // period) * period
    n_show = max(n_show, period)

    t_zoom = t_axis[-n_show:]
    tgt_zoom = targets[-n_show:]
    pos_axis = list(range(n_show))   # relative positions within the zoom window

    for ax, result, key in [(fig.add_subplot(gs[3, 0]), std_result, "standard"),
                             (fig.add_subplot(gs[3, 1]), dg_result, "delayed_grad")]:
        out_zoom = result["outputs"][-n_show:]
        ax.plot(pos_axis, tgt_zoom, color="gray", linestyle="--", linewidth=1.2,
                alpha=0.7, label="target")
        ax.plot(pos_axis, out_zoom, color=colors[key], linewidth=1.6,
                label=f"output ({key})")
        ax.scatter(pos_axis, out_zoom, s=18, color=colors[key], zorder=4)
        ax.axhline(y_mid, color="black", linestyle=":", linewidth=0.6, alpha=0.3)
        # mark period boundaries
        for k in range(0, n_show + 1, period):
            ax.axvline(k - 0.5, color="gray", linewidth=0.5, alpha=0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.5, n_show - 0.5)
        ax.set_xlabel(f"relative step (last {n_show} steps = {n_show // period} periods)")
        ax.set_ylabel("value")
        n_correct_zoom = _correct_count(out_zoom, tgt_zoom, pattern)
        ax.set_title(f"last-period zoom — {key}  |  correct={n_correct_zoom}/{n_show}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        f"MLP online learning — arch={arch}  lr={lr}  opt={opt}  params={n_params}\n"
        f"Constant input x=1 every step; target pattern={pat_str} repeating   "
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
    print(f"  target_pattern: {params.get('target_pattern', [0, 1])}")
    print(f"  seed          : {params.get('seed', 42)}")
    print()

    std_result = run_online(params, delayed_grad=False)
    dg_result  = run_online(params, delayed_grad=True)

    steps = std_result["steps"]
    targets = std_result["targets"]
    pattern = std_result["pattern"]

    for label, result in [("standard    ", std_result), ("delayed_grad", dg_result)]:
        correct = _correct_count(result["outputs"], targets, pattern)
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
