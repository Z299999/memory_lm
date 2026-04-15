"""exp0415 — MLP Online Learning Smoke Test

A shallow MLP receives a constant input (1.0) at every time step.
The targets repeat a configurable pattern cyclically, e.g.:
  [0, 1]        →  0 1 0 1 0 1 ...
  [-1, 0, 1, 0] → -1 0 1 0 -1 0 1 0 ...

Three training modes are compared side by side:

  standard      — apply current step's gradient to every layer immediately
                  W_{t+1}^(ℓ) = W_t^(ℓ) − η · g_t^(ℓ)

  delayed_grad  — apply the PREVIOUS step's gradient to ALL layers
                  W_{t+1}^(ℓ) = W_t^(ℓ) − η · g_{t-1}^(ℓ)

  layer_delay   — Layer-Delayed Backpropagation (layerDelayBP.md):
                  each layer ℓ uses a gradient whose age equals its
                  distance from the output:
                    W_{t+1}^(ℓ) = W_t^(ℓ) − η_ℓ · g_{t-(d-ℓ)}^(ℓ)
                  Output layer (ℓ=d): delay 0 — fast, short-term memory
                  Input  layer (ℓ=1): delay d-1 — slow, long-term memory

Run:  python3 run.py
Config: params.yaml
"""

from __future__ import annotations

from collections import deque

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
    """Simple MLP: [1] → hidden... → [1] with Tanh activations + Tanh output."""

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def arch_str(self) -> str:
        widths = [m.in_features for m in self.net if isinstance(m, nn.Linear)]
        widths.append(1)
        return "[" + ", ".join(map(str, widths)) + "]"

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def linear_layers(self) -> list[nn.Linear]:
        return [m for m in self.net if isinstance(m, nn.Linear)]


# ---------------------------------------------------------------------------
# Online training loop
# ---------------------------------------------------------------------------

def run_online(params: dict, mode: str = "standard") -> dict:
    """
    Run the online learning loop.

    mode="standard"    — apply grad_t to all layers at step t
    mode="delayed_grad"— apply grad_{t-1} to all layers at step t
    mode="layer_delay" — layer-delayed BP: layer ℓ uses grad_{t-(d-ℓ)}
    """
    assert mode in ("standard", "delayed_grad", "layer_delay")

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

    # For layer_delay: identify Linear layers and set up gradient cache.
    # linear_layers[i] is layer i (0=input side, d-1=output side).
    # delay for layer i = d - 1 - i
    # Cache stores G_t = list of (w_grad, b_grad) tuples, one per linear layer.
    lin_layers = model.linear_layers()
    d = len(lin_layers)
    grad_cache: deque = deque(maxlen=d)   # used for both delayed modes

    rec_outputs: list[float] = []
    rec_targets: list[float] = []
    rec_losses: list[float] = []

    for t in range(steps):
        target_val = pattern[t % period]
        y_hat = torch.tensor([[target_val]])

        # Record pre-update output
        with torch.no_grad():
            rec_outputs.append(model(x).item())
        rec_targets.append(target_val)
        rec_losses.append((rec_outputs[-1] - target_val) ** 2)

        # Forward pass + compute gradients for current step
        optimizer.zero_grad()
        y = model(x)
        loss = loss_fn(y, y_hat)
        loss.backward()

        # ---------- build and cache current gradient pack ----------
        G_t: list[tuple] = []
        for layer in lin_layers:
            w_g = layer.weight.grad.detach().clone() if layer.weight.grad is not None else None
            b_g = (layer.bias.grad.detach().clone()
                   if layer.bias is not None and layer.bias.grad is not None else None)
            G_t.append((w_g, b_g))
        grad_cache.append(G_t)

        # ---------- apply gradient according to mode ----------
        if mode == "standard":
            # .grad already holds current gradient — just step
            optimizer.step()

        elif mode == "delayed_grad":
            # Apply G_{t-1} (second-to-last in cache) to ALL layers
            if len(grad_cache) >= 2:
                G_prev = grad_cache[-2]
                for i, layer in enumerate(lin_layers):
                    w_g, b_g = G_prev[i]
                    if w_g is not None:
                        layer.weight.grad = w_g.clone()
                    if b_g is not None and layer.bias is not None:
                        layer.bias.grad = b_g.clone()
                optimizer.step()
            # else: first step, nothing to apply yet

        elif mode == "layer_delay":
            # Layer i (0=input, d-1=output) uses G_{t-(d-1-i)}.
            # In the deque (maxlen=d, just appended G_t):
            #   cache[-1] = G_t  (delay 0, for output layer)
            #   cache[-2] = G_{t-1} (delay 1)
            #   cache[-(d-i)] = G_{t-(d-1-i)}
            # Skip layer if the required old gradient is not in cache yet.
            any_updated = False
            for i, layer in enumerate(lin_layers):
                cache_pos = -(d - i)      # negative index: -d for input, -1 for output
                if abs(cache_pos) > len(grad_cache):
                    # Gradient too old, not yet available → zero this layer's grad
                    if layer.weight.grad is not None:
                        layer.weight.grad.zero_()
                    if layer.bias is not None and layer.bias.grad is not None:
                        layer.bias.grad.zero_()
                    continue
                w_g, b_g = grad_cache[cache_pos][i]
                if w_g is not None:
                    layer.weight.grad = w_g.clone()
                if b_g is not None and layer.bias is not None:
                    layer.bias.grad = b_g.clone()
                any_updated = True
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
        "mode": mode,
        "d": d,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _correct_count(outputs: list[float], targets: list[float], pattern: list[float]) -> int:
    if len(pattern) == 1:
        return sum(1 for o, t in zip(outputs, targets) if abs(o - t) < 0.1)
    vals = sorted(set(pattern))
    min_gap = min(abs(vals[i+1] - vals[i]) for i in range(len(vals)-1))
    thresh = min_gap / 2
    return sum(1 for o, t in zip(outputs, targets) if abs(o - t) < thresh)


def _avg_loss(losses: list[float], n: int = 10) -> float:
    return sum(losses[-n:]) / min(n, len(losses))


# ---------------------------------------------------------------------------
# Plotting — N-mode comparison (flexible columns)
# ---------------------------------------------------------------------------

# One colour per mode, consistent across all plots
MODE_COLORS = {
    "standard":    "steelblue",
    "delayed_grad":"darkorange",
    "layer_delay": "mediumseagreen",
}
MODE_LABELS = {
    "standard":    "standard  (grad_t → all layers)",
    "delayed_grad":"delayed   (grad_{t-1} → all layers)",
    "layer_delay": "layer-delay (grad_{t-(d-ℓ)} → layer ℓ)",
}


def save_comparison_plot(results: list[dict], output_path: Path) -> None:
    n_modes = len(results)
    steps   = results[0]["steps"]
    arch    = results[0]["arch"]
    lr      = results[0]["lr"]
    opt     = results[0]["optimizer"]
    n_params= results[0]["params"]
    pattern = results[0]["pattern"]
    targets = results[0]["targets"]
    t_axis  = list(range(steps))
    pat_str = str(pattern).replace(" ", "")

    y_lo  = min(pattern) - 0.15
    y_hi  = max(pattern) + 0.15
    y_mid = (min(pattern) + max(pattern)) / 2
    err_range = max(abs(y_hi - y_lo), 0.5) + 0.05

    fig_w = 6.5 * n_modes
    fig = plt.figure(figsize=(fig_w, 15))
    gs  = gridspec.GridSpec(4, n_modes, figure=fig, hspace=0.50, wspace=0.30)

    def _bar_colors(outs, tgts):
        return ["tomato" if o - t > 0 else "steelblue" for o, t in zip(outs, tgts)]

    for col, res in enumerate(results):
        mode  = res["mode"]
        color = MODE_COLORS.get(mode, "gray")
        label = MODE_LABELS.get(mode, mode)
        outs  = res["outputs"]
        correct = _correct_count(outs, targets, pattern)
        ll      = _avg_loss(res["losses"])

        # ── Row 0: output vs target ──────────────────────────────────────────
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t_axis, targets, color="gray", linestyle="--", lw=1, alpha=0.6)
        ax.plot(t_axis, outs, color=color, lw=1.4, label=label)
        ax.scatter(t_axis, outs, s=6, color=color, zorder=3)
        ax.axhline(y_mid, color="black", linestyle=":", lw=0.6, alpha=0.35)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{mode}\ncorrect={correct}/{steps}  last10_loss={ll:.4f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("output")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.22)

        # ── Row 1: MSE loss ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, col])
        ax.plot(t_axis, res["losses"], color=color, lw=1.2)
        ax.set_title(f"MSE loss — {mode}", fontsize=9)
        ax.set_ylim(-0.02, max(res["losses"]) * 1.05 + 0.01)
        if col == 0:
            ax.set_ylabel("MSE loss")
        ax.grid(True, alpha=0.22)

        # ── Row 2: signed error ──────────────────────────────────────────────
        errs = [o - t for o, t in zip(outs, targets)]
        ax = fig.add_subplot(gs[2, col])
        ax.bar(t_axis, errs, color=_bar_colors(outs, targets), width=0.8, alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"signed error — {mode}", fontsize=9)
        ax.set_xlabel("time step")
        ax.set_ylim(-err_range, err_range)
        if col == 0:
            ax.set_ylabel("output − target")
        ax.grid(True, alpha=0.22)

        # ── Row 3: last-period zoom ──────────────────────────────────────────
        period = len(pattern)
        n_show = 5 * period
        n_show = min(n_show, steps)
        n_show = max((n_show // period) * period, period)

        t_zoom   = t_axis[-n_show:]
        tgt_zoom = targets[-n_show:]
        out_zoom = outs[-n_show:]
        pos_axis = list(range(n_show))

        ax = fig.add_subplot(gs[3, col])
        ax.plot(pos_axis, tgt_zoom, color="gray", linestyle="--", lw=1.2, alpha=0.7, label="target")
        ax.plot(pos_axis, out_zoom, color=color, lw=1.6, label=f"output")
        ax.scatter(pos_axis, out_zoom, s=18, color=color, zorder=4)
        ax.axhline(y_mid, color="black", linestyle=":", lw=0.6, alpha=0.3)
        for k in range(0, n_show + 1, period):
            ax.axvline(k - 0.5, color="gray", lw=0.5, alpha=0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.5, n_show - 0.5)
        n_correct_zoom = _correct_count(out_zoom, tgt_zoom, pattern)
        ax.set_title(f"last-period zoom — {mode}\ncorrect={n_correct_zoom}/{n_show}", fontsize=9)
        ax.set_xlabel(f"relative step (last {n_show} = {n_show // period} periods)")
        if col == 0:
            ax.set_ylabel("value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.22)

    d = results[0]["d"]
    fig.suptitle(
        f"MLP online learning — arch={arch}  lr={lr}  opt={opt}  params={n_params}  d={d}\n"
        f"Constant input x=1; target pattern={pat_str} repeating\n"
        f"layer_delay: output layer delay=0, input layer delay={d-1}",
        fontsize=11,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    here   = Path(__file__).resolve().parent
    params = load_params(here / "params.yaml")

    print("=" * 65)
    print("exp0415 — MLP Online Learning Smoke Test (3 modes)")
    print("=" * 65)
    print(f"  hidden layers  : {params['hidden']}")
    print(f"  lr             : {params['lr']}")
    print(f"  optimizer      : {params.get('optimizer', 'sgd')}")
    print(f"  steps          : {params['steps']}")
    print(f"  target_pattern : {params.get('target_pattern', [0, 1])}")
    print(f"  seed           : {params.get('seed', 42)}")
    print()

    results = []
    for mode in ("standard", "delayed_grad", "layer_delay"):
        res = run_online(params, mode=mode)
        results.append(res)
        correct = _correct_count(res["outputs"], res["targets"], res["pattern"])
        ll = _avg_loss(res["losses"])
        d  = res["d"]
        delays = [d - 1 - i for i in range(d)]
        print(f"  [{mode:<12}]  d={d}  delays={delays}  "
              f"correct={correct}/{res['steps']}  avg_last10_loss={ll:.4f}")
    print()

    runs_dir = here / "runs"
    runs_dir.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = runs_dir / f"comparison_{timestamp}.png"

    save_comparison_plot(results, output_path)


if __name__ == "__main__":
    main()
