"""SMNFitter: high-level training and visualisation wrapper for SMNModule.

Data and loss functions are supplied by the caller; sensible defaults are
built in so the class works out of the box with no arguments::

    smn = SMNFitter()
    smn.fit()          # trains on built-in sin_mix demo data
    smn.plot()         # saves smn_plot.png

Typical usage with custom data::

    smn = SMNFitter(n=3, m=4, n_in=2, n_out=1,
                    x_bounds=[(-3.14, 3.14), (-3.14, 3.14)])
    smn.fit(x_train, y_train, x_val, y_val, epochs=500)
    smn.predict(x_test)
    smn.plot(output_path="result.png")

Comparison plot against a baseline (e.g. MLPFitter)::

    mlp = MLPFitter(layers=[16, 16], n_in=1, n_out=1)
    mlp.fit(x_train, y_train, x_val, y_val)
    smn.plot(output_path="compare.png", baseline=mlp)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smn_module import SMNModule  # type: ignore

_TWO_PI = 6.283185307179586


# ---------------------------------------------------------------------------
# Default demo data
# ---------------------------------------------------------------------------

def _make_default_data(
    n_in: int,
    x_bounds: list[tuple[float, float]],
    n_train: int = 500,
    n_val: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate built-in demo data.

    SISO  → sin_mix:  0.5·sin(x) + 0.3·sin(2x) + 0.2·sin(3x)
    MIMO  → sin_sum:  sin(x1 + x2 + …)
    """
    if n_in == 1:
        lo, hi = x_bounds[0]
        x_tr = torch.linspace(lo, hi, n_train).unsqueeze(-1)
        x_tr = x_tr[torch.randperm(n_train)]
        x_va = torch.linspace(lo, hi, n_val).unsqueeze(-1)
        y_tr = 0.5 * torch.sin(x_tr) + 0.3 * torch.sin(2 * x_tr) + 0.2 * torch.sin(3 * x_tr)
        y_va = 0.5 * torch.sin(x_va) + 0.3 * torch.sin(2 * x_va) + 0.2 * torch.sin(3 * x_va)
    else:
        lo_t = torch.tensor([b[0] for b in x_bounds])
        hi_t = torch.tensor([b[1] for b in x_bounds])
        x_tr = torch.rand(n_train, n_in) * (hi_t - lo_t) + lo_t
        x_va = torch.rand(n_val,   n_in) * (hi_t - lo_t) + lo_t
        y_tr = torch.sin(x_tr.sum(1, keepdim=True))
        y_va = torch.sin(x_va.sum(1, keepdim=True))
    return x_tr, y_tr, x_va, y_va


# ---------------------------------------------------------------------------
# Shared training logic (used by both SMNFitter and MLPFitter)
# ---------------------------------------------------------------------------

def _train_loop(
    module: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    criterion: Callable,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    verbose: bool,
) -> tuple[list[float], list[float], dict]:
    """Generic training loop.  Returns (train_losses, val_losses, best_state_dict)."""
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val = math.inf
    best_state: dict = {}

    for epoch in range(epochs):
        module.train()
        total = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(module(bx), by)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * bx.size(0)

        train_loss = total / len(loader.dataset)
        module.eval()
        with torch.no_grad():
            val_loss = float(criterion(module(x_val), y_val).item())

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"  epoch={epoch+1:04d}  train={train_loss:.6f}  val={val_loss:.6f}")

    return train_losses, val_losses, best_state


# ---------------------------------------------------------------------------
# SMNFitter
# ---------------------------------------------------------------------------

class SMNFitter:
    """High-level wrapper for training and evaluating an SMNModule.

    Args:
        n: Simplex dimension (>= 2).
        m: Resolution (>= 2).
        n_in: Number of input dimensions.
        n_out: Number of output dimensions.
        activation: Hidden-node activation ('relu', 'leaky_relu', 'gelu', 'tanh').
        x_bounds: Per-channel input bounds [(min, max), ...].
            If None, defaults to [(-2π, 2π)] × n_in.
    """

    def __init__(
        self,
        n: int = 2,
        m: int = 3,
        n_in: int = 1,
        n_out: int = 1,
        activation: str = "relu",
        x_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if x_bounds is None:
            x_bounds = [(-_TWO_PI, _TWO_PI)] * n_in
        self.x_bounds = x_bounds

        self.module = SMNModule(
            n=n, m=m, n_in=n_in, n_out=n_out,
            activation=activation, x_bounds=x_bounds,
        )

        # Populated after fit()
        self.train_losses:   list[float] = []
        self.val_losses:     list[float] = []
        self.final_train_loss: float = float("inf")
        self.final_val_loss:   float = float("inf")
        self._y_true: np.ndarray | None = None
        self._y_pred: np.ndarray | None = None

    # ------------------------------------------------------------------

    def fit(
        self,
        x_train: torch.Tensor | None = None,
        y_train: torch.Tensor | None = None,
        x_val:   torch.Tensor | None = None,
        y_val:   torch.Tensor | None = None,
        loss_fn: Callable | None = None,
        lr: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 64,
        seed: int = 42,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> "SMNFitter":
        """Train the SMN.

        Args:
            x_train: Training inputs ``(n_samples, n_in)``.
                If None, built-in demo data (sin_mix / sin_sum) is used.
            y_train: Training targets ``(n_samples, n_out)``.
            x_val:   Validation inputs.  If None and x_train is given, an
                     automatic 80/20 split is applied.
            y_val:   Validation targets.
            loss_fn: ``(pred, target) -> scalar`` callable.  Defaults to MSELoss.
            lr:          Learning rate.
            epochs:      Training epochs.
            batch_size:  Mini-batch size.
            seed:        Random seed.
            weight_decay: L2 regularisation coefficient.
            verbose:     Print progress every 50 epochs.

        Returns:
            self (for method chaining).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Resolve data
        if x_train is None or y_train is None:
            x_train, y_train, x_val, y_val = _make_default_data(self.n_in, self.x_bounds)
        elif x_val is None or y_val is None:
            n = len(x_train)
            perm = torch.randperm(n)
            split = int(0.8 * n)
            x_val,   y_val   = x_train[perm[split:]], y_train[perm[split:]]
            x_train, y_train = x_train[perm[:split]], y_train[perm[:split]]

        criterion = loss_fn if loss_fn is not None else nn.MSELoss()

        self.train_losses, self.val_losses, best_state = _train_loop(
            self.module, x_train, y_train, x_val, y_val,
            criterion, lr, epochs, batch_size, weight_decay, verbose,
        )

        if best_state:
            self.module.load_state_dict(best_state)

        self.module.eval()
        with torch.no_grad():
            self.final_train_loss = float(criterion(self.module(x_train), y_train).item())
            self.final_val_loss   = float(criterion(self.module(x_val),   y_val).item())
            self._y_true = y_val.cpu().numpy()
            self._y_pred = self.module(x_val).cpu().numpy()

        return self

    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            x: ``(n_samples, n_in)`` array or tensor, or ``(n_samples,)`` for n_in=1.

        Returns:
            Predictions as numpy array of shape ``(n_samples, n_out)``.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        self.module.eval()
        with torch.no_grad():
            return self.module(x).cpu().numpy()

    # ------------------------------------------------------------------

    def plot(
        self,
        x_ref: torch.Tensor | np.ndarray | None = None,
        y_ref: torch.Tensor | np.ndarray | None = None,
        output_path: str | Path | None = None,
        baseline: "SMNFitter | None" = None,
        title: str = "",
    ) -> None:
        """Save a visualisation plot.

        Without ``baseline``: 2-panel (scatter + loss curve) for this model alone.
        With ``baseline``:    4-panel SMN vs baseline comparison (same layout as
                              the standard exp0414 comparison plot).

        Args:
            x_ref:       Reference inputs for predictions.  Uses fit() val data if None.
            y_ref:       Reference targets.                 Uses fit() val data if None.
            output_path: PNG file path.  Defaults to ``"smn_plot.png"``.
            baseline:    A second fitter (SMNFitter or MLPFitter) for 4-panel comparison.
            title:       Optional super-title text.
        """
        import matplotlib.pyplot as plt
        from plot import _draw_scatter, save_four_panel_plot  # type: ignore

        # Resolve y_true / y_pred
        if y_ref is not None:
            if isinstance(x_ref, np.ndarray):
                x_ref = torch.from_numpy(x_ref.astype(np.float32))
            y_ref_np = y_ref if isinstance(y_ref, np.ndarray) else y_ref.cpu().numpy()
            y_pred_np = self.predict(x_ref)
        elif self._y_true is not None:
            y_ref_np  = self._y_true
            y_pred_np = self._y_pred
        else:
            raise RuntimeError("Call fit() before plot(), or pass x_ref / y_ref explicitly.")

        output_path = Path(output_path) if output_path else Path("smn_plot.png")

        if baseline is None:
            # 2-panel: scatter  |  loss curve
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            _draw_scatter(axes[0], y_ref_np, y_pred_np,
                          "SMN", self.arch_str, self.final_val_loss)

            ax = axes[1]
            ep = range(1, len(self.train_losses) + 1)
            ax.plot(ep, self.train_losses, "b-", linewidth=1, label="Train")
            ax.plot(ep, self.val_losses,   "r-", linewidth=1, label="Val")
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"SMN Losses\nfinal_val={self.final_val_loss:.6f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig.suptitle(title, fontsize=11, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {output_path}")

        else:
            # 4-panel: self vs baseline
            if baseline._y_true is None:
                raise RuntimeError("baseline.fit() must be called before plot().")
            save_four_panel_plot(
                smn_train_losses=self.train_losses,
                smn_val_losses=self.val_losses,
                smn_y_true=y_ref_np,
                smn_y_pred=y_pred_np,
                smn_architecture=self.arch_str,
                smn_final_train_loss=self.final_train_loss,
                smn_final_val_loss=self.final_val_loss,
                mlp_train_losses=baseline.train_losses,
                mlp_val_losses=baseline.val_losses,
                mlp_y_true=baseline._y_true,
                mlp_y_pred=baseline._y_pred,
                mlp_architecture=baseline.arch_str,
                mlp_final_train_loss=baseline.final_train_loss,
                mlp_final_val_loss=baseline.final_val_loss,
                figure_title=title,
                output_path=output_path,
            )

    # ------------------------------------------------------------------

    @property
    def arch_str(self) -> str:
        """Human-readable architecture description."""
        return self.module.arch_str

    def __repr__(self) -> str:
        return f"SMNFitter({self.arch_str})"
