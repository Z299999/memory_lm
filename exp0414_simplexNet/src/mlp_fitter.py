"""MLPFitter: training and visualisation wrapper for an MLP baseline.

Mirrors the SMNFitter interface so it can be dropped in as the ``baseline``
argument of ``SMNFitter.plot(baseline=mlp)``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smn_fitter import _make_default_data, _train_loop  # type: ignore


class MLPFitter:
    """High-level wrapper for training and evaluating an MLP baseline.

    Provides the same ``fit / predict / plot / arch_str`` interface as
    ``SMNFitter`` so it can be used as a drop-in baseline.

    Args:
        layers:     Hidden layer sizes, e.g. ``[16, 16, 16]``.
        n_in:       Number of input dimensions.
        n_out:      Number of output dimensions.
        activation: Hidden-node activation ('relu', 'leaky_relu', 'gelu', 'tanh').
        x_bounds:   Per-channel input bounds.  If None, defaults to [(-2π, 2π)] × n_in.
    """

    _TWO_PI = 6.283185307179586

    def __init__(
        self,
        layers: list[int] | None = None,
        n_in: int = 1,
        n_out: int = 1,
        activation: str = "relu",
        x_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        if layers is None:
            layers = [8, 8, 8]
        self.layers = layers
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        if x_bounds is None:
            x_bounds = [(-self._TWO_PI, self._TWO_PI)] * n_in
        self.x_bounds = x_bounds

        self.module = self._build_module()

        self.train_losses:     list[float] = []
        self.val_losses:       list[float] = []
        self.final_train_loss: float = float("inf")
        self.final_val_loss:   float = float("inf")
        self._y_true: np.ndarray | None = None
        self._y_pred: np.ndarray | None = None

    # ------------------------------------------------------------------

    def _build_module(self) -> nn.Module:
        act_map = {
            "relu":       nn.ReLU,
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
            "gelu":       nn.GELU,
            "tanh":       nn.Tanh,
        }
        if self.activation.lower() not in act_map:
            raise ValueError(f"Unsupported activation: {self.activation!r}")
        act_cls = act_map[self.activation.lower()]

        seq: list[nn.Module] = []
        in_dim = self.n_in
        for h in self.layers:
            seq.append(nn.Linear(in_dim, h))
            seq.append(act_cls())
            in_dim = h
        seq.append(nn.Linear(in_dim, self.n_out))
        seq.append(nn.Tanh())
        return nn.Sequential(*seq)

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
    ) -> "MLPFitter":
        """Train the MLP.  Same signature as SMNFitter.fit()."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

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
        """Run inference.  Same signature as SMNFitter.predict()."""
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
        title: str = "",
    ) -> None:
        """Save a 2-panel plot (scatter + loss) for this MLP."""
        import matplotlib.pyplot as plt
        from plot import _draw_scatter  # type: ignore

        if y_ref is not None:
            if isinstance(x_ref, np.ndarray):
                x_ref = torch.from_numpy(x_ref.astype(np.float32))
            y_ref_np  = y_ref if isinstance(y_ref, np.ndarray) else y_ref.cpu().numpy()
            y_pred_np = self.predict(x_ref)
        elif self._y_true is not None:
            y_ref_np  = self._y_true
            y_pred_np = self._y_pred
        else:
            raise RuntimeError("Call fit() before plot(), or pass x_ref / y_ref.")

        output_path = Path(output_path) if output_path else Path("mlp_plot.png")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        _draw_scatter(axes[0], y_ref_np, y_pred_np, "MLP", self.arch_str, self.final_val_loss)

        ax = axes[1]
        ep = range(1, len(self.train_losses) + 1)
        ax.plot(ep, self.train_losses, "b-", linewidth=1, label="Train")
        ax.plot(ep, self.val_losses,   "g-", linewidth=1, label="Val")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"MLP Losses\nfinal_val={self.final_val_loss:.6f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    # ------------------------------------------------------------------

    @property
    def arch_str(self) -> str:
        """Human-readable architecture description."""
        in_dim = self.n_in
        n_nodes = self.n_in + sum(self.layers) + self.n_out
        n_edges, prev = 0, self.n_in
        for h in self.layers:
            n_edges += prev * h
            prev = h
        n_edges += prev * self.n_out
        n_params = n_edges + sum(self.layers) + self.n_out   # weights + biases
        # also count input bias terms
        in_dim = self.n_in
        n_params = 0
        prev = self.n_in
        for h in self.layers:
            n_params += prev * h + h
            prev = h
        n_params += prev * self.n_out + self.n_out
        return (
            f"MLP layers={self.layers}, {self.n_in}→{self.n_out}, "
            f"nodes={n_nodes}, params={n_params}"
        )

    def __repr__(self) -> str:
        return f"MLPFitter({self.arch_str})"
