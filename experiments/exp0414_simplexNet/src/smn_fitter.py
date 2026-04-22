"""SMNFitter: high-level training and visualisation wrapper for SMN.

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

The core network module ``SMNModule`` is defined at the top of this file.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from graph import SimplexMemoryGraph, NodeKey  # type: ignore

_TWO_PI = 6.283185307179586


# =============================================================================
# SMNModule — Core PyTorch module
# =============================================================================

def _make_activation(name: str):
    """Return activation function by name."""
    name = name.lower()
    if name == "relu":
        return lambda x: F.relu(x)
    elif name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.01)
    elif name == "gelu":
        return lambda x: F.gelu(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    raise ValueError(f"Unsupported activation: {name!r}. Choose relu/leaky_relu/gelu/tanh.")


class SMNModule(nn.Module):
    """Simplex Memory Network as a pure PyTorch module.

    No dependency on Config or any experiment infrastructure — suitable for
    embedding in other projects.

    Args:
        n: Simplex dimension (>= 2).  n=2 → triangle, n=3 → tetrahedron.
        m: Resolution (>= 2).  Number of lattice points per simplex edge.
        n_in: Number of input dimensions.
        n_out: Number of output dimensions.
        activation: Hidden-node activation ('relu', 'leaky_relu', 'gelu', 'tanh').
        x_bounds: Per-channel input bounds as a list of (min, max) pairs.
            Inputs are linearly normalised to [-1, 1] per channel.
            If None, defaults to [(-1.0, 1.0)] * n_in (identity — caller is
            expected to pre-normalise their data).

    Example::

        module = SMNModule(n=3, m=4, n_in=2, n_out=1,
                           x_bounds=[(-3.14, 3.14), (-3.14, 3.14)])
        y = module(x)   # x: (batch, 2)  →  y: (batch, 1)
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
        super().__init__()
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}")
        if n_in < 1 or n_out < 1:
            raise ValueError("n_in and n_out must each be >= 1")

        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out

        # Resolve x_bounds and register as buffers so they move with the module
        # (device transfers, state_dict), avoiding torch.tensor() every forward.
        if x_bounds is None:
            x_bounds = [(-1.0, 1.0)] * n_in
        if len(x_bounds) != n_in:
            raise ValueError(f"x_bounds has {len(x_bounds)} entries but n_in={n_in}")
        self.register_buffer(
            "_x_min", torch.tensor([b[0] for b in x_bounds], dtype=torch.float32)
        )
        self.register_buffer(
            "_x_max", torch.tensor([b[1] for b in x_bounds], dtype=torch.float32)
        )

        self.graph = SimplexMemoryGraph(n=n, m=m, n_in=n_in, n_out=n_out)
        self.activation_fn = _make_activation(activation)

        n_edges = len(self.graph.edges)
        n_core = len(self.graph.core_nodes)

        # Kaiming initialisation: std = sqrt(2 / fan_in) for each edge weight
        kaiming_stds = torch.tensor([
            (2.0 / len(self.graph.preds[dst])) ** 0.5
            for _, dst in self.graph.edges
        ])
        self.ew = nn.Parameter(torch.randn(n_edges) * kaiming_stds)
        self.nb = nn.Parameter(torch.zeros(n_core))
        self.output_bias = nn.Parameter(torch.zeros(n_out))

        # Pre-build index tensors for the forward pass (no Python loops at runtime)
        (
            self._level_schedule,
            self._output_mappings,
            self._output_norm_scales,
            self._input_node_indices,
        ) = self._build_schedule()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_schedule(self):
        """Pre-compute all index tensors needed for the forward pass."""
        node_to_idx = {node: i for i, node in enumerate(self.graph.nodes)}
        edge_to_idx = {edge: i for i, edge in enumerate(self.graph.edges)}
        core_to_bias = {node: i for i, node in enumerate(self.graph.core_nodes)}

        # Assign a column in the hist buffer to every node
        hist_idx: dict = {}
        col = 0
        for node in self.graph.input_nodes:
            hist_idx[node_to_idx[node]] = col
            col += 1
        for _level, nodes in sorted(self.graph.topological_levels.items()):
            if nodes[0][0] in ("in", "out"):
                continue
            for node in nodes:
                hist_idx[node_to_idx[node]] = col
                col += 1

        # Level schedule: one entry per topological level of core nodes
        level_schedule = []
        write_offset = self.n_in
        for _level, nodes in sorted(self.graph.topological_levels.items()):
            if nodes[0][0] in ("in", "out"):
                continue
            bias_list, src_list, dst_list, ew_list = [], [], [], []
            for i, node in enumerate(nodes):
                bias_list.append(core_to_bias[node])
                for pred in self.graph.preds[node]:
                    src_list.append(hist_idx[node_to_idx[pred]])
                    dst_list.append(i)
                    ew_list.append(edge_to_idx[(pred, node)])
                hist_idx[node_to_idx[node]] = write_offset + i
            level_schedule.append((
                torch.tensor(src_list,  dtype=torch.long),
                torch.tensor(dst_list,  dtype=torch.long).unsqueeze(0),
                torch.tensor(ew_list,   dtype=torch.long),
                torch.tensor(bias_list, dtype=torch.long),
                len(nodes),
                write_offset,
            ))
            write_offset += len(nodes)

        # Output mappings: one per output node
        output_mappings = []
        output_norm_scales = []
        for out_node in self.graph.output_nodes:
            preds = self.graph.preds[out_node]
            output_mappings.append((
                torch.tensor([hist_idx[node_to_idx[p]] for p in preds], dtype=torch.long),
                torch.tensor([edge_to_idx[(p, out_node)] for p in preds], dtype=torch.long),
            ))
            output_norm_scales.append(1.0 / (len(preds) ** 0.5))

        input_node_indices = [node_to_idx[n] for n in self.graph.input_nodes]
        return level_schedule, output_mappings, output_norm_scales, input_node_indices

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, n_in)``.
               For n_in=1, also accepts shape ``(batch,)``.

        Returns:
            Output tensor of shape ``(batch, n_out)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        batch = x.shape[0]

        # Per-channel normalisation to [-1, 1] using registered buffers
        x_min = self._x_min.to(dtype=x.dtype)
        x_max = self._x_max.to(dtype=x.dtype)
        x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

        # Pre-allocated history buffer: [input cols | core cols]
        hist = x.new_zeros(batch, self.n_in + len(self.graph.core_nodes))
        for i in range(self.n_in):
            hist[:, i] = x[:, i]

        # Core nodes, level by level
        for src_t, dst_rows_0, ew_idx_t, bias_t, n_level, write_start in self._level_schedule:
            src_states = hist[:, src_t]                               # (B, n_edges)
            weighted   = src_states * self.ew[ew_idx_t]              # (B, n_edges)
            agg = weighted.new_zeros(batch, n_level).scatter_add(
                1, dst_rows_0.expand(batch, -1), weighted
            )                                                         # (B, n_level)
            out = self.activation_fn(agg + self.nb[bias_t])          # (B, n_level)
            hist[:, write_start:write_start + n_level] = out

        # Output nodes (variance-preserving aggregation + tanh)
        outputs = []
        for i, (out_src_t, out_ew_t) in enumerate(self._output_mappings):
            val = (hist[:, out_src_t] * self.ew[out_ew_t]).sum(1, keepdim=True)
            outputs.append(val * self._output_norm_scales[i])

        return torch.tanh(torch.cat(outputs, dim=1) + self.output_bias)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def arch_str(self) -> str:
        """Human-readable architecture description."""
        n_e = self.graph.edge_count
        n_c = self.graph.core_node_count
        n_p = n_e + n_c + self.n_out        # edges + core biases + output biases
        return (
            f"SMN(n={self.n}, m={self.m}, {self.n_in}→{self.n_out}), "
            f"nodes={n_c}, edges={n_e}, params={n_p}"
        )

    def __repr__(self) -> str:
        return f"SMNModule({self.arch_str})"


# =============================================================================
# Default demo data
# =============================================================================

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


# =============================================================================
# Shared training logic (used by both SMNFitter and MLPFitter)
# =============================================================================

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


# =============================================================================
# SMNFitter
# =============================================================================

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
