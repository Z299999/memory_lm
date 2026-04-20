# exp0414_simplexNet — Extension Plan

## Extension 1: MIMO Support

- [x] **config.py** — add `x_bounds: list | None` field + `resolved_x_bounds` property; falls back to `(x_min, x_max)` for all channels when unset; SISO yaml unchanged
- [x] **model.py** — replace `if n_in == 1` normalization special-case with vectorized per-channel normalization using `resolved_x_bounds`
- [x] **data.py** — `build_dataset` uses `resolved_x_bounds` for per-channel sampling (SISO and 2D MIMO)
- [x] **plot.py** — universal L2-norm scatter replaces line plots; scales to arbitrary n_out

## Extension 2: Class Encapsulation

Two-layer design: a pure PyTorch module (importable by other experiments) and a
convenience wrapper (training + visualization for exp0414 experiments).

### Layer 1 — `src/smn_module.py` (pure nn.Module)

- [ ] Extract `SMNModule(n, m, n_in=1, n_out=1, activation='relu', x_bounds=None)` from `model.py`
  - No `Config` dependency — only plain Python / torch arguments
  - `forward(x: Tensor) -> Tensor` is the full public API
  - Goal: other experiments can do `from smn_module import SMNModule` without importing anything else from this project

### Layer 2 — `src/smn_fitter.py` (training + visualization wrapper)

- [ ] Implement `SMNFitter` that wraps `SMNModule`:
  ```
  SMNFitter(n=2, m=3, n_in=1, n_out=1, activation='relu', x_bounds=None)
    .fit(x_train, y_train, x_val=None, y_val=None,
         loss_fn=None,          # default: MSELoss
         lr=0.001, epochs=300, batch_size=64)
    .predict(x) -> np.ndarray
    .plot(x_ref, y_ref, output_path=None, baseline=None)
  ```
  - If `x_train / y_train` not provided to `fit()`, generate default sin_mix SISO demo data internally
  - `loss_fn` accepts any `(pred, target) -> scalar` callable; defaults to MSE
  - `plot(baseline=None)` → 2-panel (scatter + loss curve)
  - `plot(baseline=other_fitter)` → 4-panel SMN vs baseline comparison (same layout as current `save_four_panel_plot`)
  - Stores `train_losses`, `val_losses`, `y_true`, `y_pred` after `fit()` for inspection

### Refactor existing files

- [ ] **model.py** — thin wrapper: `SMNNetwork(config)` instantiates `SMNModule` from `Config` fields, delegates `forward`
- [ ] **run.py** — replace direct `SMNNetwork` + `save_four_panel_plot` calls with `SMNFitter`; MLP comparison via `smn_fitter.plot(baseline=mlp_fitter)`
- [ ] **tests/test_smn_module.py** — standalone unit tests for `SMNModule` (no Config, no data.py dependency)
