# exp0414_simplexNet — Extension Plan

## Extension 1: MIMO Support

- [x] **config.py** — add `x_bounds: list | None` field + `resolved_x_bounds` property; falls back to `(x_min, x_max)` for all channels when unset; SISO yaml unchanged
- [x] **model.py** — replace `if n_in == 1` normalization special-case with vectorized per-channel normalization using `resolved_x_bounds`
- [x] **data.py** — `build_dataset` uses `resolved_x_bounds` for per-channel sampling (SISO and 2D MIMO)
- [x] **plot.py** — universal L2-norm scatter replaces line plots; scales to arbitrary n_out

## Extension 2: Class Encapsulation ✅

Two-layer design: a pure PyTorch module (importable by other experiments) and a
convenience wrapper (training + visualization for exp0414 experiments).

### Layer 1 — `src/smn_module.py` (pure nn.Module)

- [x] Extract `SMNModule(n, m, n_in=1, n_out=1, activation='relu', x_bounds=None)` from `model.py`
  - No `Config` dependency — only plain Python / torch arguments
  - `forward(x: Tensor) -> Tensor` is the full public API
  - `x_bounds` stored as `register_buffer` (fixes torch.tensor() per-forward regression)
  - `arch_str` and `param_count` properties for introspection

### Layer 2 — `src/smn_fitter.py` + `src/mlp_fitter.py`

- [x] `SMNFitter(n, m, n_in, n_out, activation, x_bounds)` wraps `SMNModule`:
  - `fit(x_train, y_train, x_val, y_val, loss_fn, lr, epochs, ...)` — trains; auto-generates sin_mix demo data if none provided
  - `predict(x) -> np.ndarray`
  - `plot(x_ref, y_ref, output_path, baseline=None)` — 2-panel (no baseline) or 4-panel comparison
- [x] `MLPFitter(layers, n_in, n_out, ...)` — mirrors SMNFitter interface for drop-in baseline use

### Refactored files

- [x] **model.py** — thin wrapper: `SMNNetwork(config)` instantiates `SMNModule` from `Config` fields, delegates `forward`
- [x] **run.py** — replaced direct `SMNNetwork` + `save_four_panel_plot` calls with `SMNFitter` / `MLPFitter`; comparison via `smn.plot(baseline=mlp)`
- [x] **tests/test_smn_module.py** — 12 standalone unit tests for `SMNModule` (no Config, no data.py dependency); all pass
