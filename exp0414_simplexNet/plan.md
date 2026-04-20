# exp0414_simplexNet — Extension Plan

## Extension 1: MIMO Support

- [x] **config.py** — add `x_bounds: list | None` field + `resolved_x_bounds` property; falls back to `(x_min, x_max)` for all channels when unset; SISO yaml unchanged
- [x] **model.py** — replace `if n_in == 1` normalization special-case with vectorized per-channel normalization using `resolved_x_bounds`
- [x] **data.py** — `build_dataset` uses `resolved_x_bounds` for per-channel sampling (SISO and 2D MIMO)
- [ ] **plot.py** — update visualisation to handle multi-dim input/output

## Extension 2: Class Encapsulation

- [ ] **src/smn_module.py** — extract `SMNModule(n, m, n_in, n_out, activation, x_bounds)` from `model.py`; remove all `Config` dependency; expose clean `forward(x: Tensor) -> Tensor`
- [ ] **model.py** — refactor into thin wrapper that instantiates `SMNModule` from `Config`
- [ ] **run.py** — update to use new wrapper (no change to training logic)
- [ ] **tests/test_smn_module.py** — add standalone unit tests for `SMNModule`
