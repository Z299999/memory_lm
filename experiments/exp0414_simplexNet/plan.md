# exp0414_simplexNet — Extension Plan

## Extension 1: MIMO Support ✅

- [x] **config.py** — add `x_bounds: list | None` field + `resolved_x_bounds` property
- [x] **model.py** — vectorized per-channel normalization
- [x] **data.py** — `build_dataset` uses `resolved_x_bounds`
- [x] **plot.py** — universal L2-norm scatter

## Extension 2: Class Encapsulation ✅

- [x] **smn_module.py** → merged into **smn_fitter.py**
- [x] **smn_fitter.py** — SMNModule + SMNFitter (fit/predict/plot)
- [x] **mlp_fitter.py** — MLPFitter (mirrors SMNFitter)
- [x] **model.py** — deleted (replaced by smn_fitter.py)
- [x] **tests/test_smn_module.py** → **tests/test_smn.py**

## Code Cleanup (2026-04-19) ✅

Reduced from 18 Python files to **6 core files**:

### Retained Files

| File | Purpose |
|------|---------|
| `config.py` | Config dataclass + YAML loader |
| `data.py` | Target functions + dataset builder |
| `graph.py` | SimplexMemoryGraph + lattice/potential helpers (merged) |
| `smn_fitter.py` | SMNModule + SMNFitter (merged) |
| `mlp_fitter.py` | MLPFitter baseline |
| `plot.py` | Visualization utilities |
| `run.py` | Entry point |
| `tests/test_smn.py` | SMNModule unit tests (12 tests) |

### Deleted Files

| File | Reason |
|------|--------|
| `lattice.py` | Merged into `graph.py` |
| `potential.py` | Merged into `graph.py` |
| `smn_module.py` | Merged into `smn_fitter.py` |
| `model.py` | Thin wrapper, no longer needed |
| `train.py` | Functionality moved to `smn_fitter._train_loop` |
| `mlp.py` | Functionality moved to `mlp_fitter.py` |
| `__init__.py` | Exported old API, no functional purpose |
| `tests/test_lattice.py` | Lattice is now internal to `graph.py` |
| `tests/test_potential.py` | Potential is now internal to `graph.py` |

### Result

- **Before**: 18 Python files, unclear dependency chain
- **After**: 6 core files + 1 test file, clean layered architecture:
  ```
  config.py, data.py  →  graph.py  →  smn_fitter.py  →  run.py
                                         ↑
                                   mlp_fitter.py
  ```
