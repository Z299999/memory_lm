# exp0410_triangleNet — Project Guide

## Project Overview

This is a research experiment implementing a **Triangular Memory Network (TMN)** — a novel DAG architecture where:
- Nodes are arranged in a triangle (L layers, layer r has r positions)
- Each position can have multiple neurons (depth dimension)
- **No depth connections**: neurons within the same position are NOT directly connected
- Signal flows: **layer (top→bottom)** + **position (left→right)**

## 3D Coordinate System

All nodes use `(layer, pos, depth)` notation:

| Dimension | Direction | Term | Range | Example |
|-----------|-----------|------|-------|---------|
| **Layer** | Top→Bottom | 层 | `1` to `L` | L=4 means 4 layers |
| **Position** | Left→Right | 位置 | `1` to `layer` | Layer 4 has 4 positions |
| **Depth** | Outer→Inner | 深度 | `1` to `depth[layer]` | depth=[2,2,2,2] means 2 neurons per position |

**Node examples** (L=4, depth=[2,2,2,2]):
- `(4, 1, 1)` = Layer 4, Position 1, outermost neuron
- `(4, 1, 2)` = Layer 4, Position 1, innermost neuron
- `(4, 4, 2)` = Layer 4, Position 4, bottom-right innermost neuron

## Connection Rules

### Allowed Connections (2 types)

| Type | Direction | Rule | Example |
|------|-----------|------|---------|
| **Intra-layer** | Left→Right | Adjacent positions, full connect | `(4,1,2) → (4,2,1)` |
| **Inter-layer** | Top→Bottom | Triangular structure preserved | `(3,1,2) → (4,2,1)` |

### Forbidden Connections

- **Depth connections**: `(L, pos, 1) → (L, pos, 2)` does NOT exist
- Neurons within the same position never connect directly

### Connect-Out / Connect-In Convention

- **Connect-out**: From source position's **innermost** neuron (`depth = depth[layer]`)
- **Connect-in**: To target position's **outermost** neuron (`depth = 1`)

## Key Files

```
exp0410_triangleNet/
├── params.yaml          # Main config (edit this)
├── run.py               # Entry point: python3 run.py
├── timeline.md          # Research log + dead ends + TODOs
├── readme.md            # Full documentation
└── scripts/
    ├── config.py        # Config dataclass, loads YAML
    ├── data.py          # Target functions: sin, sin_mix, etc.
    ├── train.py         # Training loop, supports trace_fn callback
    ├── plot.py          # Visualization: 4-panel + weights evolution
    └── model/
        ├── graph.py     # TMNGraph: builds DAG structure
        ├── tmn.py       # TMNNetwork: forward pass
        └── mlp.py       # MLPBaseline for comparison
```

## Running Experiments

```bash
cd exp0410_triangleNet
python3 run.py
```

Output: `runs/<task_name>_<timestamp>/comparison_4panel.png` + `weights.png`

## Key Parameters (params.yaml)

```yaml
L: 4                        # Triangle layers
depth: [1,1,1,1]           # Per-layer depth (use [2,2,2,2] for expansion)
node_activation: relu      # relu, leaky_relu, gelu, tanh
mlp_layers: [4,8,4]        # MLP baseline architecture
task_name: sin_mix         # sin, sin_mix, poly_wave, piecewise
epochs: 10000              # Training epochs
lr: 0.001
```

## Implementation Notes

### Edge Count Calculation (L=4)

| depth | Total Edges |
|-------|-------------|
| [1,1,1,1] | 15 edges |
| [2,2,2,2] | ~72 edges (d² multiplier) |

### Known Issues / Research Questions

1. **Dying neurons**: Bottom-right nodes (e.g., (4,4)) often stay at ~0 bias
2. **LeakyReLU helps**: 0.01 slope activates some dead nodes, but not all
3. **Signal attenuation**: Long paths through ReLU layers decay to 0

See `timeline.md` for detailed research progress and dead ends.

## Common Tasks

### Adding a new target function
Edit `scripts/data.py::target_function()`

### Changing activation function
Edit `params.yaml::node_activation` or add new option in `scripts/model/tmn.py`

### Modifying graph structure
Edit `scripts/model/graph.py::TMNGraph::_build_edges()`

### Adding parameter tracking
Use `trace_fn` callback in `train_with_config()` — see `run.py::build_trace_fn()` for example

## Things to Avoid

- Don't re-add `d_model` — we use scalar nodes now
- Don't add depth connections — violates TMN design
- Don't revert to string-based forward pass — use integer indices
- Don't commit `runs/` — already in .gitignore
