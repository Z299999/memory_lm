# exp0414_simplexNet — Simplex Memory Networks

A geometric feedforward neural architecture based on simplicial lattices.

## Overview

The Simplex Memory Network (SMN) is a neural architecture where:
- **Hidden nodes** are lattice points in a regular n-simplex
- **Input/output interfaces** are opposite facets of the simplex
- **Information flow** follows a potential-induced directed acyclic graph

The architecture naturally decomposes into:
- A **narrow deep backbone** (long-term memory)
- A **broad shallow buffer** (short-term memory)

## Project Structure

```
exp0414_simplexNet/
├── params.yaml          # Configuration file
├── run.py               # Entry point: python3 run.py
├── src/
│   ├── lattice.py       # V_{n,m} lattice generation
│   ├── potential.py     # Potential function H(α) and edge orientation
│   ├── graph.py         # SimplexMemoryGraph class
│   ├── config.py        # Config dataclass + YAML loader
│   ├── model.py         # SMNNetwork PyTorch module
│   ├── mlp.py           # MLPBaseline for comparison
│   ├── train.py         # Training loop
│   ├── data.py          # Target functions (1D/2D)
│   └── plot.py          # Visualization utilities
└── tests/
    ├── test_lattice.py  # Lattice generation tests
    └── test_potential.py # Potential and graph tests
```

## Quick Start

```bash
cd exp0414_simplexNet
python3 run.py
```

Output: `runs/<date>/<experiment_name>/comparison.png`

## Configuration (params.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | smn | `"smn"` or `"mlp"` |
| `n` | 2 | Simplex dimension (2=triangle, 3=tetrahedron) |
| `m` | 3 | Resolution (lattice points per edge) |
| `n_in` | 1 | Number of input dimensions |
| `n_out` | 1 | Number of output dimensions |
| `mlp_layers` | [8, 8, 8] | MLP hidden layer sizes |
| `node_activation` | relu | `relu`, `leaky_relu`, `gelu`, `tanh` |
| `task_name` | sin_mix | Target function (see Tasks below) |
| `epochs` | 300 | Training epochs |
| `lr` | 0.001 | Learning rate |
| `batch_size` | 64 | Batch size |
| `num_train` | 10000 | Training set size (points sampled from target function) |
| `x_min` / `x_max` | ±2π | Input domain for 1D tasks |
| `x_bounds` | — | Per-channel bounds for MIMO, e.g. `[[-π,π],[-1,1]]`; overrides `x_min`/`x_max` |
| `window_width` | 0.0 | Moving window width as fraction of domain (0 = disabled) |
| `window_hold` | 1 | Epochs per window position |

### Tasks

| `task_name` | n_in | Formula |
|-------------|------|---------|
| `sin` | 1 | $\sin(x)$ |
| `sin_mix` | 1 | $0.5\sin(x) + 0.3\sin(2x) + 0.2\sin(3x)$ |
| `poly_wave` | 1 | $0.1x^2\sin(x) + 0.5\cos(2x)$ |
| `piecewise` | 1 | Piecewise linear |
| `sin_cos` | 1 | $[\sin(x),\ \cos(x)]$ — **1i2o** |
| `sin_sum` | 2 | $\sin(x_1 + x_2)$ |
| `sin_product` | 2 | $\sin(x_1)\cos(x_2)$ |
| `quadratic` | 2 | $0.1(x_1^2+x_2^2) - 0.5\sin(x_1 x_2)$ |
| `trig_2d` | 2 | $[\sin(x_1+x_2),\ \cos(x_1-x_2)]$ — **2i2o** |

### Output plot

Each run saves `runs/<date>/<name>/comparison.png` — a 4-panel figure:

| Panel | Content |
|-------|---------|
| Top-left | SMN: $\|y_{\text{true}}\|_2$ vs $\|y_{\text{pred}}\|_2$ scatter — points on the diagonal `y=x` = perfect prediction |
| Top-right | MLP: same scatter for comparison |
| Bottom-left | SMN train / val loss curves (log scale) |
| Bottom-right | MLP train / val loss curves (log scale) |

The scatter uses the **L2 norm** of the output vector, so it scales to any `n_out` (including high-dimensional outputs) without needing multiple colors. For `n_out=1` the norm equals `|y|` and the axes are labelled `True` / `Predicted`.

## Mathematical Background

### Lattice V_{n,m}

$$V_{n,m} = \{\alpha \in \mathbb{Z}_{\geq 0}^{n+1} : \sum_{i=0}^n \alpha_i = m-1\}$$

Cardinality: $|V_{n,m}| = \binom{m+n-1}{n}$

### Vertex Potentials

- $\beta_0 = 0$ (vertex $a$)
- $\beta_1 = 2$ (vertex $b$)
- $\beta_k = 1$ for $k \geq 2$ (vertices $c_1, \ldots, c_{n-1}$)

### Potential Function

$$H(\alpha) = \sum_{i=0}^n \beta_i \alpha_i = 2\alpha_1 + \alpha_2 + \cdots + \alpha_n$$

### Edge Orientation

Edge $\alpha \to \alpha'$ exists iff:
- $\alpha' = \alpha + e_i - e_j$ for some $i \neq j$
- $\alpha_j \geq 1$ (can move mass from $j$)
- $\beta_i > \beta_j$ (potential increases)

Admissible edge types: $a \to c_k$, $c_k \to b$, $a \to b$

### Facets

- Input: $F_{\text{in}} = \{\alpha : \alpha_1 = 0\}$
- Output: $F_{\text{out}} = \{\alpha : \alpha_0 = 0\}$
- Shared: $F_{\text{mid}} = F_{\text{in}} \cap F_{\text{out}} = \{\alpha : \alpha_0 = \alpha_1 = 0\}$

All nodes in $F_{\text{mid}}$ have $H(\alpha) = m-1$ (isopotential).

### Backbone

$$\mathcal{B}_{n,m} = \{\alpha : \alpha_2 = \cdots = \alpha_n = 0\}$$

A chain of exactly $m$ nodes with potentials $0, 2, 4, \ldots, 2(m-1)$.

## Running Tests

```bash
cd exp0414_simplexNet
python3 tests/test_lattice.py
python3 tests/test_potential.py
```

## Examples

### Triangle motif SMN(2, 2)
- 3 hidden nodes: $a, c, b$
- Edges: $a \to c$, $c \to b$, $a \to b$
- $F_{\text{mid}} = \{c\}$ (single node)

### Tetrahedron motif SMN(3, 2)
- 4 hidden nodes: $a, c, d, b$
- Edges: $a \to c$, $a \to d$, $c \to b$, $d \to b$, $a \to b$
- $F_{\text{mid}} = \{c, d\}$ (edge)

## Reference

See `../exp0414_simplexNet/report_tex/report.pdf` for the full mathematical treatment.
