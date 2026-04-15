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
| `model_type` | smn | "smn" or "mlp" |
| `n` | 2 | Simplex dimension (2=triangle, 3=tetrahedron) |
| `m` | 3 | Resolution (m lattice points per edge) |
| `n_in` | 1 | Input dimension |
| `n_out` | 1 | Output dimension |
| `mlp_layers` | [8, 8, 8] | MLP hidden layer sizes |
| `node_activation` | relu | relu, leaky_relu, gelu, tanh |
| `task_name` | sin_mix | Target function |
| `epochs` | 300 | Training epochs |
| `lr` | 0.001 | Learning rate |
| `batch_size` | 64 | Batch size |

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
