# exp0414_simplexNet — Implementation Plan

## Context

The user wants to implement exp0414_simplexNet, a geometric feedforward architecture based on simplex lattices. The mathematical foundation is complete (report in `exp0414_simplexNet/report_tex/`). 

exp0410_triangleNet already implements a related triangular DAG architecture, but with several issues:
- Monolithic `run.py` (240 lines) with mixed concerns
- Hardcoded trace functions and architecture descriptions
- No tests
- Complex 3D coordinate system that's hard to generalize

exp0414 should be a **professional, modular reimplementation** with:
- Clean separation of concerns (graph building, model, training, visualization)
- Proper abstraction for simplex geometry
- Reusable utilities
- Test coverage

## Core Mathematical Structure (from report)

### Simplex Lattice $V_{n,m}$
- $\alpha = (\alpha_0, \ldots, \alpha_n) \in \mathbb{Z}_{\geq 0}^{n+1}$ with $\sum \alpha_i = m-1$
- Cardinality: $|V_{n,m}| = \binom{m+n-1}{n}$

### Potential Function
- $H(\alpha) = 2\alpha_1 + \alpha_2 + \cdots + \alpha_n$
- Edge $\alpha \to \alpha'$ exists iff $\alpha' = \alpha + e_i - e_j$ with $\alpha_j \geq 1$ and $\beta_i > \beta_j$
- Admissible edge types: $a \to c_k$, $c_k \to b$, $a \to b$ (where $\beta_a=0, \beta_{c_k}=1, \beta_b=2$)

### Facets
- $F_{\mathrm{in}} = \{\alpha : \alpha_1 = 0\}$ (input)
- $F_{\mathrm{out}} = \{\alpha : \alpha_0 = 0\}$ (output)
- $F_{\mathrm{mid}} = F_{\mathrm{in}} \cap F_{\mathrm{out}} = \{\alpha : \alpha_0 = \alpha_1 = 0\}$

### Backbone
- $\mathcal{B}_{n,m} = \{\alpha : \alpha_2 = \cdots = \alpha_n = 0\}$ — chain of $m$ nodes

## Recommended Approach

### Phase 1: Project Structure

```
exp0414_simplexNet/
├── params.yaml              # Main config (L, n, m, etc.)
├── run.py                   # Entry point (thin wrapper)
├── README.md                # Documentation
├── tests/                   # Unit tests
│   ├── test_graph.py
│   ├── test_potential.py
│   └── test_forward.py
└── src/
    ├── __init__.py
    ├── config.py            # Config dataclass + YAML loader
    ├── lattice.py           # V_{n,m} lattice generation
    ├── potential.py         # H(alpha), edge orientation
    ├── graph.py             # SimplexMemoryGraph class
    ├── model.py             # PyTorch nn.Module
    ├── train.py             # Training loop
    ├── data.py              # Target functions
    └── plot.py              # Visualization
```

### Phase 2: Core Modules

#### `src/lattice.py` — Lattice Generation
```python
def generate_lattice(n: int, m: int) -> list[tuple[int, ...]]:
    """Generate V_{n,m} = {α ∈ ℤ_≥0^{n+1} : Σαᵢ = m-1} using stars-and-bars."""
```

#### `src/potential.py` — Potential & Orientation
```python
def vertex_potentials(n: int) -> dict[int, int]:
    """Return β = {0: 0, 1: 2, 2..n: 1}."""

def node_potential(alpha: tuple[int, ...], beta: dict[int, int]) -> int:
    """H(α) = Σ βᵢ αᵢ."""

def is_admissible_edge(
    alpha: tuple[int, ...],
    i: int, j: int,  # move mass from j to i
    beta: dict[int, int]
) -> bool:
    """Check if α → α + eᵢ - eⱼ is admissible (αⱼ ≥ 1 and βᵢ > βⱼ)."""
```

#### `src/graph.py` — Graph Structure
```python
class SimplexMemoryGraph:
    def __init__(self, n: int, m: int, n_in: int = 1, n_out: int = 1):
        self.n = n
        self.m = m
        self.n_in = n_in
        self.n_out = n_out
        self.nodes = generate_lattice(n, m)  # core nodes
        self.input_nodes = [("in", i) for i in range(n_in)]
        self.output_nodes = [("out", i) for i in range(n_out)]
        self.edges = self._build_edges()  # using potential orientation
        self.preds = self._build_preds()
        self.topological_levels = self._build_topological_levels()
    
    def _build_edges(self) -> list[tuple[tuple, tuple]]:
        """Build directed edges using potential orientation."""
    
    @property
    def backbone(self) -> list[tuple[int, ...]]:
        """Return 𝓑_{n,m} = {α : α₂=⋯=αₙ=0}."""
    
    @property
    def F_in(self) -> list[tuple[int, ...]]:
        """Return input facet {α : α₁=0}."""
    
    @property
    def F_out(self) -> list[tuple[int, ...]]:
        """Return output facet {α : α₀=0}."""
    
    @property
    def F_mid(self) -> list[tuple[int, ...]]:
        """Return shared face {α : α₀=α₁=0}."""
```

#### `src/model.py` — PyTorch Model
Similar structure to 0410's `tmn.py` but generalized:
- Level-based forward pass using topological order
- Kaiming initialization for edges
- Configurable activation functions
- Support for scalar hidden states (d=1) initially

#### `src/config.py` — Configuration
Reuse 0410's pattern with additions:
```python
@dataclass
class Config:
    n: int = 2          # simplex dimension
    m: int = 3          # lattice resolution
    n_in: int = 1
    n_out: int = 1
    # ... same as 0410: lr, batch_size, epochs, task_name, etc.
```

### Phase 3: Visualization

#### Potential Flow Figure (like report Figure 2)
For `(n,m)=(2,3)`:
- Draw equilateral triangle with lattice points
- Color nodes by H value
- Draw arrows for edges
- Vertical dashed lines for isopotentials

#### Backbone vs Buffer Highlight
- Highlight backbone nodes in one color
- Highlight F_mid nodes in another
- Show edge directions

### Phase 4: Testing Strategy

1. **Lattice tests**: Verify |V_{n,m}| = C(m+n-1, n)
2. **Potential tests**: H increases along every edge
3. **Acyclicity tests**: No directed cycles
4. **Facet tests**: F_mid = F_in ∩ F_out, H = m-1 on F_mid
5. **Forward pass tests**: Shapes match, no NaN

### Phase 5: Milestones

| Milestone | Deliverable |
|-----------|-------------|
| 1. Lattice + Potential | `lattice.py`, `potential.py` + tests |
| 2. Graph Structure | `graph.py` with facets, backbone |
| 3. Model | `model.py` with forward pass |
| 4. Training | `train.py`, `data.py`, `config.py` |
| 5. Visualization | `plot.py` + comparison with MLP |
| 6. Integration | `run.py` + `params.yaml` |

## Critical Files to Modify/Create

**New files to create:**
- `exp0414_simplexNet/src/lattice.py`
- `exp0414_simplexNet/src/potential.py`
- `exp0414_simplexNet/src/graph.py`
- `exp0414_simplexNet/src/model.py`
- `exp0414_simplexNet/src/config.py`
- `exp0414_simplexNet/src/train.py`
- `exp0414_simplexNet/src/data.py`
- `exp0414_simplexNet/src/plot.py`
- `exp0414_simplexNet/tests/test_graph.py`
- `exp0414_simplexNet/tests/test_potential.py`
- `exp0414_simplexNet/params.yaml`
- `exp0414_simplexNet/run.py`

**Reference from 0410 (reuse patterns):**
- `exp0410_triangleNet/scripts/config.py` → config structure
- `exp0410_triangleNet/scripts/train.py` → training loop
- `exp0410_triangleNet/scripts/data.py` → target functions
- `exp0410_triangleNet/scripts/model/tmn.py` → level-based forward pass
- `exp0410_triangleNet/scripts/model/graph.py` → graph building, topological levels

## Verification

End-to-end test:
```bash
cd exp0414_simplexNet
python3 run.py  # Should train SMN(2,3) and MLP, produce comparison plot
```

Expected output:
- `runs/<timestamp>/comparison.png` — SMN vs MLP
- Loss curves decreasing
- No NaN/Inf in weights

Unit tests:
```bash
pytest tests/
```
