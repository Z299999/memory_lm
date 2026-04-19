# exp0418_brainSimplex — Research Plan

**Goal**: Analyze the simplicial structure and fractal properties of the Drosophila (fruit fly) brain connectome.

**Background**: 
- b00002 (Reimann et al., 2017) discovered rich high-dimensional directed simplices and cavities in neocortical microcircuits
- b00003 provides the Drosophila brain cell-type atlas and connectome data
- This project investigates whether similar simplicial organization exists in the fly brain

---

## Phase 1: Data Acquisition

- [x] **1.1 Obtain FlyWire connectome dataset**
  - [x] Access FlyWire database (https://flywire.ai or neuroglancer interface)
  - [x] Download full connectome adjacency matrix (neuron × neuron)
  - [x] Convert to edge list format compatible with networkx

- [x] **1.2 Data preprocessing**
  - [x] Extract excitatory/inhibitory neuron subgraphs (if annotations available)
  - [x] Generate regional subgraphs (optional, for local analysis)
  - [x] Create standard graph formats:
    - [x] Adjacency matrix (`.npy` or `.pt`)
    - [x] Edge list (`.csv` or `.edgelist`)
    - [x] Node metadata (`.csv` with cell types, brain regions)

**Deliverables**: `data/raw/`, `data/processed/` directories with connectome files

---

## Phase 2: Simplex Detection

- [x] **2.1 Implement directed simplex detection algorithm**
  - [x] Define directed clique: all-to-all connected DAG (per b00002)
  - [x] Implement k-simplex enumerator for k = 1, 2, 3, ...
  - [x] Optimize for large graphs (iterative algorithm, pruning)
  - [x] Generate simplex count distribution by dimension

- [x] **2.2 Null model comparisons**
  - [x] Generate Erdős-Rényi random graphs with matched (n, p)
  - [ ] Generate degree-preserving rewired null models
  - [x] Compare simplex distributions: biological vs. random
  - [ ] Statistical testing: is high-dimensional simplex excess significant?

**Deliverables**: `src/simplex_detection.py`, `results/simplex_counts.csv`, `plots/simplex_distribution.png`

---

## Phase 3: Cavity Analysis

- [x] **3.1 Construct directed flag complex**
  - [x] Glue all detected simplices into a simplicial complex
  - [x] Compute Euler characteristic: χ = Σ(-1)^k · (# k-simplices)
  - [x] Compute Betti numbers: β₀, β₁, β₂, ... (approximate)

- [x] **3.2 Cavity visualization**
  - [x] Visualize low-dimensional cavities (1D loops, 2D voids)
  - [x] Plot Betti number trajectories across dimensions
  - [x] Identify "hub" simplices that participate in multiple cavities

**Deliverables**: `src/cavity_analysis.py`, `plots/cavities_viz.png`, `results/betti_numbers.json`

---

## Phase 4: Fractal Analysis

- [x] **4.1 Compute fractal dimensions**
  - [x] Box-counting dimension on the simplicial complex
  - [x] Correlation dimension from simplex participation
  - [x] Compare with null models: is the brain more/less fractal?

- [x] **4.2 Test self-similarity**
  - [x] Check for power-law scaling in simplex distributions
  - [x] Analyze structure at multiple scales (full brain → regions → microcircuits)
  - [x] Fit scaling exponents and assess goodness of fit

**Deliverables**: `src/fractal_analysis.py`, `plots/scaling_laws.png`, `results/fractal_dimensions.json`

---

## Phase 5: Functional Interpretation

- [x] **5.1 Analyze simplices as computational units**
  - [x] Map input→output pathways through simplices
  - [x] Identify "source" and "sink" neurons in directed cliques
  - [x] Hypothesize information flow patterns

- [x] **5.2 Compare with exp0414_simplexNet architecture**
  - [x] Biological simplices vs. artificial SMN lattice structure
  - [x] Extract design principles for future neural architectures
  - [x] Document key similarities and differences

**Deliverables**: `report.md` or `report.pdf` with findings and interpretations

---

## Project Structure

```
exp0418_brainSimplex/
├── PLAN.md                  # This file (research plan with checklist)
├── README.md                # Project overview and quick start
├── report.md                # Analysis report (auto-generated)
├── idea.md                  # Original research idea notes
├── requirements.txt         # Python dependencies
├── data/
│   ├── DATA_DOWNLOAD.md     # Instructions for obtaining FlyWire data
│   ├── raw/                 # Raw FlyWire data
│   │   ├── sample_edge_list.csv
│   │   └── SAMPLE_DATA_README.md
│   └── processed/           # Preprocessed graphs
├── src/
│   ├── data_acquisition.py  # FlyWire data download utilities
│   ├── preprocessing.py     # Graph preprocessing pipeline
│   ├── simplex_detection.py # Directed simplex enumeration
│   ├── cavity_analysis.py   # Flag complex & topological invariants
│   ├── fractal_analysis.py  # Fractal dimension computation
│   └── report_generation.py # Report generation
├── results/
│   ├── simplex_counts.csv       # Simplex distribution
│   ├── null_model_comparison.csv # Null model analysis
│   ├── cavity_analysis.json     # Topological invariants
│   └── fractal_analysis.json    # Fral analysis results
├── plots/                   # Visualizations
│   ├── cavity_structure.png
│   └── fractal_analysis.png
└── tests/                   # Unit tests
```

---

## Suggested Tech Stack

| Module | Tools |
|--------|-------|
| Graph analysis | `networkx`, `igraph` |
| Algebraic topology | `giotto-tda`, `dionysus`, `ripser` |
| Visualization | `matplotlib`, `plotly`, `neuroglancer` |
| Data handling | `numpy`, `pandas`, `scipy` |

---

## Notes

- **Computational challenge**: FlyWire has ~10⁵ neurons; full high-dimensional simplex enumeration may be intractable. Consider sampling or approximate methods.
- **Fractal definition ambiguity**: Use multiple definitions and compare results.
- **Connection to prior work**: Leverage code from `exp0414_simplexNet` where applicable (lattice generation, potential functions).
