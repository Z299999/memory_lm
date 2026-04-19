# exp0418_brainSimplex

**Topological Analysis of Drosophila Brain Connectome**

This project analyzes the simplicial structure, cavities, and fractal properties of the Drosophila (fruit fly) brain connectome, following the methodology of Reimann et al. (2017).

## Quick Start

### 1. Install Dependencies

```bash
cd exp0418_brainSimplex
pip install -r requirements.txt
```

### 2. Download Data (Optional)

For full FlyWire connectome analysis:

```bash
# See data/DATA_DOWNLOAD.md for detailed instructions
python src/data_acquisition.py --full
```

For testing with sample data (no download required):

```bash
python src/data_acquisition.py  # Creates 100-node sample graph
```

### 3. Run Analysis Pipeline

```bash
# Step 1: Preprocess data
python src/preprocessing.py

# Step 2: Detect simplices and compare with null models
python src/simplex_detection.py 6 --compare 10

# Step 3: Analyze cavities
python src/cavity_analysis.py 3

# Step 4: Fractal analysis
python src/fractal_analysis.py

# Step 5: Generate report
python src/report_generation.py
```

## Project Overview

### Background

The paper "Cliques of Neurons Bound into Cavities" (Reimann et al., 2017) discovered that:
- Neural circuits contain high-dimensional **directed simplices** (cliques)
- These simplices organize into **cavities** (topological holes)
- This structure is non-random and functionally significant

### Goals

1. **Detect** directed simplices in the FlyWire connectome
2. **Compare** with null models (Erdős-Rényi, degree-preserving)
3. **Compute** topological invariants (Euler characteristic, Betti numbers)
4. **Analyze** fractal properties and self-similarity
5. **Interpret** functional implications for information processing

### Data

| Dataset | Neurons | Synapses | Status |
|---------|---------|----------|--------|
| FlyWire (full) | ~130,000 | ~50M | Requires download |
| Sample (test) | 100 | ~500 | Included |

## Project Structure

```
exp0418_brainSimplex/
├── PLAN.md                  # Research plan with checklist
├── README.md                # This file
├── report.md                # Auto-generated analysis report
├── idea.md                  # Original research notes
├── requirements.txt         # Dependencies
├── data/
│   ├── raw/                 # Raw data
│   └── processed/           # Preprocessed graphs
├── src/
│   ├── data_acquisition.py  # Data download
│   ├── preprocessing.py     # Graph preprocessing
│   ├── simplex_detection.py # Simplex enumeration
│   ├── cavity_analysis.py   # Topological analysis
│   ├── fractal_analysis.py  # Fractal dimension
│   └── report_generation.py # Report generation
├── results/                 # Analysis results (CSV/JSON)
└── plots/                   # Visualizations
```

## Analysis Pipeline

### Phase 1: Data Acquisition
- Download FlyWire connectome from CODEx API
- Preprocess into edge list and adjacency matrix formats

### Phase 2: Simplex Detection
- Enumerate directed k-simplices (k = 1, 2, 3, ...)
- Compare with Erdős-Rényi null models
- Compute z-scores for simplex excess

### Phase 3: Cavity Analysis
- Construct directed flag complex
- Compute Euler characteristic: χ = Σ(-1)^k · (# k-simplices)
- Estimate Betti numbers (topological holes)

### Phase 4: Fractal Analysis
- Box-counting dimension
- Correlation dimension
- Power-law scaling tests

### Phase 5: Functional Interpretation
- Map information flow pathways
- Compare with exp0414_simplexNet architecture
- Generate summary report

## Key Concepts

| Term | Definition |
|------|------------|
| **Directed k-simplex** | All-to-all connected DAG with k+1 nodes |
| **Directed clique** | Same as directed simplex |
| **Flag complex** | Simplicial complex formed by all simplices |
| **Cavity** | Topological hole in the flag complex |
| **Euler characteristic** | χ = V - E + F - ... (alternating sum) |
| **Betti number β_k** | Number of k-dimensional holes |

## References

1. Reimann, M. W., et al. (2017). Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function. *Frontiers in Computational Neuroscience*, 11, 48.

2. Schlegel, P., et al. (2024). Whole-brain annotation and multiconnectome cell typing of Drosophila. *Nature*.

## License

This project is for research purposes.
