# exp0418_brainSimplex — Research Plan

**Goal**: Analyze the simplicial structure of the Drosophila (fruit fly) brain connectome.

**Background**: Reimann et al. (2017) discovered rich high-dimensional directed simplices in neocortical microcircuits. This project investigates whether similar simplicial organization exists in the fly brain.

---

## Phase 1: Data Acquisition

- [x] **1.1 Obtain FlyWire connectome dataset**
  - [x] Create data download utilities
  - [x] Generate sample data for testing

- [x] **1.2 Data preprocessing**
  - [x] Create weighted edge list
  - [x] Generate adjacency matrix
  - [x] Compute graph statistics

**Deliverables**: `data/raw/`, `data/processed/` directories with connectome files

---

## Phase 2: Simplex Detection

- [x] **2.1 Implement directed simplex detection algorithm**
  - [x] Define directed clique: all-to-all connected DAG
  - [x] Implement k-simplex enumerator for k = 1, 2, 3, ...
  - [x] Optimize for large graphs (iterative algorithm)
  - [x] Generate simplex count distribution by dimension

- [x] **2.2 Null model comparisons**
  - [x] Generate Erdős-Rényi random graphs with matched (n, p)
  - [x] Compare simplex distributions: biological vs. random
  - [x] Compute z-scores

**Deliverables**: `src/simplex_detection.py`, `results/simplex_counts.csv`, `results/null_model_comparison.csv`

---

## Project Structure

```
exp0418_brainSimplex/
├── PLAN.md                  # This file
├── README.md                # Project overview
├── idea.md                  # Original research idea notes
├── data/
│   ├── raw/                 # Raw data
│   │   ├── synapses.csv
│   │   └── neurons.csv
│   └── processed/           # Preprocessed graphs
├── src/
│   ├── download_subset.py   # Generate subset data
│   ├── data_acquisition.py  # Data download utilities
│   ├── preprocessing.py     # Graph preprocessing
│   └── simplex_detection.py # Simplex enumeration
├── results/
│   ├── simplex_counts.csv
│   └── null_model_comparison.csv
└── plots/
```

## Suggested Tech Stack

| Module | Tools |
|--------|-------|
| Graph analysis | `networkx`, `igraph` |
| Data handling | `numpy`, `pandas`, `scipy` |
| Visualization | `matplotlib` |
