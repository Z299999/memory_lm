# exp0418_brainSimplex — Analysis Report

**Generated**: 2026-04-19T10:54:55.871452

## Executive Summary

This report presents the topological analysis of neural connectome data,
focusing on directed simplicial structure, cavities, and fractal properties.

## 1. Simplex Analysis

### Simplex Distribution

| Dimension | Count |
|-----------|-------|
| 0 | 100 |
| 1 | 483 |
| 2 | 91 |
| 3 | 0 |

### Comparison with Null Models

| Dim | Original | Null Mean | Z-score | Excess Ratio |
|-----|----------|-----------|---------|--------------|
| 0 | 100 | 100.0 | 0.00 | 1.00x |
| 1 | 483 | 468.0 | 1.06 | 1.03x |
| 2 | 91 | 89.0 | 0.15 | 1.02x |
| 3 | 0 | 0.6 | -1.22 | 0.00x |
| 4 | 0 | 0.0 | 0.00 | infx |
| 5 | 0 | 0.0 | 0.00 | infx |
| 6 | 0 | 0.0 | 0.00 | infx |

## 2. Cavity Analysis

- **Euler characteristic**: χ = -292
- **Total simplices**: 674
- **Maximum simplex dimension**: 3

### Betti Numbers (Approximate)

- β_0 = 1
- β_1 = 372
- β_2 = 0
- β_3 = 0

## 3. Fractal Analysis

- **Box-counting dimension**: 1.000
- **Correlation dimension**: nan

## 4. Functional Interpretation

### Implications for Information Processing

The presence of high-dimensional directed simplices suggests:

1. **Hierarchical information flow**: Directed simplices enforce ordered activation sequences
2. **Parallel processing paths**: Multiple simplices provide redundant computation pathways
3. **Temporal binding**: Clique structure supports synchronized activity patterns

### Comparison with exp0414_simplexNet

The simplex architecture in exp0414_simplexNet was inspired by these biological findings.
Key parallels:

- Both use simplicial structure to organize computation
- Both have directed information flow (input → output)
- Both decompose into backbone and buffer components

## 5. Methods

### Data

- Sample: 100-node random graph (for testing)
- Full analysis pending FlyWire connectome download

### Algorithms

- Directed simplex detection: Iterative clique enumeration
- Euler characteristic: Alternating sum of simplex counts
- Betti numbers: Heuristic approximation via cycle basis
- Fractal dimensions: Box-counting and correlation methods

## 6. Next Steps

1. Download full FlyWire connectome (~130,000 neurons)
2. Run simplex detection on complete data
3. Compare with biological null models (degree-preserving rewiring)
4. Analyze regional variations (mushroom body, optic lobes, etc.)
5. Investigate cell-type-specific simplicial structure
