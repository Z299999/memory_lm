# FlyWire Connectome Data (via Oxford Networks Project)

## Source
Oxford Mathematics Part C, C5.4 Networks Mini-Project

## Dataset Statistics
- **Nodes**: 32,272
- **Edges**: 849,981
- **Average degree**: 26.34

## Files
- `neurons.csv` - Node metadata (id, x, y, z, module)
- `edge_list.csv` - Edge list (pre, post, weight) - in processed/
- `node_features.csv` - Precomputed features (degree, pagerank, hodge, clusters)

## Original Source
FlyWire Project (2024), Nature
Central brain region of Drosophila connectome

## Preprocessing Notes
- Edges are unweighted (weight=1 for all)
- Node coordinates in nanometers
- Module labels from Infomap (67 modules at level 2)
