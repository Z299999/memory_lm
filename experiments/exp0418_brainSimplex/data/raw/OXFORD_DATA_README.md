# Drosophila Central Brain Data (via Oxford Networks Project)

## Source
Oxford Mathematics Part C, C5.4 Networks Mini-Project

## Dataset Statistics
- **Nodes**: 32,272
- **Edges**: 849,981
- **Average degree**: 26.34

## Files
- `pete_fly_central_edges.csv` - Oxford-provided raw edge list (`from`, `to`)
- `pete_fly_central_nodes_metadata.csv` - Oxford-provided raw node metadata
- `oxford_edge_list.csv` - Standardized edge list (`pre`, `post`, `weight`)
- `oxford_nodes.csv` - Standardized node metadata (`id`, `x`, `y`, `z`, `module`, `root_id`)

## Original Source
FlyWire Project (2024), Nature
Central brain region of Drosophila connectome

## Preprocessing Notes
- Edges are unweighted (weight=1 for all)
- Node coordinates in nanometers
- Module labels from Infomap (67 modules at level 2)
