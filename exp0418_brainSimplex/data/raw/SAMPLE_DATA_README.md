# FlyWire Sample Data

## How to obtain the full dataset

### Option 1: FlyWire Python API (Recommended)

```bash
pip install flywire
```

```python
from flywire import fetch_connectome

cn = fetch_connectome(version='783')  # or latest

# Save to CSV
cn.neurons.to_csv('neurons.csv')
cn.synapses.to_csv('synapses.csv')
```

### Option 2: Direct download from CODEx

Visit: https://codex.flywire.ai/api/v1/table

Available tables:
- `flywire_neuropil` - Full brain synapses
- `flywire_meta` - Neuron metadata

### Option 3: Google Cloud Public Datasets

The FlyWire connectome is available on Google Cloud:
https://console.cloud.google.com/marketplace/product/flywire/flywire

## Data format

### neurons.csv columns:
- id: Unique neuron identifier
- root: Root ID for merged segments
- type: Cell type label
- hemibrainType: Hemibrain atlas type (if matched)
- hemilineage: Developmental lineage
- side: Brain hemisphere (left/right)

### synapses.csv columns:
- pre: Presynaptic neuron ID
- post: Postsynaptic neuron ID
- x, y, z: Synapse location
- size: Synapse size/weight
