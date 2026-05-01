# SimplexNet

SimplexNet is now a pure PyTorch package for **Simplex Memory Networks**.

The public API is intentionally small:

```python
from simplexnet import SMN, SimplexMemoryGraph
```

- `SMN` is the trainable `torch.nn.Module`.
- `SimplexMemoryGraph` exposes the underlying simplicial DAG for analysis and research use.

This package does **not** include reinforcement learning algorithms, training wrappers, GUI tools, checkpoint helpers, logging helpers, or plotting utilities. If you want to use SMN inside an RL project, import `SMN` as a normal backbone and build the RL stack outside this package.

## Install

Requirements:

```bash
pip install torch
```

## Model API

```python
from simplexnet import SMN

model = SMN(
    n=2,
    m=4,
    n_in=3,
    n_out=2,
    activation="relu",
    output_activation="identity",
    scale_output=True,
)
```

Key constructor arguments:

- `n`: simplex dimension
- `m`: simplex resolution
- `n_in`: input dimension
- `n_out`: output dimension
- `activation`: hidden activation (`relu`, `leaky_relu`, `gelu`, `tanh`)
- `output_activation`: output activation (`identity`, `tanh`)
- `scale_output`: variance-preserving output scaling

Useful model attributes:

- `model.n`
- `model.m`
- `model.n_in`
- `model.n_out`
- `model.graph`
- `model.param_count`
- `model.arch_str`

Because `SMN` is a standard `torch.nn.Module`, it already supports:

- `model(x)`
- `model.parameters()`
- `model.train()` / `model.eval()`
- `model.state_dict()` / `model.load_state_dict()`
- `model.to(device)`

## Minimal Forward Example

```python
import torch
from simplexnet import SMN

model = SMN(n=2, m=4, n_in=3, n_out=2)
x = torch.randn(8, 3)
y = model(x)

print(y.shape)  # torch.Size([8, 2])
```

## Minimal Training Example

```python
import torch
from simplexnet import SMN

model = SMN(n=3, m=5, n_in=4, n_out=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

x = torch.randn(32, 4)
target = torch.randn(32, 1)

pred = model(x)
loss = criterion(pred, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Graph API

```python
from simplexnet import SimplexMemoryGraph

graph = SimplexMemoryGraph(n=3, m=4, n_in=2, n_out=1)

print(graph.core_node_count)
print(graph.edge_count)
print(graph.F_in)
print(graph.F_out)
print(graph.F_mid)
print(graph.backbone)
```

`SimplexMemoryGraph` keeps the main structural interfaces available for advanced use:

- `core_nodes`, `input_nodes`, `output_nodes`, `nodes`, `edges`
- `preds`, `succs`, `topological_levels`
- `F_in`, `F_out`, `F_mid`, `backbone`
- `core_node_count`, `edge_count`

## Examples

Runnable non-RL examples are kept in:

- `examples/minimal_forward.py`
- `examples/minimal_train.py`
