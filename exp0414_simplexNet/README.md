# exp0414_simplexNet — Simplex Memory Networks

A geometric feedforward neural architecture based on simplicial lattices.

## Quick Start

### 1. Run a comparison experiment

```bash
cd exp0414_simplexNet
python3 run.py
```

Output: `runs/<date>/<experiment_name>/comparison.png` — 4-panel SMN vs MLP comparison.

### 2. Use SMN in your own code

```python
import sys
sys.path.insert(0, 'src')

from smn_fitter import SMNModule, SMNFitter
import torch

# --- Option A: SMNModule (raw nn.Module) ---
# Use this if you want full control over training
module = SMNModule(n=2, m=3, n_in=1, n_out=1, activation='relu')
x = torch.randn(32, 1)
y = module(x)  # forward pass

# --- Option B: SMNFitter (high-level API) ---
# Use this for quick experiments with built-in training
smn = SMNFitter(n=2, m=3, n_in=1, n_out=1)
smn.fit(epochs=300)  # trains on built-in sin_mix data
smn.plot(output_path="result.png")  # 2-panel: scatter + loss curve
```

---

## Project Structure

```
exp0414_simplexNet/
├── params.yaml          # Configuration for run.py
├── run.py               # Entry point: trains SMN vs MLP, saves comparison plot
├── src/
│   ├── config.py        # Config dataclass + YAML loader
│   ├── data.py          # Target functions + dataset builder
│   ├── graph.py         # SimplexMemoryGraph + lattice/potential helpers
│   ├── smn_fitter.py    # SMNModule (core network) + SMNFitter (training wrapper)
│   ├── mlp_fitter.py    # MLPFitter (baseline, same interface as SMNFitter)
│   └── plot.py          # Visualization utilities
└── tests/
    └── test_smn.py      # 12 unit tests for SMNModule
```

---

## API Reference

### SMNModule — Core PyTorch Module

Use `SMNModule` when you want to integrate SMN into your own training loop.

```python
from smn_fitter import SMNModule

module = SMNModule(
    n=2,                    # Simplex dimension (2=triangle, 3=tetrahedron, ...)
    m=3,                    # Resolution (lattice points per edge)
    n_in=1,                 # Input dimensions
    n_out=1,                # Output dimensions
    activation='relu',      # 'relu', 'leaky_relu', 'gelu', 'tanh'
    x_bounds=None,          # Per-channel bounds: [(min, max), ...] or None
)
```

**Key properties:**
- `module.arch_str` — Human-readable architecture description
- `module.param_count` — Total trainable parameters

**Forward pass:**
```python
x = torch.randn(64, n_in)  # batch of 64
y = module(x)              # output: (batch, n_out)
```

**Input normalization:**
- `x_bounds` defines per-channel input ranges (e.g., `[(-3.14, 3.14)]` for SISO)
- Inputs are automatically normalized to `[-1, 1]` per channel
- If `None`, defaults to `[(-1, 1)] * n_in`

---

### SMNFitter — High-Level Training Wrapper

Use `SMNFitter` for quick experiments with built-in training and visualization.

```python
from smn_fitter import SMNFitter

smn = SMNFitter(
    n=3, m=4,               # Architecture
    n_in=2, n_out=1,        # I/O dimensions
    x_bounds=[(-3.14, 3.14), (-3.14, 3.14)],  # Per-channel bounds
)
```

#### Training

```python
# Option 1: Built-in demo data (sin_mix for SISO, sin_sum for MIMO)
smn.fit(epochs=300, lr=1e-3, batch_size=64)

# Option 2: Custom data
x_train = torch.randn(1000, n_in)
y_train = torch.randn(1000, n_out)
smn.fit(x_train, y_train, epochs=300)

# Option 3: With validation split
smn.fit(x_train, y_train, x_val, y_val, epochs=300)
```

**`fit()` parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `x_train`, `y_train` | None | Training data. If None, uses built-in demo |
| `x_val`, `y_val` | None | Validation data. Auto-splits 80/20 if None |
| `loss_fn` | `nn.MSELoss()` | Custom loss function |
| `lr` | 0.001 | Learning rate |
| `epochs` | 300 | Training epochs |
| `batch_size` | 64 | Batch size |
| `seed` | 42 | Random seed |
| `verbose` | True | Print progress every 50 epochs |

#### Inference

```python
predictions = smn.predict(x_test)  # returns numpy array
```

#### Visualization

```python
# 2-panel plot (scatter + loss curve)
smn.plot(output_path="smn_result.png")

# 4-panel comparison with MLP baseline
mlp = MLPFitter(layers=[16, 16], n_in=1, n_out=1)
mlp.fit(x_train, y_train)
smn.plot(baseline=mlp, output_path="comparison.png", title="SMN vs MLP")
```

---

### MLPFitter — Baseline Comparison

`MLPFitter` mirrors the `SMNFitter` interface for fair comparison:

```python
from smn_fitter import MLPFitter

mlp = MLPFitter(
    layers=[16, 16, 16],    # Hidden layer sizes
    n_in=1, n_out=1,        # I/O dimensions
    activation='relu',
    x_bounds=[(-3.14, 3.14)],
)
mlp.fit(x_train, y_train, epochs=300)
mlp.plot(output_path="mlp_result.png")  # 2-panel
```

---

## Advanced: Custom Loss Functions (RL, etc.)

If you have your own training loop (e.g., reinforcement learning, custom objectives), use `SMNModule` directly:

### Option 1: SMNModule only (recommended for RL)

```python
from smn_fitter import SMNModule
import torch

# Create the network (no training logic attached)
net = SMNModule(n=2, m=3, n_in=4, n_out=2, x_bounds=[(-1, 1)] * 4)

# Use in your own loop
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

for step in range(1000):
    # Your environment / data
    obs = get_observation()      # shape: (n_in,)
    action = net(obs.unsqueeze(0))  # forward pass
    
    # Your loss (policy gradient, TD error, etc.)
    loss = your_custom_loss(action, reward, next_obs)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Option 2: Access module from SMNFitter

```python
from smn_fitter import SMNFitter

smn = SMNFitter(n=2, m=3, n_in=4, n_out=2)
net = smn.module  # Access the underlying SMNModule

# Now use `net` in your own loop (same as Option 1)
```

### Example: Policy Gradient with SMN

```python
from smn_fitter import SMNModule
import torch
import torch.nn.functional as F

# SMN as policy network: state -> action probabilities
policy_net = SMNModule(n=2, m=3, n_in=4, n_out=2)  # 4-dim state, 2 actions
optimizer = torch.optim.Adam(policy_net.parameters(), lr=3e-4)

def select_action(state):
    state = torch.FloatTensor(state)
    logits = policy_net(state.unsqueeze(0))
    probs = F.softmax(logits, dim=1)
    action = probs.multinomial(num_samples=1)
    return action.item(), probs.log()[0, action.item()]

def reinforce_loss(log_prob, reward):
    return -log_prob * reward  # Gradient ascent via gradient descent

for episode in range(100):
    state = env.reset()
    log_probs = []
    rewards = []
    
    for t in range(100):
        action, log_prob = select_action(state)
        next_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        
        if done:
            break
        state = next_state
    
    # Compute returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + 0.99 * R
        returns.insert(0, R)
    
    # Policy gradient loss
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(reinforce_loss(log_prob, G))
    
    optimizer.zero_grad()
    loss = torch.cat(policy_loss).sum()
    loss.backward()
    optimizer.step()
```

---

## Configuration (params.yaml)

For `python3 run.py`, configure via `params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | 2 | Simplex dimension |
| `m` | 3 | Resolution |
| `n_in` | 1 | Input dimensions |
| `n_out` | 1 | Output dimensions |
| `mlp_layers` | [8, 8, 8] | MLP hidden sizes |
| `node_activation` | relu | `relu`, `leaky_relu`, `gelu`, `tanh` |
| `task_name` | sin_mix | Target function (see below) |
| `epochs` | 300 | Training epochs |
| `lr` | 0.001 | Learning rate |
| `batch_size` | 64 | Batch size |
| `num_train` | 10000 | Training set size |
| `x_bounds` | — | Per-channel bounds, e.g. `[[-π,π],[-1,1]]` |

### Task Registry

| `task_name` | n_in | n_out | Formula |
|-------------|------|-------|---------|
| `sin` | 1 | 1 | $\sin(x)$ |
| `sin_mix` | 1 | 1 | $0.5\sin(x) + 0.3\sin(2x) + 0.2\sin(3x)$ |
| `sin_cos` | 1 | 2 | $[\sin(x), \cos(x)]$ |
| `sin_sum` | 2 | 1 | $\sin(x_1 + x_2)$ |
| `trig_2d` | 2 | 2 | $[\sin(x_1+x_2), \cos(x_1-x_2)]$ |

---

## Output Format

### run.py Comparison Plot

`runs/<date>/<name>/comparison.png` — 4 panels:

| Panel | Content |
|-------|---------|
| Top-left | SMN: ‖y_true‖₂ vs ‖y_pred‖₂ scatter (diagonal = perfect) |
| Top-right | MLP: same scatter |
| Bottom-left | SMN train/val loss curves (log scale) |
| Bottom-right | MLP train/val loss curves |

The L2-norm scatter works for any `n_out` (1i1o to 100i100o).

### SMNFitter.plot()

- **Without baseline**: 2-panel (scatter + loss curve)
- **With baseline**: 4-panel comparison

---

## Running Tests

```bash
cd exp0414_simplexNet
PYTHONPATH=src python3 tests/test_smn.py
```

12 unit tests covering:
- Shape validation (SISO, MIMO)
- Input normalization
- Gradient flow
- State dict roundtrip

---

## Mathematical Background (Brief)

### Lattice V_{n,m}

$$V_{n,m} = \{\alpha \in \mathbb{Z}_{\geq 0}^{n+1} : \sum_{i=0}^n \alpha_i = m-1\}$$

Cardinality: $|V_{n,m}| = \binom{m+n-1}{n}$

### Edge Orientation

Edges follow a potential function $H(\alpha) = \sum \beta_i \alpha_i$:
- $\beta_0 = 0$ (input vertex)
- $\beta_1 = 2$ (output vertex)
- $\beta_k = 1$ for $k \geq 2$ (hidden vertices)

Edge $\alpha \to \alpha'$ exists iff potential increases.

---

## Examples

### Example 1: SISO sin_mix (default)

```python
from smn_fitter import SMNFitter

smn = SMNFitter(n=2, m=3)  # triangle, 3 points/edge
smn.fit(epochs=300)
smn.plot(output_path="siso.png")
print(f"Final val loss: {smn.final_val_loss:.4f}")
```

### Example 2: MIMO 2i2o

```python
from smn_fitter import SMNFitter, MLPFitter

# 2-input, 2-output with custom bounds
smn = SMNFitter(
    n=3, m=4,
    n_in=2, n_out=2,
    x_bounds=[(-3.14, 3.14), (-3.14, 3.14)],
)
smn.fit(epochs=500)

# Compare with MLP
mlp = MLPFitter(layers=[16, 16], n_in=2, n_out=2)
mlp.fit(epochs=500)

# 4-panel comparison
smn.plot(baseline=mlp, output_path="mimo_compare.png")
```

### Example 3: Custom Training Loop

```python
from smn_fitter import SMNModule
import torch.nn as nn
import torch.optim as optim

module = SMNModule(n=2, m=3, n_in=1, n_out=1)
optimizer = optim.Adam(module.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    x = torch.randn(64, 1) * 3
    y = torch.sin(x)
    
    optimizer.zero_grad()
    pred = module(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"epoch={epoch+1}, loss={loss.item():.4f}")
```

---

## Reference

Full mathematical treatment: `report_tex/report.pdf`
