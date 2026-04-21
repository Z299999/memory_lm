# TOOLS/SimplexNet - Simplex Memory Network for Reinforcement Learning

A PyTorch-based implementation of Simplex Memory Networks (SMN) for reinforcement learning tasks.

## Overview

Simplex Memory Network is a geometric neural network architecture built on simplicial lattice structures. This implementation provides:

- **SMNmodule**: Core neural network with topological forward propagation
- **SMN_RL**: High-level RL wrapper for training and testing
- **RL Algorithms**: DQN (with PPO and REINFORCE coming in Phase 2)
- **Tools**: Checkpointing, logging, and visualization utilities

## Installation

Requirements:
```bash
pip install torch numpy gymnasium matplotlib
```

## Quick Start

### High-level API (Recommended)

```python
from simplexnet import SMN_RL
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1')

# Create SMN_RL wrapper
smn_rl = SMN_RL(
    env=env,
    n=2, m=4,           # Simplex parameters
    n_in=4, n_out=2,    # obs_dim, action_dim
    gamma=0.99,
    lr=1e-3
)

# Train
rewards = smn_rl.train(num_episodes=500)

# Test
mean, std, _ = smn_rl.test(num_episodes=10)
print(f"Test reward: {mean:.2f} +/- {std:.2f}")

# Plot results
smn_rl.plot_results()
```

### Low-level API

```python
from simplexnet import SMNmodule
from simplexnet.rl.algorithms import DQN

# Create Q-network
q_network = SMNmodule(n=2, m=4, n_in=4, n_out=2)

# Create DQN agent
dqn = DQN(
    q_network=q_network,
    obs_dim=4,
    act_dim=2,
    gamma=0.99,
    lr=1e-3
)

# Training loop
for episode in range(500):
    state = env.reset()
    for step in range(500):
        action = dqn.select_action(state, training=True)
        next_state, reward, done, _, _ = env.step(action)
        dqn.store_transition(state, action, reward, next_state, done)
        loss = dqn.train_step()
        state = next_state
        if done:
            break
```

## Module API Reference

### SMNmodule

```python
from simplexnet import SMNmodule

module = SMNmodule(
    n=2,                # Simplex dimension (order of simplices)
    m=4,                # Lattice parameter (size in each dimension)
    n_in=4,             # Input dimension
    n_out=2,            # Output dimension
    activation='relu',  # Activation function
    x_bounds=None       # Input normalization bounds [[min, max], ...]
)

# Forward pass
output = module(input_tensor)  # input: [batch, n_in], output: [batch, n_out]
```

### SMN_RL

```python
from simplexnet import SMN_RL

smn_rl = SMN_RL(
    env,                    # Gymnasium environment
    n=2, m=4,               # Simplex parameters
    n_in=4, n_out=2,        # Network I/O dimensions
    gamma=0.99,             # Discount factor
    lr=1e-3,                # Learning rate
    epsilon=1.0,            # Initial exploration rate
    epsilon_decay=0.995,    # Epsilon decay per episode
    epsilon_min=0.01,       # Minimum exploration rate
    checkpoint_dir='./runs/simplexnet/checkpoints',
    log_dir='./runs/simplexnet/logs',
    plot_dir='./runs/simplexnet/plots'
)

# Train
rewards = smn_rl.train(
    num_episodes=500,
    max_steps=500,
    update_target_every=100,
    checkpoint_every=50,
    verbose=True
)

# Test
mean, std, episode_rewards = smn_rl.test(
    num_episodes=10,
    render=False,
    deterministic=True
)

# Plot
smn_rl.plot_results(window=100)
```

### DQN

```python
from simplexnet.rl.algorithms import DQN

dqn = DQN(
    q_network,              # SMNmodule instance
    obs_dim=4,              # Observation dimension
    act_dim=2,              # Action dimension (discrete)
    gamma=0.99,             # Discount factor
    lr=1e-3,                # Learning rate
    epsilon=1.0,            # Initial exploration rate
    buffer_size=10000,      # Replay buffer capacity
    train_start=1000,       # Min samples before training
    train_frequency=4       # Train every N steps
)

# Select action
action = dqn.select_action(state, training=True)

# Store transition
dqn.store_transition(state, action, reward, next_state, done)

# Training step
loss = dqn.train_step(batch_size=64, step=total_steps)

# Update target network
dqn.update_target_network()

# Decay epsilon
dqn.decay_epsilon()
```

### Tools

```python
from simplexnet.tools import CheckpointManager, TrainingLogger, plot_training_curves

# Checkpoint management
ckpt_mgr = CheckpointManager('./runs/simplexnet/checkpoints')
ckpt_mgr.save_checkpoint(module, optimizer, episode, reward)
checkpoint = ckpt_mgr.load_latest()

# Training logger
logger = TrainingLogger('./runs/simplexnet/logs')
logger.log_init(config={'lr': 1e-3})
logger.log_epoch(episode=1, reward=100, loss=0.5)
logs = logger.get_logs()

# Plotting
plot_training_curves(rewards, losses, save_path='training.png')
plot_reward_curve(rewards, window=100, save_path='reward.png')
```

## Project Structure

```
TOOLS/SimplexNet/
├── src/
│   └── simplexnet/             # Package source
│       ├── __init__.py
│       ├── core/               # Core modules
│       │   ├── __init__.py
│       │   ├── SimplexMemoryGraph.py   # Simplicial lattice DAG structure
│       │   ├── SMNmodule.py            # PyTorch neural network module
│       │   └── SMN_RL.py               # High-level RL wrapper
│       ├── rl/
│       │   ├── __init__.py
│       │   └── algorithms/
│       │       ├── __init__.py
│       │       └── dqn.py        # DQN algorithm
│       └── tools/
│           ├── __init__.py
│           ├── checkpoint.py     # Checkpoint management
│           ├── logger.py         # Training logger
│           ├── plot.py           # Visualization utilities
│           └── gui.py            # GUI (Phase 2)
├── examples/
│   └── train_rl.py       # Training example script
└── runs/                 # Run artifacts (auto-generated, ignored by git)
    └── simplexnet/
        ├── checkpoints/
        ├── logs/
        └── plots/
```

## Examples

See `examples/` directory for complete training scripts:

```bash
cd TOOLS/SimplexNet
python3 examples/train_rl.py --env CartPole-v1 --episodes 500
```

All outputs (checkpoints, logs, plots) are saved to `runs/simplexnet/` directory.

## Checkpoint Format

Checkpoints are saved as PyTorch files with the following structure:

```python
{
    'timestamp': '20260420_120000',
    'config': {'n': 2, 'm': 4, 'n_in': 4, 'n_out': 2},
    'episode': 500,
    'reward': 450.0,
    'loss': 0.02,
    'state_dict': {...},
    'optimizer_state': {...},
    'metadata': {...}
}
```

## Log Format

Training logs are stored in JSON Lines format (run.log):

```json
{"timestamp": "2026-04-20T12:00:00", "event": "init", "config": {...}}
{"timestamp": "2026-04-20T12:00:01", "event": "epoch", "episode": 1, "reward": 100, "loss": 0.5}
{"timestamp": "2026-04-20T12:00:02", "event": "checkpoint_saved", "path": "..."}
```

## Roadmap

### Phase 1 (Current)
- [x] SMNmodule implementation
- [x] DQN algorithm
- [x] Checkpoint management
- [x] Training logger
- [x] Visualization utilities
- [x] High-level SMN_RL wrapper

### Phase 2 (Next)
- [ ] PPO algorithm
- [ ] REINFORCE algorithm
- [ ] PySide-based GUI
- [ ] Multi-environment support

### Phase 3
- [ ] Unit tests
- [ ] Documentation website
- [ ] Additional environments (MountainCar, Acrobot, etc.)

## Mathematical Background

SMN operates on a simplicial lattice $V_{n,m}$ with:
- **Lattice generation**: Recursive construction of simplices
- **Vertex potentials**: $\beta = \{0: 0, 1: 2, 2..n: 1\}$ for edge orientation
- **Topological propagation**: Forward pass through DAG levels
- **Edge weight initialization**: Kaiming initialization with variance preservation

See `report_tex/report.tex` for detailed mathematical formulation.

## License

MIT License
