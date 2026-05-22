# Eco-Evolutionary Hunting Simulation (exp2-2)

Imported into this repository from `Z299999/EcoEvoLearning`, source commit `21a801f9140c8460b276c6d99c2308925f329047`.

A discrete-time eco-evolutionary simulation where neural-network-controlled agents hunt stationary prey on a 2D bounded rectangular domain.

## Quick Start

```bash
cd exp2-2
pip install -r requirements.txt
python run.py --config config.yaml
```

## Overview

- **Agents**: Neural-network-controlled individuals with energy, age, and mass (= number of edges/weights in neural network)
- **Prey**: Stationary resources with energy and shelf-life
- **Selection**: Energy/age-based natural selection with reproduction and mutation
- **Domain**: 2D bounded rectangle [0,100]×[0,100] with configurable boundaries (periodic or reflecting)

## Configuration

Edit `config.yaml` to modify simulation parameters. Key options:

- `make_video`: Set to `true` to generate MP4 video (slower), `false` for fast mode
- `T_days`: Simulation horizon in days
- `seed`: Random seed for reproducibility
- `log_every`: Print metrics every N days
- `boundary_mode`: Boundary conditions - `"periodic"` (wrap-around/torus) or `"reflect"` (elastic bouncing)
- `eta_window_days`: Rolling window size for utilization metrics (default 100)

## Output Files

Each run creates a timestamped folder in `outputs/runs/` containing:

- `config_resolved.yaml`: Copy of configuration used
- `console.log`: Terminal output log
- `timeseries.csv`: Daily metrics including:
  - Population: `N`, `B`, `D`, `b`, `d`, `M`
  - Energy: `E_mean`, `E_min`, `E_max`
  - Prey: `prey_count`, `prey_energy_total`
  - Costs: `energy_cost_motion_total`, `energy_cost_basal_total`, `energy_cost_birth_total`, `energy_cost_aging_total`
  - Utilization: `injected_energy`, `energy_gained_hunt_total`, `energy_rotted`, `eta_day`, `eta_cum`, `eta_window`, `rot_window`
- `video.mp4`: Animation (only if `make_video: true`)

## Plotting

A post-processing plotting script generates publication-quality figures from `timeseries.csv`.

### Usage

```bash
cd exp2-2

# Plot the latest run (auto-detected by timestamp)
python scripts/plot_timeseries.py

# Plot a specific run
python scripts/plot_timeseries.py --run_dir outputs/runs/20260202_203454_seed0

# With options
python scripts/plot_timeseries.py --window 100 --dpi 300 --format png

# Show plots interactively (requires display)
python scripts/plot_timeseries.py --show
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--run_dir` | latest | Path to run directory |
| `--window` | 100 | Rolling window for eta/rot metrics |
| `--dpi` | 150 | Output DPI |
| `--format` | png | Output format (png, pdf, svg) |
| `--show` | off | Display figures interactively |

### Generated Plots

Plots are saved to `<run_dir>/plots/`:

| Filename | Description |
|----------|-------------|
| `eta.png` | Energy utilization (eta{W}) and rotting fraction (rot{W}) over time |
| `mass.png` | Total biomass (M) over time |
| `birth_death.png` | Birth and death rates (b, d) over time |
| `pop_energy.png` | Population (N) and mean energy (E_mean) with dual y-axes |
| `energy_flux.png` | Energy fluxes: injected (E_in), eaten (E_eat), rotted (E_rot) |
| `energy_budget.png` | Energy costs: motion (E_move), basal (E_basal), reproduction (E_birth) |
| `age_energy.png` | Final-day age–energy distribution (mean energy per age, 5-day bins) |
| `age_distribution_final.png` | Histogram of agent ages at end of simulation |
| `plot_summary.json` | Metadata: detected columns, last-day values |

### Parameter Plots (no simulation required)

You can also visualize model parameters directly (e.g., metabolism curve)
without running a simulation:

```bash
cd exp2-2
python scripts/plot_parameters.py --config config.yaml --max_age 1000
```

This generates:

- `plots/params/metabolism_vs_age.png` — unified metabolism multiplier and
  cost-per-edge vs age for the current config.

## File Structure

```
exp2-2/
├── README.md
├── run.py              # Entry point
├── config.yaml         # Default configuration
├── requirements.txt    # Python dependencies
├── scripts/
│   └── plot_timeseries.py  # Post-processing plotting
├── src/
│   ├── __init__.py
│   ├── config.py       # Configuration loading
│   ├── sim.py          # Main simulation loop
│   ├── world.py        # World/domain handling
│   ├── agents.py       # Agent class
│   ├── prey.py         # Prey class
│   ├── observer.py     # Scent perception
│   ├── controller.py   # Neural network controller
│   ├── dynamics.py     # Motion dynamics
│   ├── hunting.py      # Hunting mechanics
│   ├── selector.py     # Natural selection
│   ├── reproduction.py # Reproduction and mutation
│   ├── checkpoint.py   # Checkpoint save/load
│   ├── metrics.py      # Metrics tracking
│   ├── console.py      # Console/file logging
│   ├── render.py       # Frame rendering
│   └── video.py        # Video generation
├── checkpoints/        # Checkpoint files (auto-created)
│   ├── ckpt_day_*.pt
│   └── latest.json
└── outputs/
    └── runs/           # Per-run output folders
        └── <run_id>/
            ├── config_resolved.yaml
            ├── console.log
            ├── timeseries.csv
            ├── video.mp4 (optional)
            └── plots/   # Generated plots
```

## Key Mechanics

### Local Perception (Nose)

Agents currently perceive their surroundings via a nose (prey scent)
within detection radius R (from prey). Eye channels are reserved for
future experiments and are disabled in this version.

Two observation modes are available:

**Vector Mode** (default in legacy, 5D input):
- S_vec: Vector sum of scent contributions (2D direction)
- S_scalar: Scalar sum of squared magnitudes (1D intensity)
- Neural input: `[S_vec_x, S_vec_y, S_scalar, energy, age]`

**Stencil Mode (nose)** (11D input):
- Nose (9D, circle + center): prey scent at 9 points:
  - Center: `(0, 0)`
  - 8 directions on a circle of radius `stencil_h`
- Internal (2D): `[energy, age]`
- Neural input（11D）：  
  `[nose_0..8, energy, age]`

Nose scent intensity at position q:
```
I(q) = Σ_j 1 / (||q - q_j||² + ε)  for prey j within radius R
```

**Nose geometry** (9 points on circle + center):
- 8 directions 等角分布在单位圆上（乘上 `stencil_h`），外加中心 `(0,0)`。

`stencil_h` 控制 nose 的采样距离 h。`stencil_geometry` 目前保留配置接口，
但在当前实现中不再区分几何：nose 固定为一圈等角采样。

### Neural Controller
MLP with ReLU activations, forward-only (no backprop). Input size depends on scent mode（5D for vector, 11D for stencil nose）。Output: velocity command (2D).

### Energy Dynamics
- Motion cost: `coef × 0.5 × mass × ||velocity||²`
- Metabolic cost: `mass × e0 × (1 + c_age × σ(age))` — unified basal + aging (see Metabolism Model)
- Hunting: energy transfer from prey to agent (limited by intake rate)

### Natural Selection (Edge-Scaled)

Agent mass is automatically computed as the number of edges (trainable weights) in the neural network. All energy thresholds are defined **per edge** and scaled by agent mass at runtime:

| Parameter | Per-Edge Config | Runtime Threshold |
|-----------|-----------------|-------------------|
| Survival | `eps_survival` | `E0_x = mass * eps_survival` |
| Birth cost | `eps_birth_cost` | `birth_cost = mass * eps_birth_cost` |
| Offspring energy | `eps_birth_energy` | `E_birth = mass * eps_birth_energy` |

**Self-Decided Reproduction**: Agents output a birth intent signal via their neural controller. Reproduction occurs when `y_birth > theta_birth`.

Selection rules:
- **Death**: `energy < 0` or `age > A`
- **Reproduction**: Agent decides via neural network output

This design ensures scale consistency when network architecture changes.

### Boundary Modes

The simulation supports two boundary modes:

- **Periodic (`boundary_mode: periodic`)**: 2D torus topology where agents and scent wrap around edges. An agent near x=0 can sense prey near x=100. This is the default mode.
- **Reflect (`boundary_mode: reflect`)**: Elastic bouncing at domain boundaries. Agents bounce off walls.

Periodic boundaries use the minimum-image convention for all distance calculations:
- Position wrapping: `x = x % Lx`
- Distance: `dx = dx - Lx * round(dx / Lx)`

### Energy Utilization Metrics

The simulation tracks how efficiently injected prey energy is harvested vs wasted by rotting:

- **E_in(t)**: Energy injected as prey each day
- **E_eat(t)**: Energy transferred from prey to agents via hunting
- **E_rot(t)**: Energy lost when prey rot (shelf-life expires)

Computed metrics:
- **eta_day**: Daily utilization = E_eat / E_in
- **eta_cum**: Cumulative utilization = Σ E_eat / Σ E_in
- **eta{W}**: Rolling window utilization over W days (default W=100)
- **rot{W}**: Rolling window rotting fraction over W days

Console output includes `eta{W}` and `rot{W}` to monitor ecosystem efficiency:
```
day=  120 | N= 187 | B=  6 D=  4 (b=0.032 d=0.021) | M= 2992 | E(mean/min/max)=42.1/10.0/133.7 | prey=  51 | eta100=0.63 rot100=0.31
```

Note: `eta{W} + rot{W}` may be < 1 because some energy remains stored in existing prey.

## New Features

### Edge-Scaled Energy Thresholds

Agent mass is automatically computed from the neural network structure as the number of edges (trainable weights). All energy quantities are defined **per edge** for scale consistency:

```yaml
# Per-edge energy parameters (mass = number of edges in neural network)
eps_survival: 0           # Energy per edge to survive
eps_birth_cost: 0.01      # Energy per edge paid for reproduction
eps_birth_energy: 0.3     # Initial energy per edge for offspring
```

**Mass Calculation**: For a network with `hidden_sizes: [8, 8]` and stencil input (12D):
- Layer 1: 12 × 8 = 96 edges
- Layer 2: 8 × 8 = 64 edges
- Output: 8 × 3 = 24 edges
- **Total mass = 184 edges**

**Motivation**: When network architecture changes, all thresholds scale automatically. No manual retuning required.

### Energy Budget Tracking

The simulation tracks detailed energy costs for ecosystem analysis:

| Column | Description |
|--------|-------------|
| `energy_cost_motion_total` | Total motion cost: Σ 0.5 * m * v² |
| `energy_cost_basal_total` | Total basal cost: Σ m * e0 |
| `energy_cost_birth_total` | Total reproduction cost: n_births * e_b |
| `energy_cost_aging_total` | Total aging cost (if enabled) |

The `energy_budget.png` plot shows E_move, E_basal, and E_birth (not E_eat/E_rot/E_age).

### Metabolism Model (Unified Basal + Aging)

A sigmoid-based model that unifies basal metabolism and aging into a single daily cost:

```
E(age, m) = m × e0 × (1 + c_age × σ(k × (age - a0(m))))
```

Where:
- `σ(x) = 1 / (1 + exp(-x))` — sigmoid function
- `a0(m) = a0_ref × (m / m_ref)^lifespan_exp` — aging onset age
- `m` = mass (number of edges in neural network)
- `e0` = basal cost per edge per day (young baseline)
- `c_age` = maximum old-age metabolic multiplier
- `lifespan_exp = 0.25` — quarter-power scaling (larger individuals age more slowly)

**Key properties:**
- Young agents: cost ≈ `m × e0` (sigmoid ≈ 0)
- Old agents: cost ≈ `m × e0 × (1 + c_age)` (sigmoid ≈ 1)
- Larger individuals have later aging onset (Kleiber's law scaling)

Configuration:
```yaml
e0: 0.0005              # Basal cost per edge per day

metabolism:
  enabled: true
  mode: sigmoid_age
  sigmoid:
    k: 0.06             # Sigmoid steepness
    a0_ref: 200         # Onset age at reference mass
    m_ref: 500          # Reference mass (edges)
    lifespan_exp: 0.25  # Quarter-power scaling
  c_age: 6.0            # Max old-age multiplier
  precompute:
    enabled: true       # Precompute lookup table for efficiency
    max_age: 2000
```

### Limited Intake Rate

Realistic time constraint on feeding - agents cannot consume prey instantly.

```yaml
hunting:
  intake_rate_per_edge: 0.2  # Max energy per edge per day
```

Daily intake limit per agent: `E_intake_max = mass * intake_rate_per_edge`

Design: With a network of 176 edges, e_prey=20, and intake_rate=0.2, a single agent can extract 176 * 0.2 = 35.2 energy/day. Selection pressure naturally favors agents that stay near prey to finish eating.

### Video: Save Last N Seconds

Reduce storage by saving only the final portion of video:

```yaml
make_video: true
save_last_seconds: 30  # Only save last 30s (0 = save all)
```

Uses a ring buffer to retain only the last N seconds of frames, then writes to MP4 at simulation end.

### Auto-Plot After Run

Automatically generate plots after simulation completes:

```yaml
postprocess:
  plot_after_run: true

plots:
  eta: true
  mass: true
  birth_death: true
  pop_energy: true
  energy_flux: true
  energy_budget: true
  age_distribution_final: true
```

Individual plots can be toggled on/off.

#### Time Series Smoothing

All time-series plots use **moving average smoothing** for visual clarity while preserving full time resolution. The default smoothing window is 10 days.

| Plot | Description |
|------|-------------|
| `eta` | Energy utilization (eta, rot, eta_cum) |
| `mass` | Total biomass over time |
| `pop_energy` | Population and mean energy |
| `birth_death` | Birth and death rates |
| `energy_budget` | Energy costs (motion, basal, birth) |
| `energy_flux` | Energy fluxes (injected, eaten, rotted) |
| `age_distribution_final` | Final age distribution (snapshot, no smoothing) |
| `rates_vs_age` | Age-dependent rates (final window, 10-day age bins) |

To adjust smoothing window via CLI:
```bash
python scripts/plot_timeseries.py --smooth_window 20
```

#### Age-Dependent Rates Plot

The `rates_vs_age.png` plot shows how birth rate k(a), death rate μ(a), and energy intake rate vary with agent age. Computed over a final window (default: last 500 days) using 10-day age bins:
- **k(a)**: births in age bin / agent-days exposure in that bin
- **μ(a)**: deaths in age bin / agent-days exposure in that bin
- **intake(a)**: total energy intake / agent-days exposure

This reveals age-dependent fitness patterns (e.g., declining reproductive rate with age).

### Scent Sensing Modes (nose)

The simulation supports two scent perception modes:

```yaml
sensing:
  scent_mode: stencil      # "vector" (5D) or "stencil" (11D: nose 9D + 2 internal)
  stencil_h: 2.0           # Sampling distance for nose stencil points
  R_eye: 16.0              # Reserved for future eyes (currently unused)
  stencil_geometry: circle # Kept for backward compatibility (ignored)
```

| Mode | Input Size | Description |
|------|------------|-------------|
| `vector`  | 5D  | Legacy mode: `[S_vec_x, S_vec_y, S_scalar, energy, age]` |
| `stencil` | 11D | Nose 9D + internal 2D: `[nose_0..8, energy, age]` |

**Stencil 模式** 提供在“自己周围一圈”的离散局部嗅觉梯度（nose），
网络可以通过比较 center vs 周围若干点来学习一阶/二阶信息（梯度、拉普拉斯）。

### Checkpointing and Resume

The simulation supports saving and loading checkpoints, allowing you to:
- Save population state at the end of a run or periodically
- Resume from a checkpoint with the same or different environment parameters
- Transfer evolved populations to harsher environments for continued selection

#### Configuration

```yaml
checkpoint:
  enabled: true                 # Enable checkpoint saving
  dir: checkpoints              # Directory for checkpoints (relative to exp2-2/)
  every_days: 500               # Save checkpoint every N days (0 = only at end)
  keep_last: 3                  # Keep only the last N checkpoints (0 = keep all)
  resume: false                 # Resume from latest checkpoint on startup
  resume_strict: true           # If true, error on incompatibility; if false, start fresh
  resume_mode: continue_population  # "continue_population" or "exact_replay"
```

#### Resume Modes

| Mode | Description |
|------|-------------|
| `continue_population` | Restore population + day, rebuild environment from NEW config. Use this to change environment parameters between runs. |
| `exact_replay` | Restore full state including prey distribution and RNG for exact reproducibility. |

#### Workflow Example: Mild → Harsh Environment

**Run 1: Mild environment (2000 days)**
```yaml
# config_mild.yaml
T_days: 2000
E_f: 1000           # Abundant food
checkpoint:
  enabled: true
  resume: false     # Fresh start
```

```bash
python run.py --config config_mild.yaml
# Checkpoint saved: checkpoints/ckpt_day_00002000.pt
```

**Run 2: Harsh environment (continue from day 2000)**
```yaml
# config_harsh.yaml
T_days: 4000        # Run until day 4000
E_f: 400            # Reduced food (harsher)
checkpoint:
  enabled: true
  resume: true      # Resume from latest checkpoint
  resume_mode: continue_population
```

```bash
python run.py --config config_harsh.yaml
# Loading checkpoint: checkpoints/ckpt_day_00002000.pt
# Resumed from checkpoint at day 2000
# (now running with E_f=400 instead of 1000)
```

#### Checkpoint Files

```
exp2-2/
└── checkpoints/
    ├── ckpt_day_00000500.pt   # Periodic checkpoint
    ├── ckpt_day_00001000.pt
    ├── ckpt_day_00002000.pt   # Final checkpoint
    └── latest.json            # Points to most recent checkpoint
```

Each checkpoint contains:
- `checkpoint_version`: Format version for compatibility
- `saved_day`: Simulation day when saved
- `timestamp`: When checkpoint was created
- `population`: All agents with their NN parameters, positions, energy, age
- `config`: Copy of the config used for that run
- (exact_replay only) `prey`: Prey positions and states
- (exact_replay only) `rng`: NumPy and PyTorch RNG states

#### Compatibility Rules

When resuming, the checkpoint must be compatible with the current config:
- **Input size**: Must match (determined by `scent_mode`: 5D for vector, 11D for stencil nose)
- **Hidden sizes**: Must match (`hidden_sizes` in config)

If incompatible and `resume_strict: true`, the simulation errors. If `resume_strict: false`, it logs a warning and starts fresh.

#### Example Logs

**Saving checkpoint:**
```
Checkpoint saved: checkpoints/ckpt_day_00002000.pt (day=2000, agents=245)
  Removed 1 old checkpoint(s)
```

**Loading checkpoint:**
```
Attempting to resume from checkpoint...
Loading checkpoint: checkpoints/ckpt_day_00002000.pt
  Restored day=2000, agents=245
  Original seed: 42, resume_mode: continue_population
Resumed from checkpoint at day 2000
```
