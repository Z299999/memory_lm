# exp0211 — Eco-Evo Workbench Reference

This directory is a **reference copy**, not an actively trained experiment in this repo.

- Source repo: `Z299999.github.io`
- Imported from: `pages/research/eco-evo/demo/`
- Demo start commit: `7c30be04736d0dda9fda5a35154ce7ed753460a2`
- Demo start date: `2026-02-11 08:48:38 -0800`

It is kept here so later experiments in `memory_lm` can directly borrow:

- layout structure
- graph interaction patterns
- help overlay / popup behavior
- control-panel organization

The files below are copied largely as-is from the original demo so this directory preserves the workbench as a UI / interaction reference.

# Eco-Evo Graph Simulation Demo

Interactive browser demo of a directed weighted graph simulation with configurable activations (`tanh`, `ReLU`, `ReLU with threshold`, `Identity`, `max |w_i x_i|`), stochastic weight dynamics, and two graph-construction modes: **bridging growth** and **random growth**.

## How to Open

Open `index.html` directly in a modern browser (Chrome, Firefox, Safari, Edge). No build step or server required.

> **Note:** Some browsers block ES module imports from `file://`. If the demo doesn't load, serve it with a local server:
> ```bash
> python3 -m http.server 8000
> # then open http://localhost:8000/pages/research/eco-evo/demo/
> ```

## Layout

- **Left panel** — Parameter controls (genesis and runtime, impulse test)
- **Center** — Interactive graph (Cytoscape.js) with zoom/pan
- **Right panel** —  
  - Output signal `‖y(t)‖₂` (value + time series)  
  - Node activation value distribution histogram (all node activations)  
  - Edge weight distribution histogram  
  - Degree distribution histogram  
- **Top toolbar** — Play/Pause, Step, Reset, Speed slider, live stats

## Parameters

This section doubles as both a user manual (what each control does) and an experiment setup reference (how to reproduce a configuration).

### Genesis (applied on Reset)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `m` | 1–50 | Number of input nodes |
| `n` | 1–50 | Number of output nodes |
| Initial architecture | core9 / mlp2 | - `core9`: 9 internal nodes (z0, ..., z8) forming a fully connected directed core; each x_i connects to every z_j, and each z_j connects to every y_l.  - `mlp2`: two hidden layers, each of size `m`; inputs fully connect to hidden layer 1, which fully connects to hidden layer 2, which fully connects to outputs (feed-forward only). |
| Input source | noise / constant / sine | Input signal generator |
| Activation | tanh / ReLU / ReLU (with threshold) / Identity / max \|w_i x_i\| | Node activation nonlinearity (applied to all non-input nodes) |
| Edge weight control | vanilla / tanh-constraint / OU / Hebbian / Hebbian+tanh / Hebb-OU | - `vanilla`: drifted Brownian motion on `w` + raw `w` in forward pass; `tanh-constraint`: same drift but contributions use `tanh(w·x)` to keep inputs bounded; `OU`: Ornstein–Uhlenbeck dynamics with user mean `m`; `Hebbian`: Brownian motion with Hebbian drift; `Hebbian+tanh`: Hebbian drift + `tanh(w·x)` constraint on contributions; `Hebb-OU`: OU dynamics whose mean is an instantaneous Hebbian function of the previous-step activations |
| Construction mode | bridge / random | - `bridge`: canonical bridging growth described below; `random`: suppresses bridging and instead applies random edge/node growth with parameters in “Random growth (runtime)” |

### Runtime (applied immediately)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `μ` (mu) | -0.1–0.1 | 0.0 | Drift term in `w += σ ξ + μ sign(w)` (used for `vanilla` / `tanh-constraint`, disabled in OU / Hebbian / Hebb-OU modes) |
| `σ` (sigma) | 0–0.05 | 0.02 | Weight noise standard deviation (used in both Brownian and OU dynamics) |
| `θ` (activation threshold) | 0–1 | 0.0 | Global threshold used only when activation is set to “ReLU (with threshold)”; nodes apply `ReLU(z - θ)` to their aggregated input |
| OU mean `m` | free | 0.0 | Target mean for OU weight dynamics (only used when edge weight control is `OU`) |
| `p_flip` | 0–1 | 0.3 | Probability of small sign-flip when `|w|` is near zero |
| `T_bridge` | 0–1 | 0.8 | Activation threshold for triggering bridges |
| `ω` (omega) | 0–0.2 | 0.05 | Bridge feedback strength for edges `z1 → z0 = -ω`, `z0 → z2 = ω` |
| `ε_zero` | 0–0.01 | 0.001 | Near-zero threshold for edge deletion / flip / structural rewiring |
| `K` (cooldown) | 0–50 | 10 | Minimum steps between two bridge events on the same node |
| Hebbian rate `η_hebb` | 0–2 | 0.1 | Hebbian strength parameter used in Hebbian and Hebb-OU edge weight controls |
| Hebbian threshold `θ_hebb` | 0–1 | 0.1 | Only edges with sufficiently strong co-activation (see Hebbian / Hebb-OU rules) receive Hebbian contributions; below threshold only noise/OU act |
| Speed | 1–200 | 10 | Simulation steps per second |

### Random growth (runtime, only when construction = random)

These parameters are visible in the UI but disabled unless you select `construction = random`.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `p_add_edge` | 0–1 | 0.01 | Per-step probability of adding a weak new edge from an internal **or input** node to another node at a sampled graph distance (see below) |
| `p_add_node` | 0–1 | 0.005 | Per-step probability of inserting a new internal node along an existing edge of type internal→internal, internal→output, or input→internal |
| `α` (alpha) | 1–3 | 1.5 | Power-law exponent for the target graph distance: smaller `α` favors longer jumps, larger `α` favors local connections |
| `d_max` | 1–6 | 4 | Maximum graph-distance considered for random edge targets |

Under the hood, the “random” mode performs:

- **Random edge growth** — with probability `p_add_edge`:
  - choose a source node `src` among internal and input nodes;
  - sample a distance `d` from the discrete power-law `P(d) ∝ 1/d^α` up to `d_max`;
  - perform a BFS in the underlying undirected graph to find candidates at distance `d`;  
    - if `src` is an **input** node, only internal nodes are valid targets;  
    - if `src` is an **internal** node, internal and output nodes are valid targets;  
  - if a candidate exists and there is no existing `src → dst` edge, add a small-weight edge `src → dst`.
- **Random node growth** — with probability `p_add_node`, choose a structural edge with one of the types
  internal→internal, internal→output, or input→internal, and insert a fresh internal node along it, adding
  two small-weight edges `src → new` and `new → dst` while keeping the original edge.

### Test mode (frozen-graph experiments)

The “Test mode (frozen graph)” panel is collapsed by default. When enabled, it freezes structural evolution and lets you probe how the current graph transforms a chosen input signal.

| Control | Range | Default | Description |
|---------|-------|---------|-------------|
| Signal type | impulse / constant | impulse | `impulse`: one-step spike on the chosen input; `constant`: same amplitude applied at all steps |
| Input index `i` | 0–(m−1) | 0 | Which input node `x_i` receives the test signal |
| Amplitude `A` | ≥ 0 | 1.0 | Magnitude of the test signal |
| Test steps | ≥ 1 | 200 | Number of time steps to run in test mode |

Workflow:

1. Choose a configuration (genesis + runtime), run until the graph and weights reach a regime of interest.
2. Freeze structural evolution (e.g. hold parameters fixed, or simply pause evolution and switch to test mode).
3. Open the **Test mode** panel, choose signal type (impulse/constant), and set `i`, `A`, `Test steps`.
4. Click **Inject** to run the test and record the output norm / distributions; click **Stop** to end playback early.

## Simulation Step Order

Each call to `step()` executes in this exact order:

1. **Set input activations** — `a[xi](t+1) = x_i(t+1)` from the selected input generator (runtime) or test signal (test mode)
2. **Forward pass (one-hop update)** — For each non-input node, compute the new activation using the **previous** step’s activations:
   - For standard activations:
     - `z_i = Σ(w_ji × a_j(t))` over all incoming edges  
     - `a_i = φ(z_i)` where `φ` is:
       - `tanh`, or  
       - `ReLU(x) = max(0, x)`, or  
       - **ReLU with threshold:** `ReLU(z_i - θ) = max(0, z_i - θ)` using the global runtime threshold `θ`, or  
       - `Identity(x) = x` (fully linear graph)
   - For `max |w_i x_i|`:
     - For each incoming edge, compute the contribution `c_j = w_ji × a_j(t)`
     - Set `a_i` to the contribution with largest absolute value (preserving sign):  
       \[
       a_i = \operatorname*{arg\,max}_j |c_j|.
       \]
3. **Bridging trigger** — For internal nodes, mark those where `|a_i| > T_bridge` (with cooldown `K` steps)
4. **Bridging action** — For each triggered node `z0`, apply the bridge construction described in the paper (creating internal nodes `z1, z2, ...`, a 2-cycle, fan-in from `xi` to `z1`, duplicated outputs from `z2`, and feedback edges of size `±ω_bridge`).
5. **Weight update** — For every edge (evolution mode only):
   - If edge weight control is `vanilla` or `tanh-constraint`:  
     \[
     w \leftarrow w + \sigma\,\xi + \mu\,\operatorname{sign}(w)
     \]
   - If edge weight control is `OU`: Ornstein–Uhlenbeck update  
     \[
     w \leftarrow m + a(w - m) + b\,\xi, \quad a = e^{-\gamma},\; b = \sigma \sqrt{\frac{1-a^2}{2\gamma}}
     \]
     with fixed mean-reversion rate `γ = 0.05`, user-controlled mean `m`, and shared `σ`.
   - If edge weight control is Hebbian (`Hebbian` or `Hebbian+tanh`): Brownian motion with activation-dependent drift  
     - Let `a_pre(t), a_post(t)` be the previous-step activations of source and target nodes, and  
       \[
       p = |a_{\text{pre}}(t)\,a_{\text{post}}(t)|.
       \]
     - If \(p > \theta_{\text{hebb}}\), set a drift magnitude  
       \[
       d = \eta_{\text{hebb}}\,p
       \]
       and update  
       \[
       w \leftarrow w + d\,\operatorname{sign}(w) + \sigma\,\xi;
       \]
       otherwise, apply only the noise term `σ ξ` (no Hebbian drift).
   - If edge weight control is Hebb-OU: OU with instantaneous Hebbian mean  
     - Let `a_pre(t), a_post(t)` be the previous-step activations and set  
       \[
       p = a_{\text{pre}}(t)\,a_{\text{post}}(t).
       \]
       Define a role sign
       \[
       s_{\text{role}}(t) =
       \begin{cases}
       \operatorname{sign}(w(t)), & w(t)\neq 0,\\
       \operatorname{sign}(p), & w(t)=0,\ p\neq 0,\\
       \pm 1 \text{ (random)}, & \text{otherwise.}
       \end{cases}
       \]
     - Require both nodes to be strongly active, i.e.  
       \(|a_{\text{pre}}(t)| > \theta_{\text{hebb}}\) and \(|a_{\text{post}}(t)| > \theta_{\text{hebb}}\).  
       If this condition holds, define an instantaneous OU mean
       \[
       m(t) = \tanh\bigl(\eta_{\text{hebb}}\,p\,s_{\text{role}}(t)\bigr);
       \]
       otherwise set \(m(t) = 0\).
     - Update
       \[
       w \leftarrow m(t) + a\bigl(w - m(t)\bigr) + b\,\xi,
       \]
       with the same `a = e^{-γ}`, `b = σ sqrt((1-a^2)/(2γ))` as in the OU rule.
6. **Near-zero event and structural changes** — For edges with `|w| < ε_zero`:
   - With probability `p_flip`: draw a small magnitude `u ~ Uniform(0, ε_zero)` and set  
     `w ← -sign(w) · u` (flip sign but keep `|w|` small).
   - Otherwise:
     - For internal→internal edges: apply a structural deletion/rewiring rule that can remove edges, rewire outputs through upstream nodes, and prune internal sinks.
     - For other edges: delete the edge; if this creates internal sinks, prune them (and possibly their predecessors) recursively.
7. **Increment** — `t++`
8. **Increment** — `t++`

At each step the UI also shows:

- `‖y(t)‖₂` — the Euclidean norm of the current output vector, computed as  
  \[
  \|y(t)\|_2 = \sqrt{\sum_j a_{yj}(t)^2}.
  \]
  plus a line chart of `‖y(t)‖₂` over time (window selectable as last 50/100/500/1000/5000 steps or all steps).
- **Node activation histogram** — a live bar chart of node activation values binned over a symmetric range `[-L, L]` (with `L` automatically chosen between 1 and 10 based on current activations).
- **Edge weight histogram** — a live bar chart of edge weights binned over a symmetric range `[-L, L]` (with `L` chosen based on the current maximum absolute weight).

## File Structure

```
demo/
├── index.html          # Entry point
├── styles.css          # All styles
├── main.js             # Orchestrator (loop, buttons, wiring)
├── sim/
│   ├── graph.js        # Graph data structure (nodes, edges, adjacency)
│   ├── input.js        # Input signal generators (noise, sine)
│   └── step.js         # Simulation step logic
├── ui/
│   ├── controls.js     # Parameter panel reader
│   ├── graph-view.js   # Cytoscape.js wrapper
│   ├── distributions.js # Chart.js histograms (degree, node activation, edge weight)
│   ├── output-view.js  # Chart.js output norm time-series
│   └── stats.js        # Read-only stat counters
└── README.md
```

## Known Limitations

- **Forward pass** uses a single discrete-time recurrent update based on the previous step’s activations (not a full topological solve for graphs with cycles).
- **Layout** uses a custom three-column preset (inputs on the left, internals in a central grid that adapts to node count, outputs on the right). Large or extremely dense graphs can still produce edge crossings.
- **Performance** degrades above ~500 nodes due to repeated layout updates and DOM work. For large graphs, consider reducing speed or bridging frequency.
- **Input generators** are simple built-in functions; file upload is not supported in this version.
- **No persistence** — simulation state is lost on page refresh.
