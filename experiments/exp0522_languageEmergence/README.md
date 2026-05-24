# exp0522 Language Emergence

Minimal experiment for testing whether a fixed language channel can act as an
external state register for a feed-forward MLP.

## Task

- Hidden world state follows a fixed sinusoidal phase trajectory.
- The agent never observes phase, time, or target value.
- The only ordinary input is a constant pulse `x_t = 1`.
- The full model also receives its previous language signal `m_{t-1}`.
- At every step the model emits one scalar, whose training target can be:
  - the waveform value `y_t`
  - the velocity `y_t - y_{t-1}`
  - the acceleration `(y_t - y_{t-1}) - (y_{t-1} - y_{t-2})`

No matter which training target is used, rollout/eval plots always display the
reconstructed waveform `y`.

Supported target kinds:

- `sine`
  - `sin(phi_t)`
- `mixed_sin`
  - `(sin(phi_t) + 0.5 * sin(2 * phi_t)) / 1.5` by default
  - used to test whether reset-mode training can launch a more complex periodic trajectory without mixing in the harder continuous handoff problem

Config convention:

- `task` defines the world itself, such as the cycle and ordinary input
- `train` defines how windows are used for optimization
- `eval` defines which rollout lengths and phase modes are measured

The repository currently supports two main experiment styles:

- `V0`
  - open-loop self-talk with `[1, m_{t-1}]`
- `V1`
  - error-corrected self-talk with `[1, e_{t-1}, m_{t-1}]`

## Comparison Groups

- `full`
  - trained and evaluated with the language channel enabled
  - receives `m_{t-1}` as input and produces `m_t` as output
  - tests whether a fixed language channel can support a useful external state

- `baseline`
  - trained from scratch without any language input or language output
  - answers the architecture-level question: how well can the task be solved with only the constant pulse input?

- `mute-deaf`
  - uses the already-trained `full` model
  - but at evaluation time the language channel is disabled:
    - language input is forced to zero
    - language output is also forced to zero
  - answers the causal question: does the trained full model actually rely on language at rollout time?

So the two main comparisons mean different things:

- `full` vs `baseline`
  - tests the value of having a language channel in the architecture

- `full` vs `mute-deaf`
  - tests whether the trained model is causally using that channel at inference time

For `V1`, the main comparison is now done inside one unified interface:

- corrected
  - closed-loop self-talk with `[1, e_{t-1}, m_{t-1}]`
- zero-error ablation
  - keeps the same `[1, e, m]` architecture
  - but forces `e=0` at every step

This keeps the target, optimizer, rollout length, and continuous-window protocol identical while only changing whether the correction signal is live.

## Quick Start

```bash
cd experiments/exp0522_languageEmergence
python3 run.py --config config.yaml
```

For a short smoke test:

```bash
cd experiments/exp0522_languageEmergence
python3 run.py --config config.yaml --epochs 50 --run-name smoke
```

For offline continuous-collapse diagnosis on a completed run:

```bash
cd experiments/exp0522_languageEmergence
python3 analyze_continuous_collapse.py --run-dir runs/20260522/20260522_215620_exp0522_clock_v0
```

## Training Rollout Length

Training window length is configured with `train.train_window_schedule`.

Examples:

```yaml
train:
  train_window_schedule: fixed(32)
```

```yaml
train:
  train_window_schedule: random_uniform(12,20)
```

Supported schedule forms:

- `fixed(L)`
  - always train on length `L`
- `random_uniform(a,b)`
  - sample an integer length uniformly from the closed interval `[a,b]` each epoch

This schedule works in both `reset` and `continuous_window` modes, though its
main research use is for `continuous_window`.

## Task Variants

The `mixed_sin` target sums an arbitrary list of `[freq_multiplier, amplitude]` components,
normalized by the sum of absolute amplitudes:

```yaml
task:
  target_kind: mixed_sin
  mixed_sin_components:
    - [1, 1.0]   # sin(phi_t)
    - [3, 0.5]   # sin(3*phi_t)  — coprime with the fundamental
```

Any number of components are supported. Frequency multipliers must be positive; amplitudes can
be any finite value. The default `[[1, 1.0], [2, 0.5]]` reproduces the original 1+2x waveform.

To use the original single-sine target:

```yaml
task:
  target_kind: sine
```

The model is directly supervised on the waveform value `y`.

## Model Variants

The MLP trunk activation is configurable from `model.activation`.

```yaml
model:
  trunk_dims: [32]
  activation: tanh
```

Supported values:

- `tanh` (default)
- `relu`
- `leaky_relu`

The primary model can also receive the previous prediction error as a separate input:

```yaml
model:
  use_error_input: true
```

When enabled, the primary model input becomes `[1, e_{t-1}, m_{t-1}]`.

## ResNet Skip Connections

When `use_residual: true` (default), each trunk layer applies a skip connection
when the input and output dimensions match:

```
new_hidden = activation(W @ hidden) + hidden
```

This is a no-op on the first layer (input_dim ≠ trunk_dim) and on dimension-mismatched
transitions between layers.

## Multi-Layer Language Readout

When `language_readout_all_layers: true`, the fixed sparse readout for language
messages samples from the concatenation of all hidden layers rather than just the
last layer. The readout matrix shape becomes `(sum(trunk_dims), language_dim)`.

```yaml
model:
  language_readout_all_layers: true
```

## Message Carry Mode

Controls how the previous message is transformed before entering the next step's
input head. Only applies to single-agent runs (`num_agents: 1`).

| mode | formula | parameters |
|---|---|---|
| `identity` | `m_in = m_prev` | none |
| `learnable_diagonal` | `m_in = (1 + d) ⊙ m_prev` | `(language_dim,)` |
| `learnable_matrix` | `m_in = m_prev + m_prev @ D` | `(language_dim, language_dim)` |

All modes initialize to `m_in = m_prev` at the start of training.

```yaml
model:
  message_carry_mode: learnable_matrix   # identity | learnable_diagonal | learnable_matrix
```

## Multi-Agent Reservoir

Setting `num_agents > 1` switches from `ExternalClockMLP` to `AgentPool`, a
reservoir-style architecture where N independent agents produce hidden states
that are read out by a single shared linear head.

### Architecture

At each time step:

1. **Error distribution** — the shared scalar error `e_prev` is distributed to
   each agent via a learnable weight vector `w_in ∈ R^N` (`w_in[i]` initialized
   to `1.0`). Agent `i` receives `w_in[i] * e_prev`.

2. **Language aggregation** — each agent `i` computes its language input from
   all agents' previous messages via a learnable matrix `D ∈ R^(N×N×d×d)`:

   ```
   language_input_i = activation( Σ_j  D[i,j] @ m_j_prev )
   ```

   `D[i,i]` is initialized to `I` (self-carry), `D[i,j≠i]` to `0` (no initial
   inter-agent signal). Gradients flow into every element of `D`.

3. **Independent trunks** — each agent runs its own MLP on
   `[pulse, w_in[i]*e, language_input_i]`.

4. **Reservoir readout** — a single learned linear head reads from the
   concatenation of all agents' last hidden states:

   ```
   y_t = W_out @ concat(h_1, ..., h_N)
   ```

5. **Language messages** — each agent generates its broadcast message via its
   own fixed sparse readout matrix (different seed per agent).

### Config

```yaml
model:
  num_agents: 2   # 1 = single agent; >1 = multi-agent reservoir
```

`message_carry_mode` is ignored when `num_agents > 1` (the D matrix handles
all carries).

### Outputs

`plots/` and `metrics/` use agent 0's messages for message-trace and message-norm
panels. The prediction is the reservoir output `y_t` (single scalar shared by all
agents).

## Sequence Modes

The experiment now supports two training-state modes:

- `sequence_mode: reset`
  - current v0 behavior
  - every training episode starts from `message = 0`
  - use `train_phase_mode: reset`

- `sequence_mode: continuous_window`
  - training runs on fixed-size windows of one discrete-time stream
  - the next window reuses the previous window's final message
  - the carried message is detached at the window boundary
  - use `train_phase_mode: continuous`

Example:

```yaml
train:
  sequence_mode: continuous_window
  train_window_schedule: fixed(32)
  train_phase_mode: continuous
  detach_error_input: true
  carry_error_between_windows: true
  force_zero_error_input: false

eval:
  eval_phase_mode: both
  continuous_eval_steps: 512
```

To run the zero-error ablation under the same `[1,e,m]` interface:

```yaml
train:
  force_zero_error_input: true
```

## Offline Collapse Analysis

Continuous-mode collapse can be analyzed after training with a dedicated offline pass.

Config:

```yaml
analysis:
  enable_continuous_collapse: true
  checkpoint_epochs: [1, 10, 50, 100, 500, 1000]
```

When enabled, the primary analyzed model saves milestone checkpoints during training:

- `checkpoints/full_language_epoch_0001.pt`
- `checkpoints/full_language_epoch_0010.pt`
- ...
- `checkpoints/full_language_final.pt`

The offline analyzer replays those checkpoints on one fixed `continuous_eval` stream and writes:

- `analysis/continuous_collapse/metrics.json`
- `analysis/continuous_collapse/collapse_metrics.png`
- `analysis/continuous_collapse/checkpoint_rollouts.png`
- `analysis/continuous_collapse/snapshots/*.json`

Legacy `v0_open_loop` and `v1_error_corrected` checkpoint families are still analyzable via `--model-name`, but the default workflow now only writes the unified primary family.

The four core collapse metrics are:

- `pred_std`
- `corr_pred_target`
- `message_temporal_variance`
- `mean_step_message_delta`

## Outputs

Each run writes a timestamped folder under a date subdirectory in `runs/`,
for example `runs/20260522/20260522_173059_exp0522_clock_v0/`, containing:

- `config.yaml`
- `resolved_config.json`
- `metrics/`
  - `summary.json`
  - `history_full_language.json`
  - `reset_eval_rollout.csv`
  - `reset_long_rollout.csv`
  - `continuous_eval_rollout.csv` when continuous evaluation is enabled
- `checkpoints/`
  - `full_language_epoch_0001.pt`, etc. when continuous-collapse checkpointing is enabled
  - `full_language_final.pt`
  - `no_language_final.pt` when `run.train_baseline: true`
- `plots/`
- `analysis/continuous_collapse/` after running the offline analyzer

The plots include:

- training curves
- training timeline for continuous-window runs
- one combined rollout diagnostics figure with:
  - short rollout comparison
  - long rollout comparison
  - short-rollout error curves
  - language channel traces
  - message norm

Most plotting behavior is configurable from `config.yaml`, including:

- figure size
- dpi
- how many short / long rollout steps are shown
- how many steps are shown in the error panel
- how many steps are shown in the message panels
- line widths, grid alpha, and legend column counts
- which training curves are shown
- whether to generate a training timeline figure
- how many training-timeline panels are shown
- how many global steps each training-timeline panel covers
- which rollout curves are shown
- whether the message-trace and message-norm panels are shown

For example, if you mostly want the cleanest rollout view, you can set:

```yaml
plot:
  plot_rollout_series: [target, full]
  plot_error_series: [full]
  plot_show_message_traces: false
  plot_show_message_norm: false
```

For continuous-window runs, you can also generate a training-time local timeline view:

```yaml
plot:
  plot_show_training_timeline: true
  plot_training_timeline_num_panels: 6
  plot_training_timeline_window_steps: 200
```

This writes:

- `metrics/training_timeline.json`
- `plots/training_timeline.png`

The timeline figure uses the **real predictions and targets seen during training**, not a later checkpoint replay.

The config is grouped into six top-level sections:

- `run`
- `model`
- `task`
- `train`
- `eval`
- `analysis`
- `plot`
