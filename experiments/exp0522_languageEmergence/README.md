# exp0522 Language Emergence

Single-agent experiment for testing whether a fixed language channel can act as
an external state register for a feed-forward MLP under continuous rollout.

## Task

- Hidden world state follows a fixed sinusoidal phase trajectory.
- The agent never observes phase, time, or target value.
- The only ordinary input is a constant pulse `x_t = 1`.
- The full model also receives its previous language signal `m_{t-1}`.
- At every step the model emits one scalar, trained directly against the
  configured prediction target.

Supported target kinds:

- `sine`
  - `sin(phi_t)`
- `mixed_sin`
  - normalized sum of any configured `[freq_multiplier, amplitude]` components
  - the current single-agent mainline config uses a richer multi-frequency mixture to stress continuous tracking rather than only the original `1x + 2x` waveform

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

The mainline single-agent workflow now uses one trained model plus evaluation
conditions and control configs, rather than the old baseline/comparator
training path.

- `full`
  - normal rollout with the language channel and error correction both live

- `sole_eye`
  - language channel disabled, error channel kept live
  - tests how much the model can do with correction alone

- `sole_speech`
  - error channel forced to zero, language channel kept live
  - tests how much the model can do with self-generated message alone

- `neither`
  - both language and error channels disabled
  - tests what remains with only the constant pulse input

- `blink(start,end)` / `stutter(start,end)`
  - temporary loss of error or language during rollout
  - test whether the learned closed loop can recover after a transient interruption

- `dim(start,ramp_end,end,pct)`
  - temporary weakening of the error channel during rollout with a gradual ramp
  - error gain linearly decreases from 1.0 at step `start` to `pct/100` by step `ramp_end`, then stays at `pct/100` until step `end`, then recovers to 1.0
  - for example, `dim(40,140,200,5)` ramps down over 100 steps to 5%, holds at 5% until step 200, then recovers
  - useful for testing whether a gradually weakened error cue is enough for the language/message loop to complete the trajectory
  - the prediction line is colored by gain level (blue = full, teal = dim) in the diagnostics plot

Architecture-level comparison against a model with no language channel is now
done by running a separate config with `language_dim: 0`.

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
- `event_triggered(threshold,min_steps,max_steps)`
  - roll forward step-by-step, accumulate train-space squared error, and end the window immediately once the cumulative total crosses `threshold`
  - the threshold is only checked after `min_steps`
  - if it never triggers, the window is forced to stop at `max_steps`

This schedule works in both `reset` and `continuous_window` modes, though its
main research use is for `continuous_window`. The default mainline config still
uses `random_uniform(...)`; `event_triggered(...)` is an optional training
protocol rather than the new default.

## Training Error Degradation

Training-time prediction-error degradation is configured with `train.error_degrade`.
It is independent of eval-only conditions such as `dim(start,end,pct)`.

```yaml
train:
  error_degrade: none
```

```yaml
train:
  error_degrade: dim(0.3,20,80,5,10)
```

```yaml
train:
  error_degrade: tail_dim(200,2000,5)
```

`dim(rate,min_steps,max_steps,pct,ramp_steps)` runs an online dim-event process
over global training time. In the example above, roughly `30%` of training steps
will fall inside an error-degradation event. Each event lasts `20..80` total
steps, including the linear ramp down and ramp back up. At the darkest point,
the fed-back prediction error is scaled to `5%`, so `e_t` becomes `0.05 * e_t`.
Setting `ramp_steps` to `0` gives a hard transition.

The degradation only weakens the error cue fed back into the next step. Targets,
losses, message carry, and eval conditions are unchanged. The intent is to reduce
over-reliance on the error input and encourage the message/internal dynamics to
support weak-cue trajectory completion.

`tail_dim(start_step,end_step,min_pct)` is a local-window curriculum. Within each
training window, the error gain stays at `1.0` before `start_step`, then linearly
falls to `min_pct / 100` by `end_step`, and remains there afterward. For example,
`tail_dim(200,2000,5)` starts dimming only after a window survives 200 local
steps and reaches a 5% error cue at local step 2000.

## Tail-Loss Window

By default, the training loss is the mean squared error over the entire rollout window.
When `tail_dim` is active, the early part of each window still has a live error cue,
so the model can "cheat" there. Setting `train_loss_tail_steps` focuses the gradient
signal on the harder, error-dimmed tail of the window.

```yaml
train:
  train_loss_tail_steps: null   # null = all steps (default)
```

```yaml
train:
  train_loss_tail_steps: 100    # only the last 100 steps contribute to the MSE loss
```

When `effective_steps <= train_loss_tail_steps`, the full window is used automatically.
BPTT still runs through the entire window, so parameters that processed early steps
still receive gradient via the message state chain. The window termination criterion
(`event_triggered`) is not affected by this setting.

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
be any finite value. A simple `[[1, 1.0], [2, 0.5]]` setting reproduces the original 1+2x waveform.

To use the original single-sine target:

```yaml
task:
  target_kind: sine
```

By default, the model is directly supervised on the waveform value `y`.

Optional prediction-target modes can change the supervised space without
changing the displayed eval plots:

```yaml
task:
  prediction_target: y          # y | v | a
```

- `y`
  - direct waveform supervision
- `v`
  - supervise `y_t - y_{t-1}`
- `a`
  - supervise `(y_t - y_{t-1}) - (y_{t-1} - y_{t-2})`

Legacy config values `velocity` and `acceleration` are still accepted as aliases
for `v` and `a`, respectively.

The loss space is controlled separately by `train_loss_space` (see below). Eval
plots, summaries, and rollout CSVs always report predictions in reconstructed `y`
space so runs remain visually comparable across target modes.

## Loss Space

`prediction_target` sets what the network outputs (`y`, `v`, or `a`), but the
quantity we actually care about is the reconstructed position `y`. With
`prediction_target: a` the network can fit `a` very well while the
double-integrated `y` drifts badly — small per-step `a` errors accumulate into
large position error. `train_loss_space` decides which space the loss is measured in:

```yaml
train:
  train_loss_space: y           # raw | y  (default: y)
```

- `y`
  - loss is `MSE(ŷ_t, y_t)` on the reconstructed position. Gradients flow back
    through the (fully differentiable) integration chain to the network's `a`/`v`
    output, so the model is optimized for the outcome, not the intermediate space.
- `raw`
  - loss is `MSE(raw_output, target)` in the configured `prediction_target` space
    (the previous behavior).

When `prediction_target: y` the two modes are identical (`ŷ == raw_output`); they
diverge only for `v`/`a`. The validation curve (`full_val`) is
reported in the matching space so train/val stay comparable.

`train_loss_space` also controls the **`event_triggered` window-termination space**:
the cumulative SSE that ends a window is measured in `y` space when
`train_loss_space: y`. Note that `y`-space errors are typically much larger in
magnitude than `a`-space errors, so an `event_triggered(threshold,...)`
threshold tuned for `raw` will trigger far sooner (shorter windows) under `y` — you
will likely need to **raise the threshold** when switching to `y`.

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

## No-Language Control

Set `language_dim: 0` to run the explicit no-language single-agent control.

- no language messages are generated
- no language state is carried between steps
- the model still keeps the constant pulse `1`
- the model can still use `e_{t-1}` if `use_error_input: true`

This is a single-run control setting, not an automatic paired comparison. If
you want to compare language vs no-language, run two separate configs.

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

## Trainable Language Readout

By default the readout matrix R is a fixed `register_buffer` and receives no gradients.
Setting `language_readout_trainable: true` promotes R to an `nn.Parameter` so that each
message head can learn which layer to read from.

```yaml
model:
  language_readout_trainable: true   # R becomes nn.Parameter; initialized from the fixed sparse matrix
```

The most useful combination is `language_readout_all_layers: true` (readout shape
`(sum(trunk_dims), language_dim)`). With three trunk layers of width 96 the rows partition
as layer-1: 0–95, layer-2: 96–191, layer-3: 192–287. Gradient pressure will push fast-timescale
heads toward layer-1 rows and slow-timescale heads toward layer-3 rows.

To prevent message dimensions from silently collapsing to zero use the column-norm penalty:

```yaml
train:
  language_readout_norm_penalty: 0.01  # penalizes (||col_j||_2 - 1)^2 for each message head j
```

**Important:** this feature is only meaningful when the language channel faces genuine training-time
pressure — i.e. error feedback is off or strongly dimmed (`force_zero_error_input`, `tail_dim`, etc.).
With a live error input the error cheat path eliminates gradient pressure on the language channel and
the learned readout will be near-arbitrary.

## Message Carry Mode

Controls how the previous message is transformed before entering the next step's
input head. This is only meaningful when `language_dim > 0`.

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

Multi-agent reservoir work is preserved on the dedicated `exp0522-multiagent`
branch and is intentionally not part of the `main` single-agent research line.

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
- `plots/`
- `analysis/continuous_collapse/` after running the offline analyzer

The plots include:

- training curves
- training timeline for continuous-window runs
- one combined rollout diagnostics figure with:
  - top rollout panels chosen by `plot_rollout_top_mode`
  - auxiliary error/message panels chosen by `plot_aux_horizon`

Most plotting behavior is configurable from `config.yaml`, including:

- figure size
- dpi
- how many short / long rollout steps are shown
- how many steps are shown in short-horizon error panels
- how many steps are shown in short-horizon message panels
- line widths, grid alpha, and legend column counts
- which training curves are shown
- whether to generate a training timeline figure
- how many training-timeline panels are shown
- how many global steps each training-timeline panel covers
- whether top rollout panels match training mode, match eval mode, or show all available families
- whether auxiliary panels use short, long, or both horizons
- whether the message-trace and message-norm panels are shown

For example, if you want the default continuous mainline view to stay focused, you can set:

```yaml
plot:
  plot_rollout_top_mode: match_train
  plot_aux_horizon: long
```

For continuous-window runs, you can also generate a training-time local timeline view:

```yaml
plot:
  plot_show_training_timeline: true
  plot_training_timeline_num_panels: 30
  plot_training_timeline_window_steps: 200
```

This writes:

- `metrics/training_timeline.json`
- `plots/training_timeline.png`

The timeline figure uses the **real predictions and targets seen during training**, not a later checkpoint replay.

By default each panel autoscales its y-axis independently. Two options allow a
uniform scale across all panels for easier visual comparison:

```yaml
plot:
  plot_training_timeline_shared_ylim: true   # auto: compute global min/max across all panels
  plot_training_timeline_ylim: null          # null = use shared_ylim logic
```

```yaml
plot:
  plot_training_timeline_ylim: [-2.0, 2.0]  # pin a fixed range; overrides shared_ylim
```

`plot_training_timeline_ylim` takes precedence over `plot_training_timeline_shared_ylim` when set.

The config is grouped into six top-level sections:

- `run`
- `model`
- `task`
- `train`
- `eval`
- `analysis`
- `plot`
