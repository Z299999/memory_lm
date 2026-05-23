# exp0522 Language Emergence

Minimal experiment for testing whether a fixed language channel can act as an
external state register for a feed-forward MLP.

## Task

- Hidden world state follows a fixed sinusoidal phase trajectory.
- The agent never observes phase, time, or target value.
- The only ordinary input is a constant pulse `x_t = 1`.
- The full model also receives its previous language signal `m_{t-1}`.
- At every step the model predicts the current target waveform.

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

Training always uses a fixed rollout length set by `train.fixed_train_steps`.

Example: always train on 32-step rollouts

```yaml
train:
  fixed_train_steps: 32
```

Example: always train on full 128-step rollouts

```yaml
train:
  fixed_train_steps: 128
```

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
run:
  train_baseline: false
  eval_mute_deaf: false

train:
  sequence_mode: continuous_window
  fixed_train_steps: 32
  train_phase_mode: continuous
  detach_error_input: true
  carry_error_between_windows: true
  force_zero_error_input: false

eval:
  eval_phase_mode: both
  continuous_eval_steps: 512
```

`run.train_baseline` controls whether the no-language baseline is trained at all.
`run.eval_mute_deaf` controls whether the trained full model is also evaluated
with the language channel forcibly disabled.

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
  - `history_no_language.json` when `run.train_baseline: true`
  - `eval_rollout.csv`
  - `long_rollout.csv`
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
