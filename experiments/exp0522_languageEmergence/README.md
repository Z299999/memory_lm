# exp0522 Language Emergence

Minimal experiment for testing whether a fixed language channel can act as an
external state register for a feed-forward MLP.

## Task

- Hidden world state follows a fixed sinusoidal phase trajectory.
- The agent never observes phase, time, or target value.
- The only ordinary input is a constant pulse `x_t = 1`.
- The full model also receives its previous language signal `m_{t-1}`.
- At every step the model predicts the current target `sin(phi_t)`.

The key comparison is:

- full language model
- no-language baseline
- mute-deaf evaluation of the full model

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

## Outputs

Each run writes a timestamped folder under `runs/` containing:

- `config.yaml`
- `resolved_config.json`
- `summary.json`
- `history_full_language.json`
- `history_no_language.json`
- `eval_rollout.csv`
- `long_rollout.csv`
- `checkpoints/`
- `plots/`

The plots include:

- training curves
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

The config is grouped into five top-level sections:

- `run`
- `model`
- `task`
- `train`
- `plot`
