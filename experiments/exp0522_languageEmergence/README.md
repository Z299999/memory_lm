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
- in-distribution rollout comparison
- long rollout comparison
- language channel traces and message norm
