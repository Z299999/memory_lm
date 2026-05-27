# exp0526 Online Neural Control

`exp0526_onlineNeuralControl` keeps the self-talk controller and continuous closed-loop training workflow from the earlier `0526` line, but resets the environment to a simpler benchmark.

## Mainline scope

- Controller: feed-forward self-talk controller with carried message state
- Environment: scalar cubic unstable system
- Training: differentiable closed-loop rollouts with one optimizer update per training window
- Evaluation: `full`, `sole_eye`, `sole_speech`, `neither`, `blink(...)`, `stutter(...)`

The old age-structured PDE version is preserved on the `exp0526-age-structured` branch.

## Environment

The first mainline environment is the scalar system from `docs/idea2.tex`:

\[
\dot x = a x + b x^3 + c u
\]

discretized with Euler as

\[
x_t = x_{t-1} + dt \left(a x_{t-1} + b x_{t-1}^3 + c u_t \right).
\]

The controller sees:

- current scalar state `x`
- a constant pulse channel `1`

The control output is bounded symmetrically:

\[
u_t = u_{\max} \tanh(\text{raw}_t).
\]

For numerical stability during early random-policy exploration, the mainline implementation also applies a soft state limit with `state_limit * tanh(raw_state / state_limit)` after each Euler step. Near the origin this leaves the intended cubic dynamics essentially unchanged while preventing runaway overflow in continuous training.

## Loss

The per-step control objective is

\[
x_t^2 + \lambda_u u_t^2.
\]

Training averages this local loss over one sampled training window.

## No-language control

Set:

```yaml
model:
  language_dim: 0
```

to run the no-language control baseline. This disables message generation and message carry while keeping the same controller/training loop.

## Outputs

Each run writes:

- `metrics/history_full_controller.json`
- `metrics/summary.json`
- `metrics/eval_rollout.csv`
- `metrics/training_timeline.json`
- `plots/training_curves.png`
- `plots/training_timeline.png`
- `plots/eval_rollout_diagnostics.png`

## Design notes

- `docs/idea2.tex` is the active design document for the simplified control benchmark.
- Future environments can be added through `src/env.py`, but the first mainline version only ships `scalar_cubic`.
