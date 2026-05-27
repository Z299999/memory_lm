# exp0526 Online Neural Control

`exp0526_onlineNeuralControl` keeps the self-talk controller and continuous closed-loop training workflow from the earlier `0526` line, but resets the environment to a simpler benchmark.

## Mainline scope

- Controller: feed-forward self-talk controller with carried message state
- Environments:
  - scalar control-affine baseline
  - planar 2D double-well benchmark
- Training: differentiable closed-loop rollouts with one optimizer update per training window
- Evaluation: `full`, `sole_eye`, `sole_speech`, `neither`, `blink(...)`, `stutter(...)`

The old age-structured PDE version is preserved on the `exp0526-age-structured` branch.

## Environments

The scalar baseline is a control-affine system from `docs/idea2.tex`:

\[
\dot x = f(x) + g(x)u
\]

discretized with Euler as

\[
x_t = x_{t-1} + dt \left(f(x_{t-1}) + g(x_{t-1})u_t \right).
\]

By default the config uses:

\[
f(x) = -x + x^3,\qquad g(x)=1,
\]

but both expressions are user-configurable in `config.yaml`.

The scalar controller sees:

- current scalar state `x`
- a constant pulse channel `1`

The new 2D environment is a fixed planar double-well family:

\[
\dot x_1 = x_2,\qquad
\dot x_2 = \alpha x_1 - \beta x_1^3 - \gamma x_2 + c u.
\]

With the default parameters `alpha = beta = 1`, `gamma = 0.4`, and `control_gain = 1`, the uncontrolled system has three equilibria:

- `(-1, 0)` stable
- `(0, 0)` unstable
- `(1, 0)` stable

The 2D controller sees the full state plus pulse:

- `x1`
- `x2`
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
- `env_kind: scalar_control_affine` lets you specify `f_expr` and `g_expr` directly in the config using a restricted math-expression syntax over `x`.
- `env_kind: scalar_cubic` is still available as a compatibility wrapper around the older `a x + b x^3 + c u` parameterization.
- `env_kind: planar_double_well` adds a fixed 2D benchmark with phase-space geometry and a phase portrait panel in `eval_rollout_diagnostics.png`.
- `docs/phaseportrait.tex` explains why the 2D system is kept alongside the 1D baseline and why the phase portrait is shown only in eval plots.
