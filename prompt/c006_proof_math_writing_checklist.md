# Proof and Mathematical Writing Checklist

This checklist records writing preferences learned from the `w00003`
revision, especially for theorem statements, proof exposition, and
technical mathematical language.

It is intended for writing and revising:

- main results sections
- theorem statements
- technical proof sections
- proof sketches and mathematical explanations

## Main Principle

Write proofs so that the mathematical action is visible immediately.

Prefer:

1. explicit objects
2. explicit intervals
3. explicit inequalities
4. one estimate per step
5. final assembly only at the end

Avoid hiding the proof behind abstract wording.

## Source Discipline

### 1. Separate facts from derivations

Always distinguish clearly between:

- facts already stated in the manuscript or literature
- explanations or conclusions derived from those facts

Do not blur sourced statements with informal interpretation.

Useful labels in discussion and revision notes:

- `manuscript/literature fact`
- `derived from the current definitions`
- `based on the current proof structure`

### 2. Keep claims exactly aligned with the mathematics

- If the theorem is state feedback, call it state feedback.
- Do not describe a proxy variable as an observer unless an actual
  observer/error-system argument is present.
- Do not use stronger words in prose than what the theorem or proof supports.

## Main Results Writing

### 1. Present predictor designs structurally

For predictor sections, especially after Krstic-style designs, present the
logic in this order:

1. reduced delayed subsystem
2. nominal delay-free feedback
3. target delayed relation
4. implementable predictor law
5. resulting closed-loop dynamics

The controller should look motivated, not guessed.

### 2. Use the paper's own language, not copied textbook language

- Follow the structure of the literature when needed.
- But explain the design in the variables and logic of the current paper.
- Avoid dropping in a formula without first stating what nominal relation
  it is meant to enforce.

## Proof Organization Checklist

### 1. Use step titles that state the mathematical action directly

Prefer step titles such as:

- `Bound V(t) by ...`
- `Bound the left-hand side by V for t\ge\tau`
- `Bound V(\tau) by the right-hand side`
- `Bound u on [t-\tau,t] for t\in[0,\tau]`
- `Combine the cases t<\tau and t\ge\tau`

Avoid abstract titles such as:

- `Recover the physical variables`
- `Transient analysis`
- `Shifted estimate`
- `Uniform bound for the neutral coordinate`

unless the mathematical object is stated immediately.

### 2. One step, one job

Each proof step should establish one self-contained estimate.

Typical pattern:

1. define what is being bounded
2. carry out the estimate
3. stop

Do not partially combine several steps before the final assembly.

### 3. Assemble only at the end

If the proof naturally breaks into:

- decay estimate
- norm equivalence
- shifted initial bound
- initial-window bound

then each should appear in its own step, and the full theorem estimate
should be assembled only in the final step.

## Displayed Mathematics Checklist

### 1. Prefer continuous `aligned` chains

Within a proof step, prefer a continuous `aligned` inequality chain when the
argument is a direct sequence of equalities/inequalities.

Use `aligned` especially when:

- differentiating a Lyapunov functional
- carrying out a norm estimate
- converting one bound into another
- assembling a final inequality

### 2. Do not over-split one chain into many displays

If several lines belong to the same uninterrupted argument, keep them in one
`aligned` block rather than breaking them into many short displays.

### 3. But do not over-pack unrelated estimates

If two lines estimate different objects with different purposes, separate them
into different blocks or different proof steps.

Rule of thumb:

- one mathematical goal -> one `aligned`
- different mathematical goals -> different displays or steps

## Language Checklist

### 1. Prefer concrete mathematical language

Prefer concrete variables, intervals, and objects over abstract wording.

Prefer:

- `bound \eta on [0,\tau]`
- `bound u on [t-\tau,t]`
- `bound the theorem left-hand side`
- `bound \Omega_0 by the initial data`

over:

- `control the transient`
- `recover the physical variables`
- `use the shifted estimate`
- `handle the neutral mode`

unless the more abstract phrase is genuinely clearer.

### 2. Use interval information explicitly

When the proof depends on time ranges, write them directly:

- `for t\in[0,\tau]`
- `for t\ge\tau`
- `on [t-\tau,t]`
- `on [-\tau,0]`

This is often clearer than prose such as `initial interval`, `shifted
window`, or `transient regime`.

### 3. Avoid undefined quantities

In delayed systems:

- do not introduce negative-time state quantities unless they are actually defined
- use explicit input history on `[-\tau,0]` when that is what the model provides
- if a state exists only for `t\ge0`, do not write window norms over negative times for that state

## Technical Narrative Checklist

### 1. Make the role of each subsection explicit

Each technical subsection should answer a clear question.

Examples:

- backstepping subsection: what analytical obstacle does this transformation remove?
- shape-variable subsection: what part of the full-state estimate does it control?
- log-density subsection: how do we return from reduced variables to the original PDE variables?

### 2. Explain why a tool appears

When introducing a technical object such as:

- a backstepping variable
- a Lyapunov functional
- a decay functional like `G_1`

state explicitly what problem it solves in the proof.

## Final Self-Check

Before finalizing a proof section, ask:

- Does each step title tell the reader exactly what is being bounded?
- Does each step perform one independent estimate?
- Are the main inequality chains written in `aligned` form when appropriate?
- Are abstract English labels replaced by concrete mathematical language where possible?
- Are all time intervals and delayed windows stated explicitly?
- Are any negative-time state quantities used without definition?
- Is the final theorem estimate assembled only after the component bounds are complete?
