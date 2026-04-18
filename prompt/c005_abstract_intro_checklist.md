# Abstract and Introduction Checklist

This checklist is distilled from comparing:

- the original `w00004` abstract/introduction draft, and
- Mamadou Diagne's revised style for the same parts.

It is not a theorem-proof checklist. It is a writing checklist for the
opening of a control/PDE paper, especially when presenting a new
predictor/observer design.

## Main Principle

Do not jump too quickly from the problem statement to `we design / we prove`.
First explain the mathematical structure that makes the design natural.

Prefer a structure-driven narrative:

1. problem
2. reduction / decomposition
3. resulting mathematical structure
4. design enabled by that structure
5. theorem-level conclusion

## Abstract Checklist

### 1. Start with the precise system class and delays

- State the model class precisely: nonlinear age-structured PDE, ODE-PDE system, etc.
- Name the delays explicitly: input delay, output delay, actuator delay, sensor delay.
- Avoid opening with only a vague application description.

### 2. State the key reduction before the controller

- Say what transformation or decomposition is introduced.
- Name the reduced mathematical objects explicitly.
- Prefer concrete phrases such as:
  - `the PDE is reduced to ...`
  - `the model is decomposed into ...`
  - `the dynamics are rewritten as a ... cascade`

Do not move directly from `we consider ...` to `we design ...` unless the design is already obvious.

### 3. Name the structure that enables the design

- Identify the exact structure used in the proof or control design.
- Use mathematical names rather than generic phrases.

Examples:

- `ODE-IDE decomposition`
- `PDE-ODE-IDE cascade`
- `transport representation of the actuator`
- `exponentially stable residual term`
- `backstepping target system`

Avoid overusing vague labels such as:

- `a key structural feature`
- `a suitable decomposition`
- `a cascade structure`

unless they are immediately followed by the precise structure.

### 4. Present the design as a consequence of the structure

- Write contributions in a `because ... therefore ...` style whenever possible.
- Show why the controller or observer is possible.

Prefer:

- `Since the delayed output can be written as ... , we construct ...`
- `Exploiting ... enables the design of ...`
- `This representation allows ...`

over:

- `We also design ...`
- `A key feature is ...`
- `We further show ...`

### 5. Separate the handling of different difficulties

- If the paper handles two different delays, two subsystems, or two mechanisms, indicate how each is treated.
- Make the reader see where each difficulty is resolved.

Example pattern:

- input delay is handled by predictor/backstepping
- output delay is handled by approximate prediction / residual decomposition

### 6. State the theorem-level result precisely

- Say what stability is proved.
- Say in which variables or norms.
- Say how the decay rate depends on the delays.

Prefer:

- `We prove exponential stability in the original PDE variables.`
- `The decay rate is independent of ...`
- `The result is stated with respect to the ... norm of ...`

Avoid broad claims that are not matched exactly by the theorem.

### 7. End with simulations briefly

- One short sentence is usually enough.
- Simulations should confirm the theorem, not replace the theorem.

## Introduction Checklist

## Recommended Four-Paragraph Introduction Structure

For this project line, a four-paragraph introduction is often the most
effective structure.

### Paragraph 1: background and literature positioning

This paragraph should:

1. define the model class and application context
2. place the paper in the main literature line
3. identify what the existing papers have already achieved
4. state the common limitation of those works

The paragraph should end by making the reader feel the exact gap.

### Paragraph 2: problem statement and high-level result

This paragraph should:

1. state the precise delayed problem considered in the paper
2. explain the main structural reason the problem is tractable
3. state the main theorem-level result

This is the paragraph where the paper says clearly what it does.

### Paragraph 3: technical proof narrative

This paragraph should explain the proof idea in English, without formulas,
symbols, or notation-heavy language.

Its job is not to restate the theorem. Its job is to let the reader see,
before entering the technical sections, how the proof actually works.

This paragraph should typically answer:

1. how the original system is transformed or decomposed
2. which subsystem carries the delay difficulty
3. which transformation or predictor handles the delay
4. what part of the dynamics is already stable or autonomous
5. what Lyapunov or energy argument closes the proof
6. how the argument is lifted back to the original PDE variables

In other words, the third paragraph should tell the technical story of the
proof from beginning to end, but in plain English.

### Paragraph 4: paper organization

This paragraph should remain short and factual.

### 1. First paragraph: build the literature bridge, not just a citation list

The first paragraph should do four jobs:

1. define the model class
2. place the paper in the main literature line
3. identify the common limitation of the existing works
4. explain what changes when delays are introduced

Good pattern:

- age-structured models are ...
- previous stabilization results established ...
- in those works the actuation is instantaneous
- with input/output delays, predictor feedback becomes relevant

Do not stop at a sequence of paper summaries.

### 2. Second paragraph: state the paper through the mechanism, not only through the result

The second paragraph should answer:

- what exact problem is solved?
- what specific structure makes the solution possible?
- what is the corresponding main theorem?

Prefer the order:

1. problem solved
2. structural reason the design works
3. consequence for the theorem

### 3. Name the mathematical objects early

- If the paper relies on amplitude/transverse coordinates, reduced ODEs, IDE residuals, predictor states, or transport PDEs, name them early.
- Let the reader know which object carries the control difficulty.

### 4. Be careful with novelty wording

- Novelty should be factual, not promotional.
- Every strong sentence in the introduction should be traceable to a theorem, proposition, or explicit derivation later in the paper.

Before keeping a sentence, ask:

- Is this proved later?
- Is this mathematically defined later?
- Is this the exact theorem statement, or am I overselling?

### 5. Prefer mechanism words over advertising words

Prefer:

- `reduced to`
- `rewritten as`
- `consists of`
- `decomposes into`
- `allows`
- `yields`
- `implies`

Use sparingly:

- `novel`
- `powerful`
- `remarkable`
- `significant`
- `key`

### 6. Keep the proof-preview paragraph structural

If a third paragraph previews the proof, it should say:

- what decomposition is used
- what transformation is used
- what Lyapunov functional or estimate closes the proof

It should not repeat the same contribution claims from the second paragraph.

It should instead read like a compact proof roadmap in words.

### 7. No formulas or symbols in the introduction proof narrative

For the introduction, especially the third paragraph:

- do not use displayed equations
- do not introduce proof-driving notation if it can be avoided
- do not force the reader to parse symbols before reaching the model section

The reader should understand the mechanism of the proof from English prose
alone.

Good introduction proof-preview language:

- `The state is decomposed into an amplitude component and a transverse shape component.`
- `The actuator delay is represented as a transport equation and compensated through predictor backstepping.`
- `The remaining shape dynamics evolve autonomously and decay exponentially.`
- `A composite Lyapunov argument then yields exponential stability in the original variables.`

Avoid introduction sentences that already look like theorem proof lines.

## Sentence-Level Style Checklist

### Prefer these habits

- Use concrete mathematical nouns.
- Make the logic visible with `because`, `since`, `thus`, `therefore`, `which yields`.
- Let the structure explain the method.
- Let the method explain the theorem.

### Avoid these habits

- announcing results before giving the structural reason
- stacking multiple contribution verbs in a row
- using generic phrases without naming the actual mathematical object
- claiming observer-based/output-feedback structure unless the observer dynamics are genuinely present

## Quick Self-Check Before Finalizing

For the abstract:

- Does the reader see the reduction before the controller?
- Does the reader see why the design is possible?
- Are the two main technical difficulties separated clearly?
- Is the theorem-level claim matched exactly by the actual theorem?

For the introduction:

- Does paragraph 1 identify the literature gap precisely?
- Does paragraph 2 explain the paper through structure, not only result?
- Does paragraph 3 tell the proof story clearly in English without formulas?
- Are the main mathematical objects named early enough?
- Could any sentence be accused of overclaiming?

## Short Template

### Abstract template

1. We consider ...
2. By defining / decomposing / rewriting ..., the system is reduced to ...
3. Exploiting ..., we design ...
4. Since / because ..., we construct ...
5. We prove ...
6. Numerical simulations ...

### Introduction template

Paragraph 1:

- model class
- main literature line
- existing limitation
- why delays change the problem

Paragraph 2:

- exact problem solved
- structural mechanism enabling the design
- theorem-level claim

Paragraph 3:

- decomposition of the system
- delay-handling mechanism
- autonomous/stable residual dynamics
- Lyapunov or energy closure
- return to original variables

Paragraph 4:

- paper organization only
