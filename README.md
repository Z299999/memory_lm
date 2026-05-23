# memory_lm

`memory_lm` is a research workspace, not a single packaged library.

The repo collects a set of small experiment folders around:

- external memory and stateless-agent workflows
- geometric / feed-forward memory architectures
- reinforcement learning backbones and control tasks
- analysis notes, literature, prompts, and writing drafts

Most code is Python. Documentation is mixed Chinese and English.

## How To Read This Repo

The repo is best treated as a **lab notebook with runnable subprojects**.

- Each experiment lives in its own folder under `experiments/`
- Most experiments have their own `README.md` and `run.py`
- There is no single root install step
- Output usually goes to a folder-local `runs/`

If you only want one thing to start from:

- for external-memory / language-channel work: `experiments/exp0522_languageEmergence/`
- for the original stateless-agent memory sandbox: `experiments/exp0404_three_world/`
- for simplex memory architectures: `experiments/exp0414_simplexNet/`

## Top-Level Layout

```text
memory_lm/
├── experiments/   # self-contained research subprojects
├── literature/    # papers, books, bib index, notes
├── prompt/        # reusable prompts / command templates
├── writing/       # drafts and longer writeups
└── TOOLS/         # shared utilities and imported tool code
```

## Experiment Map

### External Memory / Language / Training Dynamics

| Path | Theme | Entrypoint |
|---|---|---|
| `experiments/exp0202_ecoevolearning/` | eco-evolutionary hunting simulation with NN-controlled agents | `run.py` |
| `experiments/exp0404_three_world/` | stateless tested agent + host-controlled world + external memory files | `run.py` |
| `experiments/exp0406_distil/` | math supervision / distillation data generation with LLM APIs | `run.py` |
| `experiments/exp0408_memory_backprop/` | baby-case memory backpropagation over external memory updates | `run.py` |
| `experiments/exp0415_memoryforMLP/` | online-learning smoke test for whether an MLP can hold sequence memory in weights | `run.py` |
| `experiments/exp0522_languageEmergence/` | fixed language channel as external state register; reset vs continuous-window; error-corrected self-talk | `run.py` |

### Geometric / Architecture Experiments

| Path | Theme | Entrypoint |
|---|---|---|
| `experiments/exp0409_atten2017_implement/` | from-scratch Transformer reimplementation | `inference.py` |
| `experiments/exp0410_triangleNet/` | Triangular Memory Network vs MLP | `run.py` |
| `experiments/exp0414_simplexNet/` | Simplex Memory Networks (SMN), including MIMO variants | `run.py` |
| `experiments/exp0513_sigmaAlgAddrDopamine/` | self-modulated networks with hidden dopamine nodes | `run.py` |

### RL / Control

| Path | Theme | Entrypoint |
|---|---|---|
| `experiments/exp0420_RLforSMN/` | RL framework for Simplex Memory Networks | `run.py` |
| `experiments/exp0422_DQN_CartPole/` | minimal DQN on custom CartPole | docs only / no current runner at root |
| `experiments/exp0430_PPO_CartPole/` | PPO on custom CartPole | docs only / no current runner at root |
| `experiments/exp0501_PPO_LunarLander/` | PPO on customized LunarLander curriculum | docs-led subproject |
| `experiments/exp0506_PPO_SimplexLunarLander/` | SMN backbone variant of the LunarLander PPO setup | docs-led subproject |

### Reference / Analysis

| Path | Theme | Entrypoint |
|---|---|---|
| `experiments/exp0211_ecoevoWorkbenchReference/` | reference UI/workbench copy; not a main runnable experiment | reference only |
| `experiments/exp0418_brainSimplex/` | simplex analysis of Drosophila central-brain connectome | script-based pipeline |

## Recommended Entry Points

### 1. `exp0522_languageEmergence`

This is currently the clearest small-scale experiment for:

- external state via a fixed language channel
- continuous-window training
- collapse analysis
- corrected vs zero-error self-talk

Run:

```bash
cd experiments/exp0522_languageEmergence
python3 run.py --config config.yaml
```

### 2. `exp0404_three_world`

This is the original external-memory sandbox:

- the tested model is stateless across rounds
- the host owns the world, scoring, and pedagogy
- persistent state is only what gets written to external memory files

Run:

```bash
cd experiments/exp0404_three_world
python3 run.py
```

### 3. `exp0414_simplexNet`

This is the most complete geometric architecture implementation in the repo.

Run:

```bash
cd experiments/exp0414_simplexNet
python3 run.py
```

## Shared Workspaces

### `literature/`

- `pdf_papers/` — paper folders
- `pdf_books/` — books and long references
- `tex/` — TeX notes and drafts
- `bibliography.jsonl` — lightweight bibliography index

### `prompt/`

Reusable operational prompts, for example:

- reading review
- proof sketching
- LaTeX splitting
- literature registration

### `writing/`

Longer theory notes, drafts, and paper-style writeups.

## Repo Conventions

- Experiments are folder-local
- Config usually lives in `config.yaml` or `params.yaml`
- Runs usually write to `runs/<date>/<timestamp>_<name>/`
- Many folders are research-grade and intentionally exploratory
- Some folders are active code; some are preserved references or partial experiments

## Suggested Reading Order

If you want the main line of ideas in this repo, read:

1. this file
2. `experiments/exp0404_three_world/README.md`
3. `experiments/exp0415_memoryforMLP/readme.md`
4. `experiments/exp0410_triangleNet/readme.md`
5. `experiments/exp0414_simplexNet/README.md`
6. `experiments/exp0522_languageEmergence/README.md`
7. `experiments/exp0513_sigmaAlgAddrDopamine/README.md`

## Current Emphasis

The most actively iterated experiment in the current repo state is:

- `experiments/exp0522_languageEmergence/`

It currently includes:

- reset vs continuous-window training
- mixed-sine target generation
- error-corrected self-talk (`[1, e, m]`)
- zero-error ablations under the same interface
- offline continuous-collapse diagnosis
- training timeline visualizations

For the exact current behavior, always trust the local README inside the experiment folder over this root overview.
