# memory_lm

`memory_lm` is a research sandbox for experiments around external memory, memory compression, and training workflows for language models. The repository is not a single packaged library. It is a collection of small, mostly self-contained experiment folders plus a lightweight literature and prompt workspace.

The codebase is bilingual. Most implementation is in Python, while much of the surrounding documentation is in Chinese with some English mixed in.

## What Is In This Repo

### Experiments

- `exp0404_three_world/`
  - A world-based testbed for stateless agents with external memory.
  - A `host` controls the world, scoring, and prompting; the tested agent is re-called fresh each round and can only persist information through saved memory files.
  - The most complete experiment runner in the repo.

- `exp0406_distil/`
  - A math data generation pipeline that uses an LLM API to create step-by-step supervision data from datasets such as GSM8K and related math corpora.
  - Output is written as conversation-style JSONL for downstream training or experiments.

- `exp0408_memory_backprop/`
  - A follow-on experiment built around an age-structured memory model.
  - Focuses on the simplest discrete case where memory has a single slot and each new response replaces the previous one.
  - Combines forward interaction, host-generated feedback, optional backward training, and visualization.

- `exp0409_atten2017_implement/`
  - A from-scratch PyTorch reimplementation of the Transformer architecture (Vaswani et al., 2017).
  - Goal: faithful reproduction of every module in the original paper, verified on small data, as a foundation for later fine-tuning experiments.

- `exp0410_triangleNet/`
  - Implementation and validation of the **Triangular Memory Network (TMN)**: a DAG architecture where nodes are arranged in a triangle and signal flows along Y (left→right) and Z (vertical + diagonal) axes.
  - Benchmarked against an MLP baseline on 1D and 2D function-fitting tasks.

- `exp0414_simplexNet/`
  - Research report for **Simplex Memory Networks**: a geometric feedforward architecture that couples short- and long-term memory through a simplex structure.
  - Contains a full LaTeX report (`report_tex/report.pdf`); code to follow.

- `exp0415_memoryforMLP/`
  - Smoke test for whether an MLP can encode sequence memory purely through weight dynamics under online learning.
  - Setup: constant input `1`, alternating targets `0, 1, 0, 1, ...`, MLP `[1, 4, 4, 1]`, one SGD step per time step.
  - **Key finding**: MLP fails — standard SGD gradient always serves the current step and structurally conflicts with predicting the next alternating target. The network collapses to a fixed output regardless of learning rate.
  - Establishes the MLP baseline before replacing it with the simplex/triangle architecture.

### Shared Workspaces

- `literature/` — PDFs, TeX notes, and a lightweight bibliography index (`bibliography.jsonl`).
- `prompt/` — Reusable operational prompts (reading reviews, proof sketches, LaTeX splitting, registering papers).

---

## Repo Layout

```text
memory_lm/
├── exp0404_three_world/        # stateless-agent external-memory simulator
├── exp0406_distil/             # math data generation
├── exp0408_memory_backprop/    # age-structured memory backpropagation
├── exp0409_atten2017_implement/ # Transformer (2017) reimplementation
├── exp0410_triangleNet/        # Triangular Memory Network + MLP baseline
├── exp0414_simplexNet/         # Simplex Memory Networks (report)
├── exp0415_memoryforMLP/       # MLP online learning smoke test
├── literature/                 # papers, books, TeX drafts, bibliography index
└── prompt/                     # reusable repo prompts / command templates
```

---

## Quick Start

Each experiment runs from its own folder. There is no single root install step.

### `exp0404_three_world`

```bash
cd exp0404_three_world
python3 run.py
```

Configure `ACTION`, `RUN_ID`, `WORLD_PATH`, `ROUNDS`, `TESTED_MODEL`, `HOST_MODEL` at the top of `run.py`.
Requires an OpenAI-compatible API endpoint (`DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`).

### `exp0406_distil`

```bash
cd exp0406_distil
pip install -r requirements.txt
python3 run.py generate
```

Output: `data/generated/distillation_data.jsonl`.

### `exp0408_memory_backprop`

```bash
cd exp0408_memory_backprop
pip install -r requirements.txt
python3 run.py
```

Reads from `../exp0406_distil/data/generated/distillation_data.jsonl` by default.

### `exp0409_atten2017_implement`

```bash
cd exp0409_atten2017_implement
pip install -r requirements.txt
python3 inference.py
```

### `exp0410_triangleNet`

```bash
cd exp0410_triangleNet
pip install -r requirements.txt
python3 run.py
```

Configure via `params.yaml`. Output images saved to `runs/`.

### `exp0415_memoryforMLP`

```bash
cd exp0415_memoryforMLP
pip install -r requirements.txt
python3 run.py
```

Configure via `params.yaml`. Output images saved to `runs/`.

---

## Literature And Prompt Workspace

### `literature/`

- `literature/pdf_papers/` — one folder per registered paper, named `bXXXXX_short_name`
- `literature/pdf_books/` — books and longer references
- `literature/tex/` — drafts and source material
- `literature/bibliography.jsonl` — JSONL metadata index

### `prompt/`

| ID | Purpose |
|----|---------|
| `c000` | Write a reading review |
| `c001` | Read a proof and write a proof sketch |
| `c002` | Split a LaTeX project |
| `c003` | Register a new paper in `literature/` |

Registry: `prompt/command_list.jsonl`.

---

## Working Style Of The Repo

- Experiments are folder-local — each subproject carries its own docs, config, and runner.
- Runs write transcripts, metrics, and artifacts to `runs/` (gitignored by default via `**/runs/`).
- Documentation is exploratory — some folders are fairly complete, others are closer to research notes plus initial code.
- At least `exp0404_three_world` and `exp0406_distil` require a live LLM endpoint for real runs.

---

## Suggested Reading Order

1. This file.
2. `exp0404_three_world/README.md` — the original external-memory testbed.
3. `exp0408_memory_backprop/model.md` — memory dynamics motivation.
4. `exp0410_triangleNet/readme.md` — triangle network architecture and results.
5. `exp0414_simplexNet/report_tex/report.pdf` — Simplex Memory Networks paper.
6. `exp0415_memoryforMLP/readme.md` — MLP baseline and why it fails.

---

## Current State

| Experiment | Status |
|---|---|
| `exp0404_three_world` | Mature, operational |
| `exp0406_distil` | Usable utility project |
| `exp0408_memory_backprop` | In-progress, runnable + some forward-looking docs |
| `exp0409_atten2017_implement` | Runnable reimplementation |
| `exp0410_triangleNet` | Runnable, benchmarked against MLP |
| `exp0414_simplexNet` | Report complete; code to follow |
| `exp0415_memoryforMLP` | Smoke test complete — MLP baseline established |
