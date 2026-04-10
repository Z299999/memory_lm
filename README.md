# memory_lm

`memory_lm` is a research sandbox for experiments around external memory, memory compression, and training workflows for language models. The repository is not a single packaged library. It is a collection of small, mostly self-contained experiment folders plus a lightweight literature and prompt workspace.

The codebase is bilingual. Most implementation is in Python, while much of the surrounding documentation is in Chinese with some English mixed in.

## What Is In This Repo

There are four main areas:

- `exp0404_three_world/`
  - A world-based testbed for stateless agents with external memory.
  - A `host` controls the world, scoring, and prompting; the `tested agent` is re-called fresh each round and can only persist information through saved memory files.
  - This is the most complete experiment runner in the repo.

- `exp0406_distil/`
  - A math data generation pipeline that uses an LLM API to create step-by-step supervision data from datasets such as GSM8K, MATH-adjacent sources, and other math corpora.
  - Output is written as conversation-style JSONL for downstream training or experiments.

- `exp0408_memory_backprop/`
  - A follow-on experiment built around an age-structured memory model.
  - Current implementation focuses on the "baby case" where memory has a single slot and each new response replaces the previous one.
  - It combines forward interaction, host-generated feedback, optional backward training, and visualization.

- `literature/` and `prompt/`
  - `literature/` stores PDFs, TeX notes, and a lightweight bibliography index.
  - `prompt/` stores reusable operational prompts such as adding a paper, writing a reading review, or splitting LaTeX.

`exp0410_triangleNet/` currently contains only a minimal placeholder note for a future architecture idea.

## Repo Layout

```text
memory_lm/
├── exp0404_three_world/        # stateless-agent external-memory simulator
├── exp0406_distil/             # math data generation
├── exp0408_memory_backprop/    # memory backpropagation experiment
├── exp0410_triangleNet/        # early placeholder
├── literature/                 # papers, books, TeX drafts, bibliography index
└── prompt/                     # reusable repo prompts / command templates
```

## Quick Start

This repo has no single root install step. Each experiment is intended to be run from its own folder.

### 1. `exp0404_three_world`

Purpose:
- Evaluate a stateless tested agent that must rely on explicitly saved external memory across rounds.
- Explore how memory evolves from raw episodic notes into compressed rules.

Important files:
- [exp0404_three_world/run.py](/Users/shzhang/Documents/Github/memory_lm/exp0404_three_world/run.py)
- [exp0404_three_world/scripts/run_cli.py](/Users/shzhang/Documents/Github/memory_lm/exp0404_three_world/scripts/run_cli.py)
- [exp0404_three_world/scripts/run_experiment.py](/Users/shzhang/Documents/Github/memory_lm/exp0404_three_world/scripts/run_experiment.py)
- [exp0404_three_world/world/README.md](/Users/shzhang/Documents/Github/memory_lm/exp0404_three_world/world/README.md)

Typical usage:

```bash
cd exp0404_three_world
python3 run.py
```

The default workflow is to edit the constants at the top of `run.py`, especially:
- `ACTION`
- `RUN_ID`
- `WORLD_PATH`
- `ROUNDS`
- `TESTED_MODEL`
- `HOST_MODEL`

The runner writes outputs under `exp0404_three_world/runs/<run_id>/`, including transcripts, memory snapshots, host artifacts, and metrics.

Environment:
- The docs assume an OpenAI-compatible API endpoint.
- Existing docs reference `DASHSCOPE_API_KEY` and `DASHSCOPE_BASE_URL`.
- Some environments may also need `SSL_CERT_FILE` set explicitly.

### 2. `exp0406_distil`

Purpose:
- Generate math reasoning data by calling a teacher LLM over selected datasets.

Important files:
- [exp0406_distil/run.py](/Users/shzhang/Documents/Github/memory_lm/exp0406_distil/run.py)
- [exp0406_distil/config.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0406_distil/config.yaml)
- [exp0406_distil/scripts/generate_data.py](/Users/shzhang/Documents/Github/memory_lm/exp0406_distil/scripts/generate_data.py)
- [exp0406_distil/scripts/generate_parallel.py](/Users/shzhang/Documents/Github/memory_lm/exp0406_distil/scripts/generate_parallel.py)

Install and run:

```bash
cd exp0406_distil
pip install -r requirements.txt
python3 run.py debug
python3 run.py generate
python3 run.py generate-parallel --workers 32
```

Output:
- Generated data is written to `exp0406_distil/data/generated/distillation_data.jsonl`.

Notes:
- This project depends on remote API access for real generation.
- The configured teacher model and datasets live in `config.yaml`.

### 3. `exp0408_memory_backprop`

Purpose:
- Study a memory update process inspired by age-structured dynamics.
- Current implementation centers on the simplest discrete case where the latest response becomes the next memory state.

Important files:
- [exp0408_memory_backprop/run.py](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/run.py)
- [exp0408_memory_backprop/config.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/config.yaml)
- [exp0408_memory_backprop/model.md](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/model.md)
- [exp0408_memory_backprop/scripts/forward.py](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/scripts/forward.py)
- [exp0408_memory_backprop/scripts/backward.py](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/scripts/backward.py)
- [exp0408_memory_backprop/scripts/visualize.py](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/scripts/visualize.py)

Install and run:

```bash
cd exp0408_memory_backprop
pip install -r requirements.txt
python3 run.py
```

Data dependency:
- By default, this experiment reads from `../exp0406_distil/data/generated/distillation_data.jsonl`.

Outputs:
- Runs are written under `exp0408_memory_backprop/runs/`.
- The experiment supports forward generation now, with configuration knobs for backward passes and visualization cadence.

Notes:
- This folder mixes implemented code with research-planning documentation. Read the code and `config.yaml` before assuming every item in `readme.md` is fully operational.

## Literature And Prompt Workspace

### `literature/`

This folder is a lightweight paper workspace:

- `literature/pdf_papers/`
  - One folder per registered paper, named `bXXXXX_short_name`.
- `literature/pdf_books/`
  - Books and longer references.
- `literature/tex/`
  - Drafts and source material.
- `literature/bibliography.jsonl`
  - A simple JSONL metadata index.

### `prompt/`

Reusable repo prompts currently include:

- `c000`: write a reading review
- `c001`: read a proof and write a proof sketch
- `c002`: split a LaTeX project
- `c003`: register a new paper in `literature/`

The prompt registry lives in [prompt/command_list.jsonl](/Users/shzhang/Documents/Github/memory_lm/prompt/command_list.jsonl). Note that `c003` exists as [prompt/c003_add_new_paper.md](/Users/shzhang/Documents/Github/memory_lm/prompt/c003_add_new_paper.md) but is not yet listed in `command_list.jsonl`.

## Working Style Of The Repo

A few repo-level patterns are consistent across projects:

- Experiments are folder-local.
  - Each subproject carries its own docs, config, prompts, and runner.
- The repo favors transparent artifacts over hidden state.
  - Runs write transcripts, metrics, and intermediate memory snapshots to disk.
- Documentation is exploratory.
  - Some folders are fairly complete, while others are closer to research notes plus initial code.
- External APIs are expected.
  - At least the current `exp0404_three_world` and `exp0406_distil` workflows assume an LLM endpoint for real runs.

## Suggested Reading Order

If you are new to the repo, the fastest path is:

1. Read this file.
2. Read [exp0404_three_world/README.md](/Users/shzhang/Documents/Github/memory_lm/exp0404_three_world/README.md) to understand the original external-memory testbed.
3. Read [exp0408_memory_backprop/model.md](/Users/shzhang/Documents/Github/memory_lm/exp0408_memory_backprop/model.md) for the memory dynamics motivation.
4. Read [exp0406_distil/README.md](/Users/shzhang/Documents/Github/memory_lm/exp0406_distil/README.md) if you need the dataset-generation pipeline.
5. Use `literature/` and `prompt/` as the shared research workspace around the experiments.

## Current State

Pragmatically:

- `exp0404_three_world` looks like the most mature and operational experiment harness.
- `exp0406_distil` is a usable utility project for generating training data.
- `exp0408_memory_backprop` is a serious in-progress experiment with runnable code plus some forward-looking documentation.
- `exp0410_triangleNet` is not yet a runnable project.

This means the repo is best read as a living research workspace rather than a polished software product.
