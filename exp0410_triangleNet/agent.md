# Agent Notes For `exp0410_triangleNet`

## What this project is

This folder is a small research experiment for a **Triangular Memory Network (TMN)**.

The current goal is not "memory" in the full conceptual sense yet.
The current goal is much narrower:

- implement the triangular DAG network cleanly
- compare it against an MLP baseline
- test both on simple 1D function fitting
- produce one comparison figure that is easy to inspect

At the moment, this folder is intentionally optimized for **one-click running** and **low distraction**, not for feature completeness.

## What the user currently wants

The user repeatedly pushed the project toward a simpler workflow.

The current intended workflow is:

1. edit `params.yaml`
2. run `python3 run.py`
3. get exactly one output figure in a single experiment folder

The user does **not** want:

- many scattered scripts to think about
- many output files per run
- separate TMN/MLP artifact trees for normal use
- a parameter interface hidden inside Python code

So if you extend this project, preserve that simplicity unless the user explicitly asks otherwise.

## Current real entrypoint

The real entrypoint is:

- [run.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/run.py)

`run.py` currently:

- reads parameters from `params.yaml`
- trains TMN
- trains MLP
- generates one `comparison_4panel.png`

It is the main user-facing command.

## Current parameter file

The main user-editable file is:

- [params.yaml](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/params.yaml)

This file is supposed to be the only place a non-technical user needs to edit.

Important parameters there:

- `run_name`
- `L`
- `n_in`
- `n_out`
- `d_model`
- `mlp_layers`
- `task_name`
- `custom_function`
- `node_activation`
- `output_activation`
- `lr`
- `batch_size`
- `epochs`
- `x_min`
- `x_max`

## Current output convention

Current output is intentionally minimal.

Each run should produce one folder:

- `exp0410_triangleNet/runs/<run_name or task_name>/`

Inside that folder, the intended main artifact is:

- `comparison_4panel.png`

That figure contains:

- TMN loss curve
- TMN prediction
- MLP loss curve
- MLP prediction

The figure also includes architecture text:

- TMN shows `L`, triangular core node count, and `d_model`
- MLP shows `layers=[...]`

## Important implementation files

The core implementation now lives under:

- [scripts/config.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/config.py)
- [scripts/data.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/data.py)
- [scripts/train.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/train.py)
- [scripts/utils.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/utils.py)
- [scripts/model/graph.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/model/graph.py)
- [scripts/model/tmn.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/model/tmn.py)
- [scripts/model/mlp.py](/Users/shzhang/Documents/Github/memory_lm/exp0410_triangleNet/scripts/model/mlp.py)

Responsibilities:

- `config.py`: defines `Config` and loads YAML
- `data.py`: defines built-in target functions and dataset construction
- `train.py`: trains one model and returns in-memory results
- `utils.py`: plotting and small helpers
- `graph.py`: builds the triangular DAG
- `tmn.py`: TMN forward logic
- `mlp.py`: MLP baseline defined by `mlp_layers`

## Architectural assumptions

These are true in the current implementation:

- TMN is a DAG, not a relay-node-expanded MLP
- TMN node block is currently `Linear + ReLU`
- output activation is currently `tanh`
- current experiments are effectively scalar input / scalar output
- first-stage tasks are 1D regression

Because output uses `tanh`, target functions are easiest when their values are roughly in `[-1, 1]`.

## Known drift / cleanup note

`readme.md` may lag behind the latest simplification work.

In particular, if behavior and docs disagree, trust the code path:

- `params.yaml`
- `run.py`
- `scripts/train.py`
- `scripts/utils.py`

If future cleanup is needed, the README should be aligned with the current single-artifact workflow.

## Safe next steps

Reasonable next improvements, if the user asks:

- align `readme.md` with the latest simplified workflow
- keep outputs minimal and readable
- improve the aesthetics of `comparison_4panel.png`
- support more built-in target functions
- make output activation configurable beyond `tanh`
- add clearer architecture statistics to the plot

## Things to avoid unless explicitly requested

- do not reintroduce many separate output files
- do not force the user to edit Python code for ordinary runs
- do not spread configuration across many files
- do not make the default workflow depend on manual post-processing
- do not add complexity for "future extensibility" unless it clearly helps the current experiment
