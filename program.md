# autoresearch for dual-goal-representations

This file defines how an autonomous agent should run research in this repository.

Goal: improve evaluation success rate across tasks in the given RL environment (experiment with antmaze-large-navigate-v0)
Right now we are experimenting with just GCIVL : `agents/gcivl/original.py`. Check the file `eikonal.tex` for an example of good research idea, which is both mathematically interesting, intuitive and intuitive. Other files in the `agents/gcivl/state/**` contain different representation learning approaches for helping learning better value functions.
When running experiments, check the `hyperparameters.sh` file for getting best hyperparams for baselines.
The baseline for GCIVL is already computed in the `results/gcivl/antmaze-large-navigate-v0_gcivl_sd000/eval.csv`
Baseline for GCIVL + Dual `results/gcivl_dual/antmaze-large-navigate-v0_gcivl_sd000/eval.csv`

## Project structure (what to read, what to modify)

Read these first to understand execution flow and constraints and project structure:

- `main.py`:
  - Orchestrates full training loop, logging, evaluation, checkpointing.
  - Defines CLI flags, including TRAIN HYPERS and EVAL HYPERS (read-only definitions).
  - Creates output directories under `results/<agent_name>/<exp_name>/`.
  - Writes `train.csv`, `eval.csv`, and `flags.json`.
- `agents/__init__.py`:
  - Registry from `agent_name` string to agent class.
  - New/variant agent implementations must be registered here.
- `agents/**`:
  - Main research surface. Algorithm logic, losses, update rules, config (`get_config`).
  - Typical edit targets: actor/value objectives, representation losses, target updates,
    sampling behavior, optimizers, regularizers and config defaults specific to an agent
- `utils/networks.py` and related network files in `utils/**`:
  - Shared model components and neural building blocks used by agents.
  - Allowed to modify only when needed to support agent-side research changes (but not breaking backward compatibility).
- `utils/datasets.py`, `utils/env_utils.py`, `utils/evaluation.py`, `utils/flax_utils.py`,
  `utils/log_utils.py`:
  - Infrastructure/data/evaluation/logging helpers.
  - Treat as mostly stable; only network-supporting changes in `utils/**` are allowed (`networks.py`, `dual.py`, `vib.py`)

Outputs to inspect after every run:

- `results/<agent_name>/<exp_name>/eval.csv`:
  - Evaluation metrics for current experiment and algorithm.
  - Contains per-task success metrics and `evaluation/overall_success` for the given environment.
  - This file determines keep/discard decisions.
- `results/<agent_name>/<exp_name>/train.csv`:
  - Training diagnostics only (losses, value stats, etc.).
  - Use for debugging, understanding of the training evolution or idea generation, not as primary ranking metric.
- `results/<agent_name>/<exp_name>/flags.json`:
  - Exact run configuration used for reproducibility.

Out-of-scope / do not modify:

- `main.py` TRAIN HYPERS and EVAL HYPERS definitions.
- Evaluation metric semantics and task definitions.
- Artifacts in `results/`, wandb offline run artifacts, or checkpoints.

## Setup

1. Create a research tag (example: `mar11`).
2. Ensure `master` stays unchanged. Never commit on `master`.
3. Create and keep a dedicated baseline branch from `master`:
   - `git checkout master`
   - `git checkout -b autoresearch/<tag>-baseline`
4. Run one baseline experiment and record/read metrics.
5. Create a .md file in the current working branch that explains what is done, what idea is tried etc (all changes must be specified here).
6. Find best way to show which branch achieved best success rate (either by creating a file (e.g results.tsv) and commiting to it results from different branches or any other good way). But I must see which changes and what score/results were obtained for each experiment branch.
7. The code is being run on single RTX 4090
8. For each experiment, there MUST BE a description of the idea that is being implemented

## Entry point and metric

- Main entry point: `main.py`.
- Main metric: success rate in `results/<agent_name>/<exp_name>/eval.csv`.
- Primary score: latest `evaluation/overall_success` from `eval.csv`. Check also scores for intermediate steps, which should provide better signal what happend during training (decrease/increase in success rate)
- Secondary score (tie-breaker): mean of the 5 task success columns in `eval.csv`.
- Diagnostics (not ranking metric): `results/<agent_name>/<exp_name>/train.csv`.

## Hard constraints

- Run with `--wandb_mode online`.
- Do not modify `main.py` train/eval hyperparameter definitions (`TRAIN HYPERS` and `EVAL HYPERS` blocks).
- Do not change evaluation logic in `main.py`/`utils/evaluation.py`.
- Allowed code edits only:
  - `agents/**`
  - network-related files in `utils/**` when needed for agent/network changes.
- Do not install new dependencies.
- Keep `master` untouched.

## Branching policy (one experiment per branch)

Every experiment must run on a separate branch.

- Branch naming: `autoresearch/<tag>-expNNN-<short-idea>`.
- Start each experiment branch from the current best known branch (baseline first).
- Never reuse an experiment branch for a different idea.
- Keep successful branches; discard failed/non-improving branches.

Implementation rule:

- Launch training in background, so even after i have closed terminal the training must continue.
- Poll `eval.csv` in the corresponding results folder.

Notes:

- This policy must be enforced externally (process control), without editing `main.py` hyperparameter definitions.

## Experiment command

Use this template (replace agent config and env as needed):

```
uv run python main.py \
  --agent=agents/crl/dual.py \
  --env_name=antmaze-large-navigate-v0 \
  --wandb_mode=offline
  --eval_interval=100000
  --eval_episodes=30
  --wandb_mode=online
```

Ensure that eval_interval=100000 and log_interval=30000, eval_episodes=30. The run must be active until whole million iterations are performed of training.

You may pass additional CLI flags, but do not edit the train/eval hyperparameter definitions in `main.py`.

## What to optimize

Focus on new ideas and directions (not just obvious hyperparameter tuning). Try to push over boundaries and find interesting and correct approaches for experimenting. Prioritize ideas that can improve cross-task success quickly:

- representation learning losses and weighting,
- goal conditioning pathways,
- value/actor target construction,
- sampling and augmentation choices in agent code,
- network architecture changes in allowed files.

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. An improvement that adds 20 lines of hacky code? Probably not worth it. A small improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

You must understand what you have already tried (which ideas/approaches). What worked and what not.

## Evaluation and logging protocol

After each run:

1. Find latest run directory in `results/<agent_name>/...`.
2. Read `eval.csv` and extract lastest completed evaluation row (corresponding to last eval)
3. Compute:
   - `overall_success` = `evaluation/overall_success`
   - `mean_task_success` = mean of 5 per-task success columns from that row
4. Compare against current best.
5. Append one line to `results.tsv` with:
   - branch
   - short commit hash
   - overall_success
   - mean_task_success
   - status (`keep`, `discard`, `crash`)
   - short description
6. Write down your observations, thoughts regarding experiment in the `thoughts.md` which summarizes the results obtained and insights.

Status policy:

- `keep`: strictly better `overall_success`, or equal `overall_success` with better `mean_task_success` and comparable complexity.
- `discard`: valid run but not better.
- `crash`: runtime failure, NaNs (either at eval or train), or malformed outputs.
  Make a short description what was tried and insights for each branch.

## Autonomous loop

Repeat until interrupted by user:

1. Select one focused idea.
2. Create new experiment branch from current best branch.
3. Edit only allowed files.
4. Commit.
5. Parse metrics from `eval.csv` (+ inspect `train.csv` for debugging only and better understanding of the evolution of metrics).
6. Check `thoughts.md` file (if exists) and get understanding what was tried in the best branch.
7. Log in `results.tsv`.
8. Keep or discard branch by metric policy.

## Safety and reproducibility

- Keep seeds explicit (`--seed`), and track them in results descriptions.
- Avoid changes that silently alter task definitions or metric semantics.
- Do not commit artifacts from `results/`, wandb offline files, or large generated outputs.
- If an idea crashes repeatedly, record and move on.

The idea is that you are a completely autonomous researcher trying things out. Use your vast knowledge of RL literature (representation learning, offline RL, goal-conditioned RL etc.).

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause it to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
