# autoresearch for dual-goal-representations

This file defines how an autonomous agent should run research in this repository.

Goal: improve evaluation success rate across tasks in the given RL environment (experiment with antmaze-large-navigate-v0)
Right now we are experimenting with just GCIVL : `agents/gcivl/original.py`

## Project structure (what to read, what to modify)

Read these first to understand execution flow and constraints:

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
    sampling behavior, and config defaults specific to an agent.
- `utils/networks.py` and related network files in `utils/**`:
  - Shared model components and neural building blocks used by agents.
  - Allowed to modify only when needed to support agent-side research changes.
- `utils/datasets.py`, `utils/env_utils.py`, `utils/evaluation.py`, `utils/flax_utils.py`,
  `utils/log_utils.py`:
  - Infrastructure/data/evaluation/logging helpers.
  - Treat as mostly stable; only network-supporting changes in `utils/**` are allowed.

Outputs to inspect after every run:

- `results/<agent_name>/<exp_name>/eval.csv`:
  - Ground-truth evaluation metrics.
  - Contains per-task success metrics and `evaluation/overall_success`.
  - This file determines keep/discard decisions.
- `results/<agent_name>/<exp_name>/train.csv`:
  - Training diagnostics only (losses, value stats, etc.).
  - Use for debugging or idea generation, not as primary ranking metric.
- `results/<agent_name>/<exp_name>/flags.json`:
  - Exact run configuration used for reproducibility.

Out-of-scope / do not modify:

- `main.py` TRAIN HYPERS and EVAL HYPERS definitions.
- Evaluation metric semantics and task definitions.
- Artifacts in `results/`, wandb offline run artifacts, or checkpoints.

## Setup

1. Create a research tag (example: `mar10`).
2. Ensure `master` stays unchanged. Never commit on `master`.
3. Create and keep a dedicated baseline branch from `master`:
   - `git checkout master`
   - `git checkout -b autoresearch/<tag>-baseline`
4. Run one baseline experiment and record metrics.
5. Create a .md file in the current working branch that explains what is done, what idea is tried etc (all changes must be specified here).
6. Find best way to show which branch achieved best success rate (either by creating a file (e.g results.tsv) and commiting to it results from different branches or any other good way). But I must see which changes and what score/results were obtained for each experiment branch.
7. The code is being run on single RTX 4090
8. For each experiment, there MUST BE a description of the idea that is being implemented

## Entry point and metric

- Main entry point: `main.py`.
- Main metric: success rate in `results/<agent_name>/<exp_name>/eval.csv`.
- Primary score: latest `evaluation/overall_success` from `eval.csv`.
- Secondary score (tie-breaker): mean of the 5 task success columns in `eval.csv`.
- Diagnostics (not ranking metric): `results/<agent_name>/<exp_name>/train.csv`.

## Hard constraints

- Run with `--wandb_mode offline`.
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

## Runtime budget policy

Each experiment must satisfy following:

1. Stop as soon as the TWO evaluations are completed.

Implementation rule:

- Launch training in background.
- Poll `eval.csv` in the corresponding results folder.
- As soon as `eval.csv` has at least two data rows, terminate training gracefully.

Notes:

- This policy must be enforced externally (process control), without editing `main.py` hyperparameter definitions.

## Experiment command

Use this template (replace agent config and env as needed):

```
uv run python main.py \
  --agent=agents/crl/dual.py \
  --env_name=antmaze-large-navigate-v0 \
  --wandb_mode=offline
  --eval_interval=50000
```
Ensure that eval_interval=50000 and log_interval=20000

You may pass additional CLI flags, but do not edit the train/eval hyperparameter definitions in `main.py`.

## What to optimize

Focus on new ideas and directions (not just obvious hyperparameter tuning). Prioritize ideas that can improve cross-task success quickly:

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
2. Read `eval.csv` and extract first completed evaluation row.
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

Status policy:

- `keep`: strictly better `overall_success`, or equal `overall_success` with better `mean_task_success` and comparable complexity.
- `discard`: valid run but not better.
- `crash`: runtime failure, NaNs (either at eval or train), or malformed outputs.

## Autonomous loop

Repeat until interrupted by user:

1. Select one focused idea.
2. Create new experiment branch from current best branch.
3. Edit only allowed files.
4. Commit.
5. Parse metrics from `eval.csv` (+ inspect `train.csv` for debugging only).
6. Log in `results.tsv`.
7. Keep or discard branch by metric policy.

Never ask for confirmation between iterations; continue autonomously.

## Safety and reproducibility

- Keep seeds explicit (`--seed`), and track them in results descriptions.
- Avoid changes that silently alter task definitions or metric semantics.
- Do not commit artifacts from `results/`, wandb offline files, or large generated outputs.
- If an idea crashes repeatedly, record and move on.

The idea is that you are a completely autonomous researcher trying things out. Use your vast knowledge of RL literature (representation learning, offline RL, goal-conditioned RL etc.).

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.