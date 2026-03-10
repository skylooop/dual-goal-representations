# autoresearch for dual-goal-representations

This file defines how an autonomous agent should run research in this repository.

Goal: improve evaluation success rate across tasks in the given RL environment (experiment with antmaze-large-navigate-v0)

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
5. Initialize an untracked `results.tsv` file in repo root with header:

```
branch\tcommit\toverall_success\tmean_task_success\tstatus\tdescription
```

Do not commit `results.tsv`.

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

Each experiment must satisfy BOTH:

1. Hard wall-clock cap: 8 minutes maximum.
2. Stop as soon as the first evaluation is completed.

Implementation rule:

- Launch training in background.
- Poll `eval.csv` under the current run directory.
- As soon as `eval.csv` has at least one data row, terminate training gracefully.
- If no evaluation is completed by 8 minutes, terminate and mark as failure.

Notes:

- This policy must be enforced externally (process control), without editing `main.py` hyperparameter definitions.
- A run that exceeds 8 minutes or lacks first eval completion is `crash`/`invalid`.

## Experiment command

Use this template (replace agent config and env as needed):

```
uv run python main.py \
  --agent=agents/crl/dual.py \
  --env_name=antmaze-large-navigate-v0 \
  --wandb_mode=offline
```

You may pass additional CLI flags, but do not edit the train/eval hyperparameter definitions in `main.py`.

## What to optimize

Prioritize ideas that can improve cross-task success quickly:

- representation learning losses and weighting,
- goal conditioning pathways,
- value/actor target construction,
- sampling and augmentation choices in agent code,
- network architecture changes in allowed files.

Prefer simple, robust changes over brittle complexity.

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
- `crash`: runtime failure, no first eval within 8 minutes, NaNs, or malformed outputs.

## Autonomous loop

Repeat until interrupted by user:

1. Select one focused idea.
2. Create new experiment branch from current best branch.
3. Edit only allowed files.
4. Commit.
5. Run experiment under 8-minute/first-eval policy.
6. Parse metrics from `eval.csv` (+ inspect `train.csv` for debugging only).
7. Log in `results.tsv`.
8. Keep or discard branch by metric policy.

Never ask for confirmation between iterations; continue autonomously.

## Safety and reproducibility

- Keep seeds explicit (`--seed`), and track them in results descriptions.
- Avoid changes that silently alter task definitions or metric semantics.
- Do not commit artifacts from `results/`, wandb offline files, or large generated outputs.
- If an idea crashes repeatedly, record and move on.
