# Experiment: mar11 baseline

- Branch: `autoresearch/mar11-baseline`
- Agent: `agents/gcivl/original.py`
- Env: `antmaze-large-navigate-v0`
- Run mode: `--wandb_mode=offline`
- Eval cadence: `--eval_interval=100000`
- Stop condition: terminate after `eval.csv` contains 2 evaluation rows.

## Idea

Baseline sanity run for GCIVL with no algorithm changes, to establish a reference
success rate under the current code and runtime policy.

## Code changes in this branch

None for the agent itself (baseline run only).

## Result

To be filled after run completes.
