# Experiment mar11-exp001-dual-td-adv

## Idea

Switch dual GCIVL actor advantage weighting from pure value-difference to optional TD advantage.

Rationale: TD advantage includes immediate reward signal and may provide better contrast for AWR weights in sparse antmaze transitions.

## Code changes

- File: `agents/gcivl/state/dual.py`
- Added config key `actor_advantage_mode` with options:
  - `td`
  - `vdelta` (default, backward-compatible)
- Updated actor loss to branch on `actor_advantage_mode`.
- Fixed representation critic target discount access typo.

## Command

```bash
uv run python main.py \
  --agent=agents/gcivl/state/dual.py \
  --agent.alpha=10.0 \
  --agent.rep_type=bilinear \
  --agent.rep_expectile=0.9 \
  --agent.goalrep_dim=256 \
  --agent.discount=0.99 \
  --agent.actor_advantage_mode=td \
  --env_name=antmaze-large-navigate-v0 \
  --seed=1 \
  --train_steps=1000000 \
  --log_interval=30000 \
  --eval_interval=100000 \
  --eval_episodes=30 \
  --wandb_mode=online
```

## Success criterion

- Primary: final `evaluation/overall_success` > `0.32`.
- Secondary: if tied on primary, higher mean over five task success columns.
