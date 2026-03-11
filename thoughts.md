# Research notes (mar11)

## Current best reference

- Environment: `antmaze-large-navigate-v0`
- Baseline GCIVL (`agents/gcivl/original.py`): `overall_success=0.24`
- Baseline GCIVL+Dual (`agents/gcivl/state/dual.py`): `overall_success=0.32`
- Current best known target to beat: `0.32`

## Experiment 001: TD-advantage actor weighting for dual GCIVL

Hypothesis:

- In sparse-reward antmaze, using TD-shaped actor weighting (`r + gamma * V(s') - V(s)`) can sharpen policy updates relative to pure value-delta weighting (`V(s') - V(s)`), especially when the representation head changes value scale during training.

What is changed:

- Added `actor_advantage_mode` config to `agents/gcivl/state/dual.py`.
- Enabled both options in actor loss:
  - `td`: `r + gamma * V(s') - V(s)`
  - `vdelta`: `V(s') - V(s)`
- Fixed a bug in representation critic target (`self.config['discount']` was mistakenly misspelled in code path).

Run plan:

- Branch: `autoresearch/mar11-exp001-dual-td-adv`
- Seed: `1` (to avoid overwriting existing `*_sd000` baseline outputs)
- Keep required protocol: `eval_interval=100000`, `log_interval=30000`, `eval_episodes=30`, `wandb_mode=online`
- Train to `1_000_000` steps and compare final `evaluation/overall_success` against `0.32`.

Runtime note:

- Attempt with custom `--agent.agent_name=gcivl_dual_tdadv` crashed due to `KeyError` in `agents` registry (main uses `agent_name` for class lookup). Switched to standard registered `gcivl_dual` and unique seed for separate output directory.
