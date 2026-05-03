# GEMINI.md

## Project Overview

Offline goal-conditioned RL research codebase in JAX/Flax. Implements baseline algorithms (GCIVL, CRL, GCFBC, HILP, TMD, TRL, QRL, MQE) and representation learning variants (dual, BYOL-gamma, TRA, VIB, VIP). The paper's main contribution is the **dual-representation** approach in `agents/*/dual.py` files, which uses separate φ(s) and ψ(g) embeddings.

## Commands

```bash
# Setup
uv sync

# Format + lint (run after any edit)
uv run ruff format . && uv run ruff check .

# Smoke test (training dry-run, no W&B)
uv run python main.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --train_steps=5 --log_interval=5 --eval_interval=5 \
  --save_interval=1000000 --eval_episodes=1 --video_episodes=0 \
  --wandb_mode=offline

# Build-equivalent checks
uv run python -m compileall .
uv run python main.py --help
uv run python -c "from agents import agents; print(sorted(agents)[:5])"

# Full training example
uv run python main.py \
  --env_name=scene-play-v0 \
  --agent=agents/gcivl/state/dual.py \
  --agent.alpha=10.0 --agent.rep_type=bilinear --agent.discount=0.99

# Evaluate a checkpoint
uv run python eval_pretrained.py \
  --env_name=puzzle-3x3-play-v0 --agent=agents/crl/dual.py \
  --restore_path=results/.../checkpoints --restore_epoch=31000 \
  --eval_episodes=50 --num_bootstraps=50000

# Multi-agent comparison
uv run python rliable_agents.py \
  --results_dirs eval_results/crl_dual eval_results/gcfbc_dual \
  --labels "CRL-Dual" "GCFBC-Dual" --output_dir comparison_results/
```

For canonical paper reproduction hyperparameters, use `hyperparameters.sh`.

## Architecture

### Agent Pattern

All agents are `flax.struct.PyTreeNode` subclasses with a fixed interface:

```python
class MyAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any          # TrainState holding params + optimizer
    config: Any = nonpytree_field()

    def total_loss(self, batch, grad_params, rng=None): ...   # @jax.jit
    def update(self, batch): ...                               # @jax.jit
    def sample_actions(self, observations, goals, seed, temperature): ...  # @jax.jit

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config): ...

def get_config():
    return ml_collections.ConfigDict(dict(...))
```

Networks are assembled via `ModuleDict` (from `utils/flax_utils.py`) and wrapped in `TrainState`. Each named module (e.g., `"actor"`, `"value"`, `"target_value"`) is accessible via `self.network.select("actor")`. Gradients flow through `self.network.apply_loss_fn(loss_fn)`.

### Adding a New Agent

1. Copy the closest existing agent (e.g., `agents/gcivl/original.py`)
2. Modify the network assembly dict and loss functions
3. Register in `agents/__init__.py` under a new string key
4. Keep `get_config()` returning a `ml_collections.ConfigDict` with all hyperparams explicit

### Key Modules

| Module                | Role                                                               |
| --------------------- | ------------------------------------------------------------------ |
| `utils/flax_utils.py` | `ModuleDict`, `TrainState`, `nonpytree_field`                      |
| `utils/networks.py`   | `GCActor`, `GCValue`, `GCBilinearValue`, `MLP`, ensembles          |
| `utils/encoders.py`   | Visual encoders (`GCEncoder`, IMPALA) for pixel variants           |
| `utils/datasets.py`   | `GCDataset` / `HGCDataset` — goal-conditioned batch sampling       |
| `utils/dual.py`       | Dual representation modules (`GCHilbertRepresentationValue`, etc.) |
| `utils/env_utils.py`  | `make_env_and_datasets()`, `relabel_dataset()`                     |
| `utils/evaluation.py` | `evaluate()` — runs tasks, returns success rates                   |
| `utils/log_utils.py`  | CSV logger, W&B setup, checkpoint naming                           |
| `utils/rliable.py`    | Bootstrap statistics: IQM, median, mean, optimality gap            |

### Training Loop (`main.py`)

1. `make_env_and_datasets()` → load offline dataset + environment
2. `agent_class.create(seed, ex_obs, ex_actions, config)` → initialize agent
3. Background thread prefetches CPU batches onto GPU
4. Loop: `agent.update(batch)` → log metrics → `evaluate()` → checkpoint
5. Checkpoints saved via Orbax; restored by `eval_pretrained.py`

### Batch Keys

Standard `GCDataset` batch contains: `observations`, `next_observations`, `actions`, `rewards`, `masks`, `value_goals`, `actor_goals`. Optional keys (enabled by config flags): `rep_goals`, `rep_rewards`, `observation_oracles`.

### Metric Namespacing

- Training: `training/critic/loss`, `training/actor/loss`, etc.
- Evaluation: `evaluation/{task_name}_success`, `evaluation/overall_success`
- Validation: `validation/{subsystem}/*`

## Coding Conventions

- **Imports**: stdlib → third-party → local, one blank line between groups
- **Classes**: `PascalCase`; agent classes end in `Agent`
- **Functions/vars/config keys**: `snake_case`; log keys use slash namespaces
- **JAX**: `@jax.jit` on hot paths; explicit `jax.random.split` for RNG; `jax.lax.stop_gradient` only when intentional
- **Config**: all hyperparams in `get_config()` with sensible defaults; goal-sampling probabilities must sum to `1.0`
- **Seed plumbing**: preserve across Python `random`, NumPy, and JAX
- Do not silently change save directory structure, metric names, or checkpoint conventions

## Additional Notes

- See `AGENTS.md` for the full developer handbook (also read by Codex)
- The `skills/implement-gcrl-agent/` directory contains detailed templates and reference patterns for implementing new agents
- `results/`, `eval_results/`, and plot output folders are generated artifacts — do not edit unless asked
