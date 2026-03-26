# Repo Patterns

Use this reference when you are about to add a new agent file or modify an existing one.

## Canonical Files

- `agents/gcivl/original.py`: plain value-plus-actor pattern with target update.
- `agents/crl/original.py`: contrastive RL baseline.
- `agents/gcivl/state/dual.py`: state-goal dual-representation variant.
- `agents/mqe/original.py`: custom representation distance and extra batch fields.
- `agents/__init__.py`: registry and import wiring.

## Required Import Pattern

Start with this exact base import block:

```python
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
```

Then append only the extra imports needed by the algorithm. Keep import groups as stdlib, third-party, local project.

## Agent Layout

Keep the implementation in this order unless there is a clear reason not to:

1. Class docstring.
2. Fields: `rng`, `network`, `config`.
3. Small static helpers such as `expectile_loss`.
4. Per-loss helpers such as `value_loss`, `critic_loss`, `rep_loss`, `actor_loss`.
5. `total_loss`.
6. `target_update` if the algorithm uses target parameters.
7. `update`.
8. `sample_actions`.
9. `create`.
10. Module-level `get_config()`.

## Network Assembly Pattern

The common pattern in this repo is:

```python
network_info = dict(
    actor=(actor_def, (ex_observations, ex_goals)),
    critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
)
networks = {k: v[0] for k, v in network_info.items()}
network_args = {k: v[1] for k, v in network_info.items()}

network_def = ModuleDict(networks)
network_tx = optax.adam(learning_rate=config["lr"])
network_params = network_def.init(init_rng, **network_args)["params"]
network = TrainState.create(network_def, network_params, tx=network_tx)
```

If target modules are required, deep-copy the module definition and then copy the initialized parameters into the corresponding `modules_target_*` entry.

## Loss and Logging Conventions

- Return `(loss, info)` from each helper that reports metrics.
- Aggregate metrics into subsystem namespaces inside `total_loss`, for example `info[f"critic/{k}"] = v`.
- Use `rng = rng if rng is not None else self.rng` at the start of `total_loss` when stochastic components are present.
- Split RNGs locally for different stochastic losses.
- Use `jax.lax.stop_gradient` intentionally and comment the non-obvious cases.

## `update` Convention

Use `self.network.apply_loss_fn(loss_fn=loss_fn)` so gradient statistics are logged automatically. If the algorithm has target modules, update them after `apply_loss_fn` and before returning the replaced agent.

## `sample_actions` Convention

- Query the actor through `self.network.select("actor")(...)`.
- Pass encoded goal representations only when the actor expects them.
- Clip continuous actions to `[-1, 1]`.
- Respect the `temperature` argument.

## `create` Convention

- Seed with `jax.random.key(seed)` or `jax.random.PRNGKey(seed)` and split once for `init_rng`.
- Infer `action_dim` from `ex_actions`.
- Use `ex_goals = jnp.zeros(shape=(1, config["goalrep_dim"]))` only for latent-goal architectures.
- For raw-observation-goal methods, preserve the established goal setup, including `oraclerep` behavior where applicable.
- Return `flax.core.FrozenDict(**config)` in the final agent.

## `get_config` Convention

Keep every hyperparameter explicit inside `ml_collections.ConfigDict(dict(...))`.

Include:

- Agent identity such as `agent_name`.
- Optimizer and architecture keys such as `lr`, hidden dims, latent sizes, `layer_norm`.
- RL keys such as `discount`, `tau`, `alpha`, `expectile`, or algorithm-specific loss weights.
- Dataset sampling keys expected by `utils/datasets.py`.
- Flags such as `discrete`, `oraclerep`, `norm`, `p_aug`, `frame_stack`.

Keep comments short and useful. Do not silently remove compatibility keys used by training scripts.

## Registration Checklist

After creating a new agent:

1. Add the class import to `agents/__init__.py`.
2. Add the key-to-class mapping in the `agents = dict(...)` registry.
3. Keep the registry key stable and aligned with `config["agent_name"]`.

## Validation Checklist

Run the lightweight checks that fit the change:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run python -m compileall .`
- `uv run python main.py --help`
- `uv run python eval_pretrained.py --help`

If the change affects runnable training logic, prefer one short smoke run with heavily reduced steps and episodes.
