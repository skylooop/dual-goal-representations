---
name: implement-gcrl-agent
description: Implement a new goal-conditioned RL algorithm in this repository using the established JAX/Flax agent pattern. Use when Codex needs to add or update a file under agents/, wire the agent into agents/__init__.py, choose compatible modules from utils/networks.py or utils/flax_utils.py, match dataset batch shapes from utils/datasets.py, or preserve this project's required class, import, config, and validation conventions.
---

# Implement GCRL Agent

## Overview

Implement a new offline GCRL agent so it matches this repository's architecture instead of introducing a new style. Follow the shared `flax.struct.PyTreeNode` pattern, use the required import block, build networks through `ModuleDict` and `TrainState`, and keep dataset, config, and registry wiring consistent with the existing agents.

## Workflow

1. Read the closest existing agents before writing code.
- Start with the algorithm family and observation mode that match the requested method.
- Good anchor files in this repo: `agents/gcivl/original.py`, `agents/crl/original.py`, `agents/gcivl/state/dual.py`, `agents/mqe/original.py`.
- Read `references/repo-patterns.md` for the exact repository conventions.

2. Choose the target file path and the closest template.
- Place the implementation in the same subtree as related algorithms.
- Keep file naming in `snake_case.py`.
- If the method introduces a new variant, decide early how it should be registered in `agents/__init__.py`.

3. Start from the required skeleton instead of inventing a new layout.
- Use the exact base import block below first, then append only the additional local imports you actually need.
- Keep the main class as a `flax.struct.PyTreeNode` with `rng`, `network`, and `config: Any = nonpytree_field()`.
- Implement `total_loss`, `update`, `sample_actions`, `create`, and a module-level `get_config()`.

4. Build only with repo-native modules and batch keys.
- Pull neural-network building blocks from `utils/networks.py`.
- Use `utils/flax_utils.py` for `ModuleDict`, `TrainState`, and any `TrainState` customization that the algorithm actually needs.
- Match loss inputs to keys provided by `utils/datasets.py`; do not invent batch fields without updating dataset logic too.
- Read `references/networks-and-batches.md` when deciding module names, targets, or batch keys.

5. Comment only the tricky lines.
- Add short comments where the math, target construction, stop-gradient usage, or shape manipulation is not obvious from the code.
- Do not add comments for straightforward plumbing.

6. Finish the integration.
- Add the class import and registry entry in `agents/__init__.py`.
- Keep `get_config()` as an `ml_collections.ConfigDict` with explicit keys and sensible defaults.
- Preserve stable names for metrics and config fields when extending an existing family.

7. Validate with lightweight repo checks.
- Run formatting and linting.
- Run at least a smoke-level command relevant to the change.
- If full training is not practical, state that clearly and report what was verified.

## Required Skeleton

Start every new implementation with this base import block, in this order:

```python
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
```

Add extra local imports after this block only when needed, for example specific modules from `utils.networks`, `utils.encoders`, or `utils.dual`.

Use this class shape:

```python
class MyAgent(flax.struct.PyTreeNode):
    """Short algorithm description."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        ...
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch, contrastive_only=False):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch,
                grad_params,
                rng=rng,
                contrastive_only=contrastive_only,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        ...

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        ...
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
```

Outside the class, always define `get_config()` and return `ml_collections.ConfigDict`.

## Implementation Rules

- Keep the public method names and signatures aligned with the repo pattern: `create`, `update`, `sample_actions`, `total_loss`, `get_config`.
- Build all modules first, then collect them into `network_info`, `networks`, and `network_args` before calling `ModuleDict(networks)`.
- Initialize parameters with `network_def.init(init_rng, **network_args)["params"]`.
- Create the optimizer with `optax.adam(learning_rate=config["lr"])` unless the algorithm explicitly requires something else.
- Use target networks only when the algorithm needs them, and mirror the established `modules_target_*` naming.
- For continuous actions, clip sampled actions to `[-1, 1]` unless the architecture already handles squashing.
- For discrete actions, derive `action_dim = ex_actions.max() + 1`; otherwise use `ex_actions.shape[-1]`.
- If the algorithm uses learned goal representations, create placeholder `ex_goals = jnp.zeros(shape=(1, config["goalrep_dim"]))` before module initialization.
- If the algorithm uses raw observation goals, follow the repo pattern for `oraclerep`, `norm`, or observation-mode-specific goal selection.
- Prefix metrics by subsystem in `info`, for example `value/...`, `critic/...`, `actor/...`, `rep/...`.
- Register every new agent in `agents/__init__.py` immediately after implementation.

## Comments and Explanations

Add a short explanatory comment when code would otherwise be hard to reconstruct from the paper or from the tensor shapes. Typical cases:

- A target or bootstrap expression that mixes stopped and live gradients.
- A contrastive logits construction or custom distance computation.
- A shape transform such as ensemble expansion, diagonal extraction, or broadcasting over goal pairs.
- Any place where the code intentionally differs from the paper for numerical stability.

Do not comment obvious mechanics such as optimizer creation or simple dictionary assembly.

## References

- Read `references/repo-patterns.md` for the required repository structure, class layout, config rules, and registration checklist.
- Read `references/networks-and-batches.md` when choosing network modules, target modules, or dataset keys.
