# Networks and Batches

Use this reference when selecting modules from `utils/networks.py`, deciding whether the algorithm needs target networks, or matching batch keys from `utils/datasets.py`.

## Network Sources

- `utils/networks.py`: actor, critic, value, representation, scalar-parameter, normalization, and MLP building blocks.
- `utils/flax_utils.py`: `ModuleDict`, `TrainState`, and `nonpytree_field`.
- `utils/encoders.py`: observation encoders when the method needs image or state encoders.
- `utils/dual.py`: dual goal-representation constructors used by the dual variants.

## Common Module Choices in This Repo

- `GCActor` and `GCDiscreteActor`: policy heads for continuous and discrete actions.
- `GCValue`: unrestricted value network, optionally ensembled.
- `GCBilinearValue` and `GCDiscreteBilinearCritic`: bilinear contrastive-style value or critic modules.
- `StateRepresentation` and `DiscreteStateActionRepresentation`: representation modules used by methods such as MQE.
- `Param` or `LogParam`: scalar learned parameters when the algorithm needs a temperature or scale.

Match the module family to the algorithm math instead of forcing every method into the same actor-critic structure.

## `ModuleDict` Usage

`ModuleDict` expects one entry per module name. During initialization, pass one argument tuple or mapping for each module key. The keys used in `network_info` determine the parameter tree names:

- `actor` -> `modules_actor`
- `critic` -> `modules_critic`
- `target_value` -> `modules_target_value`

This is why target update helpers index `self.network.params[f"modules_{module_name}"]`.

## Target Networks

Use target modules only when the algorithm relies on lagged bootstrapping or critic stabilization.

Pattern:

```python
target_def = copy.deepcopy(source_def)
...
params["modules_target_value"] = params["modules_value"]
```

Then update with Polyak averaging inside `target_update`.

## Goal Handling Patterns

There are two common patterns:

### Raw observation goals

Use the goal tensors sampled from the dataset directly. This is common in `gcivl`, `gcfbc`, and other observation-goal methods.

### Learned goal representations

Project observations or goals first, then feed the representation into downstream actor or critic modules. This is common in `crl_dual` and related variants. When using this pattern:

- Create placeholder `ex_goals` using `config["goalrep_dim"]` inside `create`.
- Be explicit about whether `sample_actions` receives raw goals or already-encoded goals.
- Keep the encoded-goal pathway consistent across `actor_loss`, `total_loss`, and `sample_actions`.

## Batch Keys from `utils/datasets.py`

Standard `GCDataset` batches include:

- `observations`
- `next_observations`
- `actions`
- `value_goals`
- `actor_goals`
- `rewards`
- `masks`

Optional keys appear when config flags enable them:

- `rep_goals`, `rep_rewards`, `rep_masks` for representation-goal training.
- `observation_oracles` when `oraclerep=True`.
- `normed_observations` are stored in the dataset when normalization is enabled.

Agent-specific keys already supported in the repo include:

- MQE: `intermediate_value_goals`, `intermediate_value_goals_offsets`, `value_goals_offsets`.
- TRL: `value_offsets`, `value_midpoint_offsets`, `value_midpoint_observations`, `value_midpoint_actions`, `next_actions`, and goal variants.

If a new algorithm needs a new batch field, update dataset sampling in a way that preserves existing agents.

## Shape Discipline

- Batch size is always the leading dimension.
- Ensemble outputs often add a leading ensemble axis; several agents normalize this by inserting `None` when a non-ensemble tensor is returned.
- Bilinear or contrastive methods often construct `(B, B, E)` or `(E, B, B)` tensors. Add a short comment when broadcasting or diagonal extraction is not immediately obvious.
- Discrete-action modules may expect integer-coded actions or action counts derived from `ex_actions.max() + 1`.

## Comment-Worthy Cases

Add a short comment when any of these appear:

- Converting pairwise similarities into contrastive logits.
- Mixing stopped target values with live online values.
- Randomly selecting next-state or intermediate-goal targets.
- Reparameterizing a paper equation for numerical stability.
