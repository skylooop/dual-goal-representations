# LAF Refinement Patch Spec

This document describes the **required corrections** to the current `laf.py` implementation of Latent Advantage Fields (LAF) for goal-conditioned offline RL.

The intended reader is an implementation agent (Claude Code / Codex) that will modify the current codebase. The primary target file is `laf.py`. If the dataset code does not already provide the required batch fields for the actor fix, the dataset pipeline must also be updated.

The goal of this patch is **not** to turn LAF into OTA. The goal is to make the current one-step LAF implementation **mathematically self-consistent and correctly aligned with its own derivation**. Only after these corrections should we evaluate the intrinsic limits of one-step LAF on long-horizon tasks such as `antmaze-large-navigate-v0`.

---

## 1. Executive summary

There are three mandatory correctness fixes and two strongly recommended clarity/stability fixes.

### Mandatory fixes

1. **Make the critic and successor termination-consistent.**
   - The current code uses `masks` in the Bellman target `y`, but not in the online `q` or the successor regression target `x_targ`.
   - This is inconsistent with the model definition and causes the latent successor head to predict continuation through terminal / goal-reaching transitions.

2. **Make actor weighting and actor conditioning use the same goal.**
   - The current code conditions the actor on `actor_goals` but computes the AWR weight from an advantage evaluated at `value_goals`.
   - This is semantically wrong. The cloned action and the advantage weight must refer to the **same conditional policy**.

3. **Update the target network from the freshly updated online params, not from the stale pre-update params.**
   - The current `target_update()` function uses `self.network.params[...]` instead of `network.params[...]`.
   - This applies EMA toward the old online network instead of the newly updated one.

### Strongly recommended fixes

4. **Make target-parameter initialization an explicit copy, not a shared reference.**
   - Avoid aliasing between `modules_rep_value` and `modules_target_rep_value`.

5. **Make actor-side critic computations explicitly stop-gradient and parameter-consistent.**
   - Compute the actor-side advantage using `grad_params`, then detach it.
   - Explicitly detach the goal embedding fed into the actor.

---

## 2. The core LAF equations the code must implement

The implementation should follow the one-step LAF factorization:

\[
V(s,g) = z(s)^\top u(g) / \sqrt{d}
\]

\[
x(s,a) \approx \gamma \, \mathbb{E}[z(s') \mid s,a]
\]

\[
Q(s,a,g) = r(s,a,g) + x(s,a)^\top u(g) / \sqrt{d}
\]

\[
A(s,a,g) = Q(s,a,g) - V(s,g)
= r(s,a,g) + \bigl(x(s,a) - z(s)\bigr)^\top u(g) / \sqrt{d}
\]

For terminated transitions, if the environment treats goal-reaching as absorbing, then the correct masked version is:

\[
x_{\text{targ}}(s,a) = \gamma \, m \, z(s')
\]

\[
Q(s,a,g) = r(s,a,g) + m \, x(s,a)^\top u(g) / \sqrt{d}
\]

where `masks = 1` means nonterminal and `masks = 0` means terminal / absorbing.

This is not optional. If `y` uses masks but `q` and `x_targ` do not, then the critic is trained under inconsistent semantics.

---

## 3. Required correction #1: termination-consistent critic and successor

## Problem

Current code pattern:

```python
q = rewards + jnp.sum(x * u_g, axis=-1) / sqrt_d
...
y = rewards + discount * masks * v_next_bar
...
x_targ = discount * z_next_bar
```

This is wrong because:

- `y` says future continuation vanishes when `masks == 0`
- `x_targ` still asks the successor head to predict a discounted next embedding even when `masks == 0`
- `q` still lets the online successor contribute continuation value on terminal transitions

So the scalar value branch and the latent successor branch are learning different MDP semantics.

## Correct behavior

Use masks consistently in **all** future-value terms.

### Required equations

```python
q = rewards + masks * jnp.sum(x * u_g, axis=-1) / sqrt_d
v_next_bar = jnp.sum(z_next_bar * u_bar, axis=-1) / sqrt_d
y = rewards + discount * masks * v_next_bar
x_targ = discount * masks[..., None] * z_next_bar
```

### Why this is correct

The successor head is meant to approximate the **discounted future latent state contribution** that enters the Bellman equation.
If a transition is terminal, there is no continuation term. Therefore both:

- the Bellman target, and
- the successor regression target

must collapse to the immediate reward only.

Also note the logic:

- If `x` is trained to already include `gamma * mask`, then `q = r + x^T u / sqrt(d)` is sufficient in the limit.
- However, multiplying by `masks` again in the online `q` is still desirable during learning, because it enforces terminal consistency even before `x` is perfect.

So for this implementation, prefer the robust form:

```python
q = rewards + masks * jnp.sum(x * u_g, axis=-1) / sqrt_d
```

### Required code edits in `critic_loss`

Replace the current online and target computations with:

```python
sqrt_d = jnp.sqrt(self.config["latent_dim"])
discount = self.config["discount"]
mask_f = masks.astype(jnp.float32)

v, z_s, u_g = self.network.select("rep_value")(
    obs, goals, info=True, params=grad_params
)
x = self.network.select("successor")(obs, actions, params=grad_params)

q = rewards + mask_f * jnp.sum(x * u_g, axis=-1) / sqrt_d

_, z_next_bar, u_bar = self.network.select("target_rep_value")(
    next_obs, goals, info=True
)
v_next_bar = jnp.sum(z_next_bar * u_bar, axis=-1) / sqrt_d
y = rewards + discount * mask_f * v_next_bar
x_targ = discount * mask_f[..., None] * z_next_bar
```

### Diagnostics to add

Add:

```python
"terminal_frac": jnp.mean(1.0 - mask_f),
"q_terminal_mean": jnp.mean(jnp.where(mask_f == 0, q, 0.0)),
"succ_terminal_mean": jnp.mean(jnp.where(mask_f[..., None] == 0, x, 0.0)),
```

These metrics make it easy to verify that continuation terms collapse on terminal transitions.

---

## 4. Required correction #2: actor weighting and actor conditioning must use the same goal

## Problem

Current actor loss pattern:

```python
actor_goals = batch["actor_goals"]
value_goals = batch["value_goals"]

u_actor = rep_value(actor_goals)
...
v, z_s, u_g = rep_value(obs, value_goals, info=True)
x = successor(obs, actions)
q = rewards + x^T u_g / sqrt_d
adv = stop_gradient(q - v)

dist = actor(obs, u_actor)
actor_loss = -(w * log_prob(actions)).mean()
```

This is wrong because the policy is conditioned on `actor_goals`, while the weight is computed for a different goal `value_goals`.

That means the objective becomes:

- "clone action `a` for goal `g_actor`"
- weighted by "how good `a` was for goal `g_value`"

This breaks the semantics of advantage-weighted regression.

## Correct behavior

The actor loss must use a **single goal variable** `g_pi` for both:

1. policy conditioning, and
2. advantage estimation

### Correct equations

Use

\[
g_\pi := \texttt{actor_goals}
\]

Then compute:

\[
u_\pi = u(g_\pi)
\]

\[
V_\pi(s,g_\pi) = z(s)^\top u(g_\pi) / \sqrt{d}
\]

\[
Q_\pi(s,a,g_\pi) = r(s,a,g_\pi) + m \, x(s,a)^\top u(g_\pi) / \sqrt{d}
\]

\[
A_\pi(s,a,g_\pi) = Q_\pi(s,a,g_\pi) - V_\pi(s,g_\pi)
\]

and use that `A_pi` to construct the AWR weight.

## Important implementation consequence

This requires rewards and masks that correspond to `actor_goals`, not `value_goals`.

### Preferred solution

If the dataset already provides separate fields, use:

- `batch["actor_rewards"]`
- `batch["actor_masks"]`

### If those fields do not exist

Then the dataset / relabeling code must be extended to provide them.

### Temporary fallback if dataset changes are not possible immediately

If adding `actor_rewards` and `actor_masks` is nontrivial, then **temporarily** enforce:

```python
g_pi = batch["value_goals"]
```

for both actor conditioning and actor weighting.

This loses the original actor-goal sampling scheme, but it is still mathematically consistent.

Do **not** keep the current mixed-goal design.

## Required actor loss structure

Use something like:

```python
obs = batch["observations"]
actions = batch["actions"]
g_pi = batch["actor_goals"]
actor_rewards = batch["actor_rewards"]
actor_masks = batch["actor_masks"].astype(jnp.float32)

sqrt_d = jnp.sqrt(self.config["latent_dim"])

# Critic-side quantities for the SAME goal used by the actor.
v_pi, z_s, u_pi = self.network.select("rep_value")(
    obs, g_pi, info=True, params=grad_params
)
x = self.network.select("successor")(obs, actions, params=grad_params)
q_pi = actor_rewards + actor_masks * jnp.sum(x * u_pi, axis=-1) / sqrt_d
adv = jax.lax.stop_gradient(q_pi - v_pi)

w = jnp.clip(
    jnp.exp(adv / self.config["beta_adv"]),
    0.0,
    self.config["max_weight"],
)

u_actor = jax.lax.stop_gradient(u_pi)
dist = self.network.select("actor")(
    obs, u_actor, goal_encoded=True, params=grad_params
)
log_prob = dist.log_prob(actions)
actor_loss = -(w * log_prob).mean()
```

## Why this is correct

AWR is a weighted imitation loss. The weight must answer:

> how good was this action for the same policy condition that I am training the actor to imitate?

If the goal changes between weighting and conditioning, the weight no longer has a decision-theoretic meaning.

This is especially harmful in long-horizon GCRL because the sign of the advantage is already fragile. Mixing goals makes that sign even noisier.

## Additional diagnostics to add

Add:

```python
"goal_match_check": jnp.mean(jnp.all(g_pi == batch["actor_goals"], axis=-1))
```

if shapes are comparable, or at least log whether actor-side rewards are present.

Also add:

```python
"actor_q_mean": q_pi.mean(),
"actor_v_mean": v_pi.mean(),
"actor_adv_mean": adv.mean(),
"actor_adv_pos_frac": jnp.mean(adv > 0),
```

These should replace the ambiguous current actor diagnostics.

---

## 5. Required correction #3: target update must use the new online params

## Problem

Current target update:

```python
def target_update(self, network, module_name):
    new_target_params = tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        self.network.params[f"modules_{module_name}"],
        self.network.params[f"modules_target_{module_name}"],
    )
    network.params[f"modules_target_{module_name}"] = new_target_params
```

The bug is that `self.network` is the **old** network state, while the method is called after `new_network` is returned from the optimizer step.

So the target network moves toward stale parameters.

## Correct behavior

Use `network.params[...]` on both sides of the EMA.

### Required replacement

```python
def target_update(self, network, module_name):
    """Polyak-average target network parameters using the updated online params."""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
        network.params[f"modules_{module_name}"],
        network.params[f"modules_target_{module_name}"],
    )
    network.params[f"modules_target_{module_name}"] = new_target_params
```

## Why this is correct

Polyak averaging should update:

\[
\theta_{\text{targ}} \leftarrow \tau \, \theta_{\text{online,new}} + (1-\tau) \, \theta_{\text{targ,old}}
\]

not

\[
\theta_{\text{targ}} \leftarrow \tau \, \theta_{\text{online,old}} + (1-\tau) \, \theta_{\text{targ,old}}
\]

Using stale online parameters slows and distorts bootstrap tracking.

---

## 6. Strongly recommended fix #4: explicit target parameter copy at initialization

## Problem

Current initialization:

```python
params = network_params
params["modules_target_rep_value"] = params["modules_rep_value"]
```

This can create aliasing / shared references.

## Required change

Use an explicit tree copy:

```python
params = network_params
params["modules_target_rep_value"] = copy.deepcopy(params["modules_rep_value"])
```

or equivalently a `tree_map(lambda x: x, ...)` clone if needed.

## Why this is recommended

It avoids subtle bugs where online and target params accidentally share the same underlying object.

---

## 7. Strongly recommended fix #5: make actor-side detachment explicit

Even if the current framework already prevents gradients from flowing into the critic through the actor-side advantage path, the code should make this explicit.

### Required style change

- Compute `v_pi`, `u_pi`, `x` using `params=grad_params`
- Immediately detach the scalar advantage before exponentiation
- Explicitly detach the goal embedding passed to the actor

Recommended pattern:

```python
v_pi, z_s, u_pi = self.network.select("rep_value")(
    obs, g_pi, info=True, params=grad_params
)
x = self.network.select("successor")(obs, actions, params=grad_params)
q_pi = actor_rewards + actor_masks * jnp.sum(x * u_pi, axis=-1) / sqrt_d
adv = jax.lax.stop_gradient(q_pi - v_pi)
u_actor = jax.lax.stop_gradient(u_pi)
```

This keeps the code honest: the actor gets a detached goal representation, and the actor weights are treated as fixed critic outputs.

---

## 8. Suggested final structure of `critic_loss`

Below is the target structure the implementation should approximate.

```python
def critic_loss(self, batch, grad_params):
    obs = batch["observations"]
    next_obs = batch["next_observations"]
    actions = batch["actions"]
    goals = batch["value_goals"]
    rewards = batch["rewards"]
    masks = batch["masks"].astype(jnp.float32)

    sqrt_d = jnp.sqrt(self.config["latent_dim"])
    discount = self.config["discount"]

    v, z_s, u_g = self.network.select("rep_value")(
        obs, goals, info=True, params=grad_params
    )
    x = self.network.select("successor")(obs, actions, params=grad_params)

    q = rewards + masks * jnp.sum(x * u_g, axis=-1) / sqrt_d

    _, z_next_bar, u_bar = self.network.select("target_rep_value")(
        next_obs, goals, info=True
    )
    v_next_bar = jnp.sum(z_next_bar * u_bar, axis=-1) / sqrt_d
    y = rewards + discount * masks * v_next_bar
    x_targ = discount * masks[..., None] * z_next_bar

    v_goal, _, _ = self.network.select("rep_value")(
        goals, goals, info=True, params=grad_params
    )

    loss_q = jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)
    loss_v = self.expectile_loss(
        jax.lax.stop_gradient(q) - v, self.config["expectile_tau"]
    ).mean()
    loss_succ = jnp.mean((x - jax.lax.stop_gradient(x_targ)) ** 2)
    loss_goal = jnp.mean(v_goal ** 2)
    loss_reg = jnp.mean(z_s ** 2 + u_g ** 2 + x ** 2)

    total = (
        self.config["lambda_Q"] * loss_q
        + self.config["lambda_V"] * loss_v
        + self.config["lambda_succ"] * loss_succ
        + self.config["lambda_goal"] * loss_goal
        + self.config["lambda_reg"] * loss_reg
    )

    info = {
        "loss_q": loss_q,
        "loss_v": loss_v,
        "loss_succ": loss_succ,
        "loss_goal": loss_goal,
        "loss_reg": loss_reg,
        "total_critic_loss": total,
        "q_mean": q.mean(),
        "v_mean": v.mean(),
        "v_goal_mean": v_goal.mean(),
        "adv_mean": (q - v).mean(),
        "adv_pos_frac": jnp.mean((q - v) > 0),
        "z_norm": jnp.mean(jnp.linalg.norm(z_s, axis=-1)),
        "u_norm": jnp.mean(jnp.linalg.norm(u_g, axis=-1)),
        "x_norm": jnp.mean(jnp.linalg.norm(x, axis=-1)),
        "succ_error": jnp.mean((x - x_targ) ** 2),
        "bellman_target_mean": y.mean(),
        "terminal_frac": jnp.mean(1.0 - masks),
    }
    return total, info
```

---

## 9. Suggested final structure of `actor_loss`

### Preferred version if actor-specific relabeling fields exist

```python
def actor_loss(self, batch, grad_params, rng=None):
    obs = batch["observations"]
    actions = batch["actions"]
    g_pi = batch["actor_goals"]
    actor_rewards = batch["actor_rewards"]
    actor_masks = batch["actor_masks"].astype(jnp.float32)

    sqrt_d = jnp.sqrt(self.config["latent_dim"])

    v_pi, z_s, u_pi = self.network.select("rep_value")(
        obs, g_pi, info=True, params=grad_params
    )
    x = self.network.select("successor")(obs, actions, params=grad_params)
    q_pi = actor_rewards + actor_masks * jnp.sum(x * u_pi, axis=-1) / sqrt_d
    adv = jax.lax.stop_gradient(q_pi - v_pi)

    w = jnp.clip(
        jnp.exp(adv / self.config["beta_adv"]),
        0.0,
        self.config["max_weight"],
    )

    u_actor = jax.lax.stop_gradient(u_pi)
    dist = self.network.select("actor")(
        obs, u_actor, goal_encoded=True, params=grad_params
    )
    log_prob = dist.log_prob(actions)

    actor_loss = -(w * log_prob).mean()

    info = {
        "actor_loss": actor_loss,
        "actor_q_mean": q_pi.mean(),
        "actor_v_mean": v_pi.mean(),
        "actor_adv_mean": adv.mean(),
        "actor_adv_pos_frac": jnp.mean(adv > 0),
        "w_mean": w.mean(),
        "w_clip_frac": jnp.mean(
            jnp.exp(adv / self.config["beta_adv"]) > self.config["max_weight"]
        ),
        "bc_log_prob": log_prob.mean(),
    }
    if not self.config["discrete"]:
        info["mse"] = jnp.mean((dist.mode() - actions) ** 2)
        info["std"] = jnp.mean(dist.scale_diag)

    return actor_loss, info
```

### Temporary fallback if actor-specific reward fields do not exist

Use `value_goals` everywhere inside `actor_loss` until dataset support is added:

```python
g_pi = batch["value_goals"]
actor_rewards = batch["rewards"]
actor_masks = batch["masks"].astype(jnp.float32)
```

This is less flexible than the intended design, but it is consistent.

---

## 10. Update step and target update

The update step may remain structurally similar, but the target update must consume the fresh network.

Recommended implementation:

```python
@jax.jit
def update(self, batch):
    new_rng, rng = jax.random.split(self.rng)

    def loss_fn(grad_params):
        return self.total_loss(batch, grad_params, rng=rng)

    new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
    self.target_update(new_network, "rep_value")

    return self.replace(network=new_network, rng=new_rng), info
```

This is fine once `target_update()` itself is corrected.

---

## 11. What this patch fixes conceptually

After the patch, the code will correctly implement the **one-step LAF** idea:

- `u(g)` is the dual goal covector
- `z(s)` is the latent state embedding
- `x(s,a)` is the discounted one-step latent successor
- `A(s,a,g)` scores local action quality by latent alignment with the goal covector

The main conceptual effect of the patch is that it restores **internal semantic consistency**:

1. terminal transitions terminate everywhere, not only in one branch of the critic;
2. the actor is weighted using the same conditional goal it is asked to imitate;
3. the target network tracks the actual updated critic.

Without these three properties, it is impossible to judge the true quality of LAF.

---

## 12. What this patch will NOT solve

This patch does **not** remove the deeper long-horizon limitation of one-step LAF.

Even a correct one-step LAF still learns a primitive local field:

\[
A(s,a,g) = r(s,a,g) + \delta_1(s,a)^\top u(g)
\]

with

\[
\delta_1(s,a) \approx \gamma \mathbb{E}[z(s') \mid s,a] - z(s).
\]

On environments like `antmaze-large-navigate-v0`, the decisive signal is often not a one-step local displacement, but a **temporally extended corridor / path decision**. That is why OTA benefits from temporal abstraction.

So the expected workflow is:

1. **First apply this patch** and verify one-step LAF is correct.
2. **Then evaluate** whether performance is still bottlenecked by long-horizon sign instability.
3. **If yes**, build the next method on top of corrected LAF: an option- or multi-step LAF.

---

## 13. Forward-looking extension after this patch: option-LAF, derived from current LAF

This section is **not** part of the required patch, but it explains the natural next step.

The one-step model uses

\[
A(s,a,g)=r(s,a,g)+(x(s,a)-z(s))^\top u(g).
\]

The temporally abstracted extension should use an option or multi-step displacement:

\[
A_k(s,o,g)=R_k(s,o,g)+(x_k(s,o)-z(s))^\top u(g)
\]

where

\[
x_k(s,o) \approx \gamma^k \mathbb{E}[z(s_{t+k}) \mid s_t=s, o].
\]

This keeps the dual-goal geometry of LAF, but upgrades the primitive one-step field into a temporally integrated field. That is the right direction if the corrected one-step implementation still trails OTA.

Do not implement this yet inside the current patch unless explicitly requested.

---

## 14. Acceptance checklist for Codex / Claude Code

The patch is complete only if all of the following are true:

- [ ] `critic_loss()` uses `masks` consistently in `q`, `y`, and `x_targ`
- [ ] `actor_loss()` uses the same goal for actor conditioning and advantage weighting
- [ ] actor-side rewards and masks are aligned with actor-side goals, or actor uses `value_goals` consistently as an explicit fallback
- [ ] `target_update()` uses `network.params[...]`, not `self.network.params[...]`
- [ ] target params are initialized by copy, not alias
- [ ] actor-side critic quantities are detached explicitly before weighting / conditioning
- [ ] diagnostics are updated to reflect actor-side and terminal-side consistency
- [ ] existing API / class names remain stable unless dataset support forces minimal additional fields

---

## 15. Minimal implementation preference order

If only a minimal patch is possible right now, do the following in this order:

1. Fix masked `q` and masked `x_targ`
2. Fix actor-goal / actor-advantage mismatch
3. Fix stale target update
4. Add explicit target copy on init
5. Add diagnostics

That is the minimum correction set needed before making any conclusions about the quality of LAF itself.
