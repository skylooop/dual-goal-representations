# Latent Advantage Fields for Goal-Conditioned RL (LAF-GCRL)

This document specifies a simple, scalable goal-conditioned RL algorithm that combines:

1. Dayan-style local policy improvement through advantages.
2. Dual goal representations with an inner-product value parameterization.
3. A goal-independent latent action field that predicts how an action moves the agent in the learned latent geometry.

The design goal is to keep the method easy to implement in an existing offline GCRL codebase while preserving the main theoretical intuition.

---

## 1. High-level idea

We represent the goal-conditioned value function as

```math
V(s, g) = z(s)^T u(g)
```

where:
- `z(s) in R^d` is a learned state embedding,
- `u(g) in R^d` is a learned goal embedding.

We also learn an action-conditioned latent successor head

```math
x(s, a) in R^d
```

that approximates the discounted one-step latent successor:

```math
x(s, a) ~= gamma * E[z(s') | s, a]
```

This gives the structured critic:

```math
Q(s, a, g) = r(s, a, g) + x(s, a)^T u(g)
```

and therefore the advantage:

```math
A(s, a, g) = Q(s, a, g) - V(s, g)
              = r(s, a, g) + (x(s, a) - z(s))^T u(g)
```

Interpretation:
- `u(g)` tells us which latent directions are useful for reaching goal `g`.
- `x(s, a) - z(s)` tells us which latent direction action `a` induces at state `s`.
- Their inner product measures how locally useful the action is for the goal.

This is the central object of the algorithm.

---

## 2. Core derivation

### 2.1 Continuous-time intuition

Suppose the raw dynamics are

```math
ds/dt = f(s, a)
```

and the local advantage is approximated by

```math
A(s, a, g) = r(s, a, g) + f(s, a)^T grad_s V(s, g)
```

If

```math
V(s, g) = z(s)^T u(g)
```

then

```math
grad_s V(s, g) = J_z(s)^T u(g)
```

where `J_z(s)` is the Jacobian of `z` with respect to the raw state.

Therefore

```math
A(s, a, g) = r(s, a, g) + (J_z(s) f(s, a))^T u(g)
```

So the goal embedding is not the raw-state gradient itself. Instead, it is the gradient in the learned latent coordinates `z`. The relevant local dynamics object becomes the latent velocity:

```math
v_z(s, a) = J_z(s) f(s, a)
```

This is what avoids the need to explicitly learn a conservative gradient field in raw state space.

### 2.2 Discrete-time exact factorization

In practical GCRL we work in discrete time. Assume again

```math
V(s, g) = z(s)^T u(g)
```

Then for one transition `(s, a, s')`:

```math
Q(s, a, g) = r(s, a, g) + gamma * E[V(s', g) | s, a]
           = r(s, a, g) + gamma * E[z(s') | s, a]^T u(g)
```

Define the latent successor vector

```math
x(s, a) := gamma * E[z(s') | s, a]
```

Then

```math
Q(s, a, g) = r(s, a, g) + x(s, a)^T u(g)
```

and

```math
A(s, a, g) = r(s, a, g) + (x(s, a) - z(s))^T u(g)
```

This factorization is exact under the bilinear value model.

---

## 3. What the algorithm should optimize

The algorithm should learn four things:

1. `z(s)`: state embedding for the structured value.
2. `u(g)`: goal embedding for the structured value.
3. `x(s, a)`: discounted latent successor for action-conditioned local improvement.
4. `pi(a | s, u(g))`: a policy conditioned on state and goal embedding.

Important implementation principle:
- The actor should receive the goal through `u(g)`.
- The actor should still receive the full state input (or a rich state encoder), not only `z(s)`.

Reason:
- `u(g)` should carry the goal semantics.
- The raw state may still contain local control information that a compressed value embedding `z(s)` does not preserve.

---

## 4. Environment and data assumptions

This method assumes an offline dataset of trajectories

```text
D = {(s_t, a_t, s_{t+1}, traj_id, t)}
```

The method is goal relabeling based. Goals are sampled from the dataset.

You need one project-specific function:

```text
goal_reached(obs, goal) -> bool
```

This is the only environment-specific semantic primitive.

Examples:
- exact equality for discrete state tasks,
- thresholded distance in selected coordinates for continuous control,
- latent-state equality if the project already exposes a controllable-state projection.

### Reward convention

Use the sparse `-1 / 0` goal-reaching reward:

```math
r(s, a, g) = 0    if goal_reached(s', g)
           = -1   otherwise
```

where `s'` is the next state after action `a`.

This convention is recommended because:
- it matches goal-reaching as discounted time minimization,
- it gives `V(g, g) = 0` for an absorbing goal state,
- it works naturally with the factorized advantage.

---

## 5. Model components

Implement the following modules.

### 5.1 State embedding network

```text
z_theta(s) -> R^d
```

Purpose:
- used only for the structured value `V(s, g) = z(s)^T u(g)`.

Recommended:
- MLP for low-dimensional states,
- CNN/ViT encoder plus projection head for images,
- output dimension `d = 128` or `256` by default.

### 5.2 Goal embedding network

```text
u_theta(g) -> R^d
```

Purpose:
- defines the dual goal representation used everywhere else.

Notes:
- It can share a visual backbone with the state encoder if the repo already supports this.
- It should have its own projection head even if a backbone is shared.

### 5.3 Latent successor network

```text
x_eta(s, a) -> R^d
```

Purpose:
- predicts the discounted one-step latent successor.

Interpretation:
- `x(s, a) - z(s)` is the action-conditioned latent displacement field.

Implementation:
- state encoder or state trunk,
- concatenate action,
- MLP head to `R^d`.

### 5.4 Policy network

```text
pi_omega(a | s, u(g))
```

Purpose:
- outputs the action distribution or deterministic action.

Recommended default for continuous control:
- Gaussian policy with tanh squashing,
- trained with advantage-weighted behavior cloning.

For discrete action spaces:
- either use a categorical actor,
- or skip the actor and act greedily with `argmax_a A(s, a, g)` if the action space is small.

### 5.5 Target networks

Maintain target copies of at least:

```text
z_bar, u_bar
```

Optional:
- target copy of `x` if the repo already uses target critics.

Use soft target updates.

---

## 6. Structured critic definitions

Using the online networks:

```math
V_theta(s, g) = z_theta(s)^T u_theta(g)
```

```math
Q_{theta, eta}(s, a, g) = r(s, a, g) + x_eta(s, a)^T u_theta(g)
```

```math
A_{theta, eta}(s, a, g) = r(s, a, g) + (x_eta(s, a) - z_theta(s))^T u_theta(g)
```

Using the target networks:

```math
V_bar(s, g) = z_bar(s)^T u_bar(g)
```

Bellman target:

```math
y = r(s, a, g) + gamma * V_bar(s', g)
```

Since `V(g, g)` is grounded to zero, this target also behaves correctly when the sampled goal is reached at `s'`.

---

## 7. Required losses

This section is the core implementation contract.

### 7.1 Q regression loss

Train the structured Q value using TD regression:

```math
L_Q = E[(Q(s, a, g) - y)^2]
```

or Huber loss if the codebase already uses it.

Expanded:

```math
Q(s, a, g) = r + x(s, a)^T u(g)
```

```math
y = r + gamma * z_bar(s')^T u_bar(g)
```

This loss is the main global return-learning objective.

### 7.2 Value loss (expectile / IQL style)

Use an expectile regression loss to fit the structured value `V` below `Q`:

```math
L_V = E[l_tau(Q(s, a, g) - V(s, g))]
```

where

```math
l_tau(delta) = |tau - 1[delta < 0]| * delta^2
```

Recommended default:
- `tau = 0.7`

Reason:
- this is the simplest stable offline value-learning option,
- it avoids explicit action maximization inside the critic,
- it matches the standard offline IQL design.

### 7.3 Latent successor regression loss

Directly supervise the latent successor head:

```math
L_succ = E[ || x(s, a) - gamma * stopgrad(z_bar(s')) ||_2^2 ]
```

This loss is crucial.

It makes `x(s, a)` explicitly behave like the discounted one-step latent successor, which gives the advantage a direct local geometry interpretation.

Without this loss, the factorization can still work through TD, but `x` may become less interpretable and harder to optimize.

### 7.4 Goal grounding loss

For the `-1 / 0` reward, the correct value at the goal is zero. Enforce this explicitly:

```math
L_goal = E[ V(g, g)^2 ]
```

Expanded:

```math
L_goal = E[ ( z(g)^T u(g) )^2 ]
```

This helps with:
- anchoring the value scale,
- stabilizing the factorization,
- improving correctness when `goal_reached(s', g)` is true.

### 7.5 Embedding regularization

Because `V = z^T u` has scale ambiguity, add a small norm penalty:

```math
L_reg = E[ ||z(s)||_2^2 + ||u(g)||_2^2 + ||x(s, a)||_2^2 ]
```

Recommended coefficient:
- small, e.g. `1e-4` to `1e-3`

Alternative if the repo already has a preferred stabilization method:
- weight decay,
- layer norm in hidden layers,
- output norm clipping.

### 7.6 Total critic loss

Use:

```math
L_critic = lambda_Q * L_Q
         + lambda_V * L_V
         + lambda_succ * L_succ
         + lambda_goal * L_goal
         + lambda_reg * L_reg
```

Recommended starting coefficients:

```text
lambda_Q    = 1.0
lambda_V    = 1.0
lambda_succ = 1.0
lambda_goal = 0.1
lambda_reg  = 1e-4
```

---

## 8. Policy objective

The default actor objective should be advantage-weighted behavior cloning.

### 8.1 Continuous action spaces

Compute a detached advantage on dataset actions:

```math
A_detach = stopgrad(Q(s, a, g) - V(s, g))
```

Turn it into weights:

```math
w = exp(A_detach / beta)
```

Clip the weights for stability:

```math
w = min(w, w_max)
```

Then optimize:

```math
L_pi = -E[ w * log pi_omega(a | s, u(g)) ]
```

Recommended defaults:

```text
beta  = 3.0
w_max = 100.0
```

Important gradient rule:
- `A_detach` must be detached.
- Actor gradients should not backpropagate into the critic by default.
- If doing two-stage training, actor gradients should also not update `u(g)`.

### 8.2 Discrete action spaces

If the action space is small and enumerable, the simplest implementation is:

```math
pi(s, g) = argmax_a A(s, a, g)
```

If the project requires a learned categorical actor, use:

```math
L_pi = -E[ sum_a softmax(A(s, a, g) / beta)_a * log pi(a | s, u(g)) ]
```

But the greedy policy is usually enough for small discrete domains.

---

## 9. Goal sampling / relabeling

The algorithm requires hindsight goal relabeling.

For each sampled transition `(s_t, a_t, s_{t+1})`, sample a goal `g` using a mixture of:

1. `current`: `g = s_t`
2. `future`: `g = s_{t+k}` from the same trajectory, optionally with geometric sampling over `k`
3. `random`: `g` from a random state in the dataset
4. optional `trajectory_terminal`: final state from the same trajectory

Recommended simple default mixture:

```text
p_current = 0.2
p_future  = 0.5
p_random  = 0.3
```

If the project already has an OGBench-style relabeler, reuse it.

Requirements for the replay buffer / dataset interface:
- access to trajectory ids,
- access to future states within the same trajectory,
- access to random states from the full dataset.

---

## 10. Training schedule

### 10.1 Recommended default: joint training with actor warmup

Use joint critic training from step 0, and delay actor updates for a short warmup period.

Recommended:

```text
critic_warmup_steps = 10000 to 50000
```

During warmup:
- train only `z`, `u`, `x` with the critic losses.

After warmup:
- continue critic training,
- start actor training with advantage-weighted BC.

### 10.2 Optional two-stage variant

If the repo strongly separates representation learning and policy learning, use:

Stage 1:
- train `z`, `u`, `x` with `L_critic` only.

Stage 2:
- freeze `u` and optionally freeze `z`,
- train actor with `L_pi`,
- optionally continue training `x` and `z` with a smaller critic learning rate.

This is slightly more stable but less elegant than end-to-end joint training.

---

## 11. One training step: exact implementation recipe

This section is meant to be directly translatable into code.

### Inputs

A minibatch must provide:

```text
s, a, s_next, traj_id, time_index
```

The goal sampler returns:

```text
g
```

### Forward pass

1. Compute reward:

```text
r = 0 if goal_reached(s_next, g) else -1
```

2. Online embeddings and successor:

```text
z      = z_theta(s)
u      = u_theta(g)
z_goal = z_theta(g)
x      = x_eta(s, a)
```

3. Structured values:

```text
v = dot(z, u)
q = r + dot(x, u)
a = q - v
```

4. Target values:

```text
with no_grad:
    z_next_t = z_bar(s_next)
    u_t      = u_bar(g)
    v_next_t = dot(z_next_t, u_t)
    y        = r + gamma * v_next_t
    x_targ   = gamma * z_next_t
```

### Losses

```text
loss_q    = mse_or_huber(q, y)
loss_v    = expectile_loss(q.detach() - v, tau)   # or expectile on (q - v) with q detached
loss_succ = mse(x, x_targ)
loss_goal = mse(dot(z_goal, u), 0)
loss_reg  = mean(||z||^2 + ||u||^2 + ||x||^2)
loss_critic = lambda_Q * loss_q + lambda_V * loss_v + lambda_succ * loss_succ + lambda_goal * loss_goal + lambda_reg * loss_reg
```

### Critic update

```text
opt_critic.zero_grad()
loss_critic.backward()
opt_critic.step()
```

### Actor update (after warmup)

```text
with no_grad:
    adv = q - v
    w = exp(adv / beta)
    w = clip(w, max=w_max)

loss_pi = -(w * log_prob_pi(a | s, u.detach_or_not)).mean()
```

Recommended default:
- detach `u(g)` for the actor if using two-stage training,
- allow actor to use online `u(g)` only if doing fully joint training and it is stable in the codebase.

### Target update

```text
for each target param:
    target = (1 - tau_target) * target + tau_target * online
```

Recommended:

```text
tau_target = 0.005
```

---

## 12. Network interface contract for integration into an existing repo

This document is repo-agnostic. If the project already has actor/critic/trainer abstractions, map the method to the existing interfaces rather than changing the whole codebase.

Minimum required abstractions:

### 12.1 Goal sampler

```text
sample_goals(batch, dataset) -> goals
```

### 12.2 Critic module

The critic module should expose:

```text
encode_state(s) -> z
encode_goal(g) -> u
predict_successor(s, a) -> x
value(s, g) -> v
qvalue(s, a, g, reward=None) -> q
advantage(s, a, g, reward=None) -> a
```

Recommended implementation detail:
- `value`, `qvalue`, and `advantage` should be thin wrappers around `z`, `u`, `x`.
- Avoid storing redundant heads if the values are derived from the factorization.

### 12.3 Actor module

The actor should expose:

```text
log_prob(a, s, goal_embedding)
act(s, goal_embedding, deterministic=False)
```

### 12.4 Trainer

The trainer should expose:

```text
update_critic(batch)
update_actor(batch)
update_targets()
```

---

## 13. Recommended defaults

These are conservative starting points, not tuned optima.

### Global

```text
gamma = 0.99
batch_size = 1024
learning_rate = 3e-4
embedding_dim = 256
tau_target = 0.005
critic_warmup_steps = 20000
```

### Losses

```text
expectile_tau = 0.7
beta_adv = 3.0
max_weight = 100.0
lambda_Q = 1.0
lambda_V = 1.0
lambda_succ = 1.0
lambda_goal = 0.1
lambda_reg = 1e-4
```

### MLPs for state-based tasks

```text
hidden_sizes = [512, 512, 512]
activation = GELU or ReLU
layer_norm = optional but recommended in critic networks
```

### Images / pixels

Use the repo's standard visual encoder.

Important rule:
- goal input to the actor should be `u(g)` rather than raw early-fused goal pixels.
- if the repo relies heavily on early fusion, this method will need a late-fusion path for the policy and critic.

---

## 14. Why this method is simple and scalable

1. Goal dependence is always linear through `u(g)`.
2. Action dependence is isolated in a single local head `x(s, a)`.
3. The critic uses standard offline RL losses: TD regression plus expectile value fitting.
4. The actor uses standard advantage-weighted BC.
5. The extra auxiliary loss `L_succ` is only a one-step latent regression target.
6. No raw-state Jacobians, curl constraints, or explicit conservative field estimation are required.

Complexity relative to a normal goal-conditioned IQL implementation:
- replace monolithic `V(s, g)` with `z(s)^T u(g)`,
- replace monolithic `Q(s, a, g)` with `r + x(s, a)^T u(g)`,
- add one auxiliary successor regression loss.

---

## 15. Important implementation notes and pitfalls

### 15.1 Do not force the actor to use only `z(s)`

Use raw state features for the actor, plus `u(g)` as the goal input.

Reason:
- `z(s)` is for value geometry.
- the actor may need additional control-relevant state details.

### 15.2 Keep reward convention consistent everywhere

If reward is computed from `s_next`, then all Bellman targets and diagnostics must use the same convention.

### 15.3 Use goal grounding

Without `L_goal`, the value geometry is less anchored and training can drift.

### 15.4 Detach advantage weights in actor training

Never let the actor optimize the critic through the exponential weights.

### 15.5 Keep `L_succ`

This is the loss that makes the method truly local and Dayan-like in latent space.

### 15.6 Use a proper goal predicate

In continuous tasks, exact equality is usually wrong. The project must provide a correct success function.

---

## 16. Minimal ablations to include

For research code, include the following ablations if possible:

1. Full method.
2. No `L_succ`.
3. No factorization, replace with monolithic Q/V.
4. Actor conditioned on raw goal instead of `u(g)`.
5. Freeze vs jointly train `u(g)` during actor learning.
6. Different goal sampling mixtures.

This will make it easy to verify which piece is actually responsible for the gains.

---

## 17. Diagnostics and sanity checks

The following metrics should be logged.

### Critic / geometry

```text
mean(V(g, g)^2)
mean(||x - gamma * z_next_target||^2)
mean(Q - target)^2
mean(expectile residual)
mean(||z||), mean(||u||), mean(||x||)
```

### Advantage quality

```text
mean(A on dataset actions)
percent of positive advantages
correlation between A and empirical future goal reachability, if available
```

### Actor

```text
mean(exp(adv / beta)) before clipping
fraction of weights clipped
behavior cloning log-prob
success rate / normalized return
```

### Sanity checks

At convergence or during debugging, check:
- `V(g, g)` is near zero,
- `q - r` is close to `gamma * V_next_target`,
- `x` predicts `gamma * z_next` well,
- actions that actually move toward the goal have higher advantage than random actions.

---

## 18. Short pseudocode

```text
initialize z_theta, u_theta, x_eta, pi_omega
initialize target networks z_bar <- z_theta, u_bar <- u_theta

for step in training_steps:
    batch = replay.sample(batch_size)
    g = goal_sampler.sample(batch, replay)

    r = reward_from_next_state(batch.s_next, g)

    z = z_theta(batch.s)
    u = u_theta(g)
    x = x_eta(batch.s, batch.a)
    v = dot(z, u)
    q = r + dot(x, u)

    with no_grad:
        z_next_t = z_bar(batch.s_next)
        u_t = u_bar(g)
        y = r + gamma * dot(z_next_t, u_t)
        x_targ = gamma * z_next_t

    loss_q = mse(q, y)
    loss_v = expectile_loss(q.detach() - v, tau)
    loss_succ = mse(x, x_targ)
    loss_goal = mse(dot(z_theta(g), u_theta(g)), 0)
    loss_reg = mean(||z||^2 + ||u||^2 + ||x||^2)
    loss_critic = lambda_Q*loss_q + lambda_V*loss_v + lambda_succ*loss_succ + lambda_goal*loss_goal + lambda_reg*loss_reg

    update critic params

    if step >= critic_warmup_steps:
        with no_grad:
            adv = q - v
            w = clip(exp(adv / beta_adv), max_weight)
        loss_pi = -(w * log_prob_pi(batch.a | batch.s, u_theta(g))).mean()
        update actor params

    soft update z_bar, u_bar
```

---

## 19. Summary in one paragraph

LAF-GCRL learns a bilinear goal-conditioned value `V(s, g) = z(s)^T u(g)` together with an action-conditioned discounted latent successor `x(s, a) ~= gamma E[z(s') | s, a]`. This yields a structured critic `Q(s, a, g) = r + x(s, a)^T u(g)` and an explicit latent advantage `A(s, a, g) = r + (x(s, a) - z(s))^T u(g)`. The critic is trained with TD regression, expectile value fitting, latent successor regression, and goal grounding. The actor is trained with advantage-weighted behavior cloning using the detached structured advantage. The result is a simple offline GCRL algorithm in which action choice is driven by local alignment between a goal covector `u(g)` and an action-induced latent displacement `x(s, a) - z(s)`.

