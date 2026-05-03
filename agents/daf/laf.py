"""Latent Advantage Fields for Goal-Conditioned RL (LAF-GCRL).

Implements the algorithm described in agents/daf/laf_gcrl_spec.md.

Key idea:
    V(s, g) = phi(s)^T psi(g)   (bilinear value via GCBilinearRepresentationValue)
    Q(s, a, g) = r + x(s, a)^T psi(g)
    A(s, a, g) = r + (x(s, a) - phi(s))^T psi(g)

where phi = z (state embedding), psi = u (goal embedding), and x is the
discounted latent successor head trained to predict gamma * E[phi(s') | s, a].
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.dual import GCBilinearRepresentationValue
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, StateRepresentation


class LAFGCRLAgent(flax.struct.PyTreeNode):
    """Latent Advantage Fields for Goal-Conditioned RL (LAF-GCRL) agent.

    Learns a bilinear value V(s,g) = phi(s)^T psi(g) together with a latent
    successor x(s,a) ≈ gamma * E[phi(s') | s, a].  The structured advantage
    A(s,a,g) = r + (x(s,a) - phi(s))^T psi(g) drives advantage-weighted BC.

    Modules:
        rep_value        : GCBilinearRepresentationValue — V(s,g) = phi(s)^T psi(g)
        target_rep_value : EMA copy of rep_value for Bellman targets
        successor        : StateRepresentation([s, a]) -> R^d  (latent successor x)
        actor            : GCActor(s, psi(g)) — policy conditioned on goal embedding
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def expectile_loss(diff, expectile):
        """Asymmetric L2 (expectile) loss.

        l_tau(delta) = |tau - 1[delta < 0]| * delta^2
        """
        weight = jnp.where(diff >= 0, expectile, 1.0 - expectile)
        return weight * (diff**2)

    # ------------------------------------------------------------------
    # Per-subsystem losses
    # ------------------------------------------------------------------

    def critic_loss(self, batch, grad_params):
        """Compute the combined critic loss (L_Q + L_V + L_succ + L_goal + L_reg).

        All five terms from the spec §7 are included.

        Fix #1 (termination consistency): masks is applied to q, y, and x_targ
        so that terminal transitions do not contribute continuation value anywhere.
        """
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        actions = batch["actions"]
        goals = batch["value_goals"]
        rewards = batch["rewards"]
        # Cast masks to float32 explicitly; dataset may store as int.
        masks = batch["masks"].astype(jnp.float32)

        # Normalization factor matching GCBilinearValue's internal /sqrt(d).
        sqrt_d = jnp.sqrt(self.config["latent_dim"])
        discount = self.config["discount"]

        # --- Online forward pass ---
        # rep_value with info=True returns (v_scalar, phi_s, psi_g).
        # v is already divided by sqrt(d) internally.
        v, z_s, u_g = self.network.select("rep_value")(
            obs, goals, info=True, params=grad_params
        )

        # Latent successor x(s, a), shape (B, d).
        x = self.network.select("successor")(obs, actions, params=grad_params)

        # Structured Q: r + masks * x^T u / sqrt(d).
        # masks=0 on terminal transitions so continuation collapses to reward only.
        q = rewards + masks * jnp.sum(x * u_g, axis=-1) / sqrt_d

        # --- Target forward pass (no gradient) ---
        _, z_next_bar, u_bar = self.network.select("target_rep_value")(
            next_obs, goals, info=True
        )
        # Bellman target: y = r + gamma * masks * V_bar(s', g)
        v_next_bar = jnp.sum(z_next_bar * u_bar, axis=-1) / sqrt_d
        y = rewards + discount * masks * v_next_bar

        # Successor regression target: gamma * masks * phi_bar(s').
        # masks[..., None] broadcasts over the latent dimension.
        x_targ = discount * masks[..., None] * z_next_bar

        # --- Goal grounding: V(g, g) = phi(g)^T psi(g) / sqrt(d) should be 0 ---
        v_goal, _, _ = self.network.select("rep_value")(
            goals, goals, info=True, params=grad_params
        )

        # --- Individual losses ---
        loss_q = jnp.mean((q - jax.lax.stop_gradient(y)) ** 2)

        # Expectile loss on (q_detach - v): fits V below Q.
        loss_v = self.expectile_loss(
            jax.lax.stop_gradient(q) - v, self.config["expectile_tau"]
        ).mean()

        # Latent successor regression.
        loss_succ = jnp.mean((x - jax.lax.stop_gradient(x_targ)) ** 2)

        # Goal grounding.
        loss_goal = jnp.mean(v_goal**2)

        # Embedding norm regularization (scale ambiguity in bilinear model).
        loss_reg = jnp.mean(z_s**2 + u_g**2 + x**2)

        total = (
            self.config["lambda_Q"] * loss_q
            + self.config["lambda_V"] * loss_v
            + self.config["lambda_succ"] * loss_succ
            + self.config["lambda_goal"] * loss_goal
            + self.config["lambda_reg"] * loss_reg
        )

        adv = q - v
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
            "adv_mean": adv.mean(),
            "adv_pos_frac": jnp.mean(adv > 0),
            "z_norm": jnp.mean(jnp.linalg.norm(z_s, axis=-1)),
            "u_norm": jnp.mean(jnp.linalg.norm(u_g, axis=-1)),
            "x_norm": jnp.mean(jnp.linalg.norm(x, axis=-1)),
            "succ_error": jnp.mean((x - x_targ) ** 2),
            "bellman_target_mean": y.mean(),
            # Terminal-consistency diagnostics.
            "terminal_frac": jnp.mean(1.0 - masks),
        }
        return total, info

    def actor_loss(self, batch, grad_params, rng=None):
        """Advantage-weighted behavior cloning (AWR) actor loss.

        Fix #2 (goal consistency): actor conditioning and advantage weighting both
        use the same goal g_pi = value_goals.  Dataset does not provide separate
        actor_rewards / actor_masks, so we use the value-goal stream as the
        consistent fallback (patch spec §9 temporary fallback).

        Fix #5 (explicit detach): u_pi is stop_gradient'd before being passed to
        the actor so actor gradients cannot flow back into the critic.
        """
        obs = batch["observations"]
        actions = batch["actions"]
        # Use value_goals for both conditioning and weighting (consistent fallback).
        g_pi = batch["value_goals"]
        rewards = batch["rewards"]
        masks = batch["masks"].astype(jnp.float32)

        sqrt_d = jnp.sqrt(self.config["latent_dim"])

        # Critic-side quantities for the same goal used by the actor.
        # Use grad_params so gradients flow through the actor path correctly,
        # then detach the advantage scalar before weighting.
        v_pi, z_s, u_pi = self.network.select("rep_value")(
            obs, g_pi, info=True, params=grad_params
        )
        x = self.network.select("successor")(obs, actions, params=grad_params)
        q_pi = rewards + masks * jnp.sum(x * u_pi, axis=-1) / sqrt_d
        adv = jax.lax.stop_gradient(q_pi - v_pi)

        # Advantage weights: w = clip(exp(A / beta), max_weight).
        w = jnp.clip(
            jnp.exp(adv / self.config["beta_adv"]), 0.0, self.config["max_weight"]
        )

        # Detach goal embedding before passing to actor so actor gradients
        # do not flow back into the critic through u_pi.
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

    # ------------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss.

        Actor loss is gated by critic_warmup_steps: during warmup only the
        critic (rep_value + successor) is trained.  We always compute the actor
        loss for a consistent pytree structure, but multiply by a 0/1 mask so
        no gradients flow into the actor parameters during warmup.
        """
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        # Zero out actor loss during warmup so critic trains alone.
        past_warmup = jnp.array(
            self.network.step > self.config["critic_warmup_steps"], dtype=jnp.float32
        )
        loss = critic_loss + past_warmup * actor_loss
        return loss, info

    # ------------------------------------------------------------------
    # Target update
    # ------------------------------------------------------------------

    def target_update(self, network, module_name):
        """Polyak-average target network parameters using the freshly updated online params.

        Fix #3: use network.params (new) not self.network.params (old) on the online side.
        EMA: theta_targ <- tau * theta_online_new + (1 - tau) * theta_targ_old
        """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1.0 - self.config["tau"]),
            network.params[f"modules_{module_name}"],  # new online params
            network.params[f"modules_target_{module_name}"],  # old target params
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    @jax.jit
    def update(self, batch):
        """Gradient update and target network soft update."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # Soft-update target_rep_value <- rep_value.
        self.target_update(new_network, "rep_value")

        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from the actor.

        The actor receives the raw observation and the goal embedding psi(goals).
        """
        # Encode goal: psi(g) via goal-only call to rep_value.
        u_g = self.network.select("rep_value")(goals)
        dist = self.network.select("actor")(
            observations, u_g, goal_encoded=True, temperature=temperature
        )
        actions = dist.sample(seed=seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        """Instantiate the agent.

        Args:
            seed: Integer random seed.
            ex_observations: Example observation batch, shape (1, obs_dim).
            ex_actions: Example action batch. For discrete MDPs this should
                contain the maximum action index so action_dim can be inferred.
            config: ConfigDict from get_config().
            ex_goals: Unused; kept for API compatibility.
        """
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng)

        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        latent_dim = config["latent_dim"]

        # --- rep_value: V(s,g) = phi(s)^T psi(g) ---
        # GCBilinearRepresentationValue with ret_mean=True so that info=True
        # returns (v, phi_s, psi_g) with phi_s and psi_g averaged over the
        # ensemble, giving shape (B, d) tensors we can use directly.
        rep_value_def = GCBilinearRepresentationValue(
            hidden_dims=config["hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
            ensemble=True,  # 2-ensemble for stable value learning
        )

        # --- successor: x(s, a) ≈ gamma * E[phi(s') | s, a] ---
        # StateRepresentation concatenates [obs, actions] internally when
        # actions is provided, then projects to latent_dim.
        # No ensemble: x is a direct regression target.
        successor_def = StateRepresentation(
            hidden_dims=config["hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
            ensemble=False,
        )

        # --- actor: pi(a | s, psi(g)) ---
        # goal_encoded=True in __call__ means GCActor concatenates [s, psi(g)]
        # without running psi(g) through any additional encoder.
        ex_goal_rep = jnp.zeros((1, latent_dim))  # placeholder for psi(g)

        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=config["hidden_dims"],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config["hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
            )

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            target_rep_value=(
                copy.deepcopy(rep_value_def),
                (ex_observations, ex_observations),
            ),
            successor=(successor_def, (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goal_rep)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Fix #4: explicit deep copy to avoid aliasing between online and target params.
        params = network_params
        params["modules_target_rep_value"] = copy.deepcopy(params["modules_rep_value"])

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent identity.
            agent_name="laf_gcrl",
            # Optimiser.
            lr=3e-4,
            batch_size=1024,
            # Architecture.
            hidden_dims=(512, 512, 512),
            latent_dim=256,  # d — embedding dimension for phi, psi, x
            layer_norm=True,
            # RL.
            discount=0.99,
            tau=0.005,  # EMA rate for target_rep_value
            # Critic loss weights (spec §7.6).
            expectile_tau=0.7,  # tau in expectile loss L_V
            lambda_Q=1.0,
            lambda_V=1.0,
            lambda_succ=1.0,
            lambda_goal=0.1,
            lambda_reg=1e-4,
            # Actor loss (spec §8).
            beta_adv=3.0,  # temperature for advantage weighting
            max_weight=100.0,  # clip for exp(A / beta)
            # Training schedule.
            critic_warmup_steps=20000,  # steps before actor training starts
            # Actor architecture.
            const_std=True,
            discrete=False,
            # Dataset / goal sampling (GCDataset standard keys).
            dataset_class="GCDataset",
            oraclerep=False,
            norm=False,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,  # reward = 0 if goal_reached else -1
            p_aug=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
