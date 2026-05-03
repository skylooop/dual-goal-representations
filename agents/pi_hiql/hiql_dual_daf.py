"""HIQL + Dual Advantage Fields (DAF).

Hierarchical IVL with dual bilinear rep_value ψ(g) and DAF for
the low-level actor.

Architecture:
  1. rep_loss            — IQL on bilinear rep_value + rep_critic
  2. afu_loss            — AFU joint V + A with conditional gradient scaling
  3. action_effect_loss  — MSE u(s,a) ≈ γφ(s')−φ(s)  (set weight=0 to disable)
  4. value_loss          — IVL expectile on V(s, ψ(g))  (needed for high actor ΔV)
  5. low_actor_loss      — AWR weighted by dual advantage A = u^T ψ(g)/√d
  6. high_actor_loss     — AWR on subgoal prediction, weighted by ΔV from value
  7. psi_adv_loss        — regress ``u(s,a)^T ψ(g) / √d`` onto ``Q − V`` from
                           ``target_value`` (signed; trains both u and ψ;
                           set ``psi_adv_weight=0`` to disable)

State-only (encoder=None).
"""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCDiscreteActor, GCValue, LengthNormalize, MLP


class HIQLDualDAFAgent(flax.struct.PyTreeNode):
    """Hierarchical IVL with dual ψ and dual advantage fields."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _psi_for_downstream(self, goals):
        """ψ(g) for downstream nets: raw ``rep_value`` output or length-normalized.

        When ``psi_downstream_length_normalize`` is True (default), same as
        ``LengthNormalize`` in ``utils/networks.py`` (with eps).
        """
        psi = self.network.select("rep_value")(goals)
        if not self.config.get("psi_downstream_length_normalize", True):
            return psi
        return psi / (
            jnp.linalg.norm(psi, axis=-1, keepdims=True) + 1e-8
        ) * jnp.sqrt(psi.shape[-1])

    def _raw_dual_advantage(self, observations, actions, goal_reps, params=None):
        """Raw signed bilinear advantage ``u(s,a)^T ψ(g) / √d`` (proposal Eq. 4)."""
        sa_input = jnp.concatenate([observations, actions], axis=-1)
        if params is not None:
            u = self.network.select("action_effect")(sa_input, params=params)
        else:
            u = self.network.select("action_effect")(sa_input)
        D = goal_reps.shape[-1]
        return (u * goal_reps).sum(axis=-1) / jnp.sqrt(D)

    def _dual_advantage(self, observations, actions, goal_reps, params=None):
        """Softplus-capped dual advantage (≤ 0). Required by ``afu_loss``."""
        raw_adv = self._raw_dual_advantage(
            observations, actions, goal_reps, params=params
        )
        return -jax.nn.softplus(-raw_adv)

    def _apply_actor_transform(self, raw_adv):
        """Shape a signed advantage signal for actors.

        ``actor_advantage_transform`` in ``{"softplus", "raw"}``:
          * ``"softplus"`` — cap by ``-softplus(-z)`` (≤ 0). With
            ``exp(alpha*.)`` yields ``sigma(z)^alpha`` AWR weights.
          * ``"raw"``      — use signed ``z`` directly. With
            ``exp(alpha*.)`` yields the proposal's ``exp(alpha*z)`` AWR.
        """
        transform = self.config.get("actor_advantage_transform", "softplus")
        if transform == "softplus":
            return -jax.nn.softplus(-raw_adv)
        elif transform == "raw":
            return raw_adv
        raise ValueError(
            f"Unknown actor_advantage_transform={transform!r}; "
            "expected 'softplus' or 'raw'."
        )

    def _apply_awr_weight(self, adv, alpha):
        """Turn a shaped advantage into AWR weights.

        ``actor_awr_weight`` in ``{"exp", "linear_clip"}``:
          * ``"exp"``        — ``min(exp(alpha*adv), 100)``.
          * ``"linear_clip"``— ``max(alpha*adv, 0)``.
        """
        awr_mode = self.config.get("actor_awr_weight", "exp")
        if awr_mode == "exp":
            return jnp.minimum(jnp.exp(adv * alpha), 100.0)
        elif awr_mode == "linear_clip":
            return jnp.maximum(adv * alpha, 0.0)
        raise ValueError(
            f"Unknown actor_awr_weight={awr_mode!r}; "
            "expected 'exp' or 'linear_clip'."
        )

    def _goal_reps_for_actor(self, observations, goals, params=None):
        """Goal conditioning for actors: raw ψ or goal_rep([s; ψ])."""
        if self.config.get("goal_rep_for_actor", False):
            psi = self._psi_for_downstream(goals)
            gr_in = jnp.concatenate([observations, psi], axis=-1)
            if params is not None:
                return self.network.select("goal_rep")(gr_in, params=params)
            return self.network.select("goal_rep")(gr_in)
        return self._psi_for_downstream(goals)

    def _goal_reps_for_value(self, observations, goals):
        """Goal conditioning for value nets: raw ψ or goal_rep([s; ψ])."""
        if self.config.get("goal_rep_for_value", False):
            psi = self._psi_for_downstream(goals)
            gr_in = jnp.concatenate([observations, psi], axis=-1)
            return self.network.select("goal_rep")(gr_in)
        return self._psi_for_downstream(goals)

    # ------------------------------------------------------------------
    # Loss 1: IQL on dual representation
    # ------------------------------------------------------------------

    def rep_loss(self, batch, grad_params):
        discount = self.config["discount"]

        q_all_t = self.network.select("target_rep_critic")(
            batch["observations"], batch["value_goals"], batch["actions"]
        )
        if q_all_t.ndim == 1:
            q_all_t = q_all_t[None, ...]
        q_t = jnp.min(q_all_t, axis=0)

        v, phi, psi = self.network.select("rep_value")(
            batch["observations"], batch["value_goals"],
            info=True, params=grad_params,
        )
        value_loss = self.expectile_loss(
            q_t - v, q_t - v, self.config["rep_expectile"]
        ).mean()

        reg_w = float(self.config.get("rep_norm_reg", 0.01))
        rep_reg = reg_w * (jnp.mean(phi ** 2) + jnp.mean(psi ** 2))

        next_v = self.network.select("rep_value")(
            batch["next_observations"], batch["value_goals"]
        )
        q_base = batch["rewards"] + discount * batch["masks"] * next_v

        q_all = self.network.select("rep_critic")(
            batch["observations"], batch["value_goals"], batch["actions"],
            params=grad_params,
        )
        if q_all.ndim == 1:
            q_all = q_all[None, ...]

        critic_loss = ((q_all - q_base[None, ...]) ** 2).mean()

        return value_loss + critic_loss + rep_reg, {
            "rep_value_loss": value_loss,
            "rep_critic_loss": critic_loss,
            "rep_reg": rep_reg,
            "rep_v_mean": v.mean(),
            "rep_q_mean": q_base.mean(),
            "phi_norm": jnp.linalg.norm(phi, axis=-1).mean(),
            "psi_norm": jnp.linalg.norm(psi, axis=-1).mean(),
        }

    # ------------------------------------------------------------------
    # Loss 2: AFU joint V + A
    # ------------------------------------------------------------------

    def afu_loss(self, batch, grad_params):
        next_v = self.network.select("target_rep_value")(
            batch["next_observations"], batch["value_goals"]
        )
        bellman_target = jax.lax.stop_gradient(
            batch["rewards"] + self.config["discount"] * batch["masks"] * next_v
        )

        v_pred = self.network.select("rep_value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )

        goal_reps = jax.lax.stop_gradient(
            self.network.select("rep_value")(batch["value_goals"])
        )
        a_pred = self._dual_advantage(
            batch["observations"], batch["actions"], goal_reps, params=grad_params
        )

        rho = self.config["rho"]
        underestimate = (v_pred + a_pred < bellman_target).astype(v_pred.dtype)
        v_scaled = (
            (1.0 - rho * underestimate) * v_pred
            + rho * underestimate * jax.lax.stop_gradient(v_pred)
        )

        x = v_scaled - bellman_target
        y = a_pred
        overestimate = (x >= 0).astype(x.dtype)
        loss_per_sample = (
            overestimate * (x + y) ** 2
            + (1.0 - overestimate) * (x ** 2 + y ** 2)
        )
        afu_loss_val = loss_per_sample.mean()

        return self.config["afu_weight"] * afu_loss_val, {
            "afu_loss": afu_loss_val,
            "v_pred_mean": v_pred.mean(),
            "a_pred_mean": a_pred.mean(),
            "a_pred_min": a_pred.min(),
            "bellman_target_mean": bellman_target.mean(),
            "frac_underestimate": underestimate.mean(),
        }

    # ------------------------------------------------------------------
    # Loss 3: Action-effect regression (optional, set weight=0 to disable)
    # ------------------------------------------------------------------

    def action_effect_loss(self, batch, grad_params):
        w = float(self.config.get("action_effect_weight", 0.0))
        if w <= 0.0:
            z = jnp.array(0.0)
            return z, {"ae_loss": z}

        _, phi_s, _ = self.network.select("rep_value")(
            batch["observations"], batch["value_goals"], info=True
        )
        _, phi_s_next, _ = self.network.select("rep_value")(
            batch["next_observations"], batch["value_goals"], info=True
        )
        target = jax.lax.stop_gradient(
            self.config["discount"] * phi_s_next - phi_s
        )

        sa_input = jnp.concatenate(
            [batch["observations"], batch["actions"]], axis=-1
        )
        u_pred = self.network.select("action_effect")(
            sa_input, params=grad_params
        )

        loss = jnp.mean((u_pred - target) ** 2)
        return w * loss, {
            "ae_loss": loss,
            "u_norm": jnp.linalg.norm(u_pred, axis=-1).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
        }

    # ------------------------------------------------------------------
    # Loss 4: IVL on downstream V(s, ψ(g)) — provides high-actor advantage
    # ------------------------------------------------------------------

    def value_loss(self, batch, grad_params):
        goal_reps = self._goal_reps_for_value(
            batch["observations"], batch["value_goals"]
        )
        discount = self.config["discount"]

        next_v_all_t = self.network.select("target_value")(
            batch["next_observations"], goal_reps
        )
        if next_v_all_t.ndim == 1:
            next_v_all_t = next_v_all_t[None, ...]
        v_all_t = self.network.select("target_value")(
            batch["observations"], goal_reps
        )
        if v_all_t.ndim == 1:
            v_all_t = v_all_t[None, ...]

        v_all = self.network.select("value")(
            batch["observations"], goal_reps, params=grad_params
        )
        if v_all.ndim == 1:
            v_all = v_all[None, ...]

        next_v_t = jnp.min(next_v_all_t, axis=0)
        q = batch["rewards"] + discount * batch["masks"] * next_v_t

        v_t = jnp.mean(v_all_t, axis=0)
        adv = q - v_t

        q_all = (
            batch["rewards"][None, ...]
            + discount * batch["masks"][None, ...] * next_v_all_t
        )
        diff_all = q_all - v_all

        vl = self.expectile_loss(
            adv[None, ...], diff_all, self.config["expectile"]
        ).mean()

        v = jnp.mean(v_all, axis=0)
        return vl, {
            "value_loss": vl,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    # ------------------------------------------------------------------
    # Loss 5: Low actor — AWR with dual advantage
    # ------------------------------------------------------------------

    def low_actor_loss(self, batch, grad_params):
        if self.config.get("actor_td_advantage", False):
            v_next = self.network.select("rep_value")(
                batch["next_observations"], batch["low_actor_goals"]
            )
            v_curr = self.network.select("rep_value")(
                batch["observations"], batch["low_actor_goals"]
            )
            raw_adv = v_next - v_curr
        else:
            psi_adv = self._psi_for_downstream(batch["low_actor_goals"])
            raw_adv = self._raw_dual_advantage(
                batch["observations"], batch["actions"], psi_adv
            )

        adv = self._apply_actor_transform(raw_adv)
        exp_a = self._apply_awr_weight(adv, self.config["low_alpha"])

        goal_reps = self._goal_reps_for_actor(
            batch["observations"], batch["low_actor_goals"], params=grad_params
        )
        if not self.config.get("low_actor_rep_grad", False):
            goal_reps = jax.lax.stop_gradient(goal_reps)

        dist = self.network.select("low_actor")(
            batch["observations"], goal_reps, params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_prob).mean()

        info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "raw_adv_mean": raw_adv.mean(),
            "awr_weight_mean": exp_a.mean(),
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            info.update({
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            })
        return actor_loss, info

    # ------------------------------------------------------------------
    # Loss 6: High actor — subgoal prediction, ΔV from value network
    # ------------------------------------------------------------------

    def high_actor_loss(self, batch, grad_params):
        if self.config.get("goal_rep_for_actor", False):
            psi_t = self._psi_for_downstream(batch["high_actor_targets"])
            psi_targets = jax.lax.stop_gradient(
                self.network.select("goal_rep")(
                    jnp.concatenate([batch["observations"], psi_t], axis=-1)
                )
            )
        else:
            psi_targets = jax.lax.stop_gradient(
                self._psi_for_downstream(batch["high_actor_targets"])
            )

        psi_goals = self._goal_reps_for_value(
            batch["observations"], batch["high_actor_goals"]
        )
        v_all = self.network.select("value")(
            batch["observations"], psi_goals
        )
        nv_all = self.network.select("value")(
            batch["high_actor_targets"], psi_goals
        )
        if v_all.ndim == 1:
            v_all = v_all[None, ...]
        if nv_all.ndim == 1:
            nv_all = nv_all[None, ...]
        v = jnp.mean(v_all, axis=0)
        nv = jnp.mean(nv_all, axis=0)
        raw_adv = nv - v

        adv = self._apply_actor_transform(raw_adv)
        exp_a = self._apply_awr_weight(adv, self.config["high_alpha"])

        dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        log_prob = dist.log_prob(psi_targets)
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "raw_adv_mean": raw_adv.mean(),
            "awr_weight_mean": exp_a.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - psi_targets) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    # ------------------------------------------------------------------
    # Loss 7: bilinear advantage distilled from scalar Q − V.
    # ------------------------------------------------------------------

    def psi_adv_loss(self, batch, grad_params):
        """Regress ``u(s,a)^T ψ(g) / √d`` onto signed ``Q − V``.

        Mirrors the distillation loss in ``gcivl_daf_psi_adv``: the bilinear
        advantage is pushed to match the scalar advantage read off from the
        downstream value network. Both ``action_effect`` (``u``) and
        ``rep_value`` (``ψ``) receive gradients, so the bilinear form is
        trained in sign and magnitude against ``Q − V``.

        Regularized by ``psi_adv_weight``; set to 0 to disable.
        """
        w = float(self.config.get("psi_adv_weight", 0.0))
        if w <= 0.0:
            z = jnp.array(0.0)
            return z, {"psi_adv_loss": z}

        goal_reps_sg = jax.lax.stop_gradient(
            self._goal_reps_for_value(
                batch["observations"], batch["value_goals"]
            )
        )

        next_v_all_t = self.network.select("target_value")(
            batch["next_observations"], goal_reps_sg
        )
        if next_v_all_t.ndim == 1:
            next_v_all_t = next_v_all_t[None, ...]
        v_all_t = self.network.select("target_value")(
            batch["observations"], goal_reps_sg
        )
        if v_all_t.ndim == 1:
            v_all_t = v_all_t[None, ...]

        next_v_t = jnp.min(next_v_all_t, axis=0)
        v_t = jnp.mean(v_all_t, axis=0)
        q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * next_v_t
        )
        adv_target = jax.lax.stop_gradient(q - v_t)

        psi_g = self.network.select("rep_value")(
            batch["value_goals"], params=grad_params
        )
        a_dual = self._raw_dual_advantage(
            batch["observations"], batch["actions"], psi_g,
            params=grad_params,
        )

        loss = jnp.mean((a_dual - adv_target) ** 2)

        return w * loss, {
            "psi_adv_loss": loss,
            "a_dual_mean": a_dual.mean(),
            "a_dual_std": a_dual.std(),
            "adv_target_mean": adv_target.mean(),
            "adv_target_std": adv_target.std(),
            "fit_residual_abs_mean": jnp.mean(jnp.abs(a_dual - adv_target)),
        }

    # ------------------------------------------------------------------
    # Total loss & update
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f"rep/{k}"] = v

        afu_loss, afu_info = self.afu_loss(batch, grad_params)
        for k, v in afu_info.items():
            info[f"afu/{k}"] = v

        ae_loss, ae_info = self.action_effect_loss(batch, grad_params)
        for k, v in ae_info.items():
            info[f"ae/{k}"] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        low_actor_loss, low_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_info.items():
            info[f"low_actor/{k}"] = v

        high_actor_loss, high_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_info.items():
            info[f"high_actor/{k}"] = v

        psi_adv_loss_val, psi_adv_info = self.psi_adv_loss(batch, grad_params)
        for k, v in psi_adv_info.items():
            info[f"psi_adv/{k}"] = v

        loss = (
            rep_loss + afu_loss + ae_loss
            + value_loss + low_actor_loss + high_actor_loss
            + psi_adv_loss_val
        )
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "value")
        self.target_update(new_network, "rep_critic")
        self.target_update(new_network, "rep_value")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self, observations, goals=None, seed=None, temperature=1.0,
    ):
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select("high_actor")(
            observations, goals, temperature=temperature
        )
        psi = high_dist.sample(seed=high_seed)
        if self.config.get("psi_downstream_length_normalize", True):
            psi = psi / (
                jnp.linalg.norm(psi, axis=-1, keepdims=True) + 1e-8
            ) * jnp.sqrt(psi.shape[-1])

        low_dist = self.network.select("low_actor")(
            observations, psi, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        config = copy.deepcopy(config)

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        goalrep_dim = int(config["goalrep_dim"])
        value_ensemble = int(config.get("value_ensemble", 2))
        use_goal_rep = config.get("goal_rep_for_actor", False) or config.get("goal_rep_for_value", False)
        rep_dim = int(config.get("rep_dim", 10))
        actor_goal_dim = rep_dim if config.get("goal_rep_for_actor", False) else goalrep_dim
        value_goal_dim = rep_dim if config.get("goal_rep_for_value", False) else goalrep_dim

        ex_actor_goal = jnp.zeros((1, actor_goal_dim))
        ex_value_goal = jnp.zeros((1, value_goal_dim))

        rep_value_def = DualRepresentationValue(type=config["rep_type"])(
            hidden_dims=config["rep_hidden_dims"],
            latent_dim=goalrep_dim,
            layer_norm=config["layer_norm"],
        )
        rep_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=value_ensemble,
        )
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=value_ensemble,
        )
        action_effect_def = MLP(
            hidden_dims=(
                *config["action_effect_hidden_dims"], goalrep_dim
            ),
            activate_final=False,
            layer_norm=config["layer_norm"],
        )

        if config["discrete"]:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
            )

        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=actor_goal_dim,
            state_dependent_std=False,
            const_std=config["const_std"],
        )

        ex_sa = jnp.concatenate([ex_observations, ex_actions], axis=-1)

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            target_rep_value=(
                copy.deepcopy(rep_value_def),
                (ex_observations, ex_observations),
            ),
            rep_critic=(
                rep_critic_def,
                (ex_observations, ex_observations, ex_actions),
            ),
            target_rep_critic=(
                copy.deepcopy(rep_critic_def),
                (ex_observations, ex_observations, ex_actions),
            ),
            value=(value_def, (ex_observations, ex_value_goal)),
            target_value=(
                copy.deepcopy(value_def), (ex_observations, ex_value_goal)
            ),
            action_effect=(action_effect_def, (ex_sa,)),
            low_actor=(low_actor_def, (ex_observations, ex_actor_goal)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        if use_goal_rep:
            obs_dim = ex_observations.shape[-1]
            goal_rep_def = nn.Sequential([
                MLP((*config["value_hidden_dims"], rep_dim),
                    activate_final=False, layer_norm=config["layer_norm"]),
                LengthNormalize(),
            ])
            ex_gr_in = jnp.zeros((1, obs_dim + goalrep_dim))
            network_info["goal_rep"] = (goal_rep_def, (ex_gr_in,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        grad_clip = float(config.get("grad_clip", 1.0))
        # network_tx = optax.chain(
        #     optax.clip_by_global_norm(grad_clip),
        #     optax.adam(learning_rate=config["lr"]),
        # )
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_rep_critic"] = params["modules_rep_critic"]
        params["modules_target_rep_value"] = params["modules_rep_value"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="hiql_dual_daf",
            lr=3e-4,
            grad_clip=10.0,
            rep_norm_reg=0.0,
            # Dual-advantage distillation (psi_adv_loss). Regresses
            # u(s,a)^T psi(g) / sqrt(d) onto signed Q - V. Both action_effect
            # and rep_value are trained. Set to 0.0 to disable.
            psi_adv_weight=0.0,
            # Actor AWR shaping (shared by low_actor and high_actor).
            #   actor_advantage_transform: 'softplus' (default, <= 0) or 'raw'.
            #   actor_awr_weight:          'exp' (default) or 'linear_clip'.
            actor_advantage_transform="softplus",
            actor_awr_weight="exp",
            # State-dependent goal reps + TD advantage (all default to off).
            actor_td_advantage=False,
            goal_rep_for_actor=False,
            goal_rep_for_value=False,
            psi_downstream_length_normalize=True,
            rep_dim=10,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            rep_hidden_dims=(512, 512, 512),
            action_effect_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            rep_expectile=0.7,
            low_alpha=3.0,
            high_alpha=3.0,
            subgoal_steps=25,
            goalrep_dim=256,
            rep_type="bilinear",
            value_ensemble=2,
            low_actor_rep_grad=False,
            const_std=True,
            discrete=False,
            encoder=None,
            # AFU.
            rho=0.2,
            afu_weight=1.0,
            # Action-effect regression (set to 0 to disable).
            action_effect_weight=0.01,
            # Dataset.
            dataset_class="HGCDataset",
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
            oraclerep=False,
            norm=False,
        )
    )