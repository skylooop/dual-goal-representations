import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue


class GCAFUAgent(flax.struct.PyTreeNode):
    """Goal-conditioned AFU (Advantage-weighted Function Update) agent.

    Replaces the IQL expectile regression on V with the AFU Z-loss that jointly
    trains a value network V_phi(s, g) and an advantage network A_xi(s, a, g).

    AFU loss (from the notebook derivation):
        indicator = 1  if  V(s,g) + A(s,a,g) < Q(s,a,g)  else 0
        Upsilon   = V + rho * indicator * stop_grad(V - V)   # rescales gradient
        x = Upsilon - Q(s,a,g)
        y = A(s,a,g)
        Z(x, y)   = (x + y)^2  if x >= 0  else  x^2 + y^2

    When x >= 0 (V overestimates Q), both V and A are pushed down together.
    When x < 0  (V underestimates Q), A is pushed toward 0 independently and
    the gradient on V is rescaled by (1 - rho) via the Upsilon construction.

    The critic and actor losses are identical to GCIQL.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # AFU value + advantage loss
    # ------------------------------------------------------------------

    def value_loss(self, batch, grad_params):
        """Compute the AFU joint value-advantage loss."""
        # Bootstrap target from the lagged critic (no gradient).
        q1, q2 = self.network.select("target_critic")(
            batch["observations"], batch["value_goals"], batch["actions"]
        )
        q = jnp.minimum(q1, q2)

        # Live value and advantage (both receive gradients via grad_params).
        v = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        a = self.network.select("advantage")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )

        # AFU conditional gradient rescaling.
        # indicator = 1 when V + A underestimates Q (push V up less aggressively).
        indicator = jnp.where(v + a < q, 1.0, 0.0)
        # stop_gradient on v so the rho term only rescales the gradient, not the value.
        v_sg = jax.lax.stop_gradient(v)
        upsilon = v + self.config["rho"] * indicator * (v_sg - v)

        x = upsilon - q  # overestimation residual
        y = a  # advantage residual

        # Z-loss: coupled when overestimating, decoupled when underestimating.
        loss = jnp.where(x >= 0, (x + y) ** 2, x**2 + y**2)
        value_loss = loss.mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
            "a_mean": a.mean(),
            "a_abs_mean": jnp.abs(a).mean(),
            "upsilon_mean": upsilon.mean(),
            "indicator_frac": indicator.mean(),  # fraction of underestimating samples
        }

    # ------------------------------------------------------------------
    # Critic loss (identical to GCIQL)
    # ------------------------------------------------------------------

    def critic_loss(self, batch, grad_params):
        """Compute the IQL-style critic (Q) loss."""
        next_v = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        q_target = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v

        q1, q2 = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_target.mean(),
            "q_max": q_target.max(),
            "q_min": q_target.min(),
        }

    # ------------------------------------------------------------------
    # Actor loss (AWR or DDPG+BC)
    # ------------------------------------------------------------------

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC).

        AWR advantage modes (config['actor_advantage_mode']):
        - 'afu':    A_xi(s, a_data, g)           AFU advantage (≤ 0, peaks at 0 for
                                                  the optimal action) — the natural
                                                  AWR signal for this agent.
        - 'qv':     Q(s,a,g) - V(s,g)            standard IQL advantage
        - 'vdelta': V(s',g) - V(s,g)
        - 'td':     r + gamma*mask*V(s',g) - V(s,g)

        DDPG+BC uses the AFU-reconstructed Q = V(s,g) + A_xi(s, pi(s), g) as the
        policy gradient signal.  This is consistent with the AFU value decomposition
        and avoids relying on the separately trained critic Q for the actor update.
        """
        if self.config["actor_loss"] == "awr":
            mode = self.config["actor_advantage_mode"]
            actor_info = {}

            if mode == "afu":
                # A_xi is trained to satisfy A_xi(s,a,g) ≤ 0 with A_xi = 0 at the
                # optimal action.  exp(A_xi * alpha) therefore gives high weight to
                # near-optimal dataset actions and low weight to suboptimal ones —
                # exactly the AWR weighting we want.
                adv = self.network.select("advantage")(
                    batch["observations"], batch["actor_goals"], batch["actions"]
                )
            elif mode == "vdelta":
                v = self.network.select("value")(
                    batch["observations"], batch["actor_goals"]
                )
                nv = self.network.select("value")(
                    batch["next_observations"], batch["actor_goals"]
                )
                adv = nv - v
            elif mode == "td":
                v = self.network.select("value")(
                    batch["observations"], batch["actor_goals"]
                )
                nv = self.network.select("value")(
                    batch["next_observations"], batch["actor_goals"]
                )
                adv = (
                    batch["rewards"] + self.config["discount"] * batch["masks"] * nv - v
                )
            else:
                # 'qv' — standard IQL: Q(s,a,g) - V(s,g)
                v = self.network.select("value")(
                    batch["observations"], batch["actor_goals"]
                )
                q1, q2 = self.network.select("critic")(
                    batch["observations"], batch["actor_goals"], batch["actions"]
                )
                q = jnp.minimum(q1, q2)
                adv = q - v

            exp_a = jnp.exp(adv * self.config["alpha"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            log_prob = dist.log_prob(batch["actions"])
            actor_loss = -(exp_a * log_prob).mean()

            actor_info.update(
                {
                    "actor_loss": actor_loss,
                    "adv": adv.mean(),
                    "bc_log_prob": log_prob.mean(),
                }
            )
            if not self.config["discrete"]:
                actor_info.update(
                    {
                        "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                        "std": jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info

        elif self.config["actor_loss"] == "ddpgbc":
            assert not self.config["discrete"]

            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            if self.config["const_std"]:
                pi_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                pi_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            # AFU-reconstructed Q: Q_afu(s, pi(s), g) = V(s,g) + A_xi(s, pi(s), g).
            # V ≈ max_a Q by AFU training, so V + A_xi ≈ Q for any action.
            # This is more consistent than using the separate critic Q, which is
            # bootstrapped from V and can be near-constant (V - V ≈ 0 gradient).
            v = self.network.select("value")(
                batch["observations"], batch["actor_goals"]
            )
            a_pi = self.network.select("advantage")(
                batch["observations"], batch["actor_goals"], pi_actions
            )
            q_afu = jax.lax.stop_gradient(v) + a_pi  # stop_grad on V; only A gets grad

            # Normalize by the absolute mean to keep the loss scale-invariant.
            q_loss = -q_afu.mean() / jax.lax.stop_gradient(jnp.abs(q_afu).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(self.config["alpha"] * log_prob).mean()
            actor_loss = q_loss + 0.1 * bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_afu_mean": q_afu.mean(),
                "q_afu_abs_mean": jnp.abs(q_afu).mean(),
                "a_pi_mean": a_pi.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

        else:
            raise ValueError(f"Unsupported actor loss: {self.config['actor_loss']}")

    # ------------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def target_update(self, network, module_name):
        """Polyak-average the target network parameters."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with an info dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select("actor")(
            observations, goals, temperature=temperature
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
        """Create a new GCAFUAgent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. For discrete MDPs this should
                contain the maximum action value.
            config: Configuration dictionary.
            ex_goals: Unused; kept for API compatibility.
        """
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Optional visual encoders.
        encoders = {}
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = GCEncoder(concat_encoder=encoder_module())
            encoders["advantage"] = GCEncoder(concat_encoder=encoder_module())
            encoders["critic"] = GCEncoder(concat_encoder=encoder_module())
            encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

        # V_phi(s, g) — scalar, no ensemble.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
            gc_encoder=encoders.get("value"),
        )

        # A_xi(s, a, g) — scalar, no ensemble; actions are concatenated inside GCValue.
        advantage_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=False,
            gc_encoder=encoders.get("advantage"),
        )

        # Q(s, a, g) — ensemble critic for stable TD targets.
        if config["discrete"]:
            critic_def = GCDiscreteCritic(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                gc_encoder=encoders.get("critic"),
                action_dim=action_dim,
            )
        else:
            critic_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                gc_encoder=encoders.get("critic"),
            )

        # Actor.
        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                gc_encoder=encoders.get("actor"),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
                gc_encoder=encoders.get("actor"),
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            # Advantage network receives actions as the third positional arg.
            advantage=(advantage_def, (ex_observations, ex_goals, ex_actions)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialise target critic from the online critic parameters.
        network_params["modules_target_critic"] = network_params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="gcafu",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value/advantage/critic hidden dims.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network Polyak update rate.
            rho=0.4,  # AFU gradient rescaling coefficient (replaces IQL expectile).
            actor_loss="awr",  # Actor loss type ('awr' or 'ddpgbc').
            actor_advantage_mode="afu",  # AWR advantage: 'afu' (default), 'qv', 'vdelta', 'td'.
            alpha=10.0,  # AWR temperature (larger values needed since A_xi ≤ 0 and near-zero).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            norm=False,  # Whether to normalize the observations.
            oraclerep=False,  # Whether to use oracle representations.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
        )
    )
    return config
