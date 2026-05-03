import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCValue


class GCIVLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent.

    This is a variant of GCIQL that only uses a V function, without Q functions.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @staticmethod
    def z_loss_fn(x, y):
        """
        Implements Z(x, y) where:
        Z(x, y) = (x + y)^2 if x >= 0
        Z(x, y) = x^2 + y^2 otherwise
        """
        return jnp.where(x >= 0, jnp.square(x + y), jnp.square(x) + jnp.square(y))

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        if not self.config["apply_z_loss"]:
            (v1_t, v2_t) = self.network.select("target_value")(
                batch["observations"], batch["value_goals"]
            )
            v_t = (v1_t + v2_t) / 2
            adv = q - v_t

            q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
            q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
            (v1, v2) = self.network.select("value")(
                batch["observations"], batch["value_goals"], params=grad_params
            )
            v = (v1 + v2) / 2

            value_loss1 = self.expectile_loss(
                adv, q1 - v1, self.config["expectile"]
            ).mean()
            value_loss2 = self.expectile_loss(
                adv, q2 - v2, self.config["expectile"]
            ).mean()

            value_loss = value_loss1 + value_loss2
        else:
            (adv1, adv2) = self.network.select("advantage")(
                batch["observations"],
                batch["actions"],
                batch["value_goals"],
                params=grad_params,
            )
            (v1, v2) = self.network.select("value")(
                batch["observations"], batch["value_goals"], params=grad_params
            )
            v = (v1 + v2) / 2
            q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
            q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t

            indicator1 = jnp.where(adv1 + v1 < q, 1.0, 0.0)
            indicator2 = jnp.where(adv2 + v2 < q, 1.0, 0.0)
            Υ1 = (1 - 0.4 * indicator1) * v1 + 0.4 * indicator1 * jax.lax.stop_gradient(
                v1
            )
            Υ2 = (1 - 0.4 * indicator2) * v2 + 0.4 * indicator2 * jax.lax.stop_gradient(
                v2
            )
            value_loss1 = (
                self.z_loss_fn(Υ1 - q, adv1).mean() + jnp.maximum(0, adv1).mean()
            )
            value_loss2 = (
                self.z_loss_fn(Υ2 - q, adv2).mean() + jnp.maximum(0, adv2).mean()
            )
            value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def _dayan_advantage(self, batch):
        """Compute Dayan gradient-based advantage (Equation 4, Dayan & Singh 1995).

        A^w(s, a, g) = r(s, g) + (s' - s) · ∇_s V(s, g)

        Uses autodiff to compute ∇_s V(s, g) and dots it with the transition
        direction (s' - s), faithfully implementing the continuous-time advantage
        in discrete form.
        """
        value_fn = self.network.select("value")

        def scalar_value(s, g):
            v1, v2 = value_fn(s[None], g[None])
            return ((v1 + v2) / 2)[0]

        # ∇_s V(s, g) via autodiff, shape: (batch, state_dim)
        grad_v = jax.vmap(jax.grad(scalar_value, argnums=0))(
            batch["observations"], batch["actor_goals"]
        )

        # Transition direction: f(s, a) ≈ s' - s
        delta_s = batch["next_observations"] - batch["observations"]

        # Directional derivative: (s' - s) · ∇_s V(s, g)
        directional_deriv = jnp.sum(delta_s * grad_v, axis=-1)

        # Dayan advantage: r(s, g) + f(s, a) · ∇_s V(s, g)
        adv = batch["rewards"] + directional_deriv

        return adv, grad_v, directional_deriv

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss.

        Supports three advantage modes controlled by config['actor_advantage_mode']:
        - 'vdelta': V(s', g) - V(s, g)  (default, original GCIVL)
        - 'td':     r(s, g) + γ·mask·V(s', g) - V(s, g)  (standard TD advantage)
        - 'dayan':  r(s, g) + (s' - s) · ∇_s V(s, g)  (Dayan Eq. 4, gradient-based)
        """
        mode = self.config["actor_advantage_mode"]

        actor_info = {}

        if mode == "dayan":
            adv, grad_v, directional_deriv = self._dayan_advantage(batch)
            actor_info["grad_v_norm"] = jnp.linalg.norm(grad_v, axis=-1).mean()
            actor_info["directional_deriv"] = directional_deriv.mean()
        else:
            v1, v2 = self.network.select("value")(
                batch["observations"], batch["actor_goals"]
            )
            nv1, nv2 = self.network.select("value")(
                batch["next_observations"], batch["actor_goals"]
            )
            v = (v1 + v2) / 2
            nv = (nv1 + nv2) / 2

            if mode == "td":
                adv = (
                    batch["rewards"] + self.config["discount"] * batch["masks"] * nv - v
                )
            else:  # 'vdelta' original
                adv = nv - v

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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

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

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_goals=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if not config["oraclerep"]:
            ex_goals = ex_observations

        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = GCEncoder(concat_encoder=encoder_module())
            encoders["actor"] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=encoders.get("value"),
        )

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
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )

        if config["apply_z_loss"]:
            advantage_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                ensemble=True,
                gc_encoder=encoders.get("value"),
            )
            network_info.update(
                advantage=(advantage_def, (ex_observations, ex_actions, ex_goals))
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_value"] = params["modules_value"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="gcivl",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            apply_z_loss=False,  # 'expectile' or 'afu'.
            alpha=0.003,  # AWR temperature.
            actor_advantage_mode="vdelta",  # Actor advantage: 'vdelta' (V(s')-V(s)), 'td' (r+γV(s')-V(s)), 'dayan' (r+(s'-s)·∇V).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            oraclerep=False,  # Whether to use oracle goal representations.
            norm=False,  # Whether to use dataset normalization.
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
