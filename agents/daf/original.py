import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCDiscreteActor, GCValue, MLP


class DAFAgent(flax.struct.PyTreeNode):
    """Dual Advantage Fields (DAF) agent.

    Derives advantages from dual goal representations without a separate downstream
    value function. The bilinear value V(s,g) = phi(s)^T psi(g) / sqrt(d) implies
    that psi(g) = nabla_phi V, i.e. the goal embedding is the value gradient in
    representation space. An action-effect head u(s,a) approximates
    E[gamma*phi(s') - phi(s) | s,a], yielding the dual advantage
    A_hat = u(s,a)^T psi(g) / sqrt(d) for AWR policy improvement.

    Key structural advantage over baselines: the bilinear factorization means a
    single u(s,a) produces correct advantages for ALL goals simultaneously via
    dot products. The multi-goal actor loss exploits this by training the actor
    against K goals per (s,a) at marginal cost (K dot products vs K value-fn evals).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def rep_loss(self, batch, grad_params):
        """IQL loss for the dual representation value function.

        The representation value is parameterized by the bilinear inner product,
        while the critic is a standard unrestricted (MLP) Q function.
        """
        # Rep value loss: expectile regression toward target Q.
        q1, q2 = self.network.select("target_rep_critic")(
            batch["observations"], batch["rep_goals"], batch["actions"]
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select("rep_value")(
            batch["observations"], batch["rep_goals"], params=grad_params
        )
        value_loss = self.expectile_loss(
            q - v, q - v, self.config["rep_expectile"]
        ).mean()

        # Rep critic loss: Bellman regression using rep_value as bootstrap.
        next_v = self.network.select("rep_value")(
            batch["next_observations"], batch["rep_goals"]
        )
        q_target = (
            batch["rep_rewards"]
            + self.config["discount"] * batch["rep_masks"] * next_v
        )
        q1, q2 = self.network.select("rep_critic")(
            batch["observations"],
            batch["rep_goals"],
            batch["actions"],
            params=grad_params,
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return value_loss + critic_loss, {
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
            "q_mean": q_target.mean(),
            "q_max": q_target.max(),
            "q_min": q_target.min(),
        }

    def action_effect_loss(self, batch, grad_params):
        """MSE regression of u(s,a) against gamma*phi(s') - phi(s).

        Uses target_rep_value (EMA copy) for stable regression targets,
        preventing the moving-target problem from rep_value updates.
        """
        # Extract state representations from the *target* rep_value for stability.
        _, phi_s, _ = self.network.select("target_rep_value")(
            batch["observations"], batch["rep_goals"], info=True
        )
        _, phi_s_next, _ = self.network.select("target_rep_value")(
            batch["next_observations"], batch["rep_goals"], info=True
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
        return self.config["action_effect_weight"] * loss, {
            "action_effect_loss": loss,
            "u_norm": jnp.linalg.norm(u_pred, axis=-1).mean(),
            "target_norm": jnp.linalg.norm(target, axis=-1).mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Multi-goal AWR actor loss using the dual advantage.

        Exploits the bilinear factorization: a single u(s,a) is computed once,
        then dotted with psi(g_k) for K different goals. This gives K times
        more actor supervision per batch at marginal cost (K dot products),
        which baselines cannot do without K full value-function evaluations.
        """
        batch_size = batch["observations"].shape[0]
        num_goals = self.config["num_actor_goals"]

        # Compute action-effect vector once per (s, a).
        if self.config["advantage_mode"] == "oracle":
            _, phi_s, _ = self.network.select("rep_value")(
                batch["observations"], batch["actor_goals"], info=True
            )
            _, phi_s_next, _ = self.network.select("rep_value")(
                batch["next_observations"], batch["actor_goals"], info=True
            )
            u = self.config["discount"] * phi_s_next - phi_s  # (B, D)
        else:
            sa_input = jnp.concatenate(
                [batch["observations"], batch["actions"]], axis=-1
            )
            u = self.network.select("action_effect")(sa_input)  # (B, D)

        # Build K goal sets: slot 0 = trajectory actor_goals, rest = random.
        rng, goal_rng = jax.random.split(rng)
        random_keys = jax.random.split(goal_rng, num_goals - 1)
        random_indices = jax.vmap(
            lambda key: jax.random.randint(key, (batch_size,), 0, batch_size)
        )(random_keys)  # (K-1, B)
        random_goals = batch["observations"][random_indices]  # (K-1, B, obs_dim)
        multi_goals = jnp.concatenate(
            [batch["actor_goals"][None], random_goals], axis=0
        )  # (K, B, obs_dim)

        # Encode all K*B goals in one forward pass.
        goals_flat = multi_goals.reshape(-1, multi_goals.shape[-1])
        goal_reps_flat = self.network.select("rep_value")(goals_flat)
        goal_reps = goal_reps_flat.reshape(num_goals, batch_size, -1)  # (K, B, D)
        latent_dim = goal_reps.shape[-1]

        # Dual advantage for all (s, a, g_k): u^T psi(g) / sqrt(d).
        # u: (B, D), goal_reps: (K, B, D) -> adv: (K, B).
        adv = jnp.einsum("bd,kbd->kb", u, goal_reps) / jnp.sqrt(latent_dim)

        # Advantage normalization: calibrates raw inner-product scale.
        if self.config["normalize_advantage"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # Flatten to (K*B,) for vectorized AWR.
        adv_flat = adv.reshape(-1)
        exp_a = jnp.exp(adv_flat * self.config["alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Actor forward pass for all K*B (observation, goal) pairs.
        obs_repeated = jnp.tile(batch["observations"][None], (num_goals, 1, 1))
        obs_flat = obs_repeated.reshape(-1, obs_repeated.shape[-1])
        greps_flat = goal_reps.reshape(-1, latent_dim)
        dist = self.network.select("actor")(
            obs_flat, greps_flat, params=grad_params
        )
        actions_repeated = jnp.tile(
            batch["actions"][None], (num_goals, 1, 1)
        ).reshape(-1, batch["actions"].shape[-1])
        log_prob = dist.log_prob(actions_repeated)

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            "actor_loss": actor_loss,
            "adv_mean": adv_flat.mean(),
            "adv_std": adv_flat.std(),
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - actions_repeated) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f"rep/{k}"] = v

        ae_loss, ae_info = self.action_effect_loss(batch, grad_params)
        for k, v in ae_info.items():
            info[f"action_effect/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = rep_loss + ae_loss + actor_loss
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
        self.target_update(new_network, "rep_critic")
        self.target_update(new_network, "rep_value")

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
        goal_reps = self.network.select("rep_value")(goals)
        dist = self.network.select("actor")(
            observations, goal_reps, temperature=temperature
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
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = jnp.zeros(shape=(1, config["goalrep_dim"]))
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Dual representation value (bilinear: V = phi(s)^T psi(g) / sqrt(d)).
        rep_value_def = DualRepresentationValue(type=config["rep_type"])(
            hidden_dims=config["rep_hidden_dims"],
            latent_dim=config["goalrep_dim"],
            layer_norm=config["layer_norm"],
        )

        rep_critic_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
        )

        # Action-effect head: u(s, a) -> R^goalrep_dim.
        action_effect_def = MLP(
            hidden_dims=(*config["action_effect_hidden_dims"], config["goalrep_dim"]),
            activate_final=False,
            layer_norm=config["layer_norm"],
        )

        if config["discrete"]:
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
            )

        ex_action_effect_input = jnp.concatenate(
            [ex_observations, ex_actions], axis=-1
        )
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
            action_effect=(action_effect_def, (ex_action_effect_input,)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_rep_critic"] = params["modules_rep_critic"]
        params["modules_target_rep_value"] = params["modules_rep_value"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="daf",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions (for rep_critic).
            action_effect_hidden_dims=(512, 512, 512),  # Action-effect head hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            alpha=10.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            rep_expectile=0.7,  # IQL expectile for learning the representation value function.
            goalrep_dim=256,  # Dimensionality of the dual goal representation.
            rep_type="bilinear",  # Must be 'bilinear' for the gradient identity to hold.
            advantage_mode="dual",  # 'dual' (learned action-effect) or 'oracle' (true next-state).
            action_effect_weight=1.0,  # Weight for the action-effect regression loss.
            num_actor_goals=4,  # Number of goals per (s,a) in the multi-goal actor loss.
            normalize_advantage=True,  # Whether to normalize advantages before AWR exponentiation.
            # Dataset hyperparameters.
            dataset_class="GCDataset",  # Dataset class name.
            oraclerep=False,  # Always False; dummy option for compatibility.
            norm=False,  # Whether to use dataset normalization.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            rep_p_curgoal=0.2,  # Probability of using the current state as the representation goal.
            rep_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the representation goal.
            rep_p_randomgoal=0.3,  # Probability of using a random state as the representation goal.
            rep_geom_sample=True,  # Whether to use geometric sampling for future representation goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
