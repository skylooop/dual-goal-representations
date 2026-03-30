import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCValue, MLP


class DAFActorFreeAgent(flax.struct.PyTreeNode):
    """Dual Advantage Fields — actor-free variant.

    No actor network is trained. Instead, following Dayan & Singh (1995),
    actions are selected at inference time by maximizing the dual advantage
    field A(s,a,g) = u(s,a)^T psi(g) / sqrt(d) via gradient ascent w.r.t. a.

    The theoretical chain:
      1. Bilinear value: V(s,g) = phi(s)^T psi(g) / sqrt(d)
      2. Gradient identity: nabla_phi V = psi(g) / sqrt(d)
         => psi(g) is the value gradient in representation space.
      3. Action-effect model: u(s,a) ~ E[gamma*phi(s') - phi(s) | s, a]
         => u(s,a) is the expected representation-space displacement per action.
      4. Dual advantage: A(s,a,g) = u(s,a)^T psi(g) / sqrt(d)
         => dot product of "where action takes you" with "where goal pulls you."
      5. Policy: a* = argmax_a A(s,a,g) via gradient ascent through the
         differentiable u network. No actor needed.

    Training: rep_loss + action_effect_loss only (no actor loss).
    Inference: gradient-based action optimization.
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
        """IQL loss for the dual representation value function."""
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

        Targets come from the target_rep_value (EMA copy) for stability.
        Stop-gradiented so this loss does not backprop into the representation.
        """
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss. No actor loss — only rep + action_effect."""
        info = {}

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f"rep/{k}"] = v

        ae_loss, ae_info = self.action_effect_loss(batch, grad_params)
        for k, v in ae_info.items():
            info[f"action_effect/{k}"] = v

        loss = rep_loss + ae_loss
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
        """Select actions by maximizing the dual advantage field via gradient ascent.

        For each state, M random action candidates are initialized, refined by
        K gradient-ascent steps on A(s,a,g) = u(s,a)^T psi(g) / sqrt(d), and
        the highest-scoring candidate is returned.

        No actor network is involved — the "policy" is the argmax of the
        advantage field, computed on-the-fly.
        """
        # Ensure batch dimension exists (eval passes unbatched 1D inputs).
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
            goals = goals[None]

        goal_reps = self.network.select("rep_value")(goals)  # psi(g): (B, D)
        B = observations.shape[0]
        action_dim = self.config["action_dim"]
        D = goal_reps.shape[-1]
        sqrt_d = jnp.sqrt(D).astype(goal_reps.dtype)
        M = self.config["num_action_samples"]

        # Tile observations and goal_reps for M candidates: (M*B, ...).
        obs_tiled = jnp.tile(observations, (M, 1))
        gr_tiled = jnp.tile(goal_reps, (M, 1))

        # M * B random starting actions.
        actions = jax.random.uniform(
            seed, (M * B, action_dim), minval=-1.0, maxval=1.0
        )

        # Gradient ascent: maximize A(s,a,g) = u(s,a)^T psi(g) / sqrt(d).
        def step_fn(_, a):
            def adv_sum(a):
                sa = jnp.concatenate([obs_tiled, a], axis=-1)
                u = self.network.select("action_effect")(sa)
                # Sum over batch for scalar; per-sample grads separate cleanly
                # because there are no cross-sample interactions in the MLP.
                return ((u * gr_tiled) / sqrt_d).sum()

            grad_a = jax.grad(adv_sum)(a)
            # Normalize gradient: makes step size independent of scale.
            grad_a = grad_a / (
                jnp.linalg.norm(grad_a, axis=-1, keepdims=True) + 1e-8
            )
            return jnp.clip(a + self.config["action_lr"] * grad_a, -1.0, 1.0)

        refined = jax.lax.fori_loop(
            0, self.config["action_opt_steps"], step_fn, actions
        )

        # Score all M*B candidates and pick the best per state.
        sa = jnp.concatenate([obs_tiled, refined], axis=-1)
        u = self.network.select("action_effect")(sa)
        scores = (u * gr_tiled).sum(axis=-1) / sqrt_d  # (M*B,)

        scores = scores.reshape(M, B)
        refined = refined.reshape(M, B, action_dim)
        best_idx = jnp.argmax(scores, axis=0)  # (B,)
        result = refined[best_idx, jnp.arange(B)]

        if unbatched:
            result = result[0]
        return result

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
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config["discrete"]:
            action_dim = int(ex_actions.max() + 1)
        else:
            action_dim = int(ex_actions.shape[-1])

        # Store action_dim so sample_actions can create random candidates.
        config = dict(config)
        config["action_dim"] = action_dim

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
        # Maps (obs, action) to expected representation-space displacement.
        action_effect_def = MLP(
            hidden_dims=(
                *config["action_effect_hidden_dims"],
                config["goalrep_dim"],
            ),
            activate_final=False,
            layer_norm=config["layer_norm"],
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
            agent_name="daf_actor_free",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions (for rep_critic).
            action_effect_hidden_dims=(512, 512, 512),  # Action-effect head hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            discrete=False,  # Whether the action space is discrete.
            rep_expectile=0.7,  # IQL expectile for learning the representation value function.
            goalrep_dim=256,  # Dimensionality of the dual goal representation.
            rep_type="bilinear",  # Must be 'bilinear' for the gradient identity to hold.
            action_effect_weight=1.0,  # Weight for the action-effect regression loss.
            # Action optimization at inference time (no actor).
            num_action_samples=16,  # Number of random action initializations (M).
            action_opt_steps=10,  # Gradient ascent steps per candidate (K).
            action_lr=0.1,  # Step size for gradient ascent on the advantage.
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
