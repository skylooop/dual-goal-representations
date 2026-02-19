import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCDiscreteActor, GCValue


class GCIVLDualDirectAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent with direct dual representation value.

    Unlike GCIVLDualAgent which trains a separate downstream value network V(s, φ(g)),
    this agent directly uses the dual goal representation f(ψ(s), φ(g)) as the value function,
    closely following Algorithm 1 of the dual goal representations paper.

    The dual representation f(ψ(s), φ(g)) serves a dual purpose:
    - As a value function V(s, g) = f(ψ(s), φ(g)) for IQL training and advantage computation.
    - As a goal encoder φ(g) for conditioning the downstream actor.

    Networks:
        value: DualRepresentationValue f(ψ(s), φ(g)) — the value function AND goal encoder.
        target_value: EMA target copy of value (f̄).
        critic: GCValue Q(s, a, g) — the critic used for training f with IQL.
        target_critic: EMA target copy of critic (Q̄).
        actor: GCActor π(a | s, φ(g)) — the policy conditioned on goal representations.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the combined value and critic loss following Algorithm 1.

        Algorithm 1 dual goal representation learning:
        - Train ψ, φ by minimizing E[ℓ_κ²(f(ψ(s), φ(g)) - Q̄(s, a, g))]
        - Train Q by minimizing E[(Q(s, a, g) - r(s, g) - γ f(ψ(s'), φ(g)))²]
        - Update Q̄ using exponential moving averaging
        """
        # --- Value loss: expectile regression of f against Q̄ (Algorithm 1, line 8) ---
        q1, q2 = self.network.select('target_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        # --- Critic loss: TD learning with f̄ as next-state value (Algorithm 1, line 9) ---
        next_v = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss using the dual representation directly for advantages."""
        # Get goal representations for actor conditioning.
        goal_reps = self.network.select('value')(batch['actor_goals'])

        # Compute IVL advantages from the dual representation value.
        v = self.network.select('value')(batch['observations'], batch['actor_goals'])
        nv = self.network.select('value')(batch['next_observations'], batch['actor_goals'])
        v = v.mean(axis=0) if v.ndim > 1 else v
        nv = nv.mean(axis=0) if nv.ndim > 1 else nv
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], goal_reps, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
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
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'critic')

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
        goal_reps = self.network.select('value')(goals)
        dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
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

        ex_goal_reps = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # NOTE: this version does not support pixel-based observations; please refer to the visual-dedicated file.
        # Value function: dual goal representation f(ψ(s), φ(g)).
        value_def = DualRepresentationValue(type=config['rep_type'])(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'],
        )

        # Critic: Q(s, a, g) — standard MLP critic for training the value f via IQL.
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_observations)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_observations)),
            critic=(critic_def, (ex_observations, ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goal_reps)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcivl_dual_direct',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            value_hidden_dims=(512, 512, 512),  # Dual representation value hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            alpha=10.0,  # AWR temperature.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            goalrep_dim=256,  # Dimensionality of the dual goal representation.
            rep_type='bilinear',  # Parameterization of the dual goal representation (see utils/dual.py).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            oraclerep=False,  # always False; dummy option for compatibility.
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
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
