import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic, GCValue


class CRLDualDirectAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent with direct dual representation value.

    Unlike CRLDualAgent which trains a separate downstream bilinear value V(s, φ(g)),
    this agent directly uses the dual goal representation f(ψ(s), φ(g)) as the value function,
    closely following Algorithm 1 of the dual goal representations paper.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc').
    For AWR, advantages are computed as Q_contrastive(s, a, g) - f(ψ(s), φ(g)) instead of
    using a separate bilinear V network.

    Networks:
        value: DualRepresentationValue f(ψ(s), φ(g)) — the value function AND goal encoder.
        target_value: EMA target copy of value (f̄).
        iql_critic: GCValue Q_IQL(s, a, g) — MLP critic for training f via Algorithm 1.
        target_iql_critic: EMA target copy of iql_critic (Q̄_IQL).
        critic: GCBilinearValue Q(s, a, g) — contrastive critic for downstream policy extraction.
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
        """Compute the combined value and IQL critic loss following Algorithm 1.

        Algorithm 1 dual goal representation learning:
        - Train ψ, φ by minimizing E[ℓ_κ²(f(ψ(s), φ(g)) - Q̄(s, a, g))]
        - Train Q by minimizing E[(Q(s, a, g) - r(s, g) - γ f(ψ(s'), φ(g)))²]
        - Update Q̄ using exponential moving averaging
        """
        # --- Value loss: expectile regression of f against Q̄_IQL (Algorithm 1, line 8) ---
        q1, q2 = self.network.select('target_iql_critic')(batch['observations'], batch['rep_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], batch['rep_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        # --- IQL critic loss: TD learning with f̄ as next-state value (Algorithm 1, line 9) ---
        next_v = self.network.select('target_value')(batch['next_observations'], batch['rep_goals'])
        q_target = batch['rep_rewards'] + self.config['discount'] * batch['rep_masks'] * next_v
        q1, q2 = self.network.select('iql_critic')(
            batch['observations'], batch['rep_goals'], batch['actions'], params=grad_params
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

    def contrastive_loss(self, batch, grad_params):
        """Compute the contrastive value loss for the Q function."""
        batch_size = batch['observations'].shape[0]
        goal_reps = self.network.select('value')(batch['value_goals'])

        v, phi, psi = self.network.select('critic')(
            batch['observations'],
            goal_reps,
            actions=batch['actions'],
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        goal_reps = self.network.select('value')(batch['actor_goals'])

        if self.config['actor_loss'] == 'awr':
            # AWR loss: IVL-style advantages from the dual representation directly.
            v = self.network.select('value')(batch['observations'], batch['actor_goals'])
            nv = self.network.select('value')(batch['next_observations'], batch['actor_goals'])
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
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], goal_reps, params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], goal_reps, q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        contrastive_loss, contrastive_info = self.contrastive_loss(batch, grad_params)
        for k, v in contrastive_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + contrastive_loss + actor_loss
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
        self.target_update(new_network, 'iql_critic')

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
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goal_reps = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # NOTE: this version does not support pixel-based observations; please refer to the visual-dedicated file.
        # Value function: dual goal representation f(ψ(s), φ(g)).
        value_def = DualRepresentationValue(type=config['rep_type'])(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'],
        )

        # IQL critic: Q_IQL(s, a, g) — standard MLP critic for training the value f via Algorithm 1.
        iql_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        # Contrastive critic: Q(s, a, g) — bilinear contrastive critic for downstream policy extraction.
        if config['discrete']:
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                action_dim=action_dim,
            )
        else:
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
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
            iql_critic=(iql_critic_def, (ex_observations, ex_observations, ex_actions)),
            target_iql_critic=(copy.deepcopy(iql_critic_def), (ex_observations, ex_observations, ex_actions)),
            critic=(critic_def, (ex_observations, ex_goal_reps, ex_actions)),
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
        params['modules_target_iql_critic'] = params['modules_iql_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl_dual_direct',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for contrastive critic phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            expectile=0.7,  # IQL expectile for learning the dual representation value function.
            goalrep_dim=256,  # Dimensionality of the dual goal representation.
            rep_type='bilinear',  # Parameterization of the dual goal representation (see utils/dual.py).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            oraclerep=False,  # Whether to use oracle goal representations.
            norm=False,  # Whether to use dataset normalization.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            rep_p_curgoal=0.2,  # Probability of using the current state as the representation goal.
            rep_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the representation goal.
            rep_p_randomgoal=0.3,  # Probability of using a random state as the representation goal.
            rep_geom_sample=True,  # Whether to use geometric sampling for future representation goals.
            gc_negative=True,  # Used for rep_goals (not for general contrastive value learning).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config

