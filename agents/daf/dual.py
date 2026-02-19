import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic, GCValue, MLP


class DAFDualAgent(flax.struct.PyTreeNode):
    """Dual Advantage Fields (DAF) agent.

    Uses a dual goal representation to extract a policy from the learned
    embeddings.

    The key idea:
    1. Learn phi(s) and psi(g) such that V(s,g) = phi(s)^T psi(g) / sqrt(d).
    2. Learn u_xi(s,a) that predicts the expected latent displacement:
       u_xi(s,a) ≈ E[gamma * phi(s') - phi(s) | s, a].
    3. Score actions by alignment: A_hat(s,a,g) = u_xi(s,a)^T psi(g).

    For discrete actions: enumerate all actions and pick argmax (original DAF).
    For continuous actions: train a DDPG+BC actor using the contrastive critic.
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
        """Compute the IQL loss for the representation value function."""
        # Rep value loss.
        q1, q2 = self.network.select('target_rep_critic')(batch['observations'], batch['rep_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(batch['observations'], batch['rep_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        # Rep critic loss.
        next_v = self.network.select('rep_value')(batch['next_observations'], batch['rep_goals'])
        q = batch['rep_rewards'] + self.config['discount'] * batch['rep_masks'] * next_v
        q1, q2 = self.network.select('rep_critic')(
            batch['observations'], batch['rep_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

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

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss."""
        batch_size = batch['observations'].shape[0]
        goal_reps = self.network.select('rep_value')(batch['value_goals'])

        if module_name == 'critic':
            actions = batch['actions']
        else:
            actions = None
        v, phi, psi = self.network.select(module_name)(
            batch['observations'],
            goal_reps,
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

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

    def action_effect_loss(self, batch, grad_params):
        """Compute the action-effect regression loss.

        Train u_xi(s, a) to predict gamma * phi(s') - phi(s), where phi is the
        state embedding from the bilinear value network.

        We use stop-gradient on phi to prevent the action-effect head from
        influencing the representation learning.
        """
        goal_reps = self.network.select('rep_value')(batch['value_goals'])

        # Get phi(s) and phi(s') from the bilinear critic.
        # We use the critic's phi since it takes (s, a) -> phi and we need
        # a state-only embedding. Use the `value` network if available (AWR),
        # otherwise obtain from the critic by passing dummy actions.
        _, phi_s, _ = self.network.select('critic')(
            batch['observations'], goal_reps, actions=batch['actions'], info=True
        )
        _, phi_sp, _ = self.network.select('critic')(
            batch['next_observations'], goal_reps, actions=batch['actions'], info=True
        )
        # Average over ensemble dimension if present.
        if len(phi_s.shape) == 3:
            phi_s = phi_s.mean(axis=0)
            phi_sp = phi_sp.mean(axis=0)

        # Target: gamma * phi(s') - phi(s), stop gradient.
        target = jax.lax.stop_gradient(self.config['discount'] * phi_sp - phi_s)

        # Predict u_xi(s, a).
        pred = self.network.select('action_effect')(
            batch['observations'], batch['actions'], params=grad_params
        )

        ae_loss = jnp.mean((pred - target) ** 2)

        return ae_loss, {
            'ae_loss': ae_loss,
            'ae_pred_norm': jnp.mean(jnp.linalg.norm(pred, axis=-1)),
            'ae_target_norm': jnp.mean(jnp.linalg.norm(target, axis=-1)),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the DDPG+BC actor loss (continuous actions only)."""
        assert not self.config['discrete']
        goal_reps = self.network.select('rep_value')(batch['actor_goals'])

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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        ae_loss, ae_info = self.action_effect_loss(batch, grad_params)
        for k, v in ae_info.items():
            info[f'ae/{k}'] = v

        loss = critic_loss + rep_loss + ae_loss

        if not self.config['discrete']:
            rng, actor_rng = jax.random.split(rng)
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            loss = loss + actor_loss

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
        self.target_update(new_network, 'rep_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions.

        For discrete actions: enumerate all actions, score by alignment, pick argmax/Boltzmann.
        For continuous actions: sample from the learned actor.
        """
        goal_reps = self.network.select('rep_value')(goals)

        if self.config['discrete']:
            # Discrete DAF: score by alignment u_xi(s,a)^T psi(g).
            n_actions = self.config['action_dim']
            batch_shape = observations.shape[:-1]
            batch_size = 1
            for s in batch_shape:
                batch_size *= s

            obs_flat = observations.reshape(batch_size, -1)
            psi_flat = goal_reps.reshape(batch_size, -1)

            all_actions = jnp.eye(n_actions)

            def score_action(a_onehot):
                a_batch = jnp.broadcast_to(a_onehot, (batch_size, n_actions))
                u = self.network.select('action_effect')(obs_flat, a_batch)
                score = (u * psi_flat).sum(axis=-1)
                return score

            scores = jax.vmap(score_action)(all_actions).T

            def greedy_fn(_):
                return jnp.argmax(scores, axis=-1)

            def boltzmann_fn(_):
                logits = scores / jnp.maximum(temperature, 1e-8)
                return jax.random.categorical(seed, logits, axis=-1)

            actions = jax.lax.cond(temperature == 0.0, greedy_fn, boltzmann_fn, None)
            return actions.reshape(batch_shape)
        else:
            # Continuous: use the learned actor.
            dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
            actions = dist.sample(seed=seed)
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
        """Create a new DAF agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals_rep = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Bilinear critic (contrastive).
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

        # Dual representation value.
        rep_value_def = DualRepresentationValue(type=config['rep_type'])(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'],
        )

        # Rep critic (standard MLP).
        rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        # Action-effect head: MLP that takes (s, a) -> latent_dim.
        action_effect_def = ActionEffectMLP(
            hidden_dims=config['ae_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
        )

        # Actor network (for continuous actions).
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
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            rep_critic=(rep_critic_def, (ex_observations, ex_observations, ex_actions)),
            target_rep_critic=(copy.deepcopy(rep_critic_def), (ex_observations, ex_observations, ex_actions)),
            critic=(critic_def, (ex_observations, ex_goals_rep, ex_actions)),
            action_effect=(action_effect_def, (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals_rep)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_rep_critic'] = params['modules_rep_critic']

        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


class ActionEffectMLP(flax.linen.Module):
    """MLP that predicts the expected latent displacement u_xi(s, a).

    Takes concatenated (observations, actions) and outputs a vector of
    dimension `latent_dim`.
    """

    hidden_dims: tuple
    latent_dim: int
    layer_norm: bool = True

    @flax.linen.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        return x


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='daf_dual',
            lr=3e-4,
            batch_size=1024,
            rep_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            ae_hidden_dims=(512, 512, 512),
            latent_dim=512,
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            alpha=0.1,  # BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            rep_expectile=0.7,
            goalrep_dim=256,
            rep_type='bilinear',
            action_dim=ml_collections.config_dict.placeholder(int),
            # Dataset hyperparameters.
            dataset_class='GCDataset',
            oraclerep=False,
            norm=False,
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            rep_p_curgoal=0.2,
            rep_p_trajgoal=0.5,
            rep_p_randomgoal=0.3,
            rep_geom_sample=True,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
