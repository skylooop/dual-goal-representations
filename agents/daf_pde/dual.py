import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCValue, LatentDynamics


class DAFPDEDualAgent(flax.struct.PyTreeNode):
    """Dual Advantage Fields with Physics-Informed Regularization (DAF-PDE).

    Actor-free policy extraction from bilinear dual goal representations via
    latent control-affine dynamics and PDE-style regularizers (Eikonal + viscosity).

    Policy:  a*(s,g) = clip(sigma^2 * B(s)^T phi(g), [-1, 1])
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _get_phi(self, goals, params=None):
        """Extract the goal embedding φ(g) from the bilinear rep_value.

        When called with a single tensor, GCBilinearRepresentationValue
        returns the goal-side embedding (psi in GCBilinearValue terms).
        """
        return self.network.select('rep_value')(goals, params=params)

    def _get_embeddings(self, observations, goals, params=None):
        """Get (value, phi_s, psi_g) from the bilinear rep_value with info=True.

        In GCBilinearValue:
          - phi is the state encoder
          - psi is the goal encoder
        With ret_mean=True and ensemble=True, phi_s and psi_g are averaged
        across the ensemble dimension.
        """
        return self.network.select('rep_value')(
            observations, goals, info=True, params=params,
        )

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def rep_loss(self, batch, grad_params):
        """IQL loss for the bilinear representation value + MLP critic."""
        # Rep value loss (expectile regression V toward Q).
        q1, q2 = self.network.select('target_rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions'],
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(
            batch['observations'], batch['value_goals'], params=grad_params,
        )
        value_loss = self.expectile_loss(
            q - v, q - v, self.config['rep_expectile'],
        ).mean()

        # Rep critic loss (standard TD).
        next_v = self.network.select('rep_value')(
            batch['next_observations'], batch['value_goals'],
        )
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions'],
            params=grad_params,
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'critic_loss': critic_loss,
        }

    def value_loss(self, batch, grad_params):
        """Downstream IVL value loss using goal representations."""
        goal_reps = self._get_phi(batch['value_goals'])

        (next_v1_t, next_v2_t) = self.network.select('target_value')(
            batch['next_observations'], goal_reps,
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(
            batch['observations'], goal_reps,
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(
            batch['observations'], goal_reps, params=grad_params,
        )

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': ((v1 + v2) / 2).mean(),
        }

    def dynamics_loss(self, batch, grad_params):
        """Latent dynamics regression: ‖(φ(s') − φ(s)) − B(s) a‖².

        φ here is the state-side encoder of the bilinear value (called 'phi'
        in GCBilinearValue). We access it via info=True.

        IMPORTANT: We stop-gradient through rep_value so this loss only trains
        latent_dyn (B network), not the representation encoder.
        """
        # Get state embeddings φ(s) and φ(s') — stop-gradient to avoid
        # corrupting the IQL-learned representation.
        _, phi_s, _ = self._get_embeddings(
            batch['observations'], batch['value_goals'],
        )
        _, phi_s_next, _ = self._get_embeddings(
            batch['next_observations'], batch['value_goals'],
        )
        delta_z = jax.lax.stop_gradient(phi_s_next - phi_s)  # (batch, d)

        # B(s) from latent dynamics model — this IS differentiated.
        B_s = self.network.select('latent_dyn')(
            batch['observations'], params=grad_params,
        )  # (batch, d, m)

        actions = batch['actions']  # (batch, m)
        predicted_delta = jnp.einsum('...dm,...m->...d', B_s, actions)

        dyn_loss = jnp.mean(jnp.sum((delta_z - predicted_delta) ** 2, axis=-1))

        return dyn_loss, {
            'dyn_loss': dyn_loss,
            'delta_z_norm': jnp.linalg.norm(delta_z, axis=-1).mean(),
        }

    def eikonal_state_loss(self, batch, grad_params):
        """State-gradient Eikonal loss: (‖J_φ(s)^T ψ(g)‖ − 1)².

        Computed with a VJP: J_φ(s)^T ψ(g) = vjp(φ, s)(ψ(g)).
        φ = state encoder, ψ = goal encoder from the bilinear value.

        This loss backprops through rep_value to shape the representation
        geometry, using grad_params so the encoder learns Eikonal structure.
        The goal embedding is stop-gradiented as a fixed cotangent.
        """
        # Stop-gradient on goal embedding — treat as fixed cotangent vector.
        psi_g = jax.lax.stop_gradient(
            self._get_phi(batch['value_goals'])
        )  # (batch, d)

        # φ(s) uses grad_params so this loss shapes the representation.
        def phi_fn(observations):
            """State encoder φ(s) from the bilinear value network."""
            _, phi_s, _ = self._get_embeddings(
                observations, batch['value_goals'], params=grad_params,
            )
            return phi_s  # (batch, d)

        # VJP: compute J_φ(s)^T ψ(g) without forming the full Jacobian.
        _, vjp_fn = jax.vjp(phi_fn, batch['observations'])
        grad_s_V = vjp_fn(psi_g)[0]  # (batch, obs_dim) = J_φ(s)^T ψ(g)

        grad_norm = jnp.linalg.norm(grad_s_V, axis=-1)  # (batch,)
        eik_loss = jnp.mean((grad_norm - 1.0) ** 2)

        return eik_loss, {
            'eik_state_loss': eik_loss,
            'grad_s_V_norm': grad_norm.mean(),
        }

    def eikonal_ctrl_loss(self, batch, grad_params):
        """Control co-vector Eikonal: (‖u(s,g)‖ − c)² where u = σ B(s)^T ψ(g).

        The transported co-vector u = R^{-1/2} B(s)^T ψ(g).
        With R = (1/σ²)I, R^{-1/2} = σ I, so u = σ B(s)^T ψ(g).
        """
        # Stop-gradient on goal embedding — only train latent_dyn.
        psi_g = jax.lax.stop_gradient(
            self._get_phi(batch['value_goals'])
        )  # (batch, d)
        B_s = self.network.select('latent_dyn')(
            batch['observations'], params=grad_params,
        )  # (batch, d, m)

        sigma = self.config['sigma']
        # u = σ * B(s)^T ψ(g)  →  (batch, m)
        u = sigma * jnp.einsum('...dm,...d->...m', B_s, psi_g)

        u_norm = jnp.linalg.norm(u, axis=-1)  # (batch,)
        c = self.config['eik_target']
        ctrl_loss = jnp.mean((u_norm - c) ** 2)

        return ctrl_loss, {
            'eik_ctrl_loss': ctrl_loss,
            'u_norm': u_norm.mean(),
        }

    def viscosity_loss(self, batch, grad_params, rng):
        """Viscosity smoothing: ‖u(s,g) − u(s̃,g)‖² with s̃ = s + ε.

        Smooths the transported co-vector field u = B(s)^T ψ(g).
        """
        # Stop-gradient on goal embedding — only train latent_dyn.
        psi_g = jax.lax.stop_gradient(
            self._get_phi(batch['value_goals'])
        )  # (batch, d)

        B_s = self.network.select('latent_dyn')(
            batch['observations'], params=grad_params,
        )  # (batch, d, m)
        u_s = jnp.einsum('...dm,...d->...m', B_s, psi_g)  # (batch, m)

        # Perturbed observations.
        noise_std = self.config['visc_noise_std']
        eps = jax.random.normal(rng, shape=batch['observations'].shape) * noise_std
        perturbed_obs = batch['observations'] + eps

        B_s_tilde = self.network.select('latent_dyn')(
            perturbed_obs, params=grad_params,
        )  # (batch, d, m)
        u_s_tilde = jnp.einsum('...dm,...d->...m', B_s_tilde, psi_g)  # (batch, m)

        visc_loss = jnp.mean(jnp.sum((u_s - u_s_tilde) ** 2, axis=-1))

        return visc_loss, {
            'visc_loss': visc_loss,
        }

    # ------------------------------------------------------------------
    # Total loss and update
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss from Eq. (8) of the proposal."""
        info = {}
        rng = rng if rng is not None else self.rng

        # 1. Dual representation losses.
        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        # 2. Downstream value loss.
        val_loss, val_info = self.value_loss(batch, grad_params)
        for k, v in val_info.items():
            info[f'value/{k}'] = v

        # 3. Dynamics loss.
        dyn_loss, dyn_info = self.dynamics_loss(batch, grad_params)
        for k, v in dyn_info.items():
            info[f'dyn/{k}'] = v

        # 4. Eikonal state loss.
        eik_state_loss, eik_state_info = self.eikonal_state_loss(batch, grad_params)
        for k, v in eik_state_info.items():
            info[f'eik/{k}'] = v

        # 5. Eikonal control loss.
        eik_ctrl_loss, eik_ctrl_info = self.eikonal_ctrl_loss(batch, grad_params)
        for k, v in eik_ctrl_info.items():
            info[f'eik/{k}'] = v

        # 6. Viscosity loss.
        rng, visc_rng = jax.random.split(rng)
        visc_loss, visc_info = self.viscosity_loss(batch, grad_params, visc_rng)
        for k, v in visc_info.items():
            info[f'visc/{k}'] = v

        # Total objective (Eq. 8).
        loss = (
            rep_loss
            + val_loss
            + self.config['lambda_dyn'] * dyn_loss
            + self.config['lambda_eik'] * eik_state_loss
            + self.config['lambda_ctrl'] * eik_ctrl_loss
            + self.config['lambda_visc'] * visc_loss
        )
        info['total_loss'] = loss

        return loss, info

    def target_update(self, network, module_name):
        """Update the target network via Polyak averaging."""
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
        self.target_update(new_network, 'rep_critic')

        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Actor-free policy extraction
    # ------------------------------------------------------------------

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Extract actions in closed form: a*(s,g) = clip(σ² B(s)^T ψ(g)).

        Args:
            observations: Current observations.
            goals: Goal observations.
            seed: Unused (deterministic policy) but kept for API compat.
            temperature: Unused but kept for API compat.
        """
        psi_g = self._get_phi(goals)          # goal embedding (batch, d)
        B_s = self.network.select('latent_dyn')(observations)  # (batch, d, m)

        sigma_sq = self.config['sigma'] ** 2
        # a* = σ² B(s)^T ψ(g)
        actions = sigma_sq * jnp.einsum('...dm,...d->...m', B_s, psi_g)
        actions = jnp.clip(actions, -1, 1)
        return actions

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_goals=None,
    ):
        """Create a new DAF-PDE agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        batch_dim = ex_observations.shape[0]
        ex_goal_reps = jnp.zeros(shape=(batch_dim, config['goalrep_dim']))

        # ---- Bilinear dual representation value ψ(s)^T φ(g) ----
        rep_value_def = DualRepresentationValue(type='bilinear')(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'],
        )

        # ---- MLP Q-critic for training the representation ----
        rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        # ---- Downstream V(s, goal_rep) ----
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        # ---- Latent dynamics B(s) ----
        latent_dyn_def = LatentDynamics(
            hidden_dims=config['dyn_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
        )

        # ---- Assemble all modules ----
        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            rep_critic=(rep_critic_def, (ex_observations, ex_observations, ex_actions)),
            target_rep_critic=(copy.deepcopy(rep_critic_def), (ex_observations, ex_observations, ex_actions)),
            value=(value_def, (ex_observations, ex_goal_reps)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goal_reps)),
            latent_dyn=(latent_dyn_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_rep_critic'] = params['modules_rep_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='daf_pde_dual',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            discrete=False,
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dims.
            value_hidden_dims=(512, 512, 512),  # Value / critic hidden dims.
            dyn_hidden_dims=(512, 512, 512),  # Latent dynamics hidden dims.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network Polyak rate.
            expectile=0.9,  # IQL expectile for downstream V.
            rep_expectile=0.7,  # IQL expectile for representation V.
            goalrep_dim=256,  # Dimension d of the dual goal representation.
            # PDE regularization weights (Eq. 8).
            lambda_dyn=1.0,  # Weight for dynamics regression loss.
            lambda_eik=0.1,  # Weight for state Eikonal loss.
            lambda_ctrl=0.1,  # Weight for control co-vector Eikonal loss.
            lambda_visc=0.01,  # Weight for viscosity smoothing loss.
            # Policy extraction.
            sigma=1.0,  # σ in R = (1/σ²)I; controls action magnitude.
            eik_target=1.0,  # Target c for ‖u(s,g)‖ in control Eikonal.
            visc_noise_std=0.01,  # Std of Gaussian perturbation for viscosity.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
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
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
