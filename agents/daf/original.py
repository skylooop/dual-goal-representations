"""Stub module for AFUDAFAgent (Actor-Free Unified Dual Advantage Fields).

This file is a placeholder to satisfy the import in agents/__init__.py.
The full implementation is pending.
"""

from typing import Any

import flax
import jax
import ml_collections

from utils.flax_utils import nonpytree_field


class AFUDAFAgent(flax.struct.PyTreeNode):
    """Actor-Free Unified Dual Advantage Fields agent (stub).

    This is a placeholder implementation. The full algorithm is not yet implemented.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        raise NotImplementedError("AFUDAFAgent is not yet implemented.")

    @jax.jit
    def update(self, batch):
        raise NotImplementedError("AFUDAFAgent is not yet implemented.")

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        raise NotImplementedError("AFUDAFAgent is not yet implemented.")

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        raise NotImplementedError("AFUDAFAgent is not yet implemented.")


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="afu_daf",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            alpha=10.0,
            const_std=True,
            discrete=False,
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
            gc_negative=True,
            p_aug=None,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
