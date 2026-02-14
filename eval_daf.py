"""Evaluate a trained DAF agent on the T-junction environment.

Usage:
    uv run eval_daf.py --restore_path 'results/...' --restore_epoch 50000
"""

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np
import t_junction_env  # noqa: F401
import gymnasium

from agents.daf.dual import DAFDualAgent, get_config
from utils.datasets import Dataset
from utils.flax_utils import restore_agent


def junction_ranking_accuracy(agent, env, n_trials=200, seed=42):
    """Test whether DAF prefers the correct action at the junction.

    At position (3, 2) — the T-junction — the agent should prefer
    'left' (action 2) when goal is g_L and 'right' (action 3) when goal is g_R.
    """
    rng = jax.random.PRNGKey(seed)
    correct = 0
    total = 0

    for task_id in [1, 2]:
        obs, info = env.reset(options={'task_id': task_id})
        goal = info['goal']
        expected_action = 2 if task_id == 1 else 3  # left vs right

        # Move to the junction: up, up.
        for a in [0, 0]:
            obs, _, _, _, _ = env.step(a)

        # Now obs should be (0.5, 0.5) — the junction.
        obs_jax = jnp.array(obs[None])  # (1, 2)
        goal_jax = jnp.array(goal[None])  # (1, 2)

        for _ in range(n_trials):
            rng, sub = jax.random.split(rng)
            action = agent.sample_actions(
                observations=obs_jax, goals=goal_jax, seed=sub, temperature=0.0
            )
            if int(action[0]) == expected_action:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy


def success_at_h(agent, env, n_episodes=100, seed=42):
    """Run the DAF policy from the start and report success rate."""
    key = jax.random.PRNGKey(seed)

    successes = 0
    for ep in range(n_episodes):
        task_id = (ep % 2) + 1
        obs, info = env.reset(options={'task_id': task_id})
        goal = info['goal']
        done = False
        while not done:
            key, sub = jax.random.split(key)
            obs_jax = jnp.array(obs[None])
            goal_jax = jnp.array(goal[None])
            action = agent.sample_actions(
                observations=obs_jax, goals=goal_jax, seed=sub, temperature=0.0
            )
            action = int(action[0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get('success', 0.0) > 0.5:
            successes += 1

    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_path', type=str, required=True)
    parser.add_argument('--restore_epoch', type=int, required=True)
    args = parser.parse_args()

    config = get_config()
    env = gymnasium.make('t_junction-v0')

    # Create a dummy agent for shape inference.
    ex_obs = jnp.zeros((1, 2))
    ex_actions = jnp.zeros((1, 4))
    agent = DAFDualAgent.create(0, ex_obs, ex_actions, config)

    # Restore.
    agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    print('Agent restored.')

    # Junction ranking.
    jra = junction_ranking_accuracy(agent, env)
    print(f'Junction Ranking Accuracy: {jra:.2%}')

    # Success@H.
    sah = success_at_h(agent, env, n_episodes=200)
    print(f'Success@H: {sah:.2%}')

    # Random baseline.
    class RandomAgent:
        def sample_actions(self, observations, goals, seed, temperature):
            return jax.random.randint(seed, (observations.shape[0],), 0, 4)

    rand_sah = success_at_h(RandomAgent(), env, n_episodes=200, seed=123)
    print(f'Random Success@H: {rand_sah:.2%}')


if __name__ == '__main__':
    main()
