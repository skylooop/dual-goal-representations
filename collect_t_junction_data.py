"""Collect an offline dataset for the T-junction environment.

Generates a mix of random-walk and expert trajectories, then saves them
in the format expected by `Dataset.create(**fields)`.

Usage:
    uv run collect_t_junction_data.py [--n_episodes 2000] [--output t_junction_data.npz]
"""

import argparse
import numpy as np
import t_junction_env  # noqa: F401  — triggers gymnasium.register
import gymnasium


def expert_policy(obs, goal):
    """Simple hard-coded expert for the T-junction."""
    # Unnormalize.
    x = round(obs[0] * 6)
    y = round(obs[1] * 4)
    gx = round(goal[0] * 6)

    # Phase 1: go up to the junction row (y=2).
    if y < 2:
        return 0  # up
    # Phase 2: go toward the goal along the horizontal corridor.
    if x < gx:
        return 3  # right
    elif x > gx:
        return 2  # left
    else:
        return 0  # already at goal, arbitrary


def collect(n_episodes: int = 2000, expert_frac: float = 0.5, seed: int = 0):
    """Collect transitions."""
    rng = np.random.RandomState(seed)
    env = gymnasium.make('t_junction-v0')

    all_obs, all_acts, all_terminals, all_valids = [], [], [], []

    for ep in range(n_episodes):
        # Alternate goals.
        task_id = (ep % 2) + 1
        obs, info = env.reset(options={'task_id': task_id})
        goal = info['goal']
        use_expert = rng.rand() < expert_frac

        done = False
        ep_obs, ep_acts, ep_terminals = [], [], []
        while not done:
            if use_expert:
                action = expert_policy(obs, goal)
            else:
                action = int(rng.randint(env.action_space.n))

            ep_obs.append(obs.copy())
            ep_acts.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_terminals.append(float(done))

        # Append the final observation so we can compute next_observations later.
        ep_obs.append(obs.copy())
        ep_acts.append(0)  # dummy action for the terminal step
        ep_terminals[-1] = 1.0  # ensure the last step is terminal
        ep_terminals.append(1.0)

        all_obs.extend(ep_obs)
        all_acts.extend(ep_acts)
        all_terminals.extend(ep_terminals)
        all_valids.extend([1.0] * len(ep_obs))

    observations = np.array(all_obs, dtype=np.float32)
    # One-hot encode discrete actions (4 actions).
    n = len(all_acts)
    actions = np.zeros((n, 4), dtype=np.float32)
    for i, a in enumerate(all_acts):
        actions[i, a] = 1.0
    terminals = np.array(all_terminals, dtype=np.float32)
    valids = np.array(all_valids, dtype=np.float32)

    return dict(observations=observations, actions=actions, terminals=terminals, valids=valids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=2000)
    parser.add_argument('--expert_frac', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='t_junction_data.npz')
    args = parser.parse_args()

    data = collect(args.n_episodes, args.expert_frac, args.seed)
    np.savez_compressed(args.output, **data)
    print(f'Saved {len(data["observations"])} transitions to {args.output}')
    for k, v in data.items():
        print(f'  {k}: shape={v.shape} dtype={v.dtype}')


if __name__ == '__main__':
    main()
