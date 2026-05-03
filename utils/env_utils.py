import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if terminated or truncated:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if "goal" in info:
            info["goal"] = np.concatenate([info["goal"]] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(
    dataset_name, frame_stack=None, normalize_data=False, eps=1e-5
):
    """Make environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True, add_info=True
    )
    train_dataset["actions"] = np.clip(train_dataset["actions"], -1 + eps, 1 - eps)
    val_dataset["actions"] = np.clip(val_dataset["actions"], -1 + eps, 1 - eps)
    train_dataset = Dataset.create(norm=normalize_data, **train_dataset)
    val_dataset = Dataset.create(norm=normalize_data, **val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    env.reset()

    return env, train_dataset, val_dataset


def relabel_dataset(env_name, env, dataset):
    """Relabel the dataset with rewards and masks based on the fixed task of the environment.

    Args:
        env_name: Name of the environment.
        env: Environment.
        dataset: Dataset dictionary.

    Returns:
        The relabeled dataset.

    """
    if "maze" in env_name or "soccer" in env_name:
        # Locomotion environments.
        qpos_xy_start_idx = 0
        qpos_ball_start_idx = 15
        goal_xy = env.unwrapped.cur_goal_xy
        goal_tol = env.unwrapped._goal_tol

        # Compute successes.
        if "maze" in env_name:
            dists = np.linalg.norm(
                dataset["qpos"][:, qpos_xy_start_idx : qpos_xy_start_idx + 2] - goal_xy,
                axis=-1,
            )
        else:
            dists = np.linalg.norm(
                dataset["qpos"][:, qpos_ball_start_idx : qpos_ball_start_idx + 2]
                - goal_xy,
                axis=-1,
            )
        successes = (dists <= goal_tol).astype(np.float32)

        rewards = successes  # 1.0 if s == g else 0.0
        masks = 1.0 - successes
    elif "cube" in env_name or "scene" in env_name or "puzzle" in env_name:
        # Manipulation environments.
        qpos_obj_start_idx = 14
        qpos_cube_length = 7

        if "cube" in env_name:
            num_cubes = env.unwrapped._num_cubes
            target_cube_xyzs = env.unwrapped._data.mocap_pos.copy()

            # Compute successes.
            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    dataset["qpos"][
                        :,
                        qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx
                        + i * qpos_cube_length
                        + 3,
                    ]
                )
            cube_xyzs = np.stack(cube_xyzs_list, axis=1)
            successes = np.linalg.norm(target_cube_xyzs - cube_xyzs, axis=-1) <= 0.04
        elif "scene" in env_name:
            num_cubes = env.unwrapped._num_cubes
            num_buttons = env.unwrapped._num_buttons
            qpos_drawer_idx = (
                qpos_obj_start_idx + num_cubes * qpos_cube_length + num_buttons
            )
            qpos_window_idx = qpos_drawer_idx + 1
            target_cube_xyzs = env.unwrapped._data.mocap_pos.copy()
            target_button_states = env.unwrapped._target_button_states.copy()
            target_drawer_pos = env.unwrapped._target_drawer_pos
            target_window_pos = env.unwrapped._target_window_pos

            # Compute successes.
            cube_xyzs_list = []
            for i in range(num_cubes):
                cube_xyzs_list.append(
                    dataset["qpos"][
                        :,
                        qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx
                        + i * qpos_cube_length
                        + 3,
                    ]
                )
            cube_xyzs = np.stack(cube_xyzs_list, axis=1)
            cube_successes = (
                np.linalg.norm(target_cube_xyzs - cube_xyzs, axis=-1) <= 0.04
            )
            button_successes = dataset["button_states"] == target_button_states
            drawer_success = (
                np.abs(dataset["qpos"][:, qpos_drawer_idx] - target_drawer_pos) <= 0.04
            )
            window_success = (
                np.abs(dataset["qpos"][:, qpos_window_idx] - target_window_pos) <= 0.04
            )
            successes = np.concatenate(
                [
                    cube_successes,
                    button_successes,
                    drawer_success[:, None],
                    window_success[:, None],
                ],
                axis=-1,
            )
        elif "puzzle" in env_name:
            num_buttons = env.unwrapped._num_buttons
            target_button_states = env.unwrapped._target_button_states.copy()

            # Compute successes.
            successes = dataset["button_states"] == target_button_states

        rewards = successes.sum(axis=-1)  # 1.0 if s == g else 0.0
        masks = 1.0 - np.all(successes, axis=-1)
    new_fields = dict(dataset)
    new_fields["rewards"] = rewards.astype(np.float32)
    new_fields["masks"] = masks.astype(np.float32)
    new_dataset = Dataset.create(freeze=False, **new_fields)

    return new_dataset
