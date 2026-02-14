"""T-Junction gridworld environment for the DAF proof-of-concept.

A deterministic 7x5 grid with a corridor from the start to a T-junction,
then left/right corridors leading to two distinct goals.

Layout (y increases upward, x increases rightward):

    Row 4:  .  .  .  .  .  .  .
    Row 3:  .  .  .  .  .  .  .
    Row 2:  gL .  .  .  .  .  gR    <- T-junction row (left/right corridors)
    Row 1:  .  .  .  X  .  .  .     <- vertical corridor
    Row 0:  .  .  .  S  .  .  .     <- start

Walls: the agent can only move along the corridors (vertical + horizontal at row 2).
"""

import gymnasium
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Corridor mask: True means the cell is walkable.
# ---------------------------------------------------------------------------
GRID_W, GRID_H = 7, 5

_WALKABLE = np.zeros((GRID_W, GRID_H), dtype=bool)
# Vertical corridor: x=3, y=0..2
_WALKABLE[3, 0] = True
_WALKABLE[3, 1] = True
_WALKABLE[3, 2] = True
# Horizontal corridor: y=2, x=0..6
for x in range(GRID_W):
    _WALKABLE[x, 2] = True

START = np.array([3, 0], dtype=np.int32)
GOAL_L = np.array([0, 2], dtype=np.int32)
GOAL_R = np.array([6, 2], dtype=np.int32)

# Actions: 0=up, 1=down, 2=left, 3=right
ACTION_DELTAS = {
    0: np.array([0, 1], dtype=np.int32),   # up
    1: np.array([0, -1], dtype=np.int32),  # down
    2: np.array([-1, 0], dtype=np.int32),  # left
    3: np.array([1, 0], dtype=np.int32),   # right
}
NUM_ACTIONS = 4

TASK_INFOS = [
    {'task_name': 'reach_left', 'goal': GOAL_L.astype(np.float32) / np.array([GRID_W - 1, GRID_H - 1], dtype=np.float32)},
    {'task_name': 'reach_right', 'goal': GOAL_R.astype(np.float32) / np.array([GRID_W - 1, GRID_H - 1], dtype=np.float32)},
]


class TJunctionEnv(gymnasium.Env):
    """Deterministic T-junction gridworld."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}

    def __init__(self, max_steps: int = 20, render_mode=None):
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: (x, y) normalized to [0, 1].
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.task_infos = TASK_INFOS
        self._pos = START.copy()
        self._goal = GOAL_L.copy()
        self._step_count = 0
        self._task_id = 1

    # ------------------------------------------------------------------
    def _obs(self):
        return self._pos.astype(np.float32) / np.array([GRID_W - 1, GRID_H - 1], dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = START.copy()
        self._step_count = 0

        task_id = 1
        if options is not None:
            task_id = options.get('task_id', 1)
        self._task_id = task_id
        self._goal = GOAL_L.copy() if task_id == 1 else GOAL_R.copy()

        goal_obs = self._goal.astype(np.float32) / np.array([GRID_W - 1, GRID_H - 1], dtype=np.float32)
        info = {'goal': goal_obs}
        return self._obs(), info

    # ------------------------------------------------------------------
    def step(self, action):
        delta = ACTION_DELTAS[int(action)]
        new_pos = self._pos + delta
        # Clip to grid and check walkability.
        new_pos = np.clip(new_pos, [0, 0], [GRID_W - 1, GRID_H - 1])
        if _WALKABLE[new_pos[0], new_pos[1]]:
            self._pos = new_pos

        self._step_count += 1
        reached = np.array_equal(self._pos, self._goal)
        reward = 1.0 if reached else 0.0
        terminated = reached
        truncated = self._step_count >= self.max_steps

        info = {
            'success': float(reached),
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        """Render a small RGB image (for video logging compatibility)."""
        cell = 20
        img = np.full((GRID_H * cell, GRID_W * cell, 3), 40, dtype=np.uint8)
        # Draw walkable cells.
        for x in range(GRID_W):
            for y in range(GRID_H):
                if _WALKABLE[x, y]:
                    ry = (GRID_H - 1 - y) * cell
                    rx = x * cell
                    img[ry:ry + cell, rx:rx + cell] = [200, 200, 200]
        # Draw goal.
        gx, gy = self._goal
        ry = (GRID_H - 1 - gy) * cell
        rx = gx * cell
        img[ry:ry + cell, rx:rx + cell] = [0, 200, 0]
        # Draw agent.
        ax, ay = self._pos
        ry = (GRID_H - 1 - ay) * cell
        rx = ax * cell
        img[ry + 4:ry + cell - 4, rx + 4:rx + cell - 4] = [200, 50, 50]
        return img


# ---------------------------------------------------------------------------
# Register with Gymnasium
# ---------------------------------------------------------------------------
gymnasium.register(
    id='t_junction-v0',
    entry_point='t_junction_env:TJunctionEnv',
    max_episode_steps=20,
)
