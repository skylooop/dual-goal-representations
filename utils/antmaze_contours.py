"""Antmaze contour plotting: analytical Eikonal/FK solutions + learned value/policy.

Self-contained: copies the needed grid/layout and analytical solvers from 2dplot.py
so we don't depend on importing 2dplot. At the end of PI-HIQL training on antmaze,
main.py calls plot_antmaze_learned_and_analytic() to produce a 2x2 figure: Eikonal
value, Learned value, Eikonal policy (-grad V), Learned policy.
"""

from __future__ import annotations

import heapq
import os
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Grid and obstacle helpers (from 2dplot.py)
# -----------------------------------------------------------------------------


def _build_grid_bounds(x_min, x_max, y_min, y_max, nx, ny=None):
    if ny is None:
        ny = nx
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    return x, y, X, Y, hx, hy


def _obstacle_mask_from_rects(X, Y, rects):
    mask = np.zeros_like(X, dtype=bool)
    for xmin, ymin, xmax, ymax in rects:
        mask |= (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    return mask


def solve_fk_value_on_grid(x, y, obstacle_mask, goal_xy, nu):
    grid_size_y, grid_size_x = obstacle_mask.shape
    n = grid_size_y * grid_size_x

    goal_mask = np.zeros_like(obstacle_mask)
    goal_ix = np.argmin(np.abs(x - goal_xy[0]))
    goal_iy = np.argmin(np.abs(y - goal_xy[1]))
    goal_mask[goal_iy, goal_ix] = True

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    q = 1.0
    mass = q / (2.0 * nu**2)
    inv_hx2 = 1.0 / (hx * hx)
    inv_hy2 = 1.0 / (hy * hy)

    A = sp.lil_matrix((n, n))
    b = np.zeros(n)

    for iy in range(grid_size_y):
        for ix in range(grid_size_x):
            idx = iy * grid_size_x + ix

            if obstacle_mask[iy, ix]:
                A[idx, idx] = 1.0
                b[idx] = 0.0
                continue

            if goal_mask[iy, ix]:
                A[idx, idx] = 1.0
                b[idx] = 1.0
                continue

            diag = mass
            neighbors = (
                (iy - 1, ix, inv_hy2),
                (iy + 1, ix, inv_hy2),
                (iy, ix - 1, inv_hx2),
                (iy, ix + 1, inv_hx2),
            )
            for ny_, nx_, coef in neighbors:
                if 0 <= ny_ < grid_size_y and 0 <= nx_ < grid_size_x:
                    nidx = ny_ * grid_size_x + nx_
                    A[idx, nidx] = -coef
                    diag += coef

            A[idx, idx] = diag

    psi = np.asarray(spla.spsolve(A.tocsr(), b), dtype=float).reshape(
        (grid_size_y, grid_size_x)
    )
    psi = np.clip(psi, 1e-300, None)
    v = -2.0 * nu * np.log(psi)
    return v


def solve_eikonal_value_grid(obstacle_mask, goal_idx, hx, hy):
    """Solve Eikonal distance field on a grid via Dijkstra in free space."""
    h_diag = np.hypot(hx, hy)
    neighbors = [
        (-1, 0, hy),
        (1, 0, hy),
        (0, -1, hx),
        (0, 1, hx),
        (-1, -1, h_diag),
        (-1, 1, h_diag),
        (1, -1, h_diag),
        (1, 1, h_diag),
    ]

    ny, nx = obstacle_mask.shape
    gy, gx = goal_idx
    dist = np.full((ny, nx), np.inf, dtype=float)
    if obstacle_mask[gy, gx]:
        return np.full((ny, nx), np.nan, dtype=float)

    dist[gy, gx] = 0.0
    pq = [(0.0, gy, gx)]

    while pq:
        cur, iy, ix = heapq.heappop(pq)
        if cur > dist[iy, ix]:
            continue

        for dy, dx, cost in neighbors:
            nyi = iy + dy
            nxi = ix + dx
            if not (0 <= nyi < ny and 0 <= nxi < nx):
                continue
            if obstacle_mask[nyi, nxi]:
                continue
            cand = cur + cost
            if cand < dist[nyi, nxi]:
                dist[nyi, nxi] = cand
                heapq.heappush(pq, (cand, nyi, nxi))

    dist[obstacle_mask] = np.nan
    return dist


def _compute_policy(v, obstacle_mask, h_y, h_x=None):
    if h_x is None:
        h_x = h_y
    dv_dy, dv_dx = np.gradient(v, h_y, h_x)
    policy_x = -dv_dx
    policy_y = -dv_dy

    norm = np.hypot(policy_x, policy_y) + 1e-12
    policy_x = policy_x / norm
    policy_y = policy_y / norm
    policy_x[obstacle_mask] = np.nan
    policy_y[obstacle_mask] = np.nan
    return policy_x, policy_y


def _ogbench_antmaze_layout(maze_type):
    if maze_type == "medium":
        maze_map = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=int,
        )
        goal_ij = (6, 6)
    elif maze_type == "large":
        maze_map = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=int,
        )
        goal_ij = (7, 10)
    else:
        raise ValueError(f"Unsupported antmaze type: {maze_type}")

    return maze_map, goal_ij


def _ij_to_xy(ij, maze_unit=4.0, offset_x=4.0, offset_y=4.0):
    i, j = ij
    return j * maze_unit - offset_x, i * maze_unit - offset_y


def _nearest_free_idx(X, Y, obstacle_mask, goal_xy):
    free = ~obstacle_mask
    dist2 = (X - goal_xy[0]) ** 2 + (Y - goal_xy[1]) ** 2
    dist2 = np.where(free, dist2, np.inf)
    idx = np.argmin(dist2)
    return np.unravel_index(idx, X.shape)


def _style_axes_maze(ax, rects, x_min, x_max, y_min, y_max, goal_xy=None):
    for xmin, ymin, xmax, ymax in rects:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="#0d4cab",
                edgecolor="#8ed8ea",
                linewidth=0.45,
                zorder=6,
            )
        )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if goal_xy is not None:
        ax.scatter(
            goal_xy[0],
            goal_xy[1],
            marker="*",
            s=220,
            color="#ffcc00",
            edgecolor="black",
            linewidth=0.8,
            zorder=12,
            label="Goal",
        )
    ax.set_facecolor("#eef2f7")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


# -----------------------------------------------------------------------------
# Learned value and policy (agent-dependent)
# -----------------------------------------------------------------------------


def _get_value_module(agent: Any):
    """Get the value module from an agent, handling different naming conventions.

    Supports both 'value' (PI-HIQL, GCIVL) and 'rep_value' (pi_actor_free) module names.
    """
    model_def = agent.network.model_def
    if "value" in model_def.modules:
        return agent.network.select("value")
    else:
        return agent.network.select("rep_value")


def _encode_goals(agent: Any, goals: jnp.ndarray) -> jnp.ndarray:
    """Encode goals through rep_value if the agent has one, otherwise pass through."""
    model_def = agent.network.model_def
    if "rep_value" in model_def.modules and "value" in model_def.modules:
        return agent.network.select("rep_value")(goals)
    return goals


def _compute_learned_value_grid(
    agent: Any, states_flat: jnp.ndarray, goals_flat: jnp.ndarray
) -> jnp.ndarray:
    """Compute V(s,g) on a batch; return scalar value (min of ensemble)."""
    value_fn = _get_value_module(agent)
    goal_reps = _encode_goals(agent, goals_flat)
    v1, v2 = value_fn(states_flat, goal_reps)
    return jnp.minimum(v1, v2)


def _compute_learned_value_grad_grid(
    agent: Any,
    states_flat: jnp.ndarray,
    goal_vec: jnp.ndarray,
    x_index: int = 0,
    y_index: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute grad_s V(s,g); return (policy_x, policy_y) = +grad V so arrows point toward goal.

    Learned V(s,g) is higher when s is closer to g, so +∇V points toward the goal
    (same visual direction as analytical Eikonal policy −∇d where d = distance).
    """
    value_fn = _get_value_module(agent)
    goal_rep = _encode_goals(agent, goal_vec[None, :])

    def value_scalar(s: jnp.ndarray) -> jnp.ndarray:
        v1, v2 = value_fn(s[None, :], goal_rep)
        return (v1[0] + v2[0]) / 2.0

    grad_fn = jax.vmap(jax.grad(value_scalar), in_axes=0)
    # Average gradients across the sample dimension. Expects states_flat of shape (num_grid_points * num_samples, obs_dim)
    grad_s = grad_fn(states_flat)
    policy_x = grad_s[:, x_index]
    policy_y = grad_s[:, y_index]
    return policy_x, policy_y


def _build_antmaze_grid_and_analytic(
    maze_type: str,
    points_per_cell: int = 14,
    nu: float = 2.5,
    goal_xy_override: tuple[float, float] | None = None,
) -> tuple[Any, ...]:
    """Build grid (X, Y), obstacle mask, rects, bounds, and analytical Eikonal/FK values."""
    maze_map, goal_ij = _ogbench_antmaze_layout(maze_type)
    maze_unit = 4.0
    half = 0.5 * maze_unit
    rows, cols = maze_map.shape
    x_min = -4.0 - half
    x_max = (cols - 1) * maze_unit - 4.0 + half
    y_min = -4.0 - half
    y_max = (rows - 1) * maze_unit - 4.0 + half

    nx = cols * points_per_cell + 1
    ny = rows * points_per_cell + 1
    x, y, X, Y, hx, hy = _build_grid_bounds(x_min, x_max, y_min, y_max, nx, ny)

    rects = []
    for i in range(rows):
        for j in range(cols):
            if maze_map[i, j] == 1:
                cx, cy = _ij_to_xy((i, j), maze_unit=maze_unit)
                rects.append((cx - half, cy - half, cx + half, cy + half))

    obstacle_mask = _obstacle_mask_from_rects(X, Y, rects)
    goal_xy = (
        goal_xy_override
        if goal_xy_override is not None
        else _ij_to_xy(goal_ij, maze_unit=maze_unit)
    )

    v_fk = solve_fk_value_on_grid(x, y, obstacle_mask, goal_xy, nu)
    goal_idx = _nearest_free_idx(X, Y, obstacle_mask, goal_xy)
    v_eik = solve_eikonal_value_grid(obstacle_mask, goal_idx, hx, hy)

    v_eik_max = np.nanmax(v_eik[~obstacle_mask])
    v_fk_max = np.nanmax(v_fk[~obstacle_mask])
    v_eik_plot = np.where(obstacle_mask, v_eik_max, v_eik)
    v_fk_plot = np.where(obstacle_mask, v_fk_max, v_fk)
    v_eik_lines = np.ma.masked_where(obstacle_mask, v_eik)
    v_fk_lines = np.ma.masked_where(obstacle_mask, v_fk)

    v_eik_grad = np.where(obstacle_mask, np.nanmax(v_eik[~obstacle_mask]), v_eik)
    v_fk_grad = np.where(obstacle_mask, np.nanmax(v_fk[~obstacle_mask]), v_fk)
    policy_eik_x, policy_eik_y = _compute_policy(v_eik_grad, obstacle_mask, hy, hx)
    policy_fk_x, policy_fk_y = _compute_policy(v_fk_grad, obstacle_mask, hy, hx)

    return (
        X,
        Y,
        x,
        y,
        hx,
        hy,
        obstacle_mask,
        rects,
        v_eik_plot,
        v_fk_plot,
        v_eik_lines,
        v_fk_lines,
        policy_eik_x,
        policy_eik_y,
        policy_fk_x,
        policy_fk_y,
        goal_xy,
        (x_min, x_max, y_min, y_max),
    )


def plot_antmaze_learned_and_analytic(
    agent: Any,
    maze_type: str,
    save_dir: str,
    obs_dim: int,
    x_index: int = 0,
    y_index: int = 1,
    points_per_cell: int = 14,
    nu: float = 2.5,
    quiver_scale: float = 0.95,
    log_wandb: bool = False,
    goal_xy: tuple[float, float] | None = None,
    task_name: str | None = None,
    dataset_samples: np.ndarray | jnp.ndarray | None = None,
    max_dataset_samples: int = 100,
) -> str:
    """Plot 2x2: Eikonal value, Learned value; Eikonal policy, Learned policy. Save to save_dir."""
    agent = jax.device_put(agent, device=jax.devices("cpu")[0])

    (
        X,
        Y,
        x,
        y,
        hx,
        hy,
        obstacle_mask,
        rects,
        v_eik_plot,
        v_fk_plot,
        v_eik_lines,
        v_fk_lines,
        policy_eik_x,
        policy_eik_y,
        policy_fk_x,
        policy_fk_y,
        goal_xy_eval,
        (x_min, x_max, y_min, y_max),
    ) = _build_antmaze_grid_and_analytic(
        maze_type, points_per_cell=points_per_cell, nu=nu, goal_xy_override=goal_xy
    )

    ny, nx = X.shape
    num_grid_points = ny * nx

    if dataset_samples is None:
        dataset_samples_arr = np.zeros((1, obs_dim), dtype=np.float32)
    else:
        dataset_samples_arr = np.asarray(dataset_samples, dtype=np.float32)

    # Subsample to avoid memory explosion and slowness on large dataset chunks
    max_samples = max(1, max_dataset_samples)
    if dataset_samples_arr.shape[0] > max_samples:
        indices = np.random.choice(
            dataset_samples_arr.shape[0], max_samples, replace=False
        )
        dataset_samples_arr = dataset_samples_arr[indices]

    num_samples = dataset_samples_arr.shape[0]

    goal_vec = np.zeros(obs_dim, dtype=np.float32)
    goal_vec[x_index] = goal_xy_eval[0]
    goal_vec[y_index] = goal_xy_eval[1]
    goal_vec = jnp.array(goal_vec)

    grid_x = X.ravel().astype(np.float32)
    grid_y = Y.ravel().astype(np.float32)

    # Compute values in grid chunks to avoid building a huge (grid * samples) tensor.
    grid_chunk_size = 2048
    v_learned_flat = np.empty(num_grid_points, dtype=np.float32)
    for grid_start in range(0, num_grid_points, grid_chunk_size):
        grid_end = min(grid_start + grid_chunk_size, num_grid_points)
        chunk_n = grid_end - grid_start

        chunk_states = np.broadcast_to(
            dataset_samples_arr[None, :, :], (chunk_n, num_samples, obs_dim)
        ).copy()
        chunk_states[:, :, x_index] = grid_x[grid_start:grid_end, None]
        chunk_states[:, :, y_index] = grid_y[grid_start:grid_end, None]

        chunk_states_flat = jnp.asarray(chunk_states.reshape(-1, obs_dim))
        chunk_goals_flat = jnp.broadcast_to(goal_vec[None, :], chunk_states_flat.shape)
        chunk_values = np.asarray(
            _compute_learned_value_grid(agent, chunk_states_flat, chunk_goals_flat)
        )
        v_learned_flat[grid_start:grid_end] = chunk_values.reshape(
            chunk_n, num_samples
        ).mean(axis=1)

    v_learned = -v_learned_flat.reshape(ny, nx)
    v_learned[obstacle_mask] = np.nanmax(v_learned[~obstacle_mask])
    v_learned_lines = np.ma.masked_where(obstacle_mask, v_learned)

    # Computing JAX autograd at every (grid, sample) point is extremely expensive.
    # For visualization, derive policy from the learned value field directly.
    v_learned_grad = np.where(
        obstacle_mask, np.nanmax(v_learned[~obstacle_mask]), v_learned
    )
    policy_learned_x, policy_learned_y = _compute_policy(
        v_learned_grad, obstacle_mask, hy, hx
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.3), dpi=170)
    fig.patch.set_facecolor("#e9e9e9")

    levels_eik = np.linspace(np.nanmin(v_eik_plot), np.nanmax(v_eik_plot), 26)
    axes[0, 0].contourf(
        X,
        Y,
        v_eik_plot,
        levels=levels_eik,
        cmap="Blues_r",
        corner_mask=False,
        extend="max",
    )
    axes[0, 0].contour(
        X,
        Y,
        v_eik_lines,
        levels=levels_eik,
        colors="black",
        linewidths=0.35,
        alpha=0.72,
    )
    label = f"{maze_type}/{task_name}" if task_name else maze_type
    axes[0, 0].set_title(f"Eikonal Value ({label})", fontsize=18, family="monospace")
    _style_axes_maze(axes[0, 0], rects, x_min, x_max, y_min, y_max, goal_xy=goal_xy)

    levels_learned = np.linspace(np.nanmin(v_learned), np.nanmax(v_learned), 26)
    axes[0, 1].contourf(
        X,
        Y,
        v_learned,
        levels=levels_learned,
        cmap="Blues_r",
        corner_mask=False,
        extend="max",
    )
    axes[0, 1].contour(
        X,
        Y,
        v_learned_lines,
        levels=levels_learned,
        colors="black",
        linewidths=0.35,
        alpha=0.72,
    )
    axes[0, 1].set_title(
        f"Learned Value (-V) ({label})", fontsize=18, family="monospace"
    )
    _style_axes_maze(axes[0, 1], rects, x_min, x_max, y_min, y_max, goal_xy=goal_xy)

    step = max(2, points_per_cell // 2)
    axes[1, 0].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        policy_eik_x[::step, ::step],
        policy_eik_y[::step, ::step],
        color="#148f90",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0034,
        alpha=0.93,
    )
    axes[1, 0].set_title(
        rf"Eikonal Policy ($-\nabla V$) ({label})", fontsize=18, family="monospace"
    )
    _style_axes_maze(axes[1, 0], rects, x_min, x_max, y_min, y_max, goal_xy=goal_xy)

    axes[1, 1].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        policy_learned_x[::step, ::step],
        policy_learned_y[::step, ::step],
        color="#148f90",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0034,
        alpha=0.93,
    )
    axes[1, 1].set_title(
        rf"Learned Policy ($-\nabla V$) ({label})", fontsize=18, family="monospace"
    )
    _style_axes_maze(axes[1, 1], rects, x_min, x_max, y_min, y_max, goal_xy=goal_xy)

    plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=0.7)
    suffix = f"_{task_name}" if task_name else ""
    save_path = os.path.join(
        save_dir, f"antmaze_{maze_type}{suffix}_learned_analytic_contours.png"
    )
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    if log_wandb:
        try:
            import wandb

            key = (
                f"contours/antmaze_{task_name}"
                if task_name
                else "contours/antmaze_learned_analytic"
            )
            wandb.log({key: wandb.Image(save_path)})
        except Exception:
            pass

    return save_path
