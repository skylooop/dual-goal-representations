import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import heapq


def _build_grid(grid_size):
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0]
    return x, y, X, Y, h


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


def _segment_intersects_open_rectangle(p0, p1, rect):
    """True iff segment p0->p1 intersects rectangle interior (open set)."""
    xmin, ymin, xmax, ymax = rect

    def interval_open(a, da, lo, hi):
        if np.isclose(da, 0.0):
            if lo < a < hi:
                return -np.inf, np.inf
            return np.inf, -np.inf
        t1 = (lo - a) / da
        t2 = (hi - a) / da
        return min(t1, t2), max(t1, t2)

    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    tx0, tx1 = interval_open(x0, dx, xmin, xmax)
    ty0, ty1 = interval_open(y0, dy, ymin, ymax)

    t0 = max(tx0, ty0, 0.0)
    t1 = min(tx1, ty1, 1.0)
    return t0 < t1


def _is_visible(p0, p1, rects):
    for rect in rects:
        if _segment_intersects_open_rectangle(p0, p1, rect):
            return False
    return True


def _shortest_path_distance_rect_obstacles(point, goal, rects):
    """Analytic shortest path in free space around rectangular obstacles."""
    corners = []
    for rect in rects:
        corners.extend(
            [
                (rect[0], rect[1]),
                (rect[2], rect[1]),
                (rect[2], rect[3]),
                (rect[0], rect[3]),
            ]
        )

    nodes = [point, goal] + corners
    n_nodes = len(nodes)

    weights = np.full((n_nodes, n_nodes), np.inf)
    for i in range(n_nodes):
        weights[i, i] = 0.0

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if _is_visible(nodes[i], nodes[j], rects):
                dist = np.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                weights[i, j] = dist
                weights[j, i] = dist

    dist = np.full(n_nodes, np.inf)
    visited = np.zeros(n_nodes, dtype=bool)
    dist[0] = 0.0

    for _ in range(n_nodes):
        unvisited = np.where(~visited)[0]
        if unvisited.size == 0:
            break
        idx = unvisited[np.argmin(dist[unvisited])]
        if not np.isfinite(dist[idx]):
            break
        if idx == 1:
            break
        visited[idx] = True
        for j in range(n_nodes):
            if visited[j] or not np.isfinite(weights[idx, j]):
                continue
            cand = dist[idx] + weights[idx, j]
            if cand < dist[j]:
                dist[j] = cand

    return dist[1]


def solve_fk_value(grid_size, nu, obstacle_mask, goal_xy):
    x, y, _, _, _ = _build_grid(grid_size)
    return solve_fk_value_on_grid(x, y, obstacle_mask, goal_xy, nu)


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
            for ny, nx, coef in neighbors:
                if 0 <= ny < grid_size_y and 0 <= nx < grid_size_x:
                    nidx = ny * grid_size_x + nx
                    A[idx, nidx] = -coef
                    diag += coef

            A[idx, idx] = diag

    psi = np.asarray(spla.spsolve(A.tocsr(), b), dtype=float).reshape(
        (grid_size_y, grid_size_x)
    )
    psi = np.clip(psi, 1e-300, None)
    v = -2.0 * nu * np.log(psi)
    return v


def solve_eikonal_value_analytic(X, Y, obstacle_mask, goal_xy, rects):
    """Analytic Eikonal solution with unit speed and obstacle geometry."""
    v = np.full_like(X, np.nan, dtype=float)
    for iy in range(X.shape[0]):
        for ix in range(X.shape[1]):
            if obstacle_mask[iy, ix]:
                continue
            point = (float(X[iy, ix]), float(Y[iy, ix]))
            v[iy, ix] = _shortest_path_distance_rect_obstacles(point, goal_xy, rects)
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


def solve_optimal_value_grid(obstacle_mask, goal_idx, hx, hy):
    """Solve an optimal cost-to-go field on a grid via Dijkstra.

    This computes the exact optimum for the induced deterministic shortest-path MDP:
      - state: free grid cell
      - actions: 8-connected moves
      - stage cost: Euclidean move length

    The update is equivalent to a monotone HJB discretization solved exactly on the
    graph. We additionally disallow diagonal corner-cutting through obstacle corners.
    """
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

            # For diagonal moves, block "corner cutting" across two touching walls.
            if dy != 0 and dx != 0:
                if obstacle_mask[iy + dy, ix] or obstacle_mask[iy, ix + dx]:
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


def _style_axes_maze(ax, rects, x_min, x_max, y_min, y_max):
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
    ax.set_facecolor("#eef2f7")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def reproduce_antmaze_ogbench(
    maze_type,
    nu=3,
    points_per_cell=14,
    contour_percentile=99.5,
    quiver_scale=0.95,
    save_path=None,
):
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
    goal_xy = _ij_to_xy(goal_ij, maze_unit=maze_unit)

    v_fk = solve_fk_value_on_grid(x, y, obstacle_mask, goal_xy, nu)
    goal_idx = _nearest_free_idx(X, Y, obstacle_mask, goal_xy)
    v_eik = solve_eikonal_value_grid(obstacle_mask, goal_idx, hx, hy)
    v_opt = solve_optimal_value_grid(obstacle_mask, goal_idx, hx, hy)

    v_eik_grad = np.where(obstacle_mask, np.nanmax(v_eik[~obstacle_mask]), v_eik)
    v_fk_grad = np.where(obstacle_mask, np.nanmax(v_fk[~obstacle_mask]), v_fk)
    v_opt_grad = np.where(obstacle_mask, np.nanmax(v_opt[~obstacle_mask]), v_opt)
    policy_fk_x, policy_fk_y = _compute_policy(v_fk_grad, obstacle_mask, hy, hx)
    policy_eik_x, policy_eik_y = _compute_policy(v_eik_grad, obstacle_mask, hy, hx)
    policy_opt_x, policy_opt_y = _compute_policy(v_opt_grad, obstacle_mask, hy, hx)

    v_eik_max = np.nanmax(v_eik[~obstacle_mask])
    v_fk_max = np.nanmax(v_fk[~obstacle_mask])
    v_opt_max = np.nanmax(v_opt[~obstacle_mask])
    v_eik_plot = np.where(obstacle_mask, v_eik_max, v_eik)
    v_fk_plot = np.where(obstacle_mask, v_fk_max, v_fk)
    v_opt_plot = np.where(obstacle_mask, v_opt_max, v_opt)
    v_eik_lines = np.ma.masked_where(obstacle_mask, v_eik)
    v_fk_lines = np.ma.masked_where(obstacle_mask, v_fk)
    v_opt_lines = np.ma.masked_where(obstacle_mask, v_opt)

    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.3), dpi=170)
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
    axes[0, 0].set_title(
        f"Eikonal Value ({maze_type})", fontsize=18, family="monospace"
    )
    _style_axes_maze(axes[0, 0], rects, x_min, x_max, y_min, y_max)

    levels_fk = np.linspace(np.nanmin(v_fk_plot), np.nanmax(v_fk_plot), 26)
    axes[0, 1].contourf(
        X,
        Y,
        v_fk_plot,
        levels=levels_fk,
        cmap="Blues_r",
        corner_mask=False,
        extend="max",
    )
    axes[0, 1].contour(
        X, Y, v_fk_lines, levels=levels_fk, colors="black", linewidths=0.35, alpha=0.72
    )
    axes[0, 1].set_title(f"FK Value ({maze_type})", fontsize=18, family="monospace")
    _style_axes_maze(axes[0, 1], rects, x_min, x_max, y_min, y_max)

    levels_opt = np.linspace(np.nanmin(v_opt_plot), np.nanmax(v_opt_plot), 26)
    axes[0, 2].contourf(
        X,
        Y,
        v_opt_plot,
        levels=levels_opt,
        cmap="Blues_r",
        corner_mask=False,
        extend="max",
    )
    axes[0, 2].contour(
        X,
        Y,
        v_opt_lines,
        levels=levels_opt,
        colors="black",
        linewidths=0.35,
        alpha=0.72,
    )
    axes[0, 2].set_title(
        f"Optimal Value (DP/HJB grid) ({maze_type})",
        fontsize=16,
        family="monospace",
    )
    _style_axes_maze(axes[0, 2], rects, x_min, x_max, y_min, y_max)

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
        rf"Eikonal Policy ($-\nabla V$) ({maze_type})", fontsize=18, family="monospace"
    )
    _style_axes_maze(axes[1, 0], rects, x_min, x_max, y_min, y_max)

    axes[1, 1].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        policy_fk_x[::step, ::step],
        policy_fk_y[::step, ::step],
        color="#148f90",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0034,
        alpha=0.93,
    )
    axes[1, 1].set_title(
        rf"FK Policy ($-\nabla V$) ({maze_type})", fontsize=18, family="monospace"
    )
    _style_axes_maze(axes[1, 1], rects, x_min, x_max, y_min, y_max)

    axes[1, 2].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        policy_opt_x[::step, ::step],
        policy_opt_y[::step, ::step],
        color="#148f90",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0034,
        alpha=0.93,
    )
    axes[1, 2].set_title(
        rf"Optimal Policy ($-\nabla V$) ({maze_type})",
        fontsize=18,
        family="monospace",
    )
    _style_axes_maze(axes[1, 2], rects, x_min, x_max, y_min, y_max)

    plt.tight_layout(pad=0.7, w_pad=0.5, h_pad=0.7)
    if save_path is None:
        save_path = f"antmaze_{maze_type}_eikonal_fk_optimal.png"
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


def _style_axes(ax, obstacle_specs, halo_cover=0.0):
    for obstacle_x0, obstacle_y0, obstacle_size in obstacle_specs:
        if halo_cover > 0.0:
            ax.add_patch(
                Rectangle(
                    (obstacle_x0 - halo_cover, obstacle_y0 - halo_cover),
                    obstacle_size + 2.0 * halo_cover,
                    obstacle_size + 2.0 * halo_cover,
                    facecolor="#0d4cab",
                    edgecolor="none",
                    zorder=5.8,
                )
            )
        ax.add_patch(
            Rectangle(
                (obstacle_x0, obstacle_y0),
                obstacle_size,
                obstacle_size,
                facecolor="#0d4cab",
                edgecolor="#8ed8ea",
                linewidth=0.8,
                zorder=6,
            )
        )
        ax.text(
            obstacle_x0 + 0.5 * obstacle_size,
            obstacle_y0 + 0.5 * obstacle_size,
            "OBSTACLE",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            zorder=7,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor("#eef2f7")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def reproduce_figure3(
    grid_size=61,
    nu=0.08,
    goal_y=0.03,
    contour_percentile=99.0,
    quiver_scale=22,
    obstacle_specs=None,
    save_path="eikonal_fk_reproduction.png",
):
    _, _, X, Y, h = _build_grid(grid_size)

    if obstacle_specs is None:
        obstacle_specs = [(0.35, 0.35, 0.30)]

    goal_xy = (0.5, goal_y)
    rects = [(x0, y0, x0 + size, y0 + size) for x0, y0, size in obstacle_specs]
    obstacle_mask = _obstacle_mask_from_rects(X, Y, rects)

    v_fk = solve_fk_value(
        grid_size=grid_size, nu=nu, obstacle_mask=obstacle_mask, goal_xy=goal_xy
    )
    v_eik = solve_eikonal_value_analytic(X, Y, obstacle_mask, goal_xy, rects)

    policy_fk_x, policy_fk_y = _compute_policy(v_fk, obstacle_mask, h)
    policy_eik_x, policy_eik_y = _compute_policy(v_eik, obstacle_mask, h)

    # Avoid contour masking halos near obstacle edges by plotting a filled obstacle interior
    # and then overlaying the obstacle rectangle patch.
    v_eik_max = np.nanmax(v_eik[~obstacle_mask])
    v_fk_max = np.nanmax(v_fk[~obstacle_mask])
    v_eik_plot = np.where(obstacle_mask, v_eik_max, v_eik)
    v_fk_plot = np.where(obstacle_mask, v_fk_max, v_fk)
    v_fk_np = v_fk_plot
    v_eik_np = v_eik_plot
    v_eik_lines = np.ma.masked_where(obstacle_mask, v_eik)
    v_fk_lines = np.ma.masked_where(obstacle_mask, v_fk)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), dpi=180)
    fig.patch.set_facecolor("#e9e9e9")

    levels_eik = np.linspace(
        np.nanmin(v_eik_np), np.nanpercentile(v_eik_np, contour_percentile), 22
    )
    axes[0, 0].contourf(
        X, Y, v_eik_plot, levels=levels_eik, cmap="Blues_r", corner_mask=False
    )
    axes[0, 0].contour(
        X,
        Y,
        v_eik_lines,
        levels=levels_eik,
        colors="black",
        linewidths=0.42,
        alpha=0.80,
    )
    axes[0, 0].set_title("Eikonal Value", fontsize=20, family="monospace")
    _style_axes(axes[0, 0], obstacle_specs, halo_cover=0.006)

    levels_fk = np.linspace(
        np.nanmin(v_fk_np), np.nanpercentile(v_fk_np, contour_percentile), 22
    )
    axes[0, 1].contourf(
        X, Y, v_fk_plot, levels=levels_fk, cmap="Blues_r", corner_mask=False
    )
    axes[0, 1].contour(
        X, Y, v_fk_lines, levels=levels_fk, colors="black", linewidths=0.42, alpha=0.80
    )
    axes[0, 1].set_title("FK Value", fontsize=20, family="monospace")
    _style_axes(axes[0, 1], obstacle_specs, halo_cover=0.006)

    step = 3
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
        alpha=0.96,
    )
    axes[1, 0].set_title(
        r"Eikonal Policy ($-\nabla V$)", fontsize=20, family="monospace"
    )
    _style_axes(axes[1, 0], obstacle_specs)

    axes[1, 1].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        policy_fk_x[::step, ::step],
        policy_fk_y[::step, ::step],
        color="#148f90",
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        width=0.0034,
        alpha=0.96,
    )
    axes[1, 1].set_title(r"FK Policy ($-\nabla V$)", fontsize=20, family="monospace")
    _style_axes(axes[1, 1], obstacle_specs)

    plt.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.7)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    reproduce_figure3(save_path="eikonal_fk_reproduction.png")
    reproduce_figure3(
        obstacle_specs=[(0.35, 0.35, 0.30), (0.16, 0.62, 0.18)],
        save_path="eikonal_fk_reproduction_two_obstacles.png",
    )
    reproduce_antmaze_ogbench(
        maze_type="medium",
        save_path="antmaze_medium_eikonal_fk.png",
    )
    reproduce_antmaze_ogbench(
        maze_type="large",
        save_path="antmaze_large_eikonal_fk.png",
    )
