import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _build_grid(grid_size=61):
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0]
    return x, y, X, Y, h


def _obstacle_mask(X, Y, rects):
    mask = np.zeros_like(X, dtype=bool)
    for xmin, ymin, xmax, ymax in rects:
        mask |= (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    return mask


def _segment_intersects_open_rectangle(p0, p1, rect):
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


def _shortest_path_distance(point, goal, rects):
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
    n = len(nodes)
    w = np.full((n, n), np.inf)
    for i in range(n):
        w[i, i] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if _is_visible(nodes[i], nodes[j], rects):
                d = np.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                w[i, j] = d
                w[j, i] = d

    dist = np.full(n, np.inf)
    used = np.zeros(n, dtype=bool)
    dist[0] = 0.0
    for _ in range(n):
        idx = np.argmin(np.where(used, np.inf, dist))
        if used[idx] or not np.isfinite(dist[idx]):
            break
        if idx == 1:
            break
        used[idx] = True
        for j in range(n):
            if used[j] or not np.isfinite(w[idx, j]):
                continue
            cand = dist[idx] + w[idx, j]
            if cand < dist[j]:
                dist[j] = cand
    return dist[1]


def solve_hota_value_analytic(X, Y, rects, goal_xy):
    """
    HOTA-like static HJB solution via Hopf-Lax:
        H(p)=0.5||p||^2,  V(x)=0.5 * d_geo(x, g)^2
    where d_geo is geodesic distance in free space with rectangular obstacles.
    """
    obstacle_mask = _obstacle_mask(X, Y, rects)
    d_geo = np.full_like(X, np.nan, dtype=float)
    for iy in range(X.shape[0]):
        for ix in range(X.shape[1]):
            if obstacle_mask[iy, ix]:
                continue
            p = (float(X[iy, ix]), float(Y[iy, ix]))
            d_geo[iy, ix] = _shortest_path_distance(p, goal_xy, rects)

    v = 0.5 * d_geo * d_geo
    return v, obstacle_mask


def _policy_from_value(v, obstacle_mask, h):
    v_fill = np.where(obstacle_mask, np.nanmax(v[~obstacle_mask]), v)
    dv_dy, dv_dx = np.gradient(v_fill, h, h)
    px = -dv_dx
    py = -dv_dy
    norm = np.hypot(px, py) + 1e-12
    px /= norm
    py /= norm
    px[obstacle_mask] = np.nan
    py[obstacle_mask] = np.nan
    return px, py


def _draw_obstacles(ax, obstacle_specs, halo_cover=0.006):
    for x0, y0, size in obstacle_specs:
        ax.add_patch(
            Rectangle(
                (x0 - halo_cover, y0 - halo_cover),
                size + 2.0 * halo_cover,
                size + 2.0 * halo_cover,
                facecolor="#0d4cab",
                edgecolor="none",
                zorder=5.8,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0, y0),
                size,
                size,
                facecolor="#0d4cab",
                edgecolor="#8ed8ea",
                linewidth=0.8,
                zorder=6,
            )
        )
        ax.text(
            x0 + 0.5 * size,
            y0 + 0.5 * size,
            "OBSTACLE",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            zorder=7,
        )


def plot_hota(obstacle_specs, save_path):
    _, _, X, Y, h = _build_grid(grid_size=61)
    goal_xy = (0.5, 0.03)
    rects = [(x0, y0, x0 + size, y0 + size) for x0, y0, size in obstacle_specs]

    v, obstacle_mask = solve_hota_value_analytic(X, Y, rects, goal_xy)
    px, py = _policy_from_value(v, obstacle_mask, h)

    v_max = np.nanmax(v[~obstacle_mask])
    v_plot = np.where(obstacle_mask, v_max, v)
    v_lines = np.ma.masked_where(obstacle_mask, v)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9), dpi=180)
    fig.patch.set_facecolor("#e9e9e9")

    levels = np.linspace(np.nanmin(v_plot), np.nanpercentile(v_plot, 99.0), 24)
    axes[0].contourf(X, Y, v_plot, levels=levels, cmap="Blues_r", corner_mask=False)
    axes[0].contour(
        X, Y, v_lines, levels=levels, colors="black", linewidths=0.42, alpha=0.80
    )
    axes[0].set_title("HOTA Value", fontsize=19, family="monospace")

    step = 3
    axes[1].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        px[::step, ::step],
        py[::step, ::step],
        color="#148f90",
        angles="xy",
        scale_units="xy",
        scale=22,
        width=0.0034,
        alpha=0.96,
    )
    axes[1].set_title(r"HOTA Policy ($-\nabla V$)", fontsize=19, family="monospace")

    for ax in axes:
        _draw_obstacles(ax, obstacle_specs)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_facecolor("#eef2f7")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.9, w_pad=0.8)
    fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    plot_hota(obstacle_specs=[(0.35, 0.35, 0.30)], save_path="hota_2d_one_obstacle.png")
    plot_hota(
        obstacle_specs=[(0.35, 0.35, 0.30), (0.16, 0.62, 0.18)],
        save_path="hota_2d_two_obstacles.png",
    )
