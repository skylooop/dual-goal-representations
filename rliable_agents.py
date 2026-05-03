"""Compare multiple pretrained agents using RLiable metrics.

Reads score_matrix.npy files produced by eval_trained.py and generates
multi-algorithm comparison plots following the RLiable paper.

Usage:
    # Compare two agents evaluated on the same env:
    uv run compare_agents.py \
        --results_dirs \
            eval_results/puzzle-3x3-play-v0_crl_dual_sd0_step31000 \
            eval_results/puzzle-3x3-play-v0_gcfbc_dual_sd0_step50000 \
        --labels "CRL-Dual" "GCFBC-Dual" \
        --output_dir comparison_results/puzzle-3x3

    # Or point to eval_results.json files directly:
    uv run compare_agents.py \
        --results_dirs dir1 dir2 dir3 \
        --labels "Algo A" "Algo B" "Algo C"
"""

import argparse
import json
import os
import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.rliable import (
    aggregate_iqm,
    aggregate_mean,
    aggregate_median,
    aggregate_optimality_gap,
    get_interval_estimates,
    score_distribution,
    probability_of_improvement,
)


# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly, from Tol's qualitative scheme)
# ---------------------------------------------------------------------------
COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # red/pink
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]


def load_results(results_dir):
    """Load score matrix and metadata from an eval_trained.py output directory."""
    score_path = os.path.join(results_dir, "score_matrix.npy")
    meta_path = os.path.join(results_dir, "eval_results.json")

    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f"score_matrix.npy not found in {results_dir}. Run eval_trained.py first."
        )

    scores = np.load(score_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return scores, meta


# ---------------------------------------------------------------------------
# Multi-algorithm plots
# ---------------------------------------------------------------------------


def plot_aggregate_comparison(
    all_results, labels, save_path, num_bootstraps=50_000, seed=0
):
    """Grouped horizontal bar chart: IQM, Mean, Median per algorithm with CIs.

    Args:
        all_results: dict {label: score_matrix}
        labels: ordered list of algorithm labels
        save_path: output path
    """
    metric_fns = {
        "IQM": aggregate_iqm,
        "Mean": aggregate_mean,
        "Median": aggregate_median,
    }
    metric_names = list(metric_fns.keys())
    n_metrics = len(metric_names)
    n_algos = len(labels)

    # Compute all (point, CI) pairs.
    data = {}  # {label: {metric: (point, (lo, hi))}}
    for label in labels:
        data[label] = {}
        for mname, mfn in metric_fns.items():
            point, ci = get_interval_estimates(
                all_results[label],
                mfn,
                num_bootstraps=num_bootstraps,
                seed=seed,
            )
            data[label][mname] = (point, ci)

    # Plot.
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(4 * n_metrics, max(3, 0.6 * n_algos + 1.5))
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, mname in zip(axes, metric_names):
        y_pos = np.arange(n_algos)
        points = [data[l][mname][0] for l in labels]
        ci_lows = [data[l][mname][1][0] for l in labels]
        ci_highs = [data[l][mname][1][1] for l in labels]
        errs_lo = [p - lo for p, lo in zip(points, ci_lows)]
        errs_hi = [hi - p for p, hi in zip(points, ci_highs)]

        colors = [COLORS[i % len(COLORS)] for i in range(n_algos)]
        ax.barh(
            y_pos,
            points,
            xerr=[errs_lo, errs_hi],
            color=colors,
            edgecolor="white",
            height=0.5,
            capsize=4,
            error_kw=dict(lw=1.5, capthick=1.5),
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_xlim(0, max(1.05, max(ci_highs) * 1.15) if max(ci_highs) > 0 else 1.05)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(mname, fontsize=12, fontweight="bold")

    fig.suptitle("Aggregate Metrics (95% Stratified Bootstrap CI)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_performance_profiles(all_results, labels, save_path, num_points=101):
    """Overlaid performance profile curves for each algorithm.

    Args:
        all_results: dict {label: score_matrix}
        labels: ordered list of algorithm labels
        save_path: output path
    """
    thresholds = np.linspace(0, 1, num_points)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, label in enumerate(labels):
        fractions = score_distribution(all_results[label], thresholds)
        color = COLORS[i % len(COLORS)]
        ax.plot(thresholds, fractions, lw=2.2, color=color, label=label)
        ax.fill_between(thresholds, 0, fractions, alpha=0.08, color=color)

    ax.set_xlabel("Score threshold τ", fontsize=11)
    ax.set_ylabel("Fraction of runs with score ≥ τ", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Performance Profile", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_probability_of_improvement(all_results, labels, save_path):
    """Heatmap of pairwise P(X > Y) across all algorithm pairs.

    Args:
        all_results: dict {label: score_matrix}
        labels: ordered list of algorithm labels
        save_path: output path
    """
    n = len(labels)
    if n < 2:
        return  # Need at least 2 algorithms.

    matrix = np.full((n, n), 0.5)
    for i, j in itertools.combinations(range(n), 2):
        p = probability_of_improvement(all_results[labels[i]], all_results[labels[j]])
        matrix[i, j] = p
        matrix[j, i] = 1 - p

    fig, ax = plt.subplots(figsize=(max(4, 0.9 * n + 2), max(3.5, 0.9 * n + 1.5)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells.
    for i in range(n):
        for j in range(n):
            color = "white" if abs(matrix[i, j] - 0.5) > 0.3 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("P(Row > Column)", fontsize=10)
    ax.set_title("Probability of Improvement", fontsize=12)
    ax.set_xlabel("Algorithm (column)", fontsize=10)
    ax.set_ylabel("Algorithm (row)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_task_comparison(all_results, labels, task_names, save_path):
    """Grouped bar chart: per-task success rate for each algorithm.

    Args:
        all_results: dict {label: score_matrix}
        labels: ordered list of algorithm labels
        task_names: list of task names (columns of score matrix)
        save_path: output path
    """
    n_tasks = len(task_names)
    n_algos = len(labels)
    width = 0.8 / n_algos
    x = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 1.2), 4.5))
    for i, label in enumerate(labels):
        means = np.mean(all_results[label], axis=0)
        stds = np.std(all_results[label], axis=0)
        offset = (i - (n_algos - 1) / 2) * width
        color = COLORS[i % len(COLORS)]
        ax.bar(
            x + offset,
            means,
            width * 0.9,
            yerr=stds,
            color=color,
            edgecolor="white",
            label=label,
            capsize=2,
            error_kw=dict(lw=1),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Success Rate", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-Task Success Rate (mean ± std)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple agents using RLiable metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dirs",
        nargs="+",
        required=True,
        help="Paths to eval_trained.py output directories (each must contain score_matrix.npy).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Human-readable labels for each algorithm (same order as --results_dirs). "
        "If omitted, inferred from eval_results.json or directory name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Where to save comparison plots and results.",
    )
    parser.add_argument(
        "--num_bootstraps",
        type=int,
        default=50_000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap.",
    )
    args = parser.parse_args()

    n = len(args.results_dirs)
    assert n >= 2, "Need at least 2 result directories to compare."

    # Load all results.
    all_results = {}
    all_meta = {}
    labels = []

    for i, rdir in enumerate(args.results_dirs):
        scores, meta = load_results(rdir)
        # Determine label.
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        elif meta.get("agent_name"):
            label = meta["agent_name"]
        else:
            label = os.path.basename(rdir.rstrip("/"))
        labels.append(label)
        all_results[label] = scores
        all_meta[label] = meta

    # Infer task names from first meta, or generate generic ones.
    first_meta = all_meta[labels[0]]
    if "per_task" in first_meta:
        task_names = list(first_meta["per_task"].keys())
    else:
        n_tasks = all_results[labels[0]].shape[1]
        task_names = [f"task{i + 1}" for i in range(n_tasks)]

    os.makedirs(args.output_dir, exist_ok=True)

    # Print summary.
    print(f"\n{'=' * 60}")
    print(f"  Comparing {n} algorithms")
    print(f"{'=' * 60}")
    for label in labels:
        s = all_results[label]
        print(f"  {label}: score_matrix shape = {s.shape}")
    print()

    # Compute aggregate metrics.
    metric_fns = {
        "IQM": aggregate_iqm,
        "Mean": aggregate_mean,
        "Median": aggregate_median,
        "Optimality Gap": aggregate_optimality_gap,
    }

    comparison_data = {}
    for label in labels:
        comparison_data[label] = {}
        for mname, mfn in metric_fns.items():
            point, ci = get_interval_estimates(
                all_results[label],
                mfn,
                num_bootstraps=args.num_bootstraps,
                seed=args.seed,
            )
            comparison_data[label][mname] = {
                "point": point,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }

    # Print metrics table.
    print(f"  {'Algorithm':<20}", end="")
    for mname in metric_fns:
        print(f"  {mname:>22}", end="")
    print()
    print(f"  {'─' * (20 + 24 * len(metric_fns))}")
    for label in labels:
        print(f"  {label:<20}", end="")
        for mname in metric_fns:
            d = comparison_data[label][mname]
            print(
                f"  {d['point']:>6.4f} [{d['ci_low']:.3f},{d['ci_high']:.3f}]", end=""
            )
        print()

    # Pairwise P(improvement).
    if n >= 2:
        print(f"\n  Pairwise P(Row > Column):")
        print(f"  {'':>20}", end="")
        for l in labels:
            print(f"  {l:>15}", end="")
        print()
        for i, li in enumerate(labels):
            print(f"  {li:>20}", end="")
            for j, lj in enumerate(labels):
                if i == j:
                    print(f"  {'—':>15}", end="")
                else:
                    p = probability_of_improvement(all_results[li], all_results[lj])
                    print(f"  {p:>15.3f}", end="")
            print()
    print()

    # Save JSON.
    results_json = {
        "labels": labels,
        "metrics": comparison_data,
        "pairwise_improvement": {},
    }
    for i, j in itertools.combinations(range(n), 2):
        p = probability_of_improvement(all_results[labels[i]], all_results[labels[j]])
        results_json["pairwise_improvement"][f"{labels[i]} > {labels[j]}"] = p
        results_json["pairwise_improvement"][f"{labels[j]} > {labels[i]}"] = 1 - p

    with open(os.path.join(args.output_dir, "comparison_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {args.output_dir}/comparison_results.json")

    # Generate plots.
    print("Generating plots...")

    plot_aggregate_comparison(
        all_results,
        labels,
        os.path.join(args.output_dir, "aggregate_comparison.png"),
        num_bootstraps=args.num_bootstraps,
        seed=args.seed,
    )
    print(f"  ✓ {args.output_dir}/aggregate_comparison.png")

    plot_performance_profiles(
        all_results,
        labels,
        os.path.join(args.output_dir, "performance_profiles.png"),
    )
    print(f"  ✓ {args.output_dir}/performance_profiles.png")

    plot_probability_of_improvement(
        all_results,
        labels,
        os.path.join(args.output_dir, "probability_of_improvement.png"),
    )
    print(f"  ✓ {args.output_dir}/probability_of_improvement.png")

    plot_per_task_comparison(
        all_results,
        labels,
        task_names,
        os.path.join(args.output_dir, "per_task_comparison.png"),
    )
    print(f"  ✓ {args.output_dir}/per_task_comparison.png")

    print(f"\nDone! All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
