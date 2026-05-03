"""Minimal implementation of RLiable metrics (Agarwal et al., 2021).

Reference: https://agarwl.github.io/rliable/

Provides IQM, Mean, Median, Optimality Gap with stratified bootstrap
confidence intervals — no external library needed beyond numpy/scipy.
"""

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def aggregate_iqm(scores: np.ndarray) -> float:
    """Interquartile mean (25% trimmed mean) across all runs and tasks.

    Args:
        scores: (num_runs, num_tasks) matrix of per-run, per-task scores.
    """
    return float(scipy.stats.trim_mean(scores.flatten(), proportiontocut=0.25))


def aggregate_mean(scores: np.ndarray) -> float:
    """Mean of per-task means."""
    return float(np.mean(np.mean(scores, axis=0)))


def aggregate_median(scores: np.ndarray) -> float:
    """Median of per-task means."""
    return float(np.median(np.mean(scores, axis=0)))


def aggregate_optimality_gap(scores: np.ndarray, gamma: float = 1.0) -> float:
    """Optimality gap: γ − mean(min(score, γ))."""
    return float(gamma - np.mean(np.minimum(scores, gamma)))


# ---------------------------------------------------------------------------
# Stratified bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _stratified_bootstrap_sample(
    scores: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one stratified bootstrap sample.

    Resamples *runs* (rows) independently for each task (column),
    preserving the task structure as recommended by the RLiable paper.

    Args:
        scores: (num_runs, num_tasks) matrix.
        rng: numpy random generator.

    Returns:
        (num_runs, num_tasks) bootstrapped matrix.
    """
    num_runs, num_tasks = scores.shape
    boot = np.empty_like(scores)
    for t in range(num_tasks):
        idx = rng.integers(0, num_runs, size=num_runs)
        boot[:, t] = scores[idx, t]
    return boot


def get_interval_estimates(
    scores: np.ndarray,
    metric_fn,
    num_bootstraps: int = 50_000,
    confidence_level: float = 0.95,
    seed: int = 0,
):
    """Compute a metric point estimate and its bootstrap confidence interval.

    Args:
        scores: (num_runs, num_tasks) matrix.
        metric_fn: One of aggregate_iqm, aggregate_mean, etc.
        num_bootstraps: Number of bootstrap replicates.
        confidence_level: CI level (default 95%).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, (ci_low, ci_high))
    """
    rng = np.random.default_rng(seed)
    point = metric_fn(scores)

    boot_values = np.empty(num_bootstraps)
    for b in range(num_bootstraps):
        sample = _stratified_bootstrap_sample(scores, rng)
        boot_values[b] = metric_fn(sample)

    alpha = (1 - confidence_level) / 2
    ci_low = float(np.percentile(boot_values, 100 * alpha))
    ci_high = float(np.percentile(boot_values, 100 * (1 - alpha)))

    return point, (ci_low, ci_high)


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray) -> float:
    """P(X > Y) averaged across tasks using Mann-Whitney U test.

    Args:
        scores_x: (num_runs_x, num_tasks) scores for algorithm X.
        scores_y: (num_runs_y, num_tasks) scores for algorithm Y.

    Returns:
        Average probability that X beats Y across tasks.
    """
    num_tasks = scores_x.shape[1]
    probs = []
    n_x, n_y = scores_x.shape[0], scores_y.shape[0]
    for t in range(num_tasks):
        if np.array_equal(scores_x[:, t], scores_y[:, t]):
            probs.append(0.5)
        else:
            u_stat, _ = scipy.stats.mannwhitneyu(
                scores_x[:, t],
                scores_y[:, t],
                alternative="greater",
            )
            probs.append(u_stat / (n_x * n_y))
    return float(np.mean(probs))


# ---------------------------------------------------------------------------
# Score distribution (for performance profiles)
# ---------------------------------------------------------------------------


def score_distribution(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Fraction of runs×tasks achieving score ≥ τ for each threshold τ.

    Args:
        scores: (num_runs, num_tasks) matrix.
        thresholds: 1-D array of threshold values τ.

    Returns:
        1-D array of fractions, same length as thresholds.
    """
    flat = scores.flatten()
    return np.array([np.mean(flat >= tau) for tau in thresholds])
