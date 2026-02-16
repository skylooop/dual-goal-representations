"""Evaluate a trained agent from an Orbax checkpoint.

Reports results following the RLiable methodology (Agarwal et al., 2021):
  - IQM (Interquartile Mean) with 95% stratified bootstrap CIs
  - Mean, Median, Optimality Gap
  - Performance profile plot

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run eval_trained.py \
        --env_name=puzzle-3x3-play-v0 \
        --restore_path=results/.../checkpoints \
        --restore_epoch=31000 \
        --eval_episodes=50
"""

import sys
import rootutils
import warnings

ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

# Agent config (same as main.py).
config_flags.DEFINE_config_file('agent', 'agents/crl/dual.py', lock_config=False)

# General.
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('restore_path', None, 'Path to the checkpoint directory (required).', required=True)
flags.DEFINE_integer('restore_epoch', None, 'Checkpoint step to restore (None for latest).')

# Eval hypers.
flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 25, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0.0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_float('eval_goal_gaussian', None, 'Goal Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

# RLiable.
flags.DEFINE_integer('num_bootstraps', 50_000, 'Number of bootstrap replicates for CIs.')
flags.DEFINE_float('confidence_level', 0.95, 'Confidence level for bootstrap CIs.')

# Output.
flags.DEFINE_string('save_dir', 'eval_results/', 'Directory to save evaluation results.')


import json
import os
import random
from collections import defaultdict

import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from colorama import Fore, Style

from agents import agents
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.rliable import (
    aggregate_iqm,
    aggregate_mean,
    aggregate_median,
    aggregate_optimality_gap,
    get_interval_estimates,
    score_distribution,
)


def plot_aggregate_metrics(metrics_dict, save_path):
    """Bar chart of aggregate metrics with bootstrap CIs (RLiable style).

    Args:
        metrics_dict: {metric_name: (point, (ci_low, ci_high))}
        save_path: Where to save the figure.
    """
    names = list(metrics_dict.keys())
    points = [metrics_dict[n][0] for n in names]
    ci_lows = [metrics_dict[n][1][0] for n in names]
    ci_highs = [metrics_dict[n][1][1] for n in names]

    errors_low = [p - lo for p, lo in zip(points, ci_lows)]
    errors_high = [hi - p for p, hi in zip(points, ci_highs)]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    y_pos = np.arange(len(names))
    ax.barh(
        y_pos, points, xerr=[errors_low, errors_high],
        color='#5B8FF9', edgecolor='white', height=0.5,
        capsize=4, error_kw=dict(lw=1.5, capthick=1.5),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Score', fontsize=11)
    ax.set_xlim(0, max(1.05, max(ci_highs) * 1.1))
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Aggregate Metrics (95% Stratified Bootstrap CI)', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_performance_profile(scores, save_path, num_points=101):
    """Performance profile: fraction of (run, task) pairs achieving score >= τ.

    Args:
        scores: (num_runs, num_tasks) matrix.
        save_path: Where to save the figure.
        num_points: Number of threshold points.
    """
    thresholds = np.linspace(0, 1, num_points)
    fractions = score_distribution(scores, thresholds)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(thresholds, fractions, lw=2, color='#5B8FF9')
    ax.fill_between(thresholds, 0, fractions, alpha=0.15, color='#5B8FF9')
    ax.set_xlabel('Score threshold τ', fontsize=11)
    ax.set_ylabel('Fraction of runs with score ≥ τ', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Performance Profile', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_task_scores(task_names, per_task_means, per_task_stds, save_path):
    """Per-task success rate bar chart with ±1 std error bars.

    Args:
        task_names: List of task names.
        per_task_means: Per-task mean success rates.
        per_task_stds: Per-task std of success.
        save_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(task_names) * 0.8), 4))
    x = np.arange(len(task_names))
    ax.bar(
        x, per_task_means, yerr=per_task_stds,
        color='#61DDAA', edgecolor='white', width=0.6,
        capsize=3, error_kw=dict(lw=1.2),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Success Rate', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Per-Task Success Rate (mean ± std)', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    config = FLAGS.agent

    # Set up output directory.
    step_tag = FLAGS.restore_epoch if FLAGS.restore_epoch is not None else "latest"
    exp_name = f"{FLAGS.env_name}_{config['agent_name']}_sd{FLAGS.seed}_step{step_tag}"
    save_dir = os.path.join(FLAGS.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Build environment and dataset (needed to construct agent with correct shapes).
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config['frame_stack']
    )
    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'VIPDataset': VIPDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(
        Dataset.create(norm=config['norm'], **train_dataset), config
    )
    diff = train_dataset.get_diff()

    # Create agent (template for restore).
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        example_batch['actions'] = np.full_like(
            example_batch['actions'], env.action_space.n - 1
        )

    agent_class = agents[config['agent_name']]
    ex_goals = example_batch['value_goals'] if config['oraclerep'] else None
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        ex_goals=ex_goals,
    )

    # Restore trained weights.
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    print(f"{Fore.GREEN}Restored agent from step {step_tag}{Style.RESET_ALL}")
    print(f"  Network train step: {agent.network.step}")

    if FLAGS.eval_on_cpu:
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    # -----------------------------------------------------------------------
    # Evaluate — collect per-episode success for each task
    # -----------------------------------------------------------------------
    task_infos = (
        env.unwrapped.task_infos
        if hasattr(env.unwrapped, 'task_infos')
        else env.task_infos
    )
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
    num_episodes = FLAGS.eval_episodes

    print(f"\n{Fore.CYAN}Evaluating on {num_tasks} task(s), "
          f"{num_episodes} episode(s) each{Style.RESET_ALL}\n")

    task_names = []
    # Collect per-episode success: rows = episodes, cols = tasks
    all_episode_successes = []  # list of 1-D arrays, one per task
    renders = []

    for task_id in tqdm(
        range(1, num_tasks + 1),
        desc='Evaluating',
        dynamic_ncols=True,
        colour='yellow',
    ):
        task_name = task_infos[task_id - 1]['task_name']
        task_names.append(task_name)

        eval_info, trajs, cur_renders = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=num_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_temperature=FLAGS.eval_temperature,
            eval_gaussian=FLAGS.eval_gaussian,
            eval_goal_gaussian=FLAGS.eval_goal_gaussian,
            diff=diff,
        )
        renders.extend(cur_renders)

        # Extract per-episode success from trajectories.
        # Each traj's last info dict has the 'success' key.
        ep_successes = []
        for traj in trajs:
            # traj['info'] is a list of info dicts (one per step);
            # the last element contains the final 'success'.
            last_info = {k: v[-1] for k, v in traj.items() if k == 'info'}
            # flatten: traj stores info as dict-of-lists via add_to
            final_info = traj['info'][-1]  # last step info dict
            success = final_info.get('success', 0.0)
            ep_successes.append(float(success))
        all_episode_successes.append(np.array(ep_successes))

    # -----------------------------------------------------------------------
    # Build RLiable score matrix: (num_episodes, num_tasks)
    # -----------------------------------------------------------------------
    # Each column = one task, each row = one episode (treated as a "run").
    score_matrix = np.stack(all_episode_successes, axis=1)  # (num_episodes, num_tasks)

    # -----------------------------------------------------------------------
    # Compute RLiable aggregate metrics with bootstrap CIs
    # -----------------------------------------------------------------------
    print(f"\n{Fore.CYAN}Computing RLiable metrics "
          f"({FLAGS.num_bootstraps} bootstrap replicates)...{Style.RESET_ALL}")

    metrics_fns = {
        'IQM': aggregate_iqm,
        'Mean': aggregate_mean,
        'Median': aggregate_median,
        'Optimality Gap': aggregate_optimality_gap,
    }

    rliable_results = {}
    for name, fn in metrics_fns.items():
        point, ci = get_interval_estimates(
            score_matrix, fn,
            num_bootstraps=FLAGS.num_bootstraps,
            confidence_level=FLAGS.confidence_level,
            seed=FLAGS.seed,
        )
        rliable_results[name] = (point, ci)

    # Per-task stats.
    per_task_means = np.mean(score_matrix, axis=0)
    per_task_stds = np.std(score_matrix, axis=0)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n{Fore.GREEN}{Style.BRIGHT}{'='*60}")
    print(f"  RLiable Evaluation Results (step {step_tag})")
    print(f"  {FLAGS.env_name}  |  {config['agent_name']}  |  seed {FLAGS.seed}")
    print(f"  {num_episodes} episodes × {num_tasks} tasks")
    print(f"{'='*60}{Style.RESET_ALL}")

    print(f"\n  {'Metric':<20} {'Point':>8}  {'95% CI':>20}")
    print(f"  {'─'*50}")
    for name, (point, (lo, hi)) in rliable_results.items():
        print(f"  {name:<20} {point:>8.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\n  {'Task':<30} {'Mean':>8}  {'Std':>8}")
    print(f"  {'─'*50}")
    for i, tname in enumerate(task_names):
        print(f"  {tname:<30} {per_task_means[i]:>8.4f}  {per_task_stds[i]:>8.4f}")
    print()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        'env_name': FLAGS.env_name,
        'agent_name': config['agent_name'],
        'seed': FLAGS.seed,
        'step': int(step_tag) if isinstance(step_tag, int) else str(step_tag),
        'num_episodes': num_episodes,
        'num_tasks': num_tasks,
        'rliable': {
            name: {'point': point, 'ci_low': ci[0], 'ci_high': ci[1]}
            for name, (point, ci) in rliable_results.items()
        },
        'per_task': {
            tname: {
                'mean': float(per_task_means[i]),
                'std': float(per_task_stds[i]),
            }
            for i, tname in enumerate(task_names)
        },
    }
    results_path = os.path.join(save_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON saved to {results_path}")

    # Save raw score matrix for later aggregation across seeds.
    scores_path = os.path.join(save_dir, 'score_matrix.npy')
    np.save(scores_path, score_matrix)
    print(f"Score matrix ({score_matrix.shape}) saved to {scores_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_aggregate_metrics(
        rliable_results,
        os.path.join(save_dir, 'aggregate_metrics.png'),
    )
    print(f"Plot saved: {save_dir}/aggregate_metrics.png")

    plot_performance_profile(
        score_matrix,
        os.path.join(save_dir, 'performance_profile.png'),
    )
    print(f"Plot saved: {save_dir}/performance_profile.png")

    plot_per_task_scores(
        task_names, per_task_means, per_task_stds,
        os.path.join(save_dir, 'per_task_scores.png'),
    )
    print(f"Plot saved: {save_dir}/per_task_scores.png")

    # Video renders.
    if FLAGS.video_episodes > 0 and len(renders) > 0:
        np.save(
            os.path.join(save_dir, 'eval_renders.npy'),
            np.array(renders, dtype=object), allow_pickle=True,
        )
        print(f"Video renders saved to {save_dir}/eval_renders.npy")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}Evaluation complete!{Style.RESET_ALL}")


def entry(argv):
    sys.argv = argv
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user.{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}Done!{Style.RESET_ALL}")


if __name__ == "__main__":
    app.run(entry)
