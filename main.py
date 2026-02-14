import sys
import rootutils
import warnings

ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_bool('disable_jit', False, 'Whether to disable JIT compilation.')

import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
from tqdm.auto import tqdm
from colorama import Fore, Style
import wandb
from agents import agents
from ml_collections import config_flags
import threading, queue

from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict,\
    get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

# WANDB & LOGGING
flags.DEFINE_string('run_group', 'Debug', 'Run group.')

# GENERAL
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'results/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_string('wandb_mode', 'online', 'Wandb mode.')
# TRAIN HYPERS
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 10_000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

# EVAL HYPERS
flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 25, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0.0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_float('eval_goal_gaussian', None, 'Goal Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

config_flags.DEFINE_config_file('agent', 'agents/crl/dual.py', lock_config=False)


def main():
    # Set up logger.
    config = FLAGS.agent
    exp_name = get_exp_name(FLAGS.env_name, config['agent_name'], FLAGS.seed)
    setup_wandb(project='dual_goal_reprs-Research', mode=FLAGS.wandb_mode,
                group=FLAGS.agent.agent_name, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project,
                                  FLAGS.agent.agent_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    if 'oraclerep' in FLAGS.env_name and config['oraclerep'] == False:
        raise ValueError('Must enable oracle representation in config dictionary to use this environment!')

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'VIPDataset': VIPDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(norm=config['norm'], **train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(norm=config['norm'], **val_dataset), config)
    # Need to pass into evaluation functions
    diff = train_dataset.get_diff()

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    ex_goals = example_batch['value_goals'] if config['oraclerep'] else None
    agent = agent_class.create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config, ex_goals=ex_goals
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    # Prefetch: High-speed host sampling + Async staging
    import threading, queue
    prefetch_queue = queue.Queue(maxsize=4)

    def _prefetch_worker():
        while True:
            try:
                # 1. Sample on CPU (Fast NumPy RAM access)
                batch = train_dataset.sample(config['batch_size'])
                # 2. Stage on GPU (Large single H2D block transfer)
                batch = jax.device_put(batch)
                prefetch_queue.put(batch)
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
    prefetch_thread.start()

    for i in tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True, desc='Training', colour='green', leave=True, position=0):
        # Update agent.
        batch = prefetch_queue.get()
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            
            for task_id in tqdm(range(1, num_tasks + 1), desc='Evaluating',
                                leave=False, dynamic_ncols=True, colour='yellow', position=1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    eval_goal_gaussian=FLAGS.eval_goal_gaussian,
                    diff=diff,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


def entry(argv):
    sys.argv = argv
    disable_jit = FLAGS.disable_jit
    
    try:
        if disable_jit:
            with jax.disable_jit():
                main()
        else:
            main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user.{Style.RESET_ALL}")
    finally:
        wandb.finish()
    
    print(f"{Fore.GREEN}{Style.BRIGHT}Finished Experiment Run!{Style.RESET_ALL}")
    
if __name__ == "__main__":
    app.run(entry)
