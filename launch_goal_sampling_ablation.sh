#!/usr/bin/env bash
# ============================================================================
# Goal Sampling Ablation Experiments
# Tests different value/actor goal sampling strategies across envs & agents.
# Usage: bash launch_goal_sampling_ablation.sh
# ============================================================================
set -euo pipefail

# ── Common settings ─────────────────────────────────────────────────────────
SEED=0
TRAIN_STEPS=1000000
WANDB_MODE=offline
VIDEO_EPISODES=0          # disable video to speed up evals
RUN_GROUP="goal-sampling-ablation"

# ============================================================================
# Experiment 1: QRL on antmaze-large-explore
#   Hypothesis: heavy random value goals + some random actor goals can break
#   the 0% barrier on explore by forcing cross-trajectory distance learning.
# ============================================================================
CUDA_VISIBLE_DEVICES=1 uv run python main.py \
    --env_name=antmaze-large-explore-v0 \
    --agent=agents/qrl/original.py \
    --seed=${SEED} \
    --train_steps=${TRAIN_STEPS} \
    --wandb_mode=${WANDB_MODE} \
    --video_episodes=${VIDEO_EPISODES} \
    --run_group=${RUN_GROUP} \
    --wandb_tags=explore-rand-heavy \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.3 \
    --agent.value_p_randomgoal=0.7 \
    --agent.value_geom_sample=True \
    --agent.actor_p_curgoal=0.0 \
    --agent.actor_p_trajgoal=0.7 \
    --agent.actor_p_randomgoal=0.3 \
    --agent.actor_geom_sample=True

# ============================================================================
# Experiment 2: QRL on antsoccer-arena-navigate
#   Hypothesis: mixing trajectory + random value goals and using uniform actor
#   sampling (for longer-horizon ball-push goals) improves soccer performance.
# ============================================================================
CUDA_VISIBLE_DEVICES=1 uv run python main.py \
    --env_name=antsoccer-arena-navigate-v0 \
    --agent=agents/qrl/original.py \
    --seed=${SEED} \
    --train_steps=${TRAIN_STEPS} \
    --wandb_mode=${WANDB_MODE} \
    --video_episodes=${VIDEO_EPISODES} \
    --run_group=${RUN_GROUP} \
    --wandb_tags=soccer-mixed-goals \
    --agent.value_p_curgoal=0.1 \
    --agent.value_p_trajgoal=0.6 \
    --agent.value_p_randomgoal=0.3 \
    --agent.value_geom_sample=True \
    --agent.actor_p_curgoal=0.0 \
    --agent.actor_p_trajgoal=1.0 \
    --agent.actor_p_randomgoal=0.0 \
    --agent.actor_geom_sample=False

# ============================================================================
# Experiment 3: MQE on antmaze-large-explore
#   Hypothesis: MQE's multi-step backup + explore-tuned goals can learn
#   cross-trajectory distances that single-step QRL cannot.
# ============================================================================
CUDA_VISIBLE_DEVICES=1 uv run python main.py \
    --env_name=antmaze-large-explore-v0 \
    --agent=agents/mqe/original.py \
    --seed=${SEED} \
    --train_steps=${TRAIN_STEPS} \
    --wandb_mode=${WANDB_MODE} \
    --video_episodes=${VIDEO_EPISODES} \
    --run_group=${RUN_GROUP} \
    --wandb_tags=mqe-explore-rand-heavy \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.3 \
    --agent.value_p_randomgoal=0.7 \
    --agent.value_geom_sample=True \
    --agent.actor_p_curgoal=0.0 \
    --agent.actor_p_trajgoal=0.7 \
    --agent.actor_p_randomgoal=0.3 \
    --agent.actor_geom_sample=True

# ============================================================================
# Experiment 4: MQE on antmaze-large-navigate (50/50 ablation)
#   Hypothesis: adding random value goals to MQE (which defaults to trajgoal
#   only) improves cross-region distance estimates without hurting local ones.
# ============================================================================
CUDA_VISIBLE_DEVICES=1 uv run python main.py \
    --env_name=antmaze-large-navigate-v0 \
    --agent=agents/mqe/original.py \
    --seed=${SEED} \
    --train_steps=${TRAIN_STEPS} \
    --wandb_mode=${WANDB_MODE} \
    --video_episodes=${VIDEO_EPISODES} \
    --run_group=${RUN_GROUP} \
    --wandb_tags=mqe-nav-mixed-50-50 \
    --agent.value_p_curgoal=0.0 \
    --agent.value_p_trajgoal=0.5 \
    --agent.value_p_randomgoal=0.5 \
    --agent.value_geom_sample=True \
    --agent.actor_p_curgoal=0.0 \
    --agent.actor_p_trajgoal=1.0 \
    --agent.actor_p_randomgoal=0.0 \
    --agent.actor_geom_sample=True

echo "All goal-sampling ablation experiments completed."
