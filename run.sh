#!/bin/bash
# export WANDB_ENTITY="dunnolab"
export WANDB_ENTITY="zzmtsvv"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="8cf7cafd958aa2df2d9f18fa32723dddc4024806"
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

export OMP_NUM_THREADS="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HOME=/home/jovyan/shares/SR008.nfs1/.cache/huggingface

export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="/home/jovyan/.cache/torch"
export TORCH_HUB_ROOT="/home/jovyan/.cache/torch_hub"
export XDG_CACHE_HOME="/home/jovyan/.cache/xgd"

# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export JAX_PLATFORM_NAME=gpu
# export JAX_PLATFORMS=cpu

export MUJOCO_GL=egl
# export EGL_DEVICE_ID=0

# zzmtsvv proxy
export HTTPS_PROXY="http://fX3oyy:gqAsJX@161.0.20.250:8000"
export HTTP_PROXY="http://fX3oyy:gqAsJX@161.0.20.250:8000"

# wandb disabled
wandb online


# uv run --active --no-project main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
# uv run --active --no-project main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# uv run --active --no-project main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/dual_direct.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
# uv run --active --no-project main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/dual_direct.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

uv run --active --no-project main.py --env_name=puzzle-3x3-play-v0 --agent=agents/daf/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
uv run --active --no-project main.py --env_name=puzzle-4x4-play-v0 --agent=agents/daf/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# uv run --active --no-project main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
# uv run --active --no-project main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

uv run --active --no-project main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/dual_direct.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
uv run --active --no-project main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/dual_direct.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99