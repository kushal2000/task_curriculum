#!/bin/bash
#SBATCH --job-name=play_pc_rl
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=72:00:00
# SAPG from scratch on the point-cloud Play task: actor = proprio + live/goal clouds (one PointNet each,
# pc_rl.py), critic = state + clean current/goal clouds (MLP), Play's tolerance curriculum on the surface
# distance. 24576 envs = 6 SAPG blocks of 4096; minibatch 16384 (the per-point MLP runs on every point).
#   sbatch --output=experiments/nov22_play_pointcloud/slurm_%j.out experiments/nov22_play_pointcloud/rl_scratch.sbatch [overrides]
set -uo pipefail
cd /share/portal/kk837/task_curriculum
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD"
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
NAME="${NAME:-0_play_pc_scratch_${SLURM_JOB_ID:-local}}"
echo "[play_pc] host=$(hostname -s) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain --untracked-files=no)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec .venv/isaacsim/bin/python -m isaacsimenvs.scripts.training.train \
    --task Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless \
    --capture_viewer --wandb_activate --wandb_project "${WANDB_PROJECT:-play_pc}" --wandb_name "$NAME" \
    env.scene.num_envs="${ENVS:-24576}" agent.params.seed="${SEED:-1}" agent.params.config.name="$NAME" \
    agent.params.config.minibatch_size=16384 agent.params.config.central_value_config.minibatch_size=16384 "$@"
