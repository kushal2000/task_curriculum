#!/bin/bash
#SBATCH --job-name=play_pc_dagger
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=24:00:00
# Point-cloud Play student, DAgger from the keypoint Play prior (isaacsimenvs/scripts/distillation/distill_play_pc.py).
#   sbatch --output=experiments/nov22_play_pointcloud/slurm_%j.out experiments/nov22_play_pointcloud/dagger.sbatch [extra args]
# NAME (default play_pc_dagger), ENVS (24576), ITERS (2000). Writes outputs/distill_play/<NAME>_<job>/.
set -uo pipefail
cd /share/portal/kk837/task_curriculum
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD"
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
NAME="${NAME:-play_pc_dagger}"; RUN="${NAME}_${SLURM_JOB_ID:-local}"
echo "[play_pc] host=$(hostname -s) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain --untracked-files=no)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec .venv/isaacsim/bin/python -m isaacsimenvs.scripts.distillation.distill_play_pc --headless \
    --teacher pretrained_policy/flexiv_wuji/model.pth --out "outputs/distill_play/$RUN" \
    --num_envs "${ENVS:-24576}" --iters "${ITERS:-2000}" \
    --wandb_project "${WANDB_PROJECT:-play_pc}" --wandb_name "$RUN" "$@"
