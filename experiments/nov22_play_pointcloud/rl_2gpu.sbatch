#!/bin/bash
#SBATCH --job-name=play_pc_rl_2gpu
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum
#SBATCH --output=debug_outputs/train_logs/nov/slurm_%j.out
# Two GPUs (torchrun, one rank per GPU; ENVS is per rank). SAPG from scratch on the point-cloud Play task: actor = proprio + live/goal clouds (one PointNet each,
# pc_rl.py), critic = state + clean current/goal clouds (MLP), Play's tolerance curriculum on the surface
# distance. 24576 envs = 6 SAPG blocks of 4096; Play's SAPG settings otherwise (minibatch 98304). The actor
# reads 64 camera points of the tool (robot points dropped) + 64 goal points.
#   NAME=<name> [TASK=<task id>] [WANDB_PROJECT=<project>] [VIEWER=0] sbatch experiments/nov22_play_pointcloud/rl_scratch.sbatch [overrides]
# Writes debug_outputs/train_logs/nov/<date>_<NAME>_<job>/ (slurm log, nn/ checkpoints, wandb).
# RESUME_JOB=<job id>: resume that run from its newest checkpoint (latest_ckpt.py), for chaining past the time limit:
#   RESUME_JOB=123 sbatch --dependency=afterany:123 ... <the same overrides, minus --pc_init_*/--pc_actor_freeze_epochs>
# VIEWER=0: no interactive-viewer capture (a resumed run died in its wandb.log of the viewer page, 2026-11-23).
set -uo pipefail
NAME="${NAME:-0_play_pc_scratch_${SLURM_JOB_ID:-local}}"; RUN_NAME="${NAME#0_}"
cd /share/portal/kk837/task_curriculum
# One directory per run under debug_outputs/train_logs/nov/ (gitignored): the slurm log, checkpoints, wandb.
JOB="${SLURM_JOB_ID:-local}"
RUN_DIR="debug_outputs/train_logs/nov/$(date +%F)_${RUN_NAME}_${JOB}"
mkdir -p "$RUN_DIR"
[ -f "debug_outputs/train_logs/nov/slurm_${JOB}.out" ] && mv "debug_outputs/train_logs/nov/slurm_${JOB}.out" "$RUN_DIR/"
export WANDB_DIR="$PWD/$RUN_DIR"       # wandb files with the run, not ./wandb
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
echo "[play_pc] host=$(hostname -s) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain --untracked-files=no)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export OMP_NUM_THREADS=12      # the node's threads split over the two ranks
RESUME=()
if [ -n "${RESUME_JOB:-}" ]; then
    CK=$(.venv/isaacsim/bin/python experiments/nov22_play_pointcloud/latest_ckpt.py "$RESUME_JOB") || exit 1
    echo "[play_pc] resuming job $RESUME_JOB from $CK"; RESUME=(--checkpoint "$CK" --checkpoint_load_mode resume)
    TOL=$(WANDB_PROJECT="${WANDB_PROJECT:-play_pc}" .venv/isaacsim/bin/python experiments/nov22_play_pointcloud/latest_ckpt.py "$RESUME_JOB" --tolerance)
    [ -n "$TOL" ] && { echo "[play_pc] success tolerance resumes at $TOL"; RESUME+=(env.termination.success_tolerance="$TOL"); }
fi
exec .venv/isaacsim/bin/torchrun --standalone --nnodes=1 --nproc_per_node=2 -m isaacsimenvs.scripts.training.train --distributed \
    --task "${TASK:-Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0}" --agent rl_games_sapg_cfg_entry_point --headless \
    $([ "${VIEWER:-1}" = 1 ] && echo --capture_viewer) --wandb_activate --wandb_project "${WANDB_PROJECT:-play_pc}" --wandb_name "$NAME" \
    env.scene.num_envs="${ENVS:-24576}" agent.params.seed="${SEED:-1}" agent.params.config.name="$NAME" hydra.run.dir="$RUN_DIR" ${RESUME[@]+"${RESUME[@]}"} "$@"
