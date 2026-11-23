#!/bin/bash
#SBATCH --job-name=play_step_split
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum
#SBATCH --output=debug_outputs/train_logs/nov/bench/step_split_%j.out
# Per-region time of one env step (PhysX step, per-asset writes / updates, the env's phases):
#   sbatch experiments/nov22_play_pointcloud/bench/step_split.sbatch <task id> <num envs>
set -uo pipefail
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
exec .venv/isaacsim/bin/python experiments/nov22_play_pointcloud/bench/step_split.py "$@"
