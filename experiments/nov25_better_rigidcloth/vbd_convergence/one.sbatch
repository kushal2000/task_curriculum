#!/bin/bash
#SBATCH --job-name=vbd_conv1
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_convergence/one_%A_%a.out
# One bench.py run per array task: line (task id + 1) of $LIST holds its arguments.
set -uo pipefail
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${LIST:-outputs/vbd_convergence/remaining.txt}")
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export WARP_CACHE_PATH=$TROOT/${USER}_conv1_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$WARP_CACHE_PATH"
.venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/vbd_convergence/bench.py $ARGS --seconds 2.0
