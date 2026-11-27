#!/bin/bash
#SBATCH --job-name=vbd_conv
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_convergence/slurm_%A_%a.out
# One (material, test) per array task, sweeping VBD iterations 6..192. Array 0-8: 3 materials x 3 tests.
set -uo pipefail
MATS=(stable denim stiff); TESTS=(drop hang cantilever)
M=${MATS[$((SLURM_ARRAY_TASK_ID / 3))]}; T=${TESTS[$((SLURM_ARRAY_TASK_ID % 3))]}
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export WARP_CACHE_PATH=$TROOT/${USER}_conv_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$WARP_CACHE_PATH"
.venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/vbd_convergence/bench.py --material $M --test $T \
  --iters 6 12 24 48 96 192 --seconds 2.0 --out outputs/vbd_convergence/${M}_${T}.jsonl
