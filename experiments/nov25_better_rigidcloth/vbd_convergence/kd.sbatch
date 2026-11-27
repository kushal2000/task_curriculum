#!/bin/bash
#SBATCH --job-name=vbd_kd
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_convergence/kd_%A_%a.out
# Bending damping x VBD iterations. Array 0-23: 2 materials x 4 edge_kd x 3 tests; each sweeps 12 30 96 192 iterations.
set -uo pipefail
MATS=(denim stable); KDS=(9.11e-5 3e-4 1e-3 3e-3); TESTS=(drop hang cantilever)
i=$SLURM_ARRAY_TASK_ID; T=${TESTS[$((i % 3))]}; K=${KDS[$(((i / 3) % 4))]}; M=${MATS[$((i / 12))]}
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export WARP_CACHE_PATH=$TROOT/${USER}_kd_${SLURM_JOB_ID}_${i}; mkdir -p "$WARP_CACHE_PATH"
.venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/vbd_convergence/bench.py --material $M --test $T --edge_kd $K \
  --iters 12 30 96 192 --seconds 2.0 --out outputs/vbd_convergence/kd/${M}_kd${K}_${T}.jsonl
