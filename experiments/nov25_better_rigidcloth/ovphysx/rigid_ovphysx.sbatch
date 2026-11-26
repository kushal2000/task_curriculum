#!/bin/bash
#SBATCH --job-name=rigid_ovphysx
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/rigid_ovphysx/slurm_%A_%a.out
# Rung B of the sim-to-sim ladder: 64live on the PhysX plate pool through kit-less Isaac Lab 3 +
# ovphysx 0.5.9 (Isaacsimenvs-RigidClothOvPhysx-Direct-v0), box2-mid at 15 cm, 32 envs, seed
# (task id + 1); settings as the Isaac Lab 2.3 PhysX reference (outputs/rigid_physx_ref).
set -uo pipefail
S=$((SLURM_ARRAY_TASK_ID + 1)); O=outputs/rigid_ovphysx; OUT=$O/64live__ovphysx_15cm__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
experiments/nov25_better_rigidcloth/ovphysx/eval_ovphysx.sh "$OUT" $S 32 900
[ -s "$OUT" ] || { echo "[rigid_ovphysx] FAILED $OUT"; exit 1; }
