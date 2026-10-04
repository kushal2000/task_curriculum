#!/bin/bash
# Spawn height that leaves a rigid chain resting FLAT, per variant. One array task per (variant, h).
#
#   sbatch --array=0-24 scripts/cluster/bos14_rigid_cloth_spawn.sh
#
# WHY THIS IS NOT A FREE PARAMETER
#
# `ClothCfg.start_height = 0.30` is a 15 cm DROP, and its docstring records the calibration: at 0.15
# the particle sheet spawned 0.2 mm above the surface, was ejected upward at 1.7 m/s and settled
# visibly wrinkled, while a genuine drop from 0.30 lands it "PERFECTLY flat (min = mean = max to four
# decimals)". So 0.30 is not arbitrary -- it is the value at which a VBD sheet arrives flat.
#
# A limp chain of rigid slats does not. Measured at 0.30: `footprint 0.470` and `fold_err 0.047` at
# rest for box3-mid, i.e. the chain buckles on the way down and is already folded in half before the
# policy acts. That matters twice over: the task starts from the wrong state, and a crumpled rest pose
# scores HALF the fold error of a flat one, so most of the signal the policy should climb is
# already spent -- which would read as an easy task rather than a broken one.
#
# So the chain needs its own calibration of the same quantity. The guard in `ClothCfg.__post_init__`
# requires `start_height > table_half_thickness = 0.150`, so 0.151 is a 1 mm drop -- effectively
# placed on the table.
#
# The cost is a real confound and is reported as one: the rigid arms then start from a different
# spawn HEIGHT than the cloth control. What they share, and what the task actually requires, is the
# resulting STATE -- a flat sheet at rest on the table.

#SBATCH --job-name=rc_spawn
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

VARIANTS=(box3-mid box-mid-odd cyl-mid-odd box2-surface box-surface)
HEIGHTS=(0.151 0.155 0.17 0.20 0.30)

I="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$((I / 5))]}"
H="${HEIGHTS[$((I % 5))]}"
ARM="${ARM:-1e-5}"
DAMP="${DAMP:-0.05}"

export PYTHONUNBUFFERED=1
echo "[spawn] variant=$VARIANT start_height=$H armature=$ARM damping=$DAMP host=$(hostname -s)"

scripts/newton_py -m scripts.analysis.rigid_cloth_env_probe \
    --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --variant "$VARIANT" \
    --armature "$ARM" \
    --damping "$DAMP" \
    --start_height "$H" \
    --num_envs 64 \
    --settle_steps 150 \
    --fold_steps 180 \
    --timing_steps 100 \
    --nan_policy reset \
    --out "docs/results/rigid_cloth_spawn_${VARIANT}_h${H}.json"
echo "[spawn] exit=$?"
