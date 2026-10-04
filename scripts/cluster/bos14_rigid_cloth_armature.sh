#!/bin/bash
# Smallest hinge armature that integrates stably, per variant. One array task per (variant, armature).
#
#   sbatch --array=0-24 scripts/cluster/bos14_rigid_cloth_armature.sh
#
# WHY A SWEEP AND NOT A VALUE
#
# Armature is FICTITIOUS inertia. It is the standard MuJoCo remedy for a DOF whose real inertia is
# too small to integrate -- a 2 mm slat's is ~1e-8 kg m^2 at the env's dt of 1/120 -- but every unit
# of it makes the chain bend more sluggishly than its geometry says, and a cloth's bending inertia
# really is negligible. So the value is a trade, and picking it by eye would put the single largest
# deliberate departure from the cloth on a guess.
#
# The failure it fixes is not subtle: from a dead stop in FREE FALL, joint velocity reached
# 1.3e3-4.9e3 rad/s on the FIRST step, the hinges flew past their +-90 deg limits to 1648 deg, and
# all five variants took the robot non-finite with them within 2-7 steps.
#
# `nan_policy=reset` so a diverging configuration reports a COUNT instead of dying on the first env:
# "how many of 64 envs blew up" separates marginal from hopeless, which an exception cannot.

#SBATCH --job-name=rc_arm
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
ARMS=(0.0 1e-7 1e-6 1e-5 1e-4)
DAMP="${DAMP:-0.05}"

I="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$((I / 5))]}"
ARM="${ARMS[$((I % 5))]}"

export PYTHONUNBUFFERED=1
echo "[arm] variant=$VARIANT armature=$ARM damping=$DAMP host=$(hostname -s)"

scripts/newton_py -m scripts.analysis.rigid_cloth_env_probe \
    --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --variant "$VARIANT" \
    --armature "$ARM" \
    --damping "$DAMP" \
    --num_envs 64 \
    --settle_steps 120 \
    --fold_steps 150 \
    --timing_steps 100 \
    --nan_policy reset \
    --out "docs/results/rigid_cloth_arm_${VARIANT}_a${ARM}_d${DAMP}.json"
echo "[arm] exit=$?"
