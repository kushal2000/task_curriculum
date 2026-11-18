#!/bin/bash
# Finger press + drag on box2-mid plates (scripts/flexiv_wuji/probe_light_plate_press.py), one hinge
# setting per array index:  density / joint_damping / hinge_armature
#   0  2.0  / 0.05  / 1e-3   the recipe before 2026-11-10
#   1  0.1  / 0.003 / 1e-3   the recipe now
#   2  0.1  / 0.003 / 1e-4
#   3  0.02 / 0.003 / 1e-3   lightest fabric
#   4  as 1, 32 position iterations     5  as 1, 16 iterations
#   6  0.5 / 0.0125 / 1e-3    7  0.3 / 0.008 / 1e-3 (towel, the heaviest fabric)
#   sbatch --array=0-3 scripts/cluster/bos14_light_plate_press.sh
#SBATCH --job-name=plate_press
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,062,086,092,104,105,109,110,112,121]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
mkdir -p slurm_logs docs/results/light_plate
I="${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
DENS=(2.0 0.1 0.1 0.02 0.1 0.1 0.5 0.3); DAMP=(0.05 0.003 0.003 0.003 0.003 0.003 0.0125 0.008)
ARM=(0.001 0.001 0.0001 0.001 0.001 0.001 0.001 0.001); ITS=(8 8 8 8 32 16 8 8)
echo "[plate_press] index=$I density=${DENS[$I]} damping=${DAMP[$I]} armature=${ARM[$I]} host=$(hostname -s)"
.venv_isaacsim/bin/python scripts/flexiv_wuji/probe_light_plate_press.py --headless \
    --experiment flat7FlexivWujiRHL --out "docs/results/light_plate/press_$I.json" \
    "env.rigid_cloth.density=${DENS[$I]}" "env.rigid_cloth.joint_damping=${DAMP[$I]}" \
    "env.physx_chain.hinge_armature=${ARM[$I]}" \
    "env.sim.physx.max_position_iteration_count=${ITS[$I]}" \
    "env.physx_chain.solver_position_iterations=${ITS[$I]}"
echo "[plate_press] exit=$?"
