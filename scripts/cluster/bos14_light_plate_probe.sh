#!/bin/bash
# Light-plate hinge probe: PhysX box2-mid plates at fabric-like surface densities, released from a
# raised angle, over a (damping x armature) grid (scripts/analysis/hinge_release_probe.py).
#   sbatch --array=0-3 scripts/cluster/bos14_light_plate_probe.sh
#SBATCH --job-name=light_plate
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
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
DENS=(0.018 0.116 0.324 0.116); SIZES=(0.10 0.10 0.10 0.199)
D="${DENS[$I]}"; S="${SIZES[$I]}"
echo "[light_plate] index=$I density=$D size=$S host=$(hostname -s)"
.venv_isaacsim/bin/python scripts/analysis/hinge_release_probe.py --headless \
    --task Isaacsimenvs-FoldPoolPhysx-Direct-v0 --size "$S" \
    --damping 0.05,0.01,0.003,0.001 --armature 1e-3,1e-4,5e-5,1e-5 --stiffness 0 \
    --out "docs/results/light_plate/d${D}_s${S}.pt" "env.rigid_cloth.density=$D"
echo "[light_plate] exit=$?"
