#!/bin/bash
# Cloth throughput at a given vbd_iterations, holding env count fixed.
#
#   sbatch scripts/cluster/sbatch_cloth_iters.sh <iterations> [num_envs]
#
# Per-step cost is flat in env count for this scene (see docs/results/ctput_*.json), which means the
# step is dominated by fixed solver work rather than per-env compute. vbd_iterations is the largest
# such term and was inherited from cable tuning, never validated for the cloth -- so this is the
# one knob with a plausible near-linear effect on throughput.
#
# GPU generation pinned: portal is mixed hardware and per-step cost differs ~1.5x between them,
# which would swamp the effect being measured.

#SBATCH --job-name=citer
#SBATCH --partition=portal
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail
ITERS="$1"; N="${2:-256}"
cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1

echo "[citer] vbd_iterations=$ITERS num_envs=$N host=$(hostname)"
scripts/newton_py -m isaacsimenvs.eval.throughput \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --num_envs "$N" --steps 200 --warmup 60 --num_assets_per_type 1 \
    --out "docs/results/citer_${ITERS}.json" \
    'env.assets.handle_head_types=[hammer]' \
    "env.cloth.vbd_iterations=$ITERS"
echo "[citer] vbd_iterations=$ITERS exit=$?"
