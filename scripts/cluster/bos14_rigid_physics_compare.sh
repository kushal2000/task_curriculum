#!/bin/bash
# The rigid manipuland's physics on Newton vs PhysX (scripts/analysis/rigid_physics_compare.py).
#
#   sbatch --array=0-3 scripts/cluster/bos14_rigid_physics_compare.sh
#
# 0/1 box3-mid chain on Newton / PhysX, 2/3 the shape pool (nominal 10 cm) on Newton / PhysX.
# Results: docs/results/rigid_physics_compare_<backend>_<kind>.json

#SBATCH --job-name=px_cmp
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
I="${SLURM_ARRAY_TASK_ID:-0}"
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
TAG="${OUT_TAG:-}"
case "$I" in
    0) PY=(scripts/newton_py); T=Isaacsimenvs-RigidCloth-Direct-v0; OUT=newton_chain ;;
    1) PY=(.venv_isaacsim/bin/python); T=Isaacsimenvs-RigidClothPhysx-Direct-v0; OUT=physx_chain; EXTRA+=(--headless) ;;
    2) PY=(scripts/newton_py); T=Isaacsimenvs-FoldPool-Direct-v0; OUT=newton_pool ;;
    3) PY=(.venv_isaacsim/bin/python); T=Isaacsimenvs-FoldPoolPhysx-Direct-v0; OUT=physx_pool; EXTRA+=(--headless) ;;
esac
echo "[px_cmp] $T host=$(hostname -s) commit=$(git rev-parse --short HEAD) extra=${EXTRA[*]:-}"
"${PY[@]}" scripts/analysis/rigid_physics_compare.py --task "$T" \
    --out "docs/results/rigid_physics_compare_${OUT}${TAG}.json" "${EXTRA[@]}"
echo "[px_cmp] exit=$?"
