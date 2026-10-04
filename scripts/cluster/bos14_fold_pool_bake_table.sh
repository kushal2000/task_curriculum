#!/bin/bash
# Bake the stage-2 fold pool's wider table (assets/urdf/table_fold_pool.urdf) into the USD cache.
#
#   sbatch scripts/cluster/bos14_fold_pool_bake_table.sh
#
# The Newton stack converts URDFs by content-addressed cache lookup and RAISES on a miss, so a new
# table URDF has to be baked once under Isaac Sim before any FoldPool run can load it. This builds
# the Play scene with the table overridden -- every other asset is the cloth configuration's
# (hammer only, one variant), so those entries are cache hits and stay as they are.

#SBATCH --job-name=fp_bake
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=bos14-node-092,bos14-node-104,bos14-node-105
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
echo "[bake] host=$(hostname -s) commit=$(git rev-parse --short HEAD)"

BEFORE=$(find .usd_cache -maxdepth 1 -mindepth 1 -type d | wc -l)
.venv_isaacsim/bin/python -m isaacsimenvs.newton.usd_cache \
    --populate --num_assets_per_type 1 \
    'env.assets.handle_head_types=[hammer]' \
    env.assets.table_urdf=assets/urdf/table_fold_pool.urdf
RC=$?
AFTER=$(find .usd_cache -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[bake] exit=$RC cache entries ${BEFORE} -> ${AFTER}"
find .usd_cache -name "*table_fold_pool*" | head
