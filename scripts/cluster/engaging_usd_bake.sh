#!/bin/bash
# Populate .usd_cache on Engaging (Isaac Sim in the container, via scripts/isaacsim_py), so the
# kit-less Newton targets can load their assets. Covers what bos14_rigid_cloth_bake.sh and
# bos14_fold_pool_bake_table.sh baked there, plus the cloth scene itself:
#   1. Play scene, cloth configuration (hammer only, one variant, default table)
#   2. the same with the fold pool's wider table
#   3. every hinged-slat chain in rigid_cloth.VARIANTS
#   sbatch scripts/cluster/engaging_usd_bake.sh        (submit from the repo root)
# Kit exits 0 even when a bake raised, so trust the cache-entry count, not the status.
#SBATCH --job-name=usd_bake
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
mkdir -p .usd_cache
n() { find .usd_cache -maxdepth 1 -mindepth 1 -type d | wc -l; }
echo "[bake] host=$(hostname -s) $(nvidia-smi --query-gpu=name --format=csv,noheader) entries=$(n)"
BASE=(-m isaacsimenvs.newton.usd_cache --populate --num_assets_per_type 1 'env.assets.handle_head_types=[hammer]')

echo "[bake] 1/3 cloth scene";            scripts/isaacsim_py "${BASE[@]}";                                                         echo "[bake] entries=$(n)"
echo "[bake] 2/3 fold-pool table";        scripts/isaacsim_py "${BASE[@]}" env.assets.table_urdf=assets/urdf/table_fold_pool.urdf;  echo "[bake] entries=$(n)"
echo "[bake] 3/3 rigid-cloth chains";     scripts/isaacsim_py -m isaacsimenvs.newton.usd_cache --populate --skip_scene_build --rigid_cloth_variants; echo "[bake] entries=$(n)"
echo "[bake] baked lines: $(grep -c '^\[usd_cache\] baked ' "slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" || true)"
