#!/bin/bash
# Bake the five hinged-slat chain URDFs into the content-addressed USD cache.
#
#   sbatch scripts/cluster/bos14_rigid_cloth_bake.sh
#
# WHY A JOB AND NOT A LOCAL COMMAND
#
# Conversion runs `isaaclab.sim.converters.UrdfConverter`, which needs the Kit URDF importer, which
# needs `.venv_isaacsim` AND a GPU -- the login node has neither. The Newton venv that actually
# trains is kit-less and can only *read* the cache (`install_reader` raises on a miss), so without
# this step every RigidCloth run dies at construction with a cache miss.
#
# No libGLU compat guard here, unlike `bos14_g1_wuji_replay_isaacsim.sh`: the compute nodes' missing
# libGLU.so.1 breaks Isaac Sim's MDL SDK and therefore the RTX shader/camera path, which only a
# render needs. URDF -> USD conversion needs Kit but never renders a frame, so it runs clean without.
#
# `--skip_scene_build` keeps the already-baked procedural tool pool untouched: the scene build
# shuffles the pool over its whole length, so re-running it at a different --num_assets_per_type
# would write a different set of objects and silently change what a Play/Cloth run loads.
#
# The bake takes RigidClothCfg's DEFAULTS for density / joint_damping / joint_limit, because the
# URDF text is the cache key. Overriding any of those three at train time is a different asset and
# needs its own bake -- it will announce itself as a cache miss naming the variant.

#SBATCH --job-name=rc_bake
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES

echo "[bake] host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BEFORE=$(find .usd_cache -maxdepth 1 -mindepth 1 -type d | wc -l)

.venv_isaacsim/bin/python -m isaacsimenvs.newton.usd_cache \
    --populate --skip_scene_build --rigid_cloth_variants
RC=$?

# Kit's shutdown path calls os._exit(0), so a bake that raised still exits 0 -- the first
# attempt reported success having baked nothing. Trust the cache, not the status.

AFTER=$(find .usd_cache -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "[bake] exit=$RC cache entries ${BEFORE} -> ${AFTER}"

BAKED=$(grep -c "^\[usd_cache\] baked " "slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" || true)
if [ "$BAKED" -lt 5 ]; then
    echo "[bake] FAILED: only ${BAKED}/5 chains baked" >&2
    exit 1
fi
echo "[bake] OK: ${BAKED}/5 chains baked"
