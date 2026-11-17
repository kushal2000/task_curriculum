#!/bin/bash
# Bake a robot's URDF -> USD into the Newton USD cache (.cache/usd) by building its Play scene once
# under Isaac Sim (isaacsimenvs/newton/usd_cache.py). Engaging, one GPU.
#
#   sbatch experiments/nov16_vbd_cloth/usd_cache.sh Isaacsimenvs-PlayFlexivWujiRightReal-Direct-v0
#SBATCH --job-name=usdcache
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
for task in "$@"; do
    echo "[usdcache] $task"
    # Engaging: Isaac Sim runs in its container (host glibc too old), .venv_isaacsim_ct inside it
    # (the scripts/isaacsim_py wrapper, removed in 15d93c0's cleanup).
    command -v apptainer >/dev/null || module load apptainer
    apptainer exec --nv --bind /orcd --pwd "$PWD" --env OMNI_KIT_ACCEPT_EULA=YES --env ACCEPT_EULA=Y \
        "${ISAACSIM_SIF:-/orcd/scratch/orcd/011/hglin/containers/isaac-sim_5.1.0.sif}" \
        "$PWD/.venv_isaacsim_ct/bin/python" -m isaacsimenvs.newton.usd_cache --populate --task "$task" --num_assets_per_type 1
done
