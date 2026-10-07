#!/bin/bash
# Zero-action stability probe of the VBD cloth at a given size (scripts/analysis/cloth_obs_parity.py,
# pinned start): does it drop, land flat, stay finite? Output docs/results/large_cloth/probe_<TAG>.pt
#
#   TAG=s20 EXTRA_ARGS="env.cloth.size=0.2 env.cloth.resolution=13 env.cloth.reset_offset_xy=[0.0,0.0707]" \
#       sbatch scripts/cluster/bos14_large_cloth_probe.sh

#SBATCH --job-name=lc_probe
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=bos14-node-[092,104,105]
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
export PYTHONUNBUFFERED=1
echo "[lc_probe] tag=${TAG:-run} host=$(hostname -s) extra=${EXTRA_ARGS:-none}"
scripts/newton_py scripts/analysis/cloth_obs_parity.py --task Isaacsimenvs-Cloth-Direct-v0 \
    --num_envs "${ENVS:-8}" --steps "${STEPS:-150}" \
    --out "docs/results/large_cloth/probe_${TAG:-run}.pt" env.cloth.nan_policy=raise "${EXTRA[@]}"
