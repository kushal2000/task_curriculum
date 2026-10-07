#!/bin/bash
# Partly folded fold-pool starts on PhysX (scripts/analysis/fold_start_probe.py), every episode folded.
#
#   EXTRA_ARGS="env.rigid_cloth.joint_damping=0.02" OUT_TAG=d002 sbatch scripts/cluster/bos14_fold_start_probe.sh
#   .venv_isaaclab3/bin/python scripts/analysis/fold_start_probe_plot.py docs/results/fold_start_probe_d002.pt \
#       --png docs/results/fold_start_probe_d002.png

#SBATCH --job-name=px_fstart
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
echo "[fstart] host=$(hostname -s) extra=${EXTRA_ARGS:-none}"
.venv_isaacsim/bin/python scripts/analysis/fold_start_probe.py --headless \
    --num_envs "${ENVS:-60}" --steps "${STEPS:-60}" ${PROBE_ARGS:-} \
    --out "docs/results/fold_start_probe_${OUT_TAG:-run}.pt" \
    env.cloth.nan_policy=reset "${EXTRA[@]}"
