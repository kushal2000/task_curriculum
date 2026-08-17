#!/bin/bash
# Simulation throughput at one environment count.
#
#   sbatch scripts/cluster/sbatch_throughput.sh 64 [hydra overrides...]
#
# One env count per job, one GPU per job: two benchmarks sharing a device measure contention, not
# throughput. Results land in docs/results/tput_<n>[_policy].json.

#SBATCH --job-name=tput
#SBATCH --partition=portal
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail

N="$1"; shift

cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

SUFFIX=""
POLICY_FLAG=""
if [ "${WITH_POLICY:-0}" = "1" ]; then
    SUFFIX="_policy"
    POLICY_FLAG="--with_policy"
fi

echo "[tput] num_envs=$N with_policy=${WITH_POLICY:-0} host=$(hostname)"

# shellcheck disable=SC2086
scripts/newton_py -m isaacsimenvs.eval.throughput \
    --num_envs "$N" \
    --steps "${STEPS:-200}" \
    --warmup "${WARMUP:-60}" \
    --num_assets_per_type 1 \
    $POLICY_FLAG \
    --out "docs/results/tput_${N}${SUFFIX}.json" \
    'env.assets.handle_head_types=[hammer]' \
    env.cable.thickness=0.03 \
    "$@"
echo "[tput] num_envs=$N exit=$?"
