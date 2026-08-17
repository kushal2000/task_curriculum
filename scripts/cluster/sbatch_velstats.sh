#!/bin/bash
# Object speed distribution for one configuration. Submit with:
#
#   sbatch scripts/cluster/sbatch_velstats.sh <tag> <task> [hydra overrides...]
#
# The rigid-tool and rigid-rod baselines are the reference here: whatever speeds the policy
# produces while actually scoring goals are, by construction, reasonable ones.

#SBATCH --job-name=velstats
#SBATCH --partition=portal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail

TAG="$1"; TASK="$2"; shift 2

cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

echo "[job] velstats tag=$TAG task=$TASK host=$(hostname)"
echo "[job] overrides: $*"

scripts/newton_py -m isaacsimenvs.eval.velocity_stats \
    --task "$TASK" \
    --num_envs "${NUM_ENVS:-16}" \
    --steps "${STEPS:-600}" \
    --num_assets_per_type 1 \
    --single_variant \
    --out "docs/results/vel_${TAG}.json" \
    'env.assets.handle_head_types=[hammer]' \
    "$@"
echo "[job] tag=$TAG exit=$?"
