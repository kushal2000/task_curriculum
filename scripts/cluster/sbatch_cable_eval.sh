#!/bin/bash
# One cable-eval configuration per job. Submit with scripts/cluster/submit_cable_sweep.sh,
# or directly:
#
#   sbatch scripts/cluster/sbatch_cable_eval.sh stable \
#       env.cable.angular_damping=5.0 env.cable.vbd_iterations=60
#
# $1 is the result tag (docs/results/<tag>.json); everything after it is passed to Hydra.
#
# Each config is its own job because these runs are single-GPU and independent -- a sweep is
# embarrassingly parallel, and running them serially on one GPU was the bottleneck for most of a
# day. The eval harness aborts on a contact-buffer overflow (see isaacsimenvs/newton/contact_guard),
# so a job that exits non-zero means the scene was dropping contacts, not that the policy failed.

#SBATCH --job-name=cable_eval
#SBATCH --partition=portal
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail

TAG="$1"; shift

cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

NUM_ENVS="${NUM_ENVS:-64}"
MAX_STEPS="${MAX_STEPS:-6000}"

echo "[job] tag=$TAG num_envs=$NUM_ENVS host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] overrides: $*"

scripts/newton_py -m isaacsimenvs.eval.episodes \
    --task "${TASK:-Isaacsimenvs-Cable-Direct-v0}" \
    --num_envs "$NUM_ENVS" \
    --num_assets_per_type 1 \
    --single_variant \
    --max_steps "$MAX_STEPS" \
    --seed "${SEED:-0}" \
    --out "docs/results/${TAG}.json" \
    'env.assets.handle_head_types=[hammer]' \
    "$@"
status=$?

# Surface the headline without hiding the warnings: the whole log is kept in slurm_logs, and the
# harness already raises on overflow rather than leaving it to a grep.
echo "[job] tag=$TAG exit=$status"
exit $status
