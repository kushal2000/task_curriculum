#!/bin/bash
# Screen many cable configurations for stability, a few envs each.
#
#   sbatch scripts/cluster/sbatch_velgrid.sh <group> "<tag>|<overrides>" "<tag>|<overrides>" ...
#
# Screening wants MANY configs at FEW envs, not the reverse: the speed distribution is visible
# with 4 envs, so a run takes a fraction of the time and a job can sweep a whole row of the grid.
# Goal counts are the opposite -- goals are rare enough to need 64 envs -- so the flow is screen
# small here, then confirm the survivors with sbatch_cable_eval.sh at scale.
#
# The caveat that makes the second half non-optional: a few-env probe can rule a problem IN but
# never OUT. Contact-buffer overflow only appears above ~16 envs, and it is invisible at 4.

#SBATCH --job-name=velgrid
#SBATCH --partition=portal
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail

GROUP="$1"; shift

cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs docs/results

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

NUM_ENVS="${NUM_ENVS:-4}"
STEPS="${STEPS:-300}"

echo "[grid] group=$GROUP envs=$NUM_ENVS steps=$STEPS host=$(hostname) configs=$#"

for spec in "$@"; do
    tag="${spec%%|*}"
    overrides="${spec#*|}"
    echo "[grid] --- $tag : $overrides"
    # shellcheck disable=SC2086
    scripts/newton_py -m isaacsimenvs.eval.velocity_stats \
        --task Isaacsimenvs-Cable-Direct-v0 \
        --num_envs "$NUM_ENVS" \
        --steps "$STEPS" \
        --num_assets_per_type 1 \
        --single_variant \
        --out "docs/results/vel_${tag}.json" \
        'env.assets.handle_head_types=[hammer]' \
        env.cable.thickness=0.03 $overrides 2>&1 \
      | grep -E "VELSTATS|Error executing|RuntimeError" | sed "s/^/[$tag] /"
done

echo "[grid] group=$GROUP done"
