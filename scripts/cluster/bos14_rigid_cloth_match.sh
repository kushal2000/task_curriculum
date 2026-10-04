#!/bin/bash
# Measure the chain against the cloth where transfer depends on it: heading at reset, contact
# height, friction, the fold, and (for the randomised arm) what the solver actually received.
#
#   sbatch --array=0-6 scripts/cluster/bos14_rigid_cloth_match.sh
#
# Index -> configuration. The VBD cloth is the target every chain number is read against.
#
#   0  vbd                      the cloth itself
#   1  box3-mid     old         contact_margin=0, friction_priority=false (the pre-fix physics)
#   2  box2-surface old
#   3  box3-mid     fixed       defaults: cloth-matched margin + priority friction
#   4  box2-surface fixed
#   5  box3-mid     fixed + DR  randomization.enabled=true
#   6  box2-surface fixed + DR
#
# The heading fix is in the adapter and has no switch, so "old" differs from "fixed" in height and
# friction only; `rigid_cloth_yaw_probe.py` measured the heading before the fix.

#SBATCH --job-name=rc_match
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

I="${SLURM_ARRAY_TASK_ID:-0}"
NUM_ENVS="${NUM_ENVS:-32}"
OLD=(env.rigid_cloth.contact_margin=0.0 env.rigid_cloth.friction_priority=false)
case "$I" in
    0) LABEL=vbd;               ARGS=(--task Isaacsimenvs-Cloth-Direct-v0) ;;
    1) LABEL=box3-mid_old;      ARGS=(--variant box3-mid "${OLD[@]}") ;;
    2) LABEL=box2-surface_old;  ARGS=(--variant box2-surface "${OLD[@]}") ;;
    3) LABEL=box3-mid_fixed;    ARGS=(--variant box3-mid) ;;
    4) LABEL=box2-surface_fixed; ARGS=(--variant box2-surface) ;;
    5) LABEL=box3-mid_dr;       ARGS=(--variant box3-mid --randomize) ;;
    6) LABEL=box2-surface_dr;   ARGS=(--variant box2-surface --randomize) ;;
    *) echo "[match] no configuration for index $I" >&2; exit 1 ;;
esac

export PYTHONUNBUFFERED=1
echo "[match] $LABEL envs=$NUM_ENVS host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
scripts/newton_py -m scripts.analysis.rigid_cloth_match_probe \
    "${ARGS[@]}" --num_envs "$NUM_ENVS" --out "docs/results/rigid_cloth_match_${LABEL}.json"
echo "[match] exit=$?"
