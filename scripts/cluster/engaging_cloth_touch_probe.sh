#!/bin/bash
# Check the solver-contact touch test (env.cloth.release_touch=contact) against the origin distance
# on a policy's episodes (scripts/analysis/cloth_touch_probe.py). Measured fabric + plasticity,
# full-surface contact, release condition with the contact test switched on.
#
#   sbatch scripts/cluster/engaging_cloth_touch_probe.sh
# Override with CKPT=, FABRIC=, NENVS=, SEED=, EXTRA=.
# Output: docs/results/cloth_touch/<fabric>_s<seed>.json
#SBATCH --job-name=touchprobe
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUTDIR=docs/results/cloth_touch
mkdir -p slurm_logs "$OUTDIR"
export PYTHONUNBUFFERED=1
FABRIC="${FABRIC:-big_napkin}"
CKPT="${CKPT:-outputs/2026-11-09/19-29-56/0_ftfabrel3sRfs_big_napkin_28533447/nn/last_0_ftfabrel3sRfs_big_napkin_28533447_ep_500_rew__1754.8378_.pth}"
SEED="${SEED:-3}"
[ -f "$CKPT" ] || { echo "[touch] no checkpoint $CKPT" >&2; exit 1; }
read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "${FABRIC}_measured" --plastic)"
echo "[touch] fabric=${FABRIC}_measured ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
scripts/newton_py -m scripts.analysis.cloth_touch_probe \
    --checkpoint "$CKPT" --policy_config "${POLICY_CFG:-checkpoints/config.yaml}" \
    --num_envs "${NENVS:-32}" --max_steps "${MAXSTEPS:-400}" --seed "$SEED" --out "$OUTDIR/${FABRIC}${OUT_SUFFIX:-}_s${SEED}.json" \
    env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
    env.cloth.full_surface_contact=true env.cloth.release_persist=true env.episode_length_s=12.0 \
    env.cloth.persist_window_steps=180 env.cloth.persist_consecutive=true env.cloth.release_touch=contact \
    "${FAB_OV[@]}" ${EXTRA:-}
