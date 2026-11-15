#!/bin/bash
# Springback of a policy's own folds after release (scripts/analysis/cloth_springback_probe.py):
# flap opening angle per step, nearest hand link, and the hinge bend / plastic set by distance from
# the crease at the release latch. Measured fabric + plasticity, release condition, full-surface contact.
#
#   sbatch --array=0-1 scripts/cluster/engaging_cloth_springback.sh
#     0 = the training release test (fingertips + iiwa14_link_7), 1 = every hand link counts (left_.*|iiwa14_link_7)
#   SET=yank sbatch --array=0-1 scripts/cluster/engaging_cloth_springback.sh
#     the arm is taken from the policy and pulled back to its start pose at 0 = the release latch,
#     1 = the fold entry (before any press): does the fold stay folded with nothing touching it?
#   SET=vanish ...   same triggers, hand-cloth contact switched off instead of moving the arm
# Override with CKPT=, FABRIC=, NENVS=, SEED=, EXTRA=.
# Output: docs/results/cloth_springback/<fabric>_<tag>_s<seed>.{json,npz}
#SBATCH --job-name=springback
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUTDIR=docs/results/cloth_springback
mkdir -p slurm_logs "$OUTDIR"
export PYTHONUNBUFFERED=1
I="${SLURM_ARRAY_TASK_ID:-0}"
TAGS=(reltest_default reltest_allhand)
ARMS=("" "env.cloth.release_contact_bodies='left_.*|iiwa14_link_7'")
PROBE=("" "")
if [ "${SET:-}" = yank ]; then
    # Take the arm from the policy at the release latch / at the fold entry and drive it back to its
    # start pose over 0.5 s (fingers frozen); keep the episode running after a lost fold so all
    # 3 s are watched.
    TAGS=(yank_release yank_entry)
    ARMS=("env.cloth.persist_fail_terminates=false" "env.cloth.persist_fail_terminates=false")
    PROBE=("--yank release" "--yank entry")
fi
if [ "${SET:-}" = vanish ]; then
    # Same triggers, but the hand-cloth contact of that env is switched off instead (the hand passes
    # through the sheet), so nothing can drag the fold: only the cloth decides whether it holds.
    TAGS=(vanish_release vanish_entry)
    ARMS=("env.cloth.persist_fail_terminates=false" "env.cloth.persist_fail_terminates=false")
    PROBE=("--yank release --remove contact" "--yank entry --remove contact")
fi
FABRIC="${FABRIC:-big_napkin}"
CKPT="${CKPT:-outputs/2026-11-09/19-29-56/0_ftfabrel3sRfs_big_napkin_28533447/nn/last_0_ftfabrel3sRfs_big_napkin_28533447_ep_500_rew__1754.8378_.pth}"
SEED="${SEED:-3}"
[ -f "$CKPT" ] || { echo "[spring] no checkpoint $CKPT" >&2; exit 1; }
read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "${FABRIC}_measured" --plastic)"
# A key given in EXTRA replaces the preset's (Hydra rejects the same key twice).
for x in ${EXTRA:-}; do k="${x%%=*}"; for i in "${!FAB_OV[@]}"; do [ "${FAB_OV[$i]%%=*}" = "$k" ] && unset 'FAB_OV[i]'; done; done
FAB_OV=("${FAB_OV[@]}")
OUT="$OUTDIR/${FABRIC}_${TAGS[$I]}${OUT_SUFFIX:-}_s${SEED}.json"
echo "[spring] ${TAGS[$I]} fabric=${FABRIC}_measured ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
scripts/newton_py -m scripts.analysis.cloth_springback_probe \
    --checkpoint "$CKPT" --policy_config "${POLICY_CFG:-checkpoints/config.yaml}" \
    --num_envs "${NENVS:-64}" --max_steps 900 --seed "$SEED" --out "$OUT" ${PROBE[$I]} ${PROBE_EXTRA:-} \
    env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
    env.cloth.full_surface_contact=true env.cloth.release_persist=true env.episode_length_s=12.0 \
    env.cloth.persist_window_steps=180 \
    "${FAB_OV[@]}" ${ARMS[$I]} ${EXTRA:-}
