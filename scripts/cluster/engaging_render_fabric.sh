#!/bin/bash
# Film one episode of a cloth-fold policy on a MEASURED fabric (plasticity on), one array task per clip.
# Engaging port of bos14_render_cloth.sh. By default each array index is one EVAL seed (3, 4, ...) with
# the eval's fixed start (randomisation off), so the clips show the conditions behind the eval numbers;
# the clip ends one second after the filmed env's first episode.
#
#   FABRIC=towel CKPT=<path.pth> NAME=release OUTDIR=videos/fabric_release \
#       sbatch --array=0-3 scripts/cluster/engaging_render_fabric.sh
#   -> videos/fabric_release/towel/release_s3.mp4 ... release_s6.mp4
#
# The goal marker is hidden (--no_goal): in this renderer it draws as an unlit black plate on the
# stationary half (commit 81249b8). NO_GOAL= (empty) shows it again.
# STOP_AT_END= (empty) films all STEPS instead of stopping 1 s after the first episode ends.
# RENDER_ARGS: extra Hydra overrides (e.g. the release-and-persist flags). Same SEED / NUM_ENVS /
# camera = same starting sheet pose per world, so two checkpoints can be compared clip for clip.
#SBATCH --job-name=fabric_render
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=node4207
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

FABRIC="${FABRIC:?set FABRIC=<towel|denim|...>}"
CKPT="${CKPT:?set CKPT=<checkpoint.pth>}"
NAME="${NAME:-policy}"
OUTDIR="${OUTDIR:-videos/fabric_render}/${FABRIC}"
COEF="${COEF:-0}"
STEPS="${STEPS:-780}"            # 12 s episode at 60 Hz + slack; the clip stops at the episode end
NUM_ENVS="${NUM_ENVS:-8}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
STRIDE="${STRIDE:-2}"            # every 2nd policy step -> 30 fps is real time
CAM_OFFSET="${CAM_OFFSET:-0.35 -0.35 0.85}"
LOOK_AT="${LOOK_AT:-0.0 0.0 0.53}"
# Default = the eval protocol (engaging_fabric_eval.sh / episodes.py): randomisation off, fixed start
# pose, env 0, one eval seed per array task (SEED_BASE + index: 3, 4, 5, ...).
# RANDOMIZE=1: the task's own reset distribution (+-10 cm, random yaw), one env per array task, one seed.
if [ -n "${RANDOMIZE:-}" ]; then
    SEED="${SEED_BASE:-3}"; WORLD="$IDX"; RAND_OV=(--randomize_reset); TAGW="w${WORLD}"
else
    SEED=$(( ${SEED_BASE:-3} + IDX )); WORLD=0; RAND_OV=(); TAGW="s${SEED}"
fi
mkdir -p "$OUTDIR" slurm_logs
export PYTHONUNBUFFERED=1 PYGLET_HEADLESS=true OMNI_KIT_ACCEPT_EULA=YES

read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "${FABRIC}_measured" --plastic)"
[ "${#FAB_OV[@]}" -gt 0 ] || { echo "[render] no overrides for ${FABRIC}_measured" >&2; exit 1; }
read -r -a EXTRA_OV <<<"${RENDER_ARGS:-}"
OUT="$OUTDIR/${NAME}_${TAGW}.mp4"
# Rendered under a temporary name and renamed on success: a preempted (requeued) task otherwise leaves a
# 48-byte or truncated .mp4 under the final name that looks finished.
PART="$OUTDIR/.${NAME}_${TAGW}.part.mp4"
rm -f "$PART"
echo "[render] fabric=$FABRIC name=$NAME seed=$SEED world=$WORLD rand=${RANDOMIZE:-0} coef=$COEF ckpt=$CKPT host=$(hostname -s) out=$OUT"

scripts/newton_py -m isaacsimenvs.eval.render_newton \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --checkpoint "$CKPT" \
    --policy_config "${POLICY_CFG:-checkpoints/config.yaml}" \
    --sapg_expl_coef "$COEF" \
    --num_envs "$NUM_ENVS" \
    --world "$WORLD" \
    --steps "$STEPS" \
    --stride "$STRIDE" \
    --fps 30 \
    --cam_offset $CAM_OFFSET \
    --look_at $LOOK_AT \
    --seed "$SEED" \
    --num_assets_per_type 1 \
    ${RAND_OV[@]+"${RAND_OV[@]}"} \
    ${STOP_AT_END---stop_on_episode_end} \
    --end_hold_s 1.0 \
    ${NO_GOAL---no_goal} \
    --out "$PART" \
    env.cloth.keypoint_tolerance=0.015 \
    env.cloth.nan_policy=reset \
    env.termination.success_steps=10 \
    "${FAB_OV[@]}" ${EXTRA_OV[@]+"${EXTRA_OV[@]}"}
rc=$?
if [ "$rc" = 0 ] && [ -s "$PART" ]; then
    mv -f "$PART" "$OUT"
    [ -f "${PART%.mp4}.json" ] && mv -f "${PART%.mp4}.json" "${OUT%.mp4}.json"
fi
echo "[render] world=$WORLD exit=$rc out=$OUT"
exit $rc
