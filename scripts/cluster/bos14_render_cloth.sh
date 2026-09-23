#!/bin/bash
# Film one env of a cloth-fold policy on bos14, one array task per env.
#
#   CKPT=<path> OUTDIR=videos/foo sbatch --array=0-5 scripts/cluster/bos14_render_cloth.sh
#
# The array index IS the filmed env (`--world`), and `--randomize_reset` keeps the task's own reset
# distribution, so a set of renders shows different initial sheet poses rather than six copies of
# one state. Renders are cheap and independent: film several and keep whichever are informative.
#
# COEF must match the block the checkpoint actually learned in -- 0 for the prior-vs-scratch runs,
# 50 for the released SimToolReal checkpoint. See docs/results/cloth_prior_vs_scratch.md.
#
# PYGLET_HEADLESS=true, NOT xvfb-run: bos14's compute nodes have no Xvfb (only the login node does,
# so `xvfb-run` there exits 127 on every task). ViewerGL is pyglet-based and pyglet's headless mode
# uses EGL directly, which is what the earlier working renders used.
#
# Runs through outputs/fold_render/fold_render.py when present: it wraps render_newton and prints
# each fold/fold_held termination with the video timestamp, so a clip can be checked against the
# criterion instead of eyeballed.

#SBATCH --job-name=cloth_render
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum

CKPT="${CKPT:?set CKPT=<checkpoint.pth>}"
OUTDIR="${OUTDIR:-videos/cloth_render}"
COEF="${COEF:-0}"
STEPS="${STEPS:-900}"
NUM_ENVS="${NUM_ENVS:-8}"
SEED="${SEED:-1}"
STRIDE="${STRIDE:-3}"
# The default camera is framed for the rigid tool task; the sheet is 10 cm and unreadable from
# there. CLOSE is the framing used for the prior-vs-scratch clips -- keep it fixed when two
# checkpoints are to be compared side by side, along with SEED and the env count (they determine
# the reset draw, so the same world index starts from the same sheet pose).
CAM_OFFSET="${CAM_OFFSET:-1.6 -1.6 1.15}"
LOOK_AT="${LOOK_AT:-0.0 0.0 0.62}"
WORLD=$SLURM_ARRAY_TASK_ID
mkdir -p "$OUTDIR" slurm_logs

export PYTHONUNBUFFERED=1 PYGLET_HEADLESS=true

WRAPPER=outputs/fold_render/fold_render.py
if [ -f "$WRAPPER" ]; then RUN=(scripts/newton_py "$WRAPPER"); else RUN=(scripts/newton_py -m isaacsimenvs.eval.render_newton); fi
echo "[render] world=$WORLD coef=$COEF steps=$STEPS ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"

"${RUN[@]}" \
    --task Isaacsimenvs-Cloth-Direct-v0 \
    --checkpoint "$CKPT" \
    --policy_config pretrained_policy/config.yaml \
    --sapg_expl_coef "$COEF" \
    --num_envs "$NUM_ENVS" \
    --world "$WORLD" \
    --steps "$STEPS" \
    --stride "$STRIDE" \
    --cam_offset $CAM_OFFSET \
    --look_at $LOOK_AT \
    --seed "$SEED" \
    --num_assets_per_type 1 \
    --randomize_reset \
    --out "$OUTDIR/w${WORLD}.mp4" \
    env.cloth.nan_policy=reset \
    env.termination.success_steps=10
echo "[render] world=$WORLD exit=$?"
