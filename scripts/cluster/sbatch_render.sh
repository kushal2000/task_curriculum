#!/bin/bash
# Render one env of the cable task to video.
#
#   sbatch scripts/cluster/sbatch_render.sh <world> <steps> [hydra overrides...]
#
# Renders are cheap and independent, so filming several envs in parallel and keeping whichever
# actually scores beats re-rolling one env: GPU nondeterminism means a replay does not reproduce
# the episode that scored in a measured run, and the caption reports the replay's own goal count.

#SBATCH --job-name=render
#SBATCH --partition=portal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out
#SBATCH --error=/share/portal/kk837/task_curriculum/slurm_logs/%x_%j.out

set -uo pipefail

WORLD="$1"; STEPS="$2"; shift 2

cd /share/portal/kk837/task_curriculum
mkdir -p slurm_logs videos/07_replays_final_config videos/08_cloth_demo

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

xvfb-run -a scripts/newton_py -m isaacsimenvs.eval.render_newton \
    --task "${TASK:-Isaacsimenvs-Cable-Direct-v0}" \
    --num_envs "${NUM_ENVS:-64}" \
    --world "$WORLD" \
    --steps "$STEPS" \
    --seed "${SEED:-0}" \
    --stride "${STRIDE:-3}" \
    --num_assets_per_type 1 \
    --out "${OUT:-videos/07_replays_final_config/cable_w${WORLD}_s${SEED:-0}.mp4}" \
    'env.assets.handle_head_types=[hammer]' \
    "$@"
echo "[render-job] world=$WORLD exit=$?"
