#!/bin/bash
#SBATCH --job-name=vbd_stab_render
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_stability/renders/slurm_%A_%a.out
# Renders + viser recordings for one stability variant per array task (line task_id+1 of $VARIANTS), in the
# same world as vbd_stability.sbatch: vbd.yaml, pinned start, seed 1, 64live; 4 envs, each env's first episode.
#   sbatch --array=0-7 --export=ALL,VARIANTS=outputs/vbd_stability/variants.txt experiments/nov25_better_rigidcloth/eval/vbd_stability_render.sbatch
set -uo pipefail
read -r -a V < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${VARIANTS:-outputs/vbd_stability/variants.txt}")
NAME=${V[0]}; OVR=("${V[@]:1}"); OUT=outputs/vbd_stability/renders/$NAME; mkdir -p "$OUT"
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
echo "[render] $NAME on $(hostname -s): ${OVR[*]}"
.venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/eval/render_vbd.py --num_envs 4 --seed 1 --record --out "$OUT" "${OVR[@]}"
ls "$OUT"/*.npz >/dev/null 2>&1 || { echo "[render] FAILED $NAME"; exit 1; }
