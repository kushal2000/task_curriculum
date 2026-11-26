#!/bin/bash
#SBATCH --job-name=vbd_64live
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_eval_64live/slurm_%A_%a.out
# 64live on the VBD evaluation world (isaacsimenvs/tasks/cloth/config/eval/vbd.yaml), rand start.
#   index 0-4: eval.py --protocol vbd.yaml, 32 envs, seed index+1 -> outputs/vbd_eval_64live/64live__on_vbd__rand__s<seed>.json
#   index 5-9: render_vbd.py, 4 envs, seed index-4 -> outputs/vbd_renders/64live/*.mp4
# Newton venv (kit-less): this checkout's .venv/isaaclab3, the system libstdc++ preloaded (experiments/newton_py).
set -uo pipefail
I=$SLURM_ARRAY_TASK_ID
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$TMPDIR"
# Start from the shared, pre-compiled Warp kernels (~140 s of a 170 s cold start), copied into this job's
# own cache: jobs never write the shared copy, so concurrent jobs cannot read each other's half-written
# kernels (experiments/_common.sh). Kernels for another GPU architecture are compiled as usual.
WARP_SEED=.cache/warp_vbd; [ -d "$WARP_SEED" ] && cp -a "$WARP_SEED/." "$WARP_CACHE_PATH/" 2>/dev/null || mkdir -p "$WARP_CACHE_PATH"
PY=.venv/isaaclab3/bin/python; O=outputs/vbd_eval_64live; mkdir -p $O outputs/vbd_renders/64live
echo "[vbd] host $(hostname -s) index $I commit $(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain --untracked-files=no)" && echo YES || echo no)"
if [ "$I" -lt 5 ]; then
  S=$((I + 1)); OUT=$O/64live__on_vbd__rand__s$S.json
  [ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
  $PY -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-Cloth-Direct-v0 \
    --checkpoint pretrained_policy/play_pc_rigidclothRL_64live.pth --policy_config pretrained_policy/config.yaml \
    --sapg_expl_coef 0 --max_steps 900 --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --randomize \
    --protocol isaacsimenvs/tasks/cloth/config/eval/vbd.yaml --seed $S --out "$OUT"
  [ -s "$OUT" ] || { echo "[vbd] FAILED $OUT"; exit 1; }
else
  S=$((I - 4))
  $PY experiments/nov25_better_rigidcloth/eval/render_vbd.py --num_envs 4 --seed $S --randomize --out outputs/vbd_renders/64live
  ls outputs/vbd_renders/64live/vbd_seed${S}_*.mp4 >/dev/null 2>&1 || { echo "[vbd] FAILED renders seed $S"; exit 1; }
fi
