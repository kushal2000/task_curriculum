#!/bin/bash
#SBATCH --job-name=vbd_render
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/renders/slurm_%A_%a.out
# Render 4 envs (random start, seed 1) of line (task id + 1) of $RENDERS: "name|checkpoint|label|args".
# KEY=VAL tokens in args become env vars; the rest go to render_vbd.py (script flags and hydra overrides).
set -uo pipefail
IFS='|' read -r NAME CKPT LABEL ARGS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${RENDERS:-outputs/renders/renders.txt}")
read -r -a TOK <<<"$ARGS"; A=(); for t in "${TOK[@]}"; do if [[ "$t" =~ ^[A-Z_][A-Z0-9_]*= ]]; then export "$t"; else A+=("$t"); fi; done
OUT=outputs/renders/$NAME; mkdir -p "$OUT"; ls "$OUT"/*.mp4 >/dev/null 2>&1 && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
echo "[render] $NAME: ${A[*]} env: $(env | grep -E '^(NEWTON_|RN_|VBD_)' | tr '\n' ' ')"
.venv/isaaclab3/bin/python $PWD/experiments/nov25_better_rigidcloth/eval/render_vbd.py --checkpoint "$CKPT" --label "$LABEL" \
  --num_envs 4 --seed 1 $([ -n "${PINNED:-}" ] || echo --randomize) --stochastic --record --max_steps 900 --out "$OUT" "${A[@]}" > "$OUT/log.txt" 2>&1
ls "$OUT"/*.mp4 >/dev/null 2>&1 || { echo "[render] FAILED $NAME"; tail -n 20 "$OUT/log.txt"; exit 1; }
