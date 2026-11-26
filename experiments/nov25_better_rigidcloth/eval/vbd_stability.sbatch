#!/bin/bash
#SBATCH --job-name=vbd_stab
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_stability/slurm_%A_%a.out
# One VBD stability-probe variant per array task: line (task id + 1) of $VARIANTS ("name override ...").
#   sbatch --array=<lines-1> --export=ALL,VARIANTS=outputs/vbd_stability/variants.txt experiments/nov25_better_rigidcloth/eval/vbd_stability.sbatch
set -uo pipefail
read -r -a V < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${VARIANTS:-outputs/vbd_stability/variants.txt}")
NAME=${V[0]}; OVR=(); for t in "${V[@]:1}"; do if [[ "$t" =~ ^[A-Z_][A-Z0-9_]*= ]]; then export "$t"; else OVR+=("$t"); fi; done; O=outputs/vbd_stability
[ -s "$O/$NAME.json" ] && { echo "exists $O/$NAME.json"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null   # pre-compiled kernels (~140 s of a cold start)
echo "[stab] $NAME on $(hostname -s): ${OVR[*]}"
.venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/eval/vbd_stability_probe.py --name "$NAME" --out $O "${OVR[@]}"
[ -s "$O/$NAME.json" ] || { echo "[stab] FAILED $NAME"; exit 1; }
