#!/bin/bash
#SBATCH --job-name=vbd_conv_hand
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/vbd_convergence/hand_%A_%a.out
# The stable.yaml world with the hand: pinned start, sampled actions (seeded), 4 envs, 4 s, cloth recorded every
# policy step, at one VBD iteration count per array task (6 12 24 48 96 192).
set -uo pipefail
IT=(6 12 24 48 96 192); N=${IT[$SLURM_ARRAY_TASK_ID]}
OUT=outputs/vbd_convergence/hand_it$N; mkdir -p $OUT
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 NEWTON_PUSH_APPROXIMATION=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
FREE_KB=$(df -Pk /tmp | awk 'NR==2{print $4}'); if [ "${FREE_KB:-0}" -gt 8000000 ]; then TROOT=/tmp; else TROOT=$PWD/.cache/jobtmp; fi
export TMPDIR=$TROOT/${USER}_convh_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
.venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/eval/render_vbd.py --checkpoint pretrained_policy/play_pc_rigidclothRL_64live.pth \
  --protocol isaacsimenvs/tasks/cloth/config/eval/stable.yaml --label "VBD iterations $N" --num_envs 4 --seed 1 \
  --record --no_video --stochastic --max_steps 240 --out $OUT env.cloth.vbd_iterations=$N env.cloth.keypoint_tolerance=0.001 > $OUT/log.txt 2>&1
ls $OUT/*.npz
