#!/bin/bash
#SBATCH --job-name=render_pinned_rigid_cloth
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/controlled_evals/slurm_%x_%A_%a.out
# MP4s of eval_pinned_rigid_cloth: 4 episodes of env 0, pinned start, sampled actions; task 0 = 64live,
# 1 = 64live_inplace -> outputs/renders/rigid_pin_<policy>_v3/.
set -uo pipefail
P=(64live 64live_inplace); N=${P[$SLURM_ARRAY_TASK_ID]}
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/rl_games"
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$TMPDIR"
/share/portal/kk837/task_curriculum/.venv/isaacsim/bin/python experiments/nov27_controlled_evals/render_rigid_cloth.py \
  --headless --enable_cameras --checkpoint pretrained_policy/play_pc_rigidclothRL_$N.pth --episodes 4 --seed 1 \
  --label "rigid plates (PhysX) · $N" --out outputs/renders/rigid_pin_${N}_v3${PINNED_YAW_DEG:+_yaw$PINNED_YAW_DEG} "$@"
# The video: the recordings drawn in the shared look (render_recording.py), replacing Isaac's own frames
OUT=outputs/renders/rigid_pin_${N}_v3${PINNED_YAW_DEG:+_yaw$PINNED_YAW_DEG}
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/isaaclab3/bin/python experiments/nov27_controlled_evals/render_recording.py "$OUT"
