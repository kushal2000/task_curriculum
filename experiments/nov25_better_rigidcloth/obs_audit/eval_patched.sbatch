#!/bin/bash
#SBATCH --job-name=obs_audit
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/obs_audit/slurm_%A_%a.out
# Array task i: WORLD x PATCH x SEED = (plates|vbd) x (none|bodyquat) x seed 1 / 2 ; 32 envs, rand start, 900 steps.
set -uo pipefail
I=$SLURM_ARRAY_TASK_ID; WORLDS=(plates vbd); PATCHES=(none bodyquat)
W=${WORLDS[$((I / 4))]}; P=${PATCHES[$(((I / 2) % 2))]}; S=$((I % 2 + 1))
OUT=outputs/obs_audit/eval_${W}_${P}_s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 OBS_AUDIT_PATCH=$P
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TMPDIR=${OBS_TMP_BASE:-/tmp}/${USER}_${SLURM_JOB_ID}_${I}; export WARP_CACHE_PATH=$TMPDIR/warp; mkdir -p "$WARP_CACHE_PATH"
cp -a .cache/warp_vbd/. "$WARP_CACHE_PATH/" 2>/dev/null
COMMON=(--checkpoint pretrained_policy/play_pc_rigidclothRL_64live.pth --policy_config pretrained_policy/config.yaml
  --sapg_expl_coef 0 --max_steps 900 --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --randomize
  --num_envs 32 --seed $S --out "$OUT")
if [ "$W" = plates ]; then
  .venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/obs_audit/eval_patched.py --task Isaacsimenvs-RigidClothNewton-Direct-v0 \
    "${COMMON[@]}" --experiment experiments/nov24_play_pc_rl/play_pc_rl_A.yaml \
    env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
    "env.fold_pool.shapes=[box2-mid]" env.fold_pool.one_shape_per_env=false env.fold_pool.init_fold_prob=0.0 \
    env.fold_pool.goal_task.mode=half_fold "env.fold_pool.size_range=[0.10,0.10]" env.robot_self_collision=false
else
  .venv/isaaclab3/bin/python experiments/nov25_better_rigidcloth/obs_audit/eval_patched.py --task Isaacsimenvs-Cloth-Direct-v0 \
    "${COMMON[@]}" --protocol isaacsimenvs/tasks/cloth/config/eval/vbd.yaml
fi
