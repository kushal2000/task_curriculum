#!/bin/bash
#SBATCH --job-name=rigid_pin
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/share/portal/kk837/task_curriculum_nov25
#SBATCH --output=/share/portal/kk837/task_curriculum_nov25/outputs/rigid_pinned/slurm_%A_%a.out
# The training plates (PhysX, Isaac Lab 2.3) from the VBD eval's PINNED start (identity pose -> the
# sheet at 180 deg, base.reset_yaw), sampled actions, 10 cm box2-mid: does the policy fold the right
# half here? Task i: policy (i / 2), seed (i % 2 + 1).
set -uo pipefail
I=$SLURM_ARRAY_TASK_ID; S=$((I % 2 + 1)); P=(64live 64live_inplace); N=${P[$((I / 2))]}
OUT=outputs/rigid_pinned/${N}__pin_10cm__s$S.json
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/rl_games"
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}_$I; mkdir -p $TMPDIR
/share/portal/kk837/task_curriculum/.venv/isaacsim/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-RigidCloth-Direct-v0 \
  --checkpoint pretrained_policy/play_pc_rigidclothRL_$N.pth --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --max_steps 900 \
  --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --stochastic --headless \
  --experiment experiments/nov24_play_pc_rl/play_pc_rl_A.yaml --num_envs 32 --seed $S --out "$OUT" \
  env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
  "env.fold_pool.shapes=[box2-mid]" env.fold_pool.one_shape_per_env=false env.fold_pool.init_fold_prob=0.0 \
  env.fold_pool.goal_task.mode=half_fold "env.fold_pool.size_range=[0.10,0.10]"
[ -s "$OUT" ] || { echo "FAILED $OUT"; exit 1; }
