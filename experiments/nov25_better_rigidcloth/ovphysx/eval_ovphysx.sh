#!/bin/bash
# One evaluation of 64live on the PhysX plate pool through kit-less Isaac Lab 3 + ovphysx
# (Isaacsimenvs-RigidClothOvPhysx-Direct-v0), settings as the IL2 PhysX reference
# (outputs/rigid_physx_ref/64live__physx_15cm__s1.json: eval/ablate.sbatch, box2-mid at 15 cm).
#   eval_ovphysx.sh OUT.json SEED NUM_ENVS MAX_STEPS [extra hydra overrides...]
# CKPT=... overrides the checkpoint.
set -uo pipefail
cd /share/portal/kk837/task_curriculum_nov25
OUT=$1 SEED=$2 NENV=$3 MAXS=$4; shift 4
CKPT=${CKPT:-pretrained_policy/play_pc_rigidclothRL_64live.pth}
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/rl_games" LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TMPDIR=${TMPDIR:-/tmp/${USER}_ovphysx_$$}; mkdir -p "$TMPDIR"
exec .venv/isaaclab3_ovphysx/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-RigidClothOvPhysx-Direct-v0 \
  --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --max_steps "$MAXS" \
  --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --randomize \
  --experiment experiments/nov24_play_pc_rl/play_pc_rl_A.yaml --num_envs "$NENV" --seed "$SEED" --out "$OUT" \
  env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset env.termination.success_steps=10 \
  "env.fold_pool.shapes=[box2-mid]" env.fold_pool.one_shape_per_env=false env.fold_pool.init_fold_prob=0.0 \
  env.fold_pool.goal_task.mode=half_fold "env.fold_pool.size_range=[0.15,0.15]" "$@"
