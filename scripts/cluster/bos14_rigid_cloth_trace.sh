#!/bin/bash
# Run `episodes.py` and `render_newton.py` back to back on the SAME node with the same settings,
# printing the shared RC_TRACE line from both, so the two rollouts can be diffed step by step.
#
#   sbatch scripts/cluster/bos14_rigid_cloth_trace.sh
#   diff <(grep '^\[trace\]' slurm_logs/rc_trace_<id>.out | head -20) ...
#
# Exists because a clip and the evaluation table disagreed -- the same box3-mid checkpoint folds at
# step 81 under `episodes.py` and was still flat at step 81 on film -- and three rounds of reading
# both files produced three wrong guesses (reset randomisation, env count, the priming zero tick).
# Two traces answer it directly: if step 0 already differs the observation differs, and if the
# observations match but the actions do not the player is configured differently.

#SBATCH --job-name=rc_trace
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs

CKPT="${CKPT:-outputs/2026-09-30/02-54-22/0_rc_box3-mid_s1_155046/nn/0_rc_box3-mid_s1_155046.pth}"
TASK="${TASK:-Isaacsimenvs-RigidCloth-Direct-v0}"
VARIANT="${VARIANT:-box3-mid}"
NUM_ENVS="${NUM_ENVS:-8}"
STEPS="${STEPS:-20}"

export PYTHONUNBUFFERED=1 PYGLET_HEADLESS=true RC_TRACE="$STEPS"
echo "[trace] host=$(hostname -s) task=$TASK variant=$VARIANT envs=$NUM_ENVS steps=$STEPS"

echo "[trace] ===== episodes.py ====="
scripts/newton_py -m isaacsimenvs.eval.episodes \
    --task "$TASK" --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
    --sapg_expl_coef 0 --num_envs "$NUM_ENVS" --max_steps "$STEPS" \
    --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --seed 1 \
    --out /dev/null \
    env.rigid_cloth.variant="$VARIANT" env.cloth.nan_policy=reset env.termination.success_steps=10 \
    2>&1 | grep -E "^\[trace\]|Traceback|Error"

echo "[trace] ===== render_newton.py ====="
scripts/newton_py -m isaacsimenvs.eval.render_newton \
    --task "$TASK" --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
    --sapg_expl_coef 0 --num_envs "$NUM_ENVS" --world 0 --steps "$STEPS" --stride 1000 \
    --success_tolerance 0.01 --num_assets_per_type 1 --seed 1 \
    --out videos/rigid_cloth/_trace.mp4 \
    env.rigid_cloth.variant="$VARIANT" env.cloth.nan_policy=reset env.termination.success_steps=10 \
    2>&1 | grep -E "^\[trace\]|Traceback|Error"

echo "[trace] done"
