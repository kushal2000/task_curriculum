#!/bin/bash
# Held-fold evaluation of the prior-vs-scratch runs on bos14, one checkpoint per array task.
#
#   COEF=0 sbatch --array=0-6 scripts/cluster/bos14_cloth_eval_pvs.sh     # results tagged _c0
#
# COEF is the SAPG block conditioning (`--sapg_expl_coef`): 0 = block 5, the only block that learned
# to fold in these runs (per-block training successes 0.55-0.64 vs ~0.001 for blocks 0-4); 50 = block
# 0, the released checkpoint's value, which is what the first pass (job 146704, `_c50`) measured.
#
# Protocol is sbatch_cloth_eval.sh's (5 seeds x 32 envs x 900 steps, success_tolerance 0.01 so the
# inherited rigid-tool criterion cannot fire) WITH env.termination.success_steps=10. Training ran at
# success_steps=1, where the episode ends on the first folded step: that counts transient folds, and
# "held" can structurally never fire. Only a fold that persists 10 steps counts here.
#
# Task 0 is the untrained SimToolReal policy, re-measured at the current commit rather than taken
# from cloth_baseline_held10_s*, which predates the fold-criterion fixes in e01d7f8.

#SBATCH --job-name=cloth_eval_pvs
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p docs/results

R=outputs/2026-09-13
TAGS=(pvs_untrained pvs_pretrained_s1 pvs_pretrained_s2 pvs_pretrained_s3 pvs_scratch_s1 pvs_scratch_s2 pvs_scratch_s3)
CKPTS=(
    pretrained_policy/model.pth
    "$(ls $R/*/0_pretrained_s1_146678/nn/*_ep_509_*.pth)"
    "$(ls $R/*/0_pretrained_s2_146680/nn/*_ep_509_*.pth)"
    "$(ls $R/*/0_pretrained_s3_146682/nn/*_ep_509_*.pth)"
    "$(ls $R/*/0_scratch_s1_146679/nn/*_ep_509_*.pth)"
    "$(ls $R/*/0_scratch_s2_146681/nn/*_ep_509_*.pth)"
    "$(ls $R/*/0_scratch_s3_146683/nn/*_ep_509_*.pth)"
)
COEF="${COEF:-0}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}_c${COEF}"
CKPT="${CKPTS[$SLURM_ARRAY_TASK_ID]}"
[ -f "$CKPT" ] || { echo "[eval] checkpoint not found: $CKPT" >&2; exit 1; }

export PYTHONUNBUFFERED=1
echo "[eval] tag=$TAG sapg_expl_coef=$COEF checkpoint=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"

for SEED in 1 2 3 4 5; do
    OUT="docs/results/cloth_${TAG}_s${SEED}.json"
    echo "[eval] === seed $SEED -> $OUT ==="
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef "$COEF" \
        --num_envs 32 \
        --max_steps 900 \
        --success_tolerance 0.01 \
        --num_assets_per_type 1 \
        --single_variant \
        --seed "$SEED" \
        --out "$OUT" \
        env.termination.success_steps=10 \
        || echo "[eval] seed $SEED FAILED"
done

echo "[eval] === summary ==="
.venv_isaaclab3/bin/python scripts/analysis/cloth_fold_summary.py --label "$TAG" docs/results/cloth_${TAG}_s*.json
