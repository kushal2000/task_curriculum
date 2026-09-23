#!/bin/bash
# Held-fold evaluation of ONE checkpoint, one array task per SAPG block coefficient.
#
#   CKPT=/path/model.pth TAG=ref COEFS="50 40 30 20 10 0" SEEDS=1 sbatch --array=0-5 \
#       scripts/cluster/bos14_cloth_eval_ckpt.sh                  # block probe
#   CKPT=... TAG=ref COEFS=0 SEEDS="1 2 3 4 5" sbatch --array=0-0 scripts/cluster/bos14_cloth_eval_ckpt.sh
#
# Same protocol as bos14_cloth_eval_pvs.sh (32 envs x 900 steps, success_tolerance 0.01 so the
# inherited rigid-tool criterion cannot fire, env.termination.success_steps=10 so a fold must hold
# 10 steps), so numbers are directly comparable to docs/results/cloth_prior_vs_scratch.md.
#
# The block matters: a SAPG checkpoint holds six policies conditioned on an exploration coefficient
# (linspace(50, 0, 6)), and a finetune need not have learned in the same block as the checkpoint it
# started from. Probe first, then evaluate the winner across seeds.
#
# Results: docs/results/cloth_${TAG}_c${COEF}_s${SEED}.json

#SBATCH --job-name=cloth_eval_ckpt
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p docs/results

CKPT="${CKPT:?set CKPT=<checkpoint.pth>}"
TAG="${TAG:?set TAG=<short name>}"
read -r -a COEF_LIST <<<"${COEFS:-0}"
COEF="${COEF_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"
SEEDS="${SEEDS:-1}"
[ -f "$CKPT" ] || { echo "[eval] checkpoint not found: $CKPT" >&2; exit 1; }

export PYTHONUNBUFFERED=1
echo "[eval] tag=$TAG coef=$COEF seeds='$SEEDS' checkpoint=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"

for SEED in $SEEDS; do
    OUT="docs/results/cloth_${TAG}_c${COEF}_s${SEED}.json"
    echo "[eval] === coef $COEF seed $SEED -> $OUT ==="
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
        || echo "[eval] coef $COEF seed $SEED FAILED"
done

echo "[eval] === summary ==="
.venv_isaaclab3/bin/python scripts/analysis/cloth_fold_summary.py --label "${TAG}_c${COEF}" docs/results/cloth_${TAG}_c${COEF}_s*.json
