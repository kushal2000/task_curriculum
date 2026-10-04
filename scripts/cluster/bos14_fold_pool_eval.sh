#!/bin/bash
# Evaluate the stage-2 fold-pool policies: per shape and per sheet size, pinned and randomised.
#
#   TRAIN_JOB=156603 sbatch --array=0-5 scripts/cluster/bos14_fold_pool_eval.sh
#   .venv_isaaclab3/bin/python scripts/analysis/fold_pool_eval_summary.py --train_job 156603
#
# A training array launched with RUN_TAG needs the same RUN_TAG here.
#
# Array index = (policy seed - 1) * 2 + mode; mode 0 = pinned, 1 = randomised. Each task runs two
# evaluation seeds. One episode per env (isaacsimenvs.eval.episodes), success = the fold HELD for
# 10 steps at the task's own 1.5 cm tolerance, as every stage-1 cloth number was reported.
#
#   pinned      reset noise and obs/action DR off (eval.protocol.disable_randomization) AND the
#               contact-shell DR off: nominal physics. Shape and size still vary -- they ARE the
#               task.
#   randomised  the training regime, contact shell included.
#
# 320 envs, not stage 1's 32: one episode per env split over five shapes leaves ~64 per shape per
# run instead of ~6. Sheet sizes are drawn per env from fold_pool.size_seed, so every run evaluates
# the same 320 sizes. The contact buffer is 4x training's: stage 1 found evaluation of a policy that
# folds overflowing the training size (548k pairs/env against 524288).

#SBATCH --job-name=fp_eval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results
: "${TRAIN_JOB:?set TRAIN_JOB to the job id of the fold-pool training array}"

I="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( I / 2 + 1 ))
MODE=$(( I % 2 ))
NAME="0_fp_${RUN_TAG:-pool}_s${SEED}_${TRAIN_JOB}"
CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs')) or '')" "$NAME")
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[fp_eval] no checkpoint for $NAME" >&2; exit 1; }

if [ "$MODE" -eq 1 ]; then
    MODE_ARGS=(--randomize); TAG=rand
else
    MODE_ARGS=(env.rigid_cloth.randomization.enabled=false); TAG=pinned
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
LABEL="${LABEL_PREFIX:-fp_eval_}s${SEED}_${TRAIN_JOB}__${TAG}"

export PYTHONUNBUFFERED=1
echo "[fp_eval] label=$LABEL ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD) extra=${EXTRA_ARGS:-none}"
for ES in 1 2; do
    OUT="docs/results/${LABEL}_s${ES}.json"
    echo "[fp_eval] === eval seed $ES -> $OUT ==="
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task Isaacsimenvs-FoldPool-Direct-v0 \
        --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef "${EXPL_COEF:-0}" \
        --num_envs "${ENVS:-320}" \
        --max_steps 900 \
        --success_tolerance 0.01 \
        --num_assets_per_type 1 \
        --single_variant \
        "env.newton.per_env_triangle_pairs=${TRI_PAIRS:-2097152}" \
        "${MODE_ARGS[@]}" \
        --seed "$ES" \
        --out "$OUT" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        "${EXTRA[@]}" \
        || echo "[fp_eval] $LABEL seed $ES FAILED"
done
