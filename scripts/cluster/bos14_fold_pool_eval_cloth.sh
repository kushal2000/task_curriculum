#!/bin/bash
# Stage-2 fold-pool policies, zero-shot on the VBD cloth, at both fold cutoffs.
#
#   sbatch --array=0-23 scripts/cluster/bos14_fold_pool_eval_cloth.sh
#
# Index = policy * 4 + cutoff * 2 + start, with
#   policy 0..2 = box3-mid prior, seeds 1-3 (train 156746, 0_fp_box3_s<n>_156746)
#          3..5 = generic prior,  seeds 1-3 (train 156744, 0_fp_pool_s<n>_156744)
#   cutoff 0 = 4 cm (the cloth task's keypoint_tolerance), 1 = 1.5 cm (what the pool trained on)
#   start  0 = pinned (`disable_randomization`), 1 = `--randomize`
#
# Protocol identical to bos14_rigid_cloth_eval_rc2.sh's cloth target (32 envs x 900 steps,
# success_tolerance 0.01, success_steps=10, nan_policy=reset, SAPG block 5, eval seeds 1 and 2), so
# the rows sit in the same table as the stage-1 chains and the cloth-trained reference. The 1.5 cm
# cutoff is a separate RUN, not a re-score (keypoint_tolerance gates termination).
#
# Results: docs/results/rc_eval_[tol15_]fp_<tag>_s<n>__on_vbd[__rand]_c0_s<seed>.json

#SBATCH --job-name=fp_cloth
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=bos14-node-[104,105]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

I="${SLURM_ARRAY_TASK_ID:-0}"
P=$((I / 4)); C=$(((I % 4) / 2)); S=$((I % 2))
TAGS=(box3 box3 box3 pool pool pool); JOBS=(156746 156746 156746 156744 156744 156744)
SEED=$(( (P % 3) + 1 ))
NAME="0_fp_${TAGS[$P]}_s${SEED}_${JOBS[$P]}"
CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs')) or '')" "$NAME")
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[eval] no checkpoint for $NAME" >&2; exit 1; }

PREFIX=docs/results/rc_eval_; TOL_ARGS=()
[ "$C" -eq 1 ] && { PREFIX=docs/results/rc_eval_tol15_; TOL_ARGS=(env.cloth.keypoint_tolerance=0.015); }
RANDOM_ARGS=(); SUFFIX=""
[ "$S" -eq 1 ] && { RANDOM_ARGS=(--randomize); SUFFIX="__rand"; }
LABEL="fp_${TAGS[$P]}_s${SEED}__on_vbd${SUFFIX}"

export PYTHONUNBUFFERED=1
echo "[eval] label=$LABEL ckpt=$CKPT prefix=$PREFIX host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in 1 2; do
    OUT="${PREFIX}${LABEL}_c0_s${ES}.json"
    echo "[eval] === seed $ES -> $OUT ==="
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" \
        --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef 0 \
        --num_envs 32 \
        --max_steps 900 \
        --success_tolerance 0.01 \
        --num_assets_per_type 1 \
        --single_variant \
        "${RANDOM_ARGS[@]}" \
        --seed "$ES" \
        --out "$OUT" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        "${TOL_ARGS[@]}" \
        || echo "[eval] $LABEL seed $ES FAILED"
done
echo "[eval] done $LABEL"
