#!/bin/bash
# Zero-shot: the released SimToolReal policy (pretrained_policy/model.pth), never trained on cloth or
# on any chain, evaluated directly on the VBD cloth at both fold cutoffs and every SAPG block.
#
#   sbatch --array=0-23 scripts/cluster/bos14_cloth_eval_pretrained.sh
#
# Index = block * 4 + cutoff * 2 + start, with
#   block  0..5 = --sapg_expl_coef 50 40 30 20 10 0 (50 = block 0, the released checkpoint's default)
#   cutoff 0 = 4 cm (the task's keypoint_tolerance), 1 = 1.5 cm
#   start  0 = pinned (`disable_randomization`), 1 = `--randomize`
#
# Protocol identical to bos14_rigid_cloth_eval_rc2.sh's cloth target (32 envs x 900 steps,
# success_tolerance 0.01, success_steps=10, nan_policy=reset, eval seeds 1 and 2), so these rows sit
# in the same table as the chains and the cloth-trained reference. As there, the 1.5 cm cutoff is a
# separate RUN, not a re-score: keypoint_tolerance gates is_success() and therefore termination.
#
# Results: docs/results/rc_eval_[tol15_]pretrained__on_vbd[__rand]_c<coef>_s<seed>.json

#SBATCH --job-name=pre_eval
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
COEFS=(50 40 30 20 10 0)
COEF="${COEFS[$((I / 4))]}"; C=$(((I % 4) / 2)); S=$((I % 2))
CKPT=pretrained_policy/model.pth
[ -f "$CKPT" ] || { echo "[eval] checkpoint not found: $CKPT" >&2; exit 1; }

PREFIX=docs/results/rc_eval_; TOL_ARGS=()
[ "$C" -eq 1 ] && { PREFIX=docs/results/rc_eval_tol15_; TOL_ARGS=(env.cloth.keypoint_tolerance=0.015); }
RANDOM_ARGS=(); SUFFIX=""
[ "$S" -eq 1 ] && { RANDOM_ARGS=(--randomize); SUFFIX="__rand"; }
LABEL="pretrained__on_vbd${SUFFIX}"

export PYTHONUNBUFFERED=1
echo "[eval] label=$LABEL coef=$COEF prefix=$PREFIX host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in 1 2; do
    OUT="${PREFIX}${LABEL}_c${COEF}_s${ES}.json"
    echo "[eval] === seed $ES -> $OUT ==="
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
        "${RANDOM_ARGS[@]}" \
        --seed "$ES" \
        --out "$OUT" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        "${TOL_ARGS[@]}" \
        || echo "[eval] $LABEL coef $COEF seed $ES FAILED"
done
echo "[eval] done $LABEL coef $COEF"
