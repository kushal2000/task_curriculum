#!/bin/bash
# Evaluate the retrained chains (bos14_rigid_cloth_retrain.sh) on their own chain and on the cloth.
#
#   sbatch --dependency=afterany:<train_job> --array=0-47 \
#       --export=ALL,TRAIN_JOB=<train_job> scripts/cluster/bos14_rigid_cloth_eval_rc2.sh
#
# Index = policy * 4 + target * 2 + start, with
#   policy = group * 6 + arm * 3 + (seed - 1)       exactly the training array's index
#   target 0 = the policy's own chain (nominal physics: DR off, the cloth-matched fixes on)
#   target 1 = the VBD cloth
#   start  0 = pinned (`disable_randomization`: centred, heading as the cloth decodes the identity)
#   start  1 = `--randomize`: the training reset distribution, uniform heading included
# Indices 48-53 evaluate the cloth-trained reference instead (see below).
#
# Protocol identical to bos14_rigid_cloth_eval.sh (32 envs x 900 steps, success_tolerance 0.01,
# success_steps=10, SAPG block 5 via --sapg_expl_coef 0, eval seeds 1 and 2), so these numbers sit
# in the same table as the first round's.

#SBATCH --job-name=rc2_eval
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

: "${TRAIN_JOB:?set TRAIN_JOB to the job id of the retraining array}"
I="${SLURM_ARRAY_TASK_ID:-0}"
P=$((I / 4)); T=$(((I % 4) / 2)); S=$((I % 2))
ARMS=(box3-mid box2-surface)
GROUPS_=(fix fixdr)
GROUP="${GROUPS_[$(( (P / 6) % 2 ))]}"   # % 2: indices 48+ (reference) are overwritten below
ARM="${ARMS[$(((P % 6) / 3))]}"
SEED=$(( (P % 3) + 1 ))
NAME="0_rc2_${ARM}_${GROUP}_s${SEED}_${TRAIN_JOB}"
POLICY="rc2_${ARM}_${GROUP}_s${SEED}"

if [ "$I" -ge 48 ]; then
    # 48-53: the cloth-trained reference on the cloth, box3-mid, box2-surface (pinned, random), so
    # a re-scored protocol (EXTRA_ARGS) has its upper bound measured under the same settings.
    R=$((I - 48)); S=$((R % 2)); TGT=(vbd box3-mid box2-surface); TARGET_I="${TGT[$((R / 2))]}"
    CKPT=outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth
    # `reference_rc2`, NOT `reference`: round 1 already evaluated this same checkpoint on box3-mid
    # and box2-surface BEFORE the fixes, and those files are the "OLD chain" row. A bare `reference`
    # label writes `rc_eval_reference__on_box3-mid_c0_s1.json` and destroys it -- the fixed chain is
    # a different manipuland and needs its own tag.
    POLICY=reference_rc2; T=1; [ "$TARGET_I" != vbd ] && { T=0; ARM="$TARGET_I"; }
else
    CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, \"scripts/analysis\"); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path(\"outputs\")) or \"\")" "$NAME")
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[eval] no checkpoint for $NAME" >&2; exit 1; }
# Extra hydra overrides, e.g. EXTRA_ARGS="env.cloth.keypoint_tolerance=0.015" with a matching
# OUT_PREFIX so the re-scored files never overwrite the 4 cm ones.
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"

if [ "$T" -eq 0 ]; then
    TASK=Isaacsimenvs-RigidCloth-Direct-v0; TARGET="$ARM"; VARIANT_ARGS=("env.rigid_cloth.variant=${ARM}")
else
    TASK=Isaacsimenvs-Cloth-Direct-v0; TARGET=vbd; VARIANT_ARGS=()
fi
RANDOM_ARGS=(); SUFFIX=""
[ "$S" -eq 1 ] && { RANDOM_ARGS=(--randomize); SUFFIX="__rand"; }
LABEL="${POLICY}__on_${TARGET}${SUFFIX}"

export PYTHONUNBUFFERED=1
echo "[eval] label=$LABEL ckpt=$CKPT host=$(hostname -s)"
for ES in 1 2; do
    OUT="${OUT_PREFIX:-docs/results/rc_eval_}${LABEL}_c0_s${ES}.json"
    echo "[eval] === seed $ES -> $OUT ==="
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task "$TASK" \
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
        "${VARIANT_ARGS[@]}" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        "${EXTRA[@]}" \
        || echo "[eval] $LABEL seed $ES FAILED"
done
echo "[eval] done $LABEL"
