#!/bin/bash
# Evaluate the round-3 chains (bos14_rigid_cloth_retrain_rc3.sh) on their own chain and on the cloth,
# at BOTH fold cutoffs.
#
#   # 4 cm, the cutoff every earlier round was scored at
#   sbatch --dependency=afterany:<train_job> --array=0-77 \
#       --export=ALL,TRAIN_JOB=<train_job> scripts/cluster/bos14_rigid_cloth_eval_rc3.sh
#
#   # 1.5 cm, the precision cutoff
#   sbatch --dependency=afterany:<train_job> --array=0-77 \
#       --export=ALL,TRAIN_JOB=<train_job>,OUT_PREFIX=docs/results/rc_eval_tol15_,EXTRA_ARGS=env.cloth.keypoint_tolerance=0.015 \
#       scripts/cluster/bos14_rigid_cloth_eval_rc3.sh
#
# THE 1.5 cm PASS IS NOT A RE-SCORE OF THE 4 cm ROLLOUTS. `keypoint_tolerance` feeds
# `ClothEnv.is_success()`, which gates `success_steps` and therefore termination -- so a tighter
# cutoff ends episodes differently and the rollouts genuinely diverge. Both passes have to run, and
# `OUT_PREFIX` is what keeps them from overwriting each other.
#
# Index = policy * 4 + target * 2 + start, with
#   policy = group * 9 + arm * 3 + (seed - 1)       exactly the training array's index
#   target 0 = the policy's own chain (nominal physics: DR off, the cloth-matched fixes on)
#   target 1 = the VBD cloth
#   start  0 = pinned (`disable_randomization`: centred, heading as the cloth decodes the identity)
#   start  1 = `--randomize`: the training reset distribution, uniform heading included
# Indices 72-77 evaluate the cloth-trained reference on the three round-3 chains instead, which is
# the upper bound those arms are read against. The reference ON THE CLOTH is not repeated here --
# it is already measured by bos14_rigid_cloth_eval_rc2.sh indices 48-49 under both cutoffs.
#
# Protocol identical to rounds 1 and 2 (32 envs x 900 steps, success_tolerance 0.01,
# success_steps=10, SAPG block 5 via --sapg_expl_coef 0, eval seeds 1 and 2), so these numbers sit
# in the same table.
#
# RUNS ON A 5090, unlike training. Training needs the 96 GB RTX PRO 6000 only because 524288 pairs
# x 1536 envs is a 66 GB allocation; evaluation is 32 envs, so the same budget is well under 1 GB and
# fits a 32 GB card with room to spare. Keeping eval on gpu_5090 also stops 156 eval tasks from
# queueing behind the 18 training runs for the same scarce PRO 6000s.
#
# `per_env_triangle_pairs` IS overridden here, to 2097152 -- RigidCloth.yaml's 524288 is NOT enough
# for a 16-17 slat chain under a policy that actually folds it. Measured on the first attempt at this
# array (155730): "Triangle pair buffer overflowed 17542368 > 16777216", i.e. 548k pairs/env against
# a 524k budget, on `box-mid-odd` and `box-surface` own-chain tasks.
#
# Note how far that is above every earlier estimate, and why each was low:
#   ~119k/env  round 1 training   -- measured on runs that NEVER fold; the slats never stack.
#   ~285k/env  round 1 eval       -- measured with the CLOTH-trained reference, which folds these
#                                    chains only ~28% of the time.
#   ~161k/env  round 3 gate probe -- a driven 180 deg fold, but driven through the HINGES with the
#                                    robot not gripping; no fingertip-slat contact pile-up.
#   ~548k/env  round 3 eval       -- a chain policy that folds nearly every episode, with the hand
#                                    still on the slats. This is the true worst case.
#
# Which is the doc's own warning turned on itself: "the overflow appears exactly when the measurement
# succeeds", so every estimate taken from a less successful policy under-reads. 2097152 is 3.8x the
# measured peak and costs ~3.2 GB at 32 envs -- nothing on a 32 GB card. Over-size this deliberately;
# under-sizing corrupts the successful half of the matrix and leaves the failing half looking clean.

#SBATCH --job-name=rc3_eval
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Node-health exclusion from 2026-09-30; drop it once the two nodes are confirmed good. 092 is a
# gpu_a100 node and cannot be picked for this gpu_5090 job, so naming it here excluded nothing.
#SBATCH --exclude=bos14-node-[104,105]
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results

: "${TRAIN_JOB:?set TRAIN_JOB to the job id of the round-3 training array}"
I="${SLURM_ARRAY_TASK_ID:-0}"
P=$((I / 4)); T=$(((I % 4) / 2)); S=$((I % 2))
ARMS=(cyl-mid-odd box-mid-odd box-surface)
GROUPS_=(fix fixdr)
GROUP="${GROUPS_[$(( (P / 9) % 2 ))]}"   # % 2: indices 72+ (reference) are overwritten below
ARM="${ARMS[$(((P % 9) / 3))]}"
SEED=$(( (P % 3) + 1 ))
NAME="0_rc3_${ARM}_${GROUP}_s${SEED}_${TRAIN_JOB}"
POLICY="rc3_${ARM}_${GROUP}_s${SEED}"

if [ "$I" -ge 72 ]; then
    # 72-77: the cloth-trained reference on each round-3 chain (pinned, randomised), so every
    # cutoff has its upper bound measured under its own settings.
    R=$((I - 72)); S=$((R % 2)); ARM="${ARMS[$((R / 2))]}"
    CKPT=outputs/2026-09-23/17-31-13/0_pretrained_s1_resume_150911/nn/0_pretrained_s1_resume_150911.pth
    # `reference_rc3`, NOT `reference`: round 1 already evaluated this same checkpoint on these same
    # three chains BEFORE the fixes, and those files are the "OLD chain" row. A bare `reference`
    # label writes `rc_eval_reference__on_cyl-mid-odd_c0_s1.json` and destroys it. Round 2 tagged
    # its own re-measurement `reference_rc2` for exactly this reason.
    POLICY=reference_rc3; T=0
else
    CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, \"scripts/analysis\"); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path(\"outputs\")) or \"\")" "$NAME")
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[eval] no checkpoint for $NAME" >&2; exit 1; }
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
echo "[eval] label=$LABEL ckpt=$CKPT host=$(hostname -s) extra=${EXTRA_ARGS:-none}"
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
        "env.newton.per_env_triangle_pairs=${TRI_PAIRS:-2097152}" \
        "${RANDOM_ARGS[@]}" \
        --seed "$ES" \
        --out "$OUT" \
        "${VARIANT_ARGS[@]}" \
        env.cloth.nan_policy=reset \
        env.termination.success_steps=10 \
        "${EXTRA[@]}" \
        || echo "[eval] $LABEL seed $ES FAILED"
done
# `:-$SLURM_JOB_ID` because a non-array `sbatch` of this script leaves SLURM_ARRAY_JOB_ID unset,
# and `set -u` then killed the task here -- after the eval had already finished. Grepping a log that
# is not there also used to print 0, i.e. no overflow on a run nobody checked, so say so instead.
LOG="slurm_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${I}.out"
if [ -f "$LOG" ]; then
    echo "[eval] overflow_warnings=$(grep -ciE 'buffer overflow|overflowed' "$LOG" || true)"
else
    echo "[eval] WARNING: log $LOG not found -- overflow_warnings NOT CHECKED" >&2
fi
echo "[eval] done $LABEL"
