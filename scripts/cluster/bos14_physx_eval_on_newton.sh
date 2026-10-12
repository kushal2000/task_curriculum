#!/bin/bash
# Evaluate PhysX-trained chain policies on NEWTON: their own chain there, and the VBD cloth -- the
# number the port is for. Same protocol as bos14_rigid_cloth_eval_rc2.sh, so rows line up with the
# Newton-trained ones in docs/reports/2_rigid_chain/rigid_cloth_rl_comparison.md.
#
#   TRAIN_JOB=157222 RUN_TAG=rc2fix sbatch --array=0-11 scripts/cluster/bos14_physx_eval_on_newton.sh
#   KIND=pool TRAIN_JOB=... RUN_TAG=box3 sbatch --array=0-11 scripts/cluster/bos14_physx_eval_on_newton.sh
#
# Index = (seed-1)*4 + target*2 + start; start 0 = pinned, 1 = randomised; two eval seeds each.
#   KIND=chain (default): target 0 = the Newton box3-mid chain, 1 = the VBD cloth
#   KIND=pool: the VBD cloth at target 0 = the 4 cm cutoff, 1 = 1.5 cm (bos14_fold_pool_eval_cloth.sh)
# Labels [tol15_]px[pool]_<RUN_TAG>_s<seed>__on_<target>.

#SBATCH --job-name=px_eval
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs docs/results
: "${TRAIN_JOB:?set TRAIN_JOB to the PhysX training array}"
: "${RUN_TAG:?set RUN_TAG to the training run tag}"
I="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( I / 4 + 1 )); T=$(( (I % 4) / 2 )); S=$(( I % 2 ))
KIND="${KIND:-chain}"
NAME="0_px${KIND}_${RUN_TAG}_s${SEED}_${TRAIN_JOB}"
CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs')) or '')" "$NAME")
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[px_eval] no checkpoint for $NAME" >&2; exit 1; }
PREFIX="${OUT_PREFIX:-docs/results/rc_eval_}"; POLICY="px_${RUN_TAG}"
if [ "$KIND" = pool ]; then
    TASK=Isaacsimenvs-Cloth-Direct-v0; TARGET=vbd; TARGET_ARGS=(); POLICY="pxpool_${RUN_TAG}"
    [ "$T" -eq 1 ] && { PREFIX="${OUT_PREFIX:-docs/results/rc_eval_}tol15_"; TARGET_ARGS=(env.cloth.keypoint_tolerance=0.015); }
elif [ "$T" -eq 0 ]; then
    TASK=Isaacsimenvs-RigidCloth-Direct-v0; TARGET=box3-mid; TARGET_ARGS=(env.rigid_cloth.variant=box3-mid)
else
    TASK=Isaacsimenvs-Cloth-Direct-v0; TARGET=vbd; TARGET_ARGS=()
fi
START_ARGS=(); SUFFIX=""
[ "$S" -eq 1 ] && { START_ARGS=(--randomize); SUFFIX="__rand"; }
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
LABEL="${POLICY}_s${SEED}__on_${TARGET}${SUFFIX}"
export PYTHONUNBUFFERED=1
echo "[px_eval] label=$LABEL ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in 1 2; do
    OUT="${PREFIX}${LABEL}_c0_s${ES}.json"
    scripts/newton_py -m isaacsimenvs.eval.episodes \
        --task "$TASK" --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
        --sapg_expl_coef 0 --num_envs 32 --max_steps 900 --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant "${START_ARGS[@]}" --seed "$ES" --out "$OUT" \
        "${TARGET_ARGS[@]}" env.cloth.nan_policy=reset env.termination.success_steps=10 \
        "${EXTRA[@]}" || echo "[px_eval] $LABEL seed $ES FAILED"
done
echo "[px_eval] done $LABEL"
