#!/bin/bash
# Evaluate one PhysX training array on every target: the rigid pool and the box3-mid chain on BOTH
# simulators, and the VBD cloth (Newton) -- pinned and randomised starts, 4 cm and 1.5 cm cutoffs.
#
#   KIND=pool  TRAIN_JOB=<id> RUN_TAG=<tag> sbatch --array=0-41 scripts/cluster/bos14_px_eval_matrix.sh
#   KIND=chain TRAIN_JOB=<id> RUN_TAG=<tag> sbatch --array=0-41 scripts/cluster/bos14_px_eval_matrix.sh
#   python3 scripts/analysis/px_eval_matrix_summary.py <tag> [<tag> ...]
#
# Checkpoint: CKPT_SEL=best (default) takes nn/<name>.pth, the BEST-reward checkpoint, which is not
# necessarily the trained policy (the 10-07 ft7lr finetunes wrote it once at epoch ~10). CKPT_SEL=last
# takes the highest-epoch nn/last_<name>_ep_*.pth -- use it after a finetune. CKPT_SEL=ep:<N> takes
# epoch N's checkpoint (last_ / periodic_ / snap_, in that order). CKPT=<path> overrides
# both for every index, so pair it with one seed's indices (e.g. --array=0-9). OUT_TAG (default
# RUN_TAG) names the output files; set it when re-evaluating a run under another checkpoint, or the
# existing files are skipped as done:
#   KIND=pool TRAIN_JOB=<id> RUN_TAG=<tag> CKPT_SEL=last OUT_TAG=<tag>_last sbatch --array=0-29 ...
#
# Index = (seed-1)*14 + target*2 + start; start 0 = pinned, 1 = randomised. Targets:
#   0 PhysX pool    1 PhysX box3-mid    2 Newton pool    3 Newton box3-mid    4 VBD cloth (Newton)
#   5 PhysX 7-shape pool, flat starts   6 PhysX 7-shape pool, non-target folded starts
# (Before 2026-10-12 there were only targets 0-4 and the stride was 10: --array=0-29. Targets 0-4 are
# unchanged, and so are their output names.) 5 and 6 are the pool the ft7 fine-tunes train on: the 5
# shapes plus box4-mid and star8-mid, one shape per env round-robin (env i -> shape i % 7, 64 envs
# each). 6 starts every episode from a non-target fold (init_fold_starts=other, 15-180 deg) except
# box2-mid, which has none and starts flat.
# Each task runs both cutoffs x two eval seeds. Protocol as every earlier eval (episodes.py, 900
# steps, a fold counts once HELD 10 steps, SAPG block 0 coef 0). Pool targets use 320 envs (5 shapes
# x 10-20 cm), the chain and cloth 32. The physics is each target's nominal one: a policy trained
# with another hinge damping (run 1: 0.02) is scored at the yaml's 0.05 like everyone else.
# Output: docs/results/px_matrix/<OUT_TAG>_s<seed>__on_<target>__<pinned|rand>__<4cm|1p5cm>_s<es>.json

#SBATCH --job-name=px_mx
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
: "${TRAIN_JOB:?set TRAIN_JOB to the PhysX training array}"
: "${RUN_TAG:?set RUN_TAG to the training run tag}"
KIND="${KIND:-pool}"
CKPT_SEL="${CKPT_SEL:-best}"
OUT_TAG="${OUT_TAG:-$RUN_TAG}"
OUTDIR="${OUTDIR:-docs/results/px_matrix}"
mkdir -p slurm_logs "$OUTDIR"
I="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( I / 14 + 1 )); T=$(( (I % 14) / 2 )); S=$(( I % 2 ))
NAME="0_px${KIND}_${RUN_TAG}_s${SEED}_${TRAIN_JOB}"
if [ -z "${CKPT:-}" ]; then
    CKPT=$(.venv_isaaclab3/bin/python -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs'), sys.argv[2]) or '')" "$NAME" "$CKPT_SEL")
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[px_mx] no checkpoint for $NAME" >&2; exit 1; }

PHYSX=(.venv_isaacsim/bin/python -m isaacsimenvs.eval.episodes --headless)
NEWTON=(scripts/newton_py -m isaacsimenvs.eval.episodes)
POOL7=("env.fold_pool.shapes=[box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid]"
       env.fold_pool.one_shape_per_env=true)
case "$T" in
    0) TARGET=physx_pool;  PY=("${PHYSX[@]}");  TASK=Isaacsimenvs-FoldPoolPhysx-Direct-v0;  ENVS=320; TARGS=() ;;
    1) TARGET=physx_box3;  PY=("${PHYSX[@]}");  TASK=Isaacsimenvs-RigidClothPhysx-Direct-v0; ENVS=32
       TARGS=(env.rigid_cloth.variant=box3-mid) ;;
    2) TARGET=newton_pool; PY=("${NEWTON[@]}"); TASK=Isaacsimenvs-FoldPool-Direct-v0;       ENVS=320
       TARGS=(env.newton.per_env_triangle_pairs=2097152) ;;
    3) TARGET=newton_box3; PY=("${NEWTON[@]}"); TASK=Isaacsimenvs-RigidCloth-Direct-v0;     ENVS=32
       TARGS=(env.rigid_cloth.variant=box3-mid env.newton.per_env_triangle_pairs=524288) ;;
    4) TARGET=vbd;         PY=("${NEWTON[@]}"); TASK=Isaacsimenvs-Cloth-Direct-v0;          ENVS=32; TARGS=() ;;
    5) TARGET=physx_pool7; PY=("${PHYSX[@]}");  TASK=Isaacsimenvs-FoldPoolPhysx-Direct-v0;  ENVS=448
       TARGS=("${POOL7[@]}") ;;
    6) TARGET=physx_pool7of; PY=("${PHYSX[@]}"); TASK=Isaacsimenvs-FoldPoolPhysx-Direct-v0; ENVS=448
       TARGS=("${POOL7[@]}" env.fold_pool.init_fold_starts=other env.fold_pool.init_fold_prob=1.0
              "env.fold_pool.init_fold_deg=[15.0,180.0]") ;;
esac
if [ "$S" -eq 1 ]; then
    START=rand; SARGS=(--randomize)
elif [ "$T" -eq 4 ]; then
    START=pinned; SARGS=()
else
    START=pinned; SARGS=(env.rigid_cloth.randomization.enabled=false)
fi
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
echo "[px_mx] $NAME on $TARGET/$START ckpt=$CKPT (sel=${CKPT_SEL}) out_tag=$OUT_TAG host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for CUT in 0.04 0.015; do
    [ "$CUT" = 0.04 ] && CT=4cm || CT=1p5cm
    for ES in 1 2; do
        OUT="$OUTDIR/${OUT_TAG}_s${SEED}__on_${TARGET}__${START}__${CT}_s${ES}.json"
        [ -s "$OUT" ] && { echo "[px_mx] exists: $OUT"; continue; }
        "${PY[@]}" --task "$TASK" --checkpoint "$CKPT" --policy_config pretrained_policy/config.yaml \
            --sapg_expl_coef 0 --num_envs "$ENVS" --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$OUT" \
            "${TARGS[@]}" "env.cloth.keypoint_tolerance=$CUT" env.cloth.nan_policy=reset \
            env.termination.success_steps=10 "${EXTRA[@]}" \
            || echo "[px_mx] $OUT FAILED"
    done
done
echo "[px_mx] done"
