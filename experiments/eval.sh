#!/bin/bash
# Fold evaluation of fold-pool / VBD policies: one or many checkpoints x targets x start states, each at
# two cutoffs (4 cm, 1.5 cm) and over EVAL_SEEDS. One array index = one (checkpoint, target, start).
#
#   one checkpoint:     CKPT=<pth> TAG=<tag> experiments/eval.sh                        (index 0 .. nT*nS-1)
#   a training array:   RUN_TAG=<tag> TRAIN_JOB=<job> SEEDS=3 experiments/eval.sh       (index 0 .. 3*nT*nS-1)
#                       run names 0_<KIND>_<tag>_s<seed>_<job> (KIND default pxpool), checkpoint by CKPT_SEL
#   a list:             LIST=<file> experiments/eval.sh                                 (index 0 .. lines*nT*nS-1)
#                       one line per checkpoint: "<tag> <seed> <pth>" or "<tag> <seed> <run name> <select>"
#                       (select resolved when the job runs, so a list can wait for a training in progress)
#   Index = (item * nT + target) * nS + start, nT / nS = the number of TARGETS / STARTS.
#
# TARGETS (default "pool7 vbd"):
#   pool     PhysX fold pool, its 5 default shapes, 320 envs
#   pool7    PhysX fold pool, the 7 flat7 shapes one per env, 448 envs
#   pool7of  pool7 with every episode from a non-target fold (init_fold_starts=other, 15-180 deg)
#   vbd      the VBD cloth on Newton, 32 envs (12 robot / 1 cloth substep)
# STARTS (default "pinned rand"): pinned = DR off (one start state); rand = the env's randomisation on.
# CKPT_SEL (default best): best | last | ep:<N> (isaacsimenvs/scripts/evaluation/checkpoints.py).
# START_HEIGHT (default 0.20): the sheet spawn height, as in the policy's training recipe. EVAL_SEEDS (default "1 2"); EXTRA_ARGS: more
# hydra overrides for every run. Protocol: eval.py, 900 steps, a fold counts once HELD 10 steps, SAPG
# block 0 with coefficient 0.
# Output: $OUTDIR/<tag>_s<seed>__on_<physx_pool|physx_pool7|physx_pool7of|vbd>__<start>__<4cm|1p5cm>_s<es>.json
# (default OUTDIR docs/results/px_matrix); existing files are skipped. Summaries:
#   python3 isaacsimenvs/scripts/evaluation/matrix_summary.py <tag> [<tag> ...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
read -r -a TARGET_LIST <<<"${TARGETS:-pool7 vbd}"
read -r -a START_LIST <<<"${STARTS:-pinned rand}"
NT=${#TARGET_LIST[@]}; NS=${#START_LIST[@]}
ITEM=$(( INDEX / (NT * NS) )); T=${TARGET_LIST[$(( (INDEX / NS) % NT ))]}; START=${START_LIST[$(( INDEX % NS ))]}

checkpoint() {  # <run name> <select>
    .venv_isaacsim/bin/python -m isaacsimenvs.scripts.evaluation.checkpoints "$1" "$2" 2>/dev/null
}
if [ -n "${LIST:-}" ]; then
    read -r TAG SEED CKPT SEL < <(sed -n "$(( ITEM + 1 ))p" "$LIST")
    [ -n "${SEL:-}" ] && CKPT="$(checkpoint "$CKPT" "$SEL")"
elif [ -n "${TRAIN_JOB:-}" ]; then
    TAG="${RUN_TAG:?set RUN_TAG with TRAIN_JOB}"; SEED=$(( ITEM + 1 ))
    CKPT="$(checkpoint "0_${KIND:-pxpool}_${RUN_TAG}_s${SEED}_${TRAIN_JOB}" "${CKPT_SEL:-best}")"
else
    TAG="${TAG:?set TAG (with CKPT), RUN_TAG + TRAIN_JOB, or LIST}"; SEED="${SEED:-1}"
fi
[ -n "${CKPT:-}" ] && [ -f "$CKPT" ] || { echo "[eval] item $ITEM: no checkpoint '${CKPT:-}'" >&2; exit 1; }

POOL7=("env.fold_pool.shapes=[box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid]"
       env.fold_pool.one_shape_per_env=true)
case "$T" in
    pool)    TASK=Isaacsimenvs-RigidCloth-Direct-v0; NAME=physx_pool;    ENVS=320; TARGS=() ;;
    pool7)   TASK=Isaacsimenvs-RigidCloth-Direct-v0; NAME=physx_pool7;   ENVS=448; TARGS=("${POOL7[@]}") ;;
    pool7of) TASK=Isaacsimenvs-RigidCloth-Direct-v0; NAME=physx_pool7of; ENVS=448
             TARGS=("${POOL7[@]}" env.fold_pool.init_fold_starts=other env.fold_pool.init_fold_prob=1.0
                    "env.fold_pool.init_fold_deg=[15.0,180.0]") ;;
    vbd)     TASK=Isaacsimenvs-Cloth-Direct-v0;      NAME=vbd;           ENVS=32
             TARGS=(env.robot_self_collision=false env.newton.njmax=32768) ;;
    *) echo "[eval] unknown target $T (pool, pool7, pool7of, vbd)" >&2; exit 2 ;;
esac
# _common.sh picked the interpreter from the (absent) --task; pick it for this target.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" --task "$TASK"
# Pinned on the rigid pool: the plates' physics DR off too (eval.py turns off the rest).
C=env.domain_randomization.cloth
PINNED_PLATES=("$C.slat_friction_scale=[1.0,1.0]" "$C.robot_friction_scale=[1.0,1.0]" "$C.table_friction_scale=[1.0,1.0]"
               "$C.hinge_damping_scale=[1.0,1.0]" "$C.hinge_stiffness=[0.0,0.0]" "$C.mass_scale=[1.0,1.0]"
               "$C.contact_margin_offset=[0.0,0.0]")
case "$START" in
    rand)   SARGS=(--randomize) ;;
    pinned) SARGS=(); [ "$T" != vbd ] && SARGS=("${PINNED_PLATES[@]}") ;;
    *) echo "[eval] unknown start $START (pinned, rand)" >&2; exit 2 ;;
esac
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
OUTDIR="${OUTDIR:-docs/results/px_matrix}"
mkdir -p "$OUTDIR"
echo "[eval] tag=$TAG seed=$SEED target=$NAME start=$START ckpt=$CKPT"
provenance eval

RC=0
for CUT in 0.04 0.015; do
    [ "$CUT" = 0.04 ] && CT=4cm || CT=1p5cm
    for ES in ${EVAL_SEEDS:-1 2}; do
        OUT="$OUTDIR/${TAG}_s${SEED}__on_${NAME}__${START}__${CT}_s${ES}.json"
        [ -s "$OUT" ] && { echo "[eval] exists: $OUT"; continue; }
        run_watched "eval_$(basename "$OUT" .json)" "${PY[@]}" -m isaacsimenvs.scripts.evaluation.eval "${HEADLESS[@]}" \
            --task "$TASK" --checkpoint "$CKPT" --policy_config "${POLICY_CONFIG:-pretrained_policy/config.yaml}" \
            --sapg_expl_coef 0 --num_envs "$ENVS" --max_steps 900 --success_tolerance 0.01 \
            --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$OUT" \
            "env.cloth.start_height=${START_HEIGHT:-0.20}" \
            "${TARGS[@]}" "env.cloth.keypoint_tolerance=$CUT" env.cloth.nan_policy=reset \
            env.termination.success_steps=10 "${EXTRA[@]}" \
            || { echo "[eval] $OUT FAILED"; RC=1; }
    done
done
echo "[eval] done exit=$RC"
exit $RC
