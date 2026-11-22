#!/bin/bash
# Fold evaluation of every fold policy -- keypoint (rl_games) or point-cloud student (distilled or RL),
# recognised from the checkpoint -- at the 1.5 cm cutoff over EVAL_SEEDS. One array index = one
# (checkpoint, cell).
#
#   one checkpoint:     CKPT=<pth> TAG=<tag> [RECIPE=<recipe>] experiments/eval.sh        (index 0 .. nC-1)
#   a training array:   RUN_TAG=<tag> TRAIN_JOB=<job> SEEDS=3 experiments/eval.sh         (index 0 .. 3*nC-1)
#                       run names 0_<KIND>_<tag>_s<seed>_<job> (KIND default pxpool), checkpoint by CKPT_SEL
#   a list:             LIST=<file> experiments/eval.sh                                   (index 0 .. lines*nC-1)
#                       one line per checkpoint: "<tag> <seed> <recipe> <pth>" or
#                       "<tag> <seed> <recipe> <run name> <select>" (select resolved when the job runs, so a
#                       list can wait for a training in progress)
#   Index = item * nC + cell, nC = the number of CELLS (3 by default).
#
# CELLS (default "rigidbody_flat:rand vbd:pinned vbd:rand"), <target>:<start>:
#   rigidbody_flat     PhysX rigid sheet in the policy's OWN training world (its recipe: physics, DR,
#                      start height, accept_lookalike), the 7 flat7 shapes one per env, 448 envs, flat
#                      starts, the half fold (goal_task half_fold: the stage-1 task with the goal tables)
#   rigidbody_folded   (opt-in) the same, every episode from a random reachable folded state (each hinge
#                      flat or fully folded, flat included) with ONE more fold as the goal: goal_task
#                      any_fold, the goal point-cloud students' hard mode
#   vbd                the VBD cloth on Newton, 32 envs, the VBD task's own physics for every policy (the
#                      ns6 protocol: num_substeps 6 / cloth substeps 2); a student brings its cloud
#   start: rand = the env's randomisation on; pinned = DR off (one start state). Rigid body is rand only.
#   A goal point-cloud student has no VBD cell: eval.py exits 3 and the cell is skipped.
# RECIPE (default auto): the policy's training recipe, for the rigid cells. auto = run.json (train.py,
#   since 2026-11-22) / args.json (distill_pc) next to the checkpoint, with the training's env overrides.
#   Older checkpoints: name it (a list line's third column, or RECIPE=).
# CKPT_SEL (default best): best | last | ep:<N> (isaacsimenvs/scripts/evaluation/checkpoints.py).
# START_HEIGHT: the sheet spawn height on VBD (default 0.20; rigid: the recipe's unless set).
# EVAL_SEEDS (default "1 2"); EXTRA_ARGS: more hydra overrides for every run. Protocol: eval.py, 900
# steps, a fold counts once HELD 10 steps within 1.5 cm, SAPG block 0 with coefficient 0.
# Output: $OUTDIR/<tag>_s<seed>__on_<target>__<start>__1p5cm_s<es>.json (the cutoff stays in the name:
# docs/reports' run_record_sync.py reads it; default OUTDIR docs/results/px_matrix); existing files
# are skipped. Summaries:
#   python3 isaacsimenvs/scripts/evaluation/matrix_summary.py <tag> [<tag> ...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
read -r -a CELL_LIST <<<"${CELLS:-rigidbody_flat:rand vbd:pinned vbd:rand}"
NC=${#CELL_LIST[@]}
ITEM=$(( INDEX / NC )); CELL=${CELL_LIST[$(( INDEX % NC ))]}
T=${CELL%%:*}; START=${CELL#*:}

checkpoint() {  # <run name> <select>
    .venv_isaacsim/bin/python -m isaacsimenvs.scripts.evaluation.checkpoints "$1" "$2" 2>/dev/null
}
if [ -n "${LIST:-}" ]; then
    read -r TAG SEED RECIPE CKPT SEL < <(sed -n "$(( ITEM + 1 ))p" "$LIST")
    [ -n "${SEL:-}" ] && CKPT="$(checkpoint "$CKPT" "$SEL")"
elif [ -n "${TRAIN_JOB:-}" ]; then
    TAG="${RUN_TAG:?set RUN_TAG with TRAIN_JOB}"; SEED=$(( ITEM + 1 ))
    CKPT="$(checkpoint "0_${KIND:-pxpool}_${RUN_TAG}_s${SEED}_${TRAIN_JOB}" "${CKPT_SEL:-best}")"
else
    TAG="${TAG:?set TAG (with CKPT), RUN_TAG + TRAIN_JOB, or LIST}"; SEED="${SEED:-1}"
fi
RECIPE="${RECIPE:-auto}"
[ -n "${CKPT:-}" ] && [ -f "$CKPT" ] || { echo "[eval] item $ITEM: no checkpoint '${CKPT:-}'" >&2; exit 1; }

RIGID=(Isaacsimenvs-RigidCloth-Direct-v0 448
       "env.fold_pool.shapes=[box2-mid,box2diag4-mid,box2x8-mid,cross4-mid,corners8-mid,box4-mid,star8-mid]"
       env.fold_pool.one_shape_per_env=true env.fold_pool.init_fold_prob=0.0)
case "$T" in
    rigidbody_flat)   TASK=${RIGID[0]}; ENVS=${RIGID[1]}; TARGS=("${RIGID[@]:2}" env.fold_pool.goal_task.mode=half_fold) ;;
    rigidbody_folded) TASK=${RIGID[0]}; ENVS=${RIGID[1]}; TARGS=("${RIGID[@]:2}" env.fold_pool.goal_task.mode=any_fold) ;;
    vbd)              TASK=Isaacsimenvs-Cloth-Direct-v0; ENVS=32
                      TARGS=(env.robot_self_collision=false env.newton.njmax=32768) ;;
    *) echo "[eval] unknown target $T (rigidbody_flat, rigidbody_folded, vbd)" >&2; exit 2 ;;
esac
case "$START" in
    rand)   SARGS=(--randomize) ;;
    pinned) [ "$T" = vbd ] || { echo "[eval] $T is rand only" >&2; exit 2; }; SARGS=() ;;
    *) echo "[eval] unknown start $START (pinned, rand)" >&2; exit 2 ;;
esac
if [ "$T" = vbd ]; then
    TARGS+=("env.cloth.start_height=${START_HEIGHT:-0.20}")
else
    SARGS+=(--experiment "$RECIPE")
    [ -n "${START_HEIGHT:-}" ] && TARGS+=("env.cloth.start_height=$START_HEIGHT")
fi
# _common.sh picked the interpreter from the (absent) --task; pick it for this target.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" --task "$TASK"
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
OUTDIR="${OUTDIR:-docs/results/px_matrix}"
mkdir -p "$OUTDIR"
echo "[eval] tag=$TAG seed=$SEED target=$T start=$START recipe=$([ "$T" = vbd ] && echo - || echo "$RECIPE") ckpt=$CKPT"
provenance eval

RC=0
for ES in ${EVAL_SEEDS:-1 2}; do
    OUT="$OUTDIR/${TAG}_s${SEED}__on_${T}__${START}__1p5cm_s${ES}.json"
    [ -s "$OUT" ] && { echo "[eval] exists: $OUT"; continue; }
    run_watched "eval_$(basename "$OUT" .json)" "${PY[@]}" -m isaacsimenvs.scripts.evaluation.eval "${HEADLESS[@]}" \
        --task "$TASK" --checkpoint "$CKPT" --policy_config "${POLICY_CONFIG:-pretrained_policy/config.yaml}" \
        --sapg_expl_coef 0 --num_envs "$ENVS" --max_steps 900 --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant "${SARGS[@]}" --seed "$ES" --out "$OUT" \
        "${TARGS[@]}" env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset \
        env.termination.success_steps=10 "${EXTRA[@]}"
    rc=$?
    if [ "$rc" -eq 3 ]; then echo "[eval] $T not applicable to this policy: skipped"; break; fi
    # eval.py's exit code can be 0 after an exception inside the Hydra-wrapped run: the file decides.
    [ "$rc" -eq 0 ] && [ ! -s "$OUT" ] && rc=1
    [ "$rc" -ne 0 ] && { echo "[eval] $OUT FAILED"; RC=1; }
done
echo "[eval] done exit=$RC"
exit $RC
