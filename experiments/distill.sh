#!/bin/bash
# Point-cloud student distillation (isaacsimenvs/scripts/distillation/distill_pc.py) on the PhysX fold pool,
# and evaluations of a student / the teacher on PhysX or the VBD cloth (Newton).
#
#   TEACHER=<pth> NAME=B_g1 VIS=cams GOAL=g1 experiments/distill.sh                       (train)
#   TEACHER=<pth> MODE=eval SIM=vbd STUDENT=<pth> TAG=<tag> START=rand experiments/distill.sh
#   TEACHER=<pth> MODE=evalall STUDENT=<pth> TAG=<tag> experiments/distill.sh   (PhysX rand + VBD pinned/rand x 2 seeds)
#   TEACHER=<pth> MODE=evalx STUDENT=<pth> TAG=<tag> SIMS="physx vbd" SEEDS="1 2 3" EVAL_ARGS="--eval_noise ..." ...
#   TEACHER=<pth> MODE=smoke [PART=physx|scene|vbd] experiments/distill.sh
#
# TEACHER (required): the state policy's checkpoint; EXPERIMENT (default rigid_cloth_right_wuji): its
# training recipe, which builds the PhysX env. The VBD env is the Cloth task at the ns6 protocol.
# Train: NAME, VIS (cams|none), GOAL (g0|g3|g1|gl|first|none), NOISE=1, G3_NOISE=1, ENVS (4096),
#        ITERS (2000), SEED (default INDEX + 1), EXTRA (more distill_pc args). Writes outputs/distill/<NAME>_s<seed>_<job>/.
# Eval:  SIM (physx|vbd), POLICY (student|teacher), STUDENT, TAG, START (pinned|rand), CUT (0.015),
#        EVAL_ENVS, EVAL_SEED. Writes docs/results/distill/<TAG>__on_<sim>__<start>_s<seed>.json.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TEACHER="${TEACHER:?set TEACHER to the teacher checkpoint}"
EXPERIMENT="${EXPERIMENT:-rigid_cloth_right_wuji}"
PX_TASK=Isaacsimenvs-RigidCloth-Direct-v0
VBD_TASK=Isaacsimenvs-Cloth-Direct-v0
PXPY=(.venv_isaacsim/bin/python); NEWTONPY=(experiments/newton_py)
CUT="${CUT:-0.015}"
FOLD=("env.cloth.keypoint_tolerance=$CUT" env.termination.success_steps=10 env.cloth.nan_policy=reset)
VBD_EXTRA=(env.cloth.start_height=0.20 env.robot_self_collision=false env.newton.njmax=32768 env.newton.num_substeps=6 env.cloth.substeps=2)
SEED="${SEED:-$(( INDEX + 1 ))}"
echo "[distill] mode=${MODE:-train} teacher=$TEACHER experiment=$EXPERIMENT"
BACKEND=physx provenance distill

train() {  # name envs iters <extra args...>
    local name=$1 envs=$2 iters=$3; shift 3
    local args=(--vis "${VIS:-cams}" --goal "${GOAL:-g0}")
    [ "${NOISE:-0}" = 1 ] && args+=(--noise)
    [ "${G3_NOISE:-0}" = 1 ] && args+=(--g3_noise)
    read -r -a extra <<<"${EXTRA:-}"
    "${PXPY[@]}" -m isaacsimenvs.scripts.distillation.distill_pc --mode train --headless \
        --task "$PX_TASK" --experiment "$EXPERIMENT" --teacher "$TEACHER" --seed "$SEED" \
        --out "outputs/distill/$name" --iters "$iters" "${args[@]}" "$@" "${extra[@]}" \
        "env.scene.num_envs=$envs" "${FOLD[@]}"
}

evaluate() {  # sim policy student out start envs seed <extra args...>
    local sim=$1 policy=$2 student=$3 out=$4 start=$5 envs=$6 seed=$7; shift 7
    local sargs=() stu=()
    [ "$start" = rand ] && sargs=(--randomize)
    [ -n "$student" ] && stu=(--student "$student")
    mkdir -p "$(dirname "$out")"
    if [ "$sim" = vbd ]; then
        "${NEWTONPY[@]}" -m isaacsimenvs.scripts.distillation.distill_pc --mode eval --policy "$policy" "${stu[@]}" \
            --task "$VBD_TASK" --teacher "$TEACHER" --num_envs "$envs" --seed "$seed" \
            --json "$out" "${sargs[@]}" "$@" "${FOLD[@]}" "${VBD_EXTRA[@]}"
    else
        "${PXPY[@]}" -m isaacsimenvs.scripts.distillation.distill_pc --mode eval --headless --policy "$policy" \
            "${stu[@]}" --task "$PX_TASK" --experiment "$EXPERIMENT" --teacher "$TEACHER" --num_envs "$envs" \
            --seed "$seed" --json "$out" "${sargs[@]}" "$@" "${FOLD[@]}"
    fi
}

case "${MODE:-train}" in
train)
    RUN="${NAME:?set NAME}_s${SEED}_${JOB}"
    train "$RUN" "${ENVS:-4096}" "${ITERS:-2000}"; RC=$?
    # a failed start must not release a chained evaluation
    [ -f "outputs/distill/$RUN/student_last.pth" ] || RC=1
    echo "[distill] train exit=$RC"; exit $RC ;;
eval)
    OUT="docs/results/distill/${TAG:?set TAG}__on_${SIM:-vbd}__${START:-pinned}_s${EVAL_SEED:-1}.json"
    evaluate "${SIM:-vbd}" "${POLICY:-student}" "${STUDENT:-}" "$OUT" "${START:-pinned}" "${EVAL_ENVS:-64}" "${EVAL_SEED:-1}" ;;
evalall)  # one student (or the teacher: POLICY=teacher, STUDENT unset): PhysX rand, VBD pinned + rand x 2 seeds
    D="docs/results/distill/${TAG:?set TAG}"; P="${POLICY:-student}"; ST="${STUDENT:-}"
    evaluate physx "$P" "$ST" "${D}__on_physx__rand_s1.json" rand 322 1; echo "[evalall] physx exit=$?"
    for ES in 1 2; do for START in pinned rand; do
        evaluate vbd "$P" "$ST" "${D}__on_vbd__${START}_s${ES}.json" "$START" 64 "$ES"; echo "[evalall] vbd $START s$ES exit=$?"
    done; done ;;
evalx)  # stress / extra seeds: SIMS, PX_STARTS, STARTS, SEEDS, EVAL_ARGS (e.g. --eval_noise); TAG carries a suffix
    D="docs/results/distill/${TAG:?set TAG}"; P="${POLICY:-student}"; ST="${STUDENT:-}"
    read -r -a eargs <<<"${EVAL_ARGS:-}"
    for SIM in ${SIMS:-physx vbd}; do
        if [ "$SIM" = physx ]; then
            for START in ${PX_STARTS:-rand}; do
                evaluate physx "$P" "$ST" "${D}__on_physx__${START}_s1.json" "$START" 322 1 "${eargs[@]}"; echo "[evalx] physx $START exit=$?"
            done
        else
            for ES in ${SEEDS:-1 2}; do for START in ${STARTS:-pinned rand}; do
                evaluate vbd "$P" "$ST" "${D}__on_vbd__${START}_s${ES}.json" "$START" 64 "$ES" "${eargs[@]}"; echo "[evalx] vbd $START s$ES exit=$?"
            done; done
        fi
    done ;;
smoke)
    S="smoke_${JOB}"
    case "${PART:-physx}" in
    physx)
        VIS=cams GOAL=g1 NOISE=1 train "${S}_small" 256 30 --save_every 15 --log_every 5 --beta_iters 10; echo "[smoke] small train exit=$?"
        VIS=cams GOAL=g1 NOISE=1 train "${S}_big" 4096 12 --log_every 4 --save_every 1000; echo "[smoke] big train exit=$?"
        evaluate physx student "outputs/distill/${S}_small/student_last.pth" "docs/results/distill/${S}_student_px.json" rand 64 1 --max_steps 300
        echo "[smoke] student eval exit=$?"
        evaluate physx teacher "" "docs/results/distill/${S}_teacher_px.json" rand 322 1; echo "[smoke] teacher eval exit=$?" ;;
    scene)   # the scene cloud (everything in the box, xyz only)
        SC="--cloud scene --aux_coef 0"; DL="--pc_delay 6 --proprio_delay 3 --action_delay 3"
        GOAL=first NOISE=1 EXTRA="$SC $DL --fk_points 128" train "${S}_small" 256 30 --save_every 15 --log_every 5 --beta_iters 10; echo "[smoke] small train exit=$?"
        evaluate physx student "outputs/distill/${S}_small/student_last.pth" "docs/results/distill/${S}_student_px.json" rand 64 1 --max_steps 300
        echo "[smoke] student px eval exit=$?"
        evaluate vbd student "outputs/distill/${S}_small/student_last.pth" "docs/results/distill/${S}_student_vbd.json" pinned 32 1 --max_steps 200
        echo "[smoke] student vbd eval exit=$?" ;;
    vbd)
        evaluate vbd teacher "" "docs/results/distill/${S}_teacher_vbd_pinned.json" pinned 64 1; echo "[smoke] vbd pinned exit=$?" ;;
    esac ;;
*) echo "MODE must be train, eval, evalall, evalx or smoke" >&2; exit 2 ;;
esac
