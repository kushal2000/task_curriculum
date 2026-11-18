#!/bin/bash
# Point-cloud student distillation (isaacsimenvs/distill/distill_pc.py) on the PhysX fold pool, or an
# evaluation of a student / the teacher on PhysX or the VBD cloth (Newton).
#   NAME=B_g1 VIS=cams GOAL=g1 sbatch scripts/cluster/bos14_distill_pc.sh                 (train)
#   MODE=eval SIM=vbd POLICY=student STUDENT=<pth> TAG=<tag> START=rand sbatch ...          (eval)
#   MODE=evalall STUDENT=<pth> TAG=<tag> sbatch --time=03:00:00 ...   (PhysX rand + VBD pinned/rand x 2 seeds)
#   TEACHER=<pth> EXPERIMENT=<recipe>: another teacher (default flat7RW_base epoch 1500).
#   MODE=smoke sbatch --time=01:00:00 scripts/cluster/bos14_distill_pc.sh
# Train vars: NAME, VIS (cams|none), GOAL (g0|g3|g1|gl), NOISE=1, G3_NOISE=1, ENVS, ITERS, SEED, EXTRA.
# Eval vars: SIM (physx|vbd), POLICY, STUDENT, TAG, START (pinned|rand), CUT (0.015), EVAL_ENVS, EVAL_SEED.
#SBATCH --job-name=distill_pc
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,059,062,086,092,104,105,109,110,112,120,121,123]
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
mkdir -p slurm_logs
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
# Isaac Lab logs to $TMPDIR/isaaclab, and on some nodes /tmp/isaaclab belongs to another user.
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
TEACHER="${TEACHER:-outputs/2026-11-11/02-51-52/0_pxpool_flat7RW_base_s1_277681/nn/snap_0_pxpool_flat7RW_base_s1_277681_ep_1500.pth}"
EXPERIMENT="${EXPERIMENT:-flat7RW_base}"   # the teacher's recipe (PhysX env)
PX_TASK=Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0
VBD_TASK=Isaacsimenvs-ClothFlexivWujiRightHLRL-Direct-v0
CUT="${CUT:-0.015}"
FOLD=("env.cloth.keypoint_tolerance=$CUT" env.termination.success_steps=10 env.cloth.nan_policy=reset)
VBD_EXTRA=(env.cloth.start_height=0.20 env.robot_self_collision=false env.newton.njmax=32768
           env.newton.num_substeps=6 env.cloth.substeps=2)
echo "[distill_pc] mode=${MODE:-train} host=$(hostname -s) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

train() {  # name envs iters <extra args...>
    local name=$1 envs=$2 iters=$3; shift 3
    local args=(--vis "${VIS:-cams}" --goal "${GOAL:-g0}")
    [ "${NOISE:-0}" = 1 ] && args+=(--noise)
    [ "${G3_NOISE:-0}" = 1 ] && args+=(--g3_noise)
    read -r -a extra <<<"${EXTRA:-}"
    .venv_isaacsim/bin/python -m isaacsimenvs.distill.distill_pc --mode train --headless \
        --task "$PX_TASK" --experiment "$EXPERIMENT" --teacher "$TEACHER" --seed "${SEED:-1}" \
        --out "outputs/distill/$name" --iters "$iters" "${args[@]}" "$@" "${extra[@]}" \
        "env.scene.num_envs=$envs" "${FOLD[@]}"
}

evaluate() {  # sim policy student out start envs seed <extra args...>
    local sim=$1 policy=$2 student=$3 out=$4 start=$5 envs=$6 seed=$7; shift 7
    local sargs=() stu=()
    [ "$start" = rand ] && sargs=(--randomize)
    [ -n "$student" ] && stu=(--student "$student")
    if [ "$sim" = vbd ]; then
        scripts/newton_py -m isaacsimenvs.distill.distill_pc --mode eval --policy "$policy" "${stu[@]}" \
            --task "$VBD_TASK" --teacher "$TEACHER" --num_envs "$envs" --seed "$seed" \
            --json "$out" "${sargs[@]}" "$@" "${FOLD[@]}" "${VBD_EXTRA[@]}"
    else
        .venv_isaacsim/bin/python -m isaacsimenvs.distill.distill_pc --mode eval --headless --policy "$policy" \
            "${stu[@]}" --task "$PX_TASK" --experiment "$EXPERIMENT" --teacher "$TEACHER" --num_envs "$envs" \
            --seed "$seed" --json "$out" "${sargs[@]}" "$@" "${FOLD[@]}"
    fi
}

case "${MODE:-train}" in
train)
    train "${NAME:?set NAME}_s${SEED:-1}_${SLURM_JOB_ID}" "${ENVS:-4096}" "${ITERS:-2000}"; RC=$?
    # a failed start must not release the chained evaluation (afterok)
    [ -f "outputs/distill/${NAME}_s${SEED:-1}_${SLURM_JOB_ID}/student_last.pth" ] || RC=1
    echo "[distill_pc] train exit=$RC"; exit $RC ;;
eval)
    OUT="docs/results/distill/${TAG:?set TAG}__on_${SIM:-vbd}__${START:-pinned}_s${EVAL_SEED:-1}.json"
    evaluate "${SIM:-vbd}" "${POLICY:-student}" "${STUDENT:-}" "$OUT" "${START:-pinned}" "${EVAL_ENVS:-64}" "${EVAL_SEED:-1}" ;;
evalall)  # one student (or the teacher: POLICY=teacher, STUDENT unset): PhysX rand, VBD pinned + rand x 2 seeds
    D="docs/results/distill/${TAG:?set TAG}"; P="${POLICY:-student}"; ST="${STUDENT:-}"
    evaluate physx "$P" "$ST" "${D}__on_physx__rand_s1.json" rand 322 1; echo "[evalall] physx exit=$?"
    for ES in 1 2; do for START in pinned rand; do
        evaluate vbd "$P" "$ST" "${D}__on_vbd__${START}_s${ES}.json" "$START" 64 "$ES"; echo "[evalall] vbd $START s$ES exit=$?"
    done; done ;;
evalx)  # stress / extra seeds: SIMS="physx vbd", STARTS, SEEDS, EVAL_ARGS (e.g. --eval_noise); TAG carries a suffix
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
    S="smoke_${SLURM_JOB_ID}"
    case "${PART:-physx}" in
    physx)
        VIS=cams GOAL=g1 NOISE=1 train "${S}_small" 256 30 --save_every 15 --log_every 5 --beta_iters 10; echo "[smoke] small train exit=$?"
        VIS=cams GOAL=g1 NOISE=1 train "${S}_big" 4096 12 --log_every 4 --save_every 1000; echo "[smoke] big train exit=$?"
        VIS=cams GOAL=g3 G3_NOISE=1 train "${S}_g3" 256 6 --log_every 3 --save_every 1000; echo "[smoke] g3 train exit=$?"
        VIS=none GOAL=gl train "${S}_gl" 256 6 --log_every 3 --save_every 1000; echo "[smoke] gl train exit=$?"
        evaluate physx student "outputs/distill/${S}_small/student_last.pth" "docs/results/distill/${S}_student_px.json" rand 64 1 --max_steps 300
        echo "[smoke] student eval exit=$?"
        evaluate physx teacher "" "docs/results/distill/${S}_teacher_px.json" rand 322 1; echo "[smoke] teacher eval exit=$?" ;;
    scene)   # the scene cloud (everything in the box, xyz only): train small + big, PhysX and VBD evals
        SC="--cloud scene --aux_coef 0"; DL="--pc_delay 6 --proprio_delay 3 --action_delay 3"
        GOAL=first NOISE=1 EXTRA="$SC $DL --fk_points 128" train "${S}_small" 256 30 --save_every 15 --log_every 5 --beta_iters 10; echo "[smoke] small train exit=$?"
        GOAL=none NOISE=1 EXTRA="$SC $DL" train "${S}_big" "${BIG_ENVS:-4096}" 12 --log_every 4 --save_every 1000; echo "[smoke] big train exit=$?"
        GOAL=none EXTRA="$SC" train "${S}_clean" 256 6 --log_every 3 --save_every 1000; echo "[smoke] clean train exit=$?"
        evaluate physx student "outputs/distill/${S}_small/student_last.pth" "docs/results/distill/${S}_student_px.json" rand 64 1 --max_steps 300 --dump_cloud "outputs/distill/${S}_small/cloud_px.npz"
        echo "[smoke] student px eval exit=$?"
        evaluate physx student "outputs/distill/${S}_clean/student_last.pth" "docs/results/distill/${S}_clean_noise_px.json" rand 64 1 --max_steps 200 --eval_noise $DL
        echo "[smoke] clean student, noise eval exit=$?"
        evaluate vbd student "outputs/distill/${S}_small/student_last.pth" "docs/results/distill/${S}_student_vbd.json" pinned 32 1 --max_steps 200 --dump_cloud "outputs/distill/${S}_small/cloud_vbd.npz"
        echo "[smoke] student vbd eval exit=$?" ;;
    vbd)
        evaluate vbd teacher "" "docs/results/distill/${S}_teacher_vbd_pinned.json" pinned 64 1; echo "[smoke] vbd pinned exit=$?"
        evaluate vbd teacher "" "docs/results/distill/${S}_teacher_vbd_rand.json" rand 64 1; echo "[smoke] vbd rand exit=$?"
        [ -n "${STUDENT:-}" ] && evaluate vbd student "$STUDENT" "docs/results/distill/${S}_student_vbd.json" pinned 32 1 --max_steps 200
        echo "[smoke] vbd done" ;;
    esac ;;
*) echo "MODE must be train, eval, evalall or smoke"; exit 2 ;;
esac
echo "[distill_pc] exit=$?"
