#!/bin/bash
# Point-cloud student distillation (DAgger, isaacsimenvs/scripts/distillation/distill_pc.py) on the PhysX
# fold pool. Training only: evaluate a student with experiments/eval.sh like any fold policy (its recipe
# is read from the args.json next to it).
#
#   TEACHER=<pth> NAME=<name> [EXPERIMENT=<point-cloud recipe>] experiments/distill.sh
#   TEACHER=<pth> MODE=smoke experiments/distill.sh          (30 iterations at 256 envs)
#
# TEACHER (required): the state policy's checkpoint. EXPERIMENT: a point-cloud recipe (env.obs.object:
# pointcloud: the cloud, its DR), default experiments/nov16_pointcloud_dagger/rigidcloth_pc_dagger.
# NAME, ENVS (4096), ITERS (2000), SEED (default INDEX + 1), EXTRA (more distill_pc args or hydra
# overrides, e.g. env.obs.pointcloud.goal=none). Writes outputs/distill/<NAME>_s<seed>_<job>/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
TEACHER="${TEACHER:?set TEACHER to the teacher checkpoint}"
EXPERIMENT="${EXPERIMENT:-experiments/nov16_pointcloud_dagger/rigidcloth_pc_dagger}"
FOLD=(env.cloth.keypoint_tolerance=0.015 env.termination.success_steps=10 env.cloth.nan_policy=reset)
SEED="${SEED:-$(( INDEX + 1 ))}"
echo "[distill] mode=${MODE:-train} teacher=$TEACHER experiment=$EXPERIMENT"
BACKEND=physx provenance distill

train() {  # name envs iters <extra args...>
    local name=$1 envs=$2 iters=$3; shift 3
    read -r -a extra <<<"${EXTRA:-}"
    .venv_isaacsim/bin/python -m isaacsimenvs.scripts.distillation.distill_pc --headless \
        --task Isaacsimenvs-RigidCloth-Direct-v0 --experiment "$EXPERIMENT" --teacher "$TEACHER" --seed "$SEED" \
        --out "outputs/distill/$name" --iters "$iters" "$@" "${extra[@]}" \
        "env.scene.num_envs=$envs" "${FOLD[@]}"
}

case "${MODE:-train}" in
train)
    RUN="${NAME:?set NAME}_s${SEED}_${JOB}"
    train "$RUN" "${ENVS:-4096}" "${ITERS:-2000}"; RC=$?
    # a failed start must not release a chained evaluation
    [ -f "outputs/distill/$RUN/student_last.pth" ] || RC=1
    echo "[distill] train exit=$RC"; exit $RC ;;
smoke)
    train "smoke_${JOB}" 256 30 --save_every 15 --log_every 5 --beta_iters 10
    echo "[distill] smoke exit=$?" ;;
eval|evalall|evalx)
    echo "[distill] MODE=$MODE was removed on 2026-11-22: evaluate students with experiments/eval.sh" >&2; exit 2 ;;
*) echo "MODE must be train or smoke" >&2; exit 2 ;;
esac
