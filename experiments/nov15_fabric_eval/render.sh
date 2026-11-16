#!/bin/bash
# Videos for the fabric eval: each policy x {rigid plates (PhysX pool), cotton (VBD)} x {fold, unfold},
# one scored episode per clip (--stop_on_episode_end), same protocol as job.sh: no DR, deterministic,
# SAPG coef 0, 1.5 cm cutoff held 10 steps. The rigid clips film env 0 = the box2-mid sheet, the shape
# closest to the VBD square sheet. Runs on the local GPU; every call is appended to submissions.log.
#
#   experiments/nov15_fabric_eval/render.sh <prior ckpt> <scratch ckpt>
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
D=experiments/nov15_fabric_eval
# Videos are run artifacts (gitignored): debug_outputs/videos/<month>/<date>_fabric_eval/, like default_video_path.
OUT=debug_outputs/videos/$(date +%b | tr A-Z a-z)/$(date +%F)_fabric_eval; mkdir -p "$OUT"
declare -A CK=([prior]="$1" [scratch]="$2")
CAM=(--cam_offset 0.6 -0.6 1.05 --look_at 0.0 0.0 0.5)
COMMON=(--policy_config pretrained_policy/flexiv_wuji/config.yaml --sapg_expl_coef 0 --steps "${STEPS:-900}"
        --stop_on_episode_end "${CAM[@]}")
EVAL_OV=(env.cloth.keypoint_tolerance=0.015 env.termination.success_steps=10 env.cloth.nan_policy=reset)
read -r -a FAB <<<"$(.venv/isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py cotton_measured --plastic)"
{ echo "## $(date '+%Y-%m-%d %H:%M:%S') host=$(hostname -s) commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
  echo "\$ $D/render.sh $*"; } >>"$D/submissions.log"

run() {  # name, command...
    local name="$1"; shift
    echo "render $name: $*" >>"$D/submissions.log"
    "$@" >"$OUT/$name.log" 2>&1 && echo "[render] $name ok" || echo "[render] $name FAILED (see $OUT/$name.log)"
}
for P in prior scratch; do
    run "${P}__rigid__fold" .venv/isaacsim/bin/python scripts/visualization/render_physx.py --headless \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment rigid_cloth_flexiv_wuji_full_dr \
        --checkpoint "${CK[$P]}" --single_variant --num_envs 8 "${COMMON[@]}" --out "$OUT/${P}__rigid__fold.mp4" \
        env.cloth.task=fold env.fold_pool.init_fold_prob=0.0 "${EVAL_OV[@]}"
    run "${P}__rigid__unfold" .venv/isaacsim/bin/python scripts/visualization/render_physx.py --headless \
        --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0 --experiment rigid_cloth_flexiv_wuji_full_dr \
        --checkpoint "${CK[$P]}" --single_variant --num_envs 8 "${COMMON[@]}" --out "$OUT/${P}__rigid__unfold.mp4" \
        env.cloth.task=unfold "${EVAL_OV[@]}"
    for T in fold unfold; do
        run "${P}__cotton__$T" xvfb-run -a .venv/isaaclab3/bin/python scripts/visualization/render_newton.py \
            --task Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0 --checkpoint "${CK[$P]}" --num_envs 4 \
            "${COMMON[@]}" --out "$OUT/${P}__cotton__$T.mp4" env.cloth.task=$T "${EVAL_OV[@]}" "${FAB[@]}"
    done
done
