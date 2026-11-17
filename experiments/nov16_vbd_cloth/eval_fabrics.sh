#!/bin/bash
# Rigid-checkpoint eval over fabric presets, one job, sequentially (Engaging, one GPU).
#
#   PRESET_SUFFIX=_measured HOLD=1 OUTDIR=outputs/vbd_eval/legacy sbatch experiments/nov16_vbd_cloth/eval_fabrics.sh chiffon silk ...
#
# PRESET_SUFFIX picks isaacsimenvs/cfg/fabric/<fabric><suffix>.yaml (_measured = today's 7x7 cloth with
# its plasticity; _vbd = the irregular-mesh hinge-rheology cloth). HOLD=1 scores a fold only once the
# keypoints have stayed within tolerance for 180 consecutive steps (termination.success_steps=180);
# HOLD=0 scores the first step they touch it (success_steps=1). Contact is fixed for every preset:
# 8 mm particle radius (thickness 16 mm) and 2 mm self-contact radius. TOL = keypoint tolerance [m] (0.04).
# TASK / CKPT / CFG pass through to eval.sh (another robot or policy).
#SBATCH --job-name=vbdevalf
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
SUFFIX="${PRESET_SUFFIX:-_measured}"
HOLDS="${HOLD:-0}"   # "0", "1", or "0 1" (touch then hold in one job)
OUTDIR="${OUTDIR:?set OUTDIR}"
mkdir -p "$OUTDIR"
rc=0
for HOLD in $HOLDS; do
STEPS=1; [[ "$HOLD" == 1 ]] && STEPS=180
for fab in "$@"; do
    OUT_FILE="$OUTDIR/${fab}_hold${HOLD}.json"
    [[ -f "$OUT_FILE" && "${RESUME:-1}" == 1 ]] && { echo "[evalf] $OUT_FILE exists, skipping"; continue; }  # requeue-safe
    mapfile -t OV < <(.venv_isaaclab3/bin/python isaacsimenvs/tasks/cloth/utils/fabrics.py "${fab}${SUFFIX}" --plastic | tr ' ' '\n' | grep -v -E '^$|self_contact_radius|self_contact_margin|thickness')
    OV+=(env.cloth.thickness=0.016 env.cloth.self_contact_radius=0.002 env.cloth.self_contact_margin=0.003
         env.termination.success_steps=$STEPS env.cloth.keypoint_tolerance=${TOL:-0.04})
    # EXTRA_OV: more Hydra overrides, space separated (e.g. a policy's training-time settings)
    [[ -n "${EXTRA_OV:-}" ]] && OV+=($EXTRA_OV)
    echo "[evalf] ${fab}${SUFFIX} hold=$HOLD overrides: ${OV[*]}"
    # CKPT_MAP: a file of "<fabric> <checkpoint>" lines, one policy per fabric (e.g. per-fabric finetunes)
    if [[ -n "${CKPT_MAP:-}" ]]; then export CKPT="$(awk -v f="$fab" '$1 == f {print $2}' "$CKPT_MAP")"; echo "[evalf] $fab ckpt $CKPT"; fi
    OUT="$OUT_FILE" bash experiments/nov16_vbd_cloth/eval.sh "${OV[@]}" || rc=$?
done
done
exit $rc
