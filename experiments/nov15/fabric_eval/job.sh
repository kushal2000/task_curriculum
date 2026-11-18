#!/bin/bash
# One array task of the fabric eval: one policy x one task (fold / unfold) x one measured fabric,
# eval seeds 3-8 (6 x 32 = 192 episodes). Submitted by submit.sh, which fixes the checkpoints.
#
#   job.sh <tag> <checkpoint>      (array index = task * 7 + fabric)
#
# Protocol (the 2026-11 measured-fabric eval, engaging_fabric_eval.sh): VBD sheet on Newton,
# measured fabric preset with its plasticity, no DR (eval_play.py pins it), deterministic actions,
# SAPG block coef 0, 1.5 cm keypoint cutoff, a fold counts once held 10 steps, 900 steps.
#SBATCH --job-name=fabeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
TASK_ID=Isaacsimenvs-ClothFlexivWujiRightReal-Direct-v0
POLICY_TAG="$1"; CKPT="$2"
source experiments/_common.sh --task "$TASK_ID"

FABRICS=(big_napkin chiffon cotton denim napkin silk towel)
TASKS=(fold unfold)
I="${SLURM_ARRAY_TASK_ID:-0}"
FABRIC="${FABRICS[$((I % 7))]}"; CLOTH_TASK="${TASKS[$((I / 7))]:?index $I out of range}"
OUT=experiments/nov15/fabric_eval/results
adopt_log "$OUT"
[[ -f "$CKPT" ]] || { echo "[fabeval] no checkpoint $CKPT" >&2; exit 1; }
read -r -a FAB_OV <<<"$("${PY[@]}" -m isaacsimenvs.tasks.cloth.deformable_env_cfg "${FABRIC}_measured" --plastic)"
[[ "${#FAB_OV[@]}" -gt 0 ]] || { echo "[fabeval] no overrides for ${FABRIC}_measured" >&2; exit 1; }
echo "[fabeval] policy=$POLICY_TAG task=$CLOTH_TASK fabric=${FABRIC}_measured ckpt=$CKPT"

gpu_log_start
for ES in ${SEEDS:-3 4 5 6 7 8}; do
    JSON="$OUT/${POLICY_TAG}__${CLOTH_TASK}__${FABRIC}_s${ES}.json"
    [[ -s "$JSON" ]] && { echo "[fabeval] exists: $JSON"; continue; }
    "${PY[@]}" isaacsimenvs/scripts/evaluation/eval_play.py --task "$TASK_ID" \
        --checkpoint "$CKPT" --policy_config pretrained_policy/flexiv_wuji/config.yaml \
        --sapg_expl_coef 0 --num_envs "${NENVS:-32}" --max_steps "${MAXSTEPS:-900}" \
        --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --seed "$ES" --out "$JSON" \
        env.task.mode="$CLOTH_TASK" env.task.keypoint_tolerance=0.015 env.task.nan_policy=reset \
        env.termination.success_steps=10 "${FAB_OV[@]}" \
        || echo "[fabeval] $JSON FAILED"
done
gpu_log_stop
