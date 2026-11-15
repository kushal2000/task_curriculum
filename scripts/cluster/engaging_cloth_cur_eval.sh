#!/bin/bash
# Independent check of a contact-curriculum checkpoint: deterministic actions, pinned start, the exact
# thin stage config of engaging_fabric_curriculum.sh for FABRIC (real-thickness self-contact, two-stage
# plasticity, full-surface contact, the thin study's stretch/bending, release-persist, 12 s) at
# hand/table contact CONTACT. cloth_pierce_probe.py records each env's first termination reason
# (persist / persist_fail / fold_held / fall / nonfinite / timeout) plus finger piercing, so a held
# fold is told apart from NaN resets and an impaled sheet.
#   FABRIC=cotton CKPTS="a.pth b.pth" TAGS="a b" CONTACT=0.012 sbatch --array=0-1 scripts/cluster/engaging_cloth_cur_eval.sh
# Per-task overrides: FABRICS / CONTACTS (space-separated, indexed like CKPTS) instead of FABRIC / CONTACT.
# Output: docs/results/cloth_cur_eval/<tag>.json
#SBATCH --job-name=cureval
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
PY=.venv_isaaclab3/bin/python
I="${SLURM_ARRAY_TASK_ID:-0}"
read -r -a CK <<<"$CKPTS"; read -r -a TG <<<"$TAGS"
read -r -a FB <<<"${FABRICS:-}"; read -r -a CT <<<"${CONTACTS:-}"
CKPT="${CK[$I]}"; TAG="${TG[$I]}"
FAB="${FB[$I]:-${FABRIC:-cotton}}"; T_CONTACT="${CT[$I]:-${CONTACT:-0.012}}"
OUT=docs/results/cloth_cur_eval; mkdir -p "$OUT"
# Same table as engaging_fabric_finetune.sh THIN=1: thickness, yield, set.
declare -A TH=([big_napkin]="0.0003 20 80" [chiffon]="0.0001 100 80" [cotton]="0.0003 20 80"
               [denim]="0.00035 100 80" [napkin]="0.00015 20 120" [silk]="0.0003 42 120" [towel]="0.001 100 80")
read -r T Y SET <<<"${TH[$FAB]:?no thin preset for $FAB}"
# Self-contact margin as engaging_fabric_finetune.sh THIN_MARGIN (default 3 mm; 1.5x = old 1.5 x T).
M="${THIN_MARGIN:-0.003}"; [ "$M" = 1.5x ] && M=$($PY -I -c "print(1.5 * $T)")
read -r -a PRE <<<"$($PY -I isaacsimenvs/tasks/cloth/utils/fabrics.py "${FAB}_measured" --plastic)"
SET_OV=(env.cloth.thickness=$T_CONTACT env.cloth.self_contact_radius=$T env.cloth.self_contact_margin=$M
        env.cloth.plastic_yield_curvature=$Y.0 env.cloth.plastic_set_curvature=$SET.0 env.cloth.full_surface_contact=true)
case $FAB in
    chiffon) SET_OV+=(env.cloth.tri_ke=100.0 env.cloth.tri_ka=100.0 env.cloth.edge_ke=0.0012076974591784867);;
    cotton|silk) SET_OV+=(env.cloth.tri_ke=100.0 env.cloth.tri_ka=100.0);;
esac
KEEP=(); for o in "${PRE[@]}"; do dup=0; for v in "${SET_OV[@]}"; do [ "${o%%=*}" = "${v%%=*}" ] && dup=1; done; [ $dup = 0 ] && KEEP+=("$o"); done
CFG=("${KEEP[@]}" "${SET_OV[@]}" env.cloth.nan_policy=reset env.cloth.keypoint_tolerance=0.015
     env.termination.success_steps=10 env.cloth.release_persist=true env.episode_length_s=12.0)
echo "[cureval] $TAG fabric=$FAB ckpt=$CKPT contact=$T_CONTACT: ${CFG[*]}"
scripts/newton_py -m scripts.analysis.cloth_pierce_probe --checkpoint "$CKPT" \
    --policy_config checkpoints/config.yaml --num_envs "${ENVS:-128}" --max_steps 730 --out "$OUT/$TAG.json" \
    "${CFG[@]}" 2>&1 | grep -E "^\[(pierce|grip)|Error|Traceback|line [0-9]+|ValueError" | tail -15
