#!/bin/bash
# Score a policy on the VBD cloth set up as a MEASURED fabric, with bos14_fabric_eval.sh's protocol:
# 1.5 cm cutoff, pinned, 32 envs x 900 steps, a fold counts once held 10 steps, eval seeds 3-8
# (192 episodes), plasticity ON (fabrics.py --plastic). Index = fabric * NCOEF + coef.
#
#   # baseline: the untuned prior on every fabric, block 5 (coef 0)
#   TAG=flat7DRABC_s1 CKPT=checkpoints/<prior>.pth sbatch --array=0-6 scripts/cluster/engaging_fabric_eval.sh
#   # block probe of each fabric's own finetune (CKPT unset -> that fabric's final ftfab checkpoint)
#   TAG=ft FT_JOB=<array job> COEFS="50 40 30 20 10 0" SEEDS=3 sbatch --array=0-41 scripts/cluster/engaging_fabric_eval.sh
#   # final: the chosen block, all seeds (one fabric: FABRICS=cotton --array=0)
#   TAG=ft FT_JOB=<array job> COEFS=0 sbatch --array=0-6 scripts/cluster/engaging_fabric_eval.sh
#   # a tagged finetune (engaging_fabric_finetune.sh RUN_TAG=rel3s): FT_TAG=rel3s FT_JOB=<its job>
#
# Output: docs/results/px_matrix/fabric_<TAG>__<fabric>_measured_plast_c<coef>_s<es>.json;
# existing files are skipped, so a requeued task only redoes what it had not finished.
#SBATCH --job-name=fabeval
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
OUTDIR=docs/results/px_matrix
mkdir -p slurm_logs "$OUTDIR"
export PYTHONUNBUFFERED=1

TAG="${TAG:?set TAG=<short name>}"
read -r -a FAB_LIST <<<"${FABRICS:-big_napkin chiffon cotton denim napkin silk towel}"
# COEFS=best: index = fabric; use the block with the most held folds in this TAG's seed-3 probe
# (ties -> the smaller coef, i.e. toward the c0 leader). Lets the final eval chain on the probe
# with --dependency=afterok, no one watching.
if [ "${COEFS:-}" = best ]; then
    F="${FAB_LIST[${SLURM_ARRAY_TASK_ID:-0}]:?no fabric for index ${SLURM_ARRAY_TASK_ID:-0}}"
    COEFS=$(.venv_isaaclab3/bin/python -I -c "
import json, sys
best = None
for c in (0, 10, 20, 30, 40, 50):
    try: h = json.load(open(f'$OUTDIR/fabric_${TAG}__${F}_measured_plast_c{c}_s3.json'))['termination_reasons'].get('fold_held', 0)
    except (OSError, ValueError): continue
    if best is None or h > best[1]: best = (c, h)
print(best[0] if best else '')")
    [ -n "$COEFS" ] || { echo "[fabeval] COEFS=best: no seed-3 probe files for $F" >&2; exit 1; }
    echo "[fabeval] COEFS=best -> c$COEFS for $F"
fi
read -r -a COEF_LIST <<<"${COEFS:-0}"
I="${SLURM_ARRAY_TASK_ID:-0}"
FABRIC="${FAB_LIST[$((I / ${#COEF_LIST[@]}))]:?no fabric for index $I}"
COEF="${COEF_LIST[$((I % ${#COEF_LIST[@]}))]}"
PRESET="${FABRIC}_measured"

if [ -z "${CKPT:-}" ]; then
    : "${FT_JOB:?set CKPT, or FT_JOB to the engaging_fabric_finetune.sh array job}"
    CKPT=$(.venv_isaaclab3/bin/python -I -c "import sys; sys.path.insert(0, 'scripts/analysis'); from pathlib import Path; from rigid_cloth_eval_manifest import find_checkpoint; print(find_checkpoint(sys.argv[1], Path('outputs'), 'last') or '')" "0_ftfab${FT_TAG:-}_${FABRIC}_${FT_JOB}")
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "[fabeval] no checkpoint for $FABRIC (${CKPT:-none})" >&2; exit 1; }

read -r -a FAB_OV <<<"$(.venv_isaaclab3/bin/python -I isaacsimenvs/tasks/cloth/utils/fabrics.py "$PRESET" --plastic)"
[ "${#FAB_OV[@]}" -gt 0 ] || { echo "[fabeval] no overrides for $PRESET" >&2; exit 1; }

echo "[fabeval] tag=$TAG fabric=$PRESET coef=$COEF ckpt=$CKPT host=$(hostname -s) commit=$(git rev-parse --short HEAD)"
for ES in ${SEEDS:-3 4 5 6 7 8}; do
    OUT="$OUTDIR/fabric_${TAG}__${PRESET}_plast_c${COEF}_s${ES}.json"
    [ -s "$OUT" ] && { echo "[fabeval] exists: $OUT"; continue; }
    scripts/newton_py -m isaacsimenvs.eval.episodes --task Isaacsimenvs-Cloth-Direct-v0 \
        --checkpoint "$CKPT" --policy_config "${POLICY_CFG:-checkpoints/config.yaml}" \
        --sapg_expl_coef "$COEF" --num_envs "${NENVS:-32}" --max_steps "${MAXSTEPS:-900}" --success_tolerance 0.01 \
        --num_assets_per_type 1 --single_variant --seed "$ES" --out "$OUT" \
        env.cloth.keypoint_tolerance=0.015 env.cloth.nan_policy=reset \
        env.termination.success_steps="${SUCCESS_STEPS:-10}" "${FAB_OV[@]}" ${EVAL_ARGS:-} \
        || echo "[fabeval] $OUT FAILED"
done
echo "[fabeval] === summary ==="
.venv_isaaclab3/bin/python -I scripts/analysis/cloth_fold_summary.py --label "${TAG}__${PRESET}_c${COEF}" \
    "$OUTDIR"/fabric_${TAG}__${PRESET}_plast_c${COEF}_s*.json
